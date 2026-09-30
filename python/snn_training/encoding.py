"""Dependency-free input encoders for hardware-oriented spike streams.

Every encoder returns events in the timestamp/neuron-ID order consumed by the
RTL interface.  The helper functions intentionally use 8-bit lookup tables for
non-linear mappings.  The PyTorch path builds its tensors from the same tables,
so host encoding and QAT cannot disagree at floating-point bin boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

from snn_reference import SpikeEvent


_RATE_RANDOM_MASK = (1 << 31) - 1
_RATE_RANDOM_DENOMINATOR = float(1 << 31)

INPUT_ENCODINGS = (
    "latency",
    "rate",
    "sparse-latency",
    "nonlinear-latency",
    "deterministic-rate",
    "top-k-rank-order",
    "burst",
)

VARIABLE_EVENT_ENCODINGS = frozenset(
    {
        "rate",
        "sparse-latency",
        "deterministic-rate",
        "top-k-rank-order",
        "burst",
    }
)


def _validate_pixels(
    pixels: Sequence[int], expected_pixels: int | None
) -> tuple[int, ...]:
    if expected_pixels is not None and len(pixels) != expected_pixels:
        raise ValueError(f"expected {expected_pixels} pixels, got {len(pixels)}")
    normalized = tuple(int(pixel) for pixel in pixels)
    invalid = next((pixel for pixel in normalized if not 0 <= pixel <= 255), None)
    if invalid is not None:
        raise ValueError(f"pixel={invalid} is outside unsigned 8-bit range")
    return normalized


def rate_random_u31(
    seed: int, timestamp: int, neuron_id: int, *, sample_index: int = 0
) -> int:
    """Return a stable counter-based pseudo-random integer in ``[0, 2**31)``.

    The same integer-only mixing function is implemented by the PyTorch tensor
    encoder. Its result therefore depends only on the supplied coordinates and
    not on iteration order, batch size or global random-number-generator state.
    """
    if timestamp < 0 or neuron_id < 0 or sample_index < 0:
        raise ValueError("rate-encoding coordinates must be non-negative")
    value = (
        ((int(seed) * 1_103_515_245) & _RATE_RANDOM_MASK)
        ^ (((sample_index + 1) * 73_856_093) & _RATE_RANDOM_MASK)
        ^ (((timestamp + 1) * 19_349_663) & _RATE_RANDOM_MASK)
        ^ (((neuron_id + 1) * 83_492_791) & _RATE_RANDOM_MASK)
    )
    value ^= (value << 13) & _RATE_RANDOM_MASK
    value ^= value >> 17
    value ^= (value << 5) & _RATE_RANDOM_MASK
    return value & _RATE_RANDOM_MASK


def latency_timestamp(pixel: int, timesteps: int) -> int:
    """Map an unsigned 8-bit pixel to an inverse-intensity timestamp.

    White (255) spikes at timestamp zero and black (0) at the final timestamp.
    Integer floor division makes the mapping identical on host and target.
    """
    if not 0 <= pixel <= 255:
        raise ValueError(f"pixel={pixel} is outside unsigned 8-bit range")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    return ((255 - pixel) * (timesteps - 1)) // 255


def nonlinear_latency_timestamp(pixel: int, timesteps: int, gamma: float) -> int:
    """Map intensity to latency with ``(1 - (pixel/255)**gamma)``."""
    if not 0 <= pixel <= 255:
        raise ValueError(f"pixel={pixel} is outside unsigned 8-bit range")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("latency_gamma must be finite and positive")
    if gamma == 1.0:
        return latency_timestamp(pixel, timesteps)
    if pixel == 0:
        return timesteps - 1
    if pixel == 255:
        return 0
    return min(
        timesteps - 1,
        int((1.0 - math.pow(pixel / 255.0, gamma)) * (timesteps - 1)),
    )


@lru_cache(maxsize=64)
def nonlinear_latency_lut(timesteps: int, gamma: float) -> tuple[int, ...]:
    """Return the canonical 256-entry non-linear timestamp lookup table."""
    return tuple(
        nonlinear_latency_timestamp(pixel, timesteps, gamma)
        for pixel in range(256)
    )


def _validate_threshold(threshold: int) -> int:
    threshold = int(threshold)
    if not 0 <= threshold <= 255:
        raise ValueError("encoding_threshold must be in [0, 255]")
    return threshold


def _sorted(events: list[SpikeEvent]) -> tuple[SpikeEvent, ...]:
    events.sort(key=lambda event: (event.timestamp, event.neuron_id))
    return tuple(events)


def encode_latency_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Encode exactly one event per pixel and return RTL-ready sorted events."""
    normalized = _validate_pixels(pixels, expected_pixels)

    events = [
        SpikeEvent(latency_timestamp(pixel, timesteps), neuron_id)
        for neuron_id, pixel in enumerate(normalized)
    ]
    return _sorted(events)


def encode_sparse_latency_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    encoding_threshold: int = 0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Emit one linearly timed event only for pixels above ``threshold``."""
    normalized = _validate_pixels(pixels, expected_pixels)
    threshold = _validate_threshold(encoding_threshold)
    return _sorted(
        [
            SpikeEvent(latency_timestamp(pixel, timesteps), neuron_id)
            for neuron_id, pixel in enumerate(normalized)
            if pixel > threshold
        ]
    )


def encode_nonlinear_latency_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    latency_gamma: float = 2.0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Emit exactly one event per pixel using a gamma latency curve."""
    normalized = _validate_pixels(pixels, expected_pixels)
    timestamps = nonlinear_latency_lut(timesteps, latency_gamma)
    return _sorted(
        [
            SpikeEvent(timestamps[pixel], neuron_id)
            for neuron_id, pixel in enumerate(normalized)
        ]
    )


def encode_rate_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    rate_scale: float = 1.0,
    seed: int = 0,
    sample_index: int = 0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Encode pixels as a reproducible Bernoulli rate-coded event stream.

    At every timestep, input neuron ``i`` emits at most one event with
    ``P(spike) = clamp(rate_scale * pixel_i / 255, 0, 1)``. Events are returned
    in the timestamp-then-neuron-ID order required by the RTL interface.
    """
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not math.isfinite(rate_scale) or rate_scale < 0.0:
        raise ValueError("rate_scale must be finite and non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    normalized = _validate_pixels(pixels, expected_pixels)

    events: list[SpikeEvent] = []
    for timestamp in range(timesteps):
        for neuron_id, pixel in enumerate(normalized):
            probability = min(1.0, rate_scale * pixel / 255.0)
            if probability <= 0.0:
                continue
            if probability >= 1.0 or (
                rate_random_u31(
                    seed,
                    timestamp,
                    neuron_id,
                    sample_index=sample_index,
                )
                / _RATE_RANDOM_DENOMINATOR
                < probability
            ):
                events.append(SpikeEvent(timestamp, neuron_id))
    return tuple(events)


def deterministic_rate_count(pixel: int, timesteps: int, rate_scale: float) -> int:
    """Return the rounded, clipped deterministic spike count for one pixel."""
    if not 0 <= pixel <= 255:
        raise ValueError(f"pixel={pixel} is outside unsigned 8-bit range")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not math.isfinite(rate_scale) or rate_scale < 0.0:
        raise ValueError("rate_scale must be finite and non-negative")
    return min(timesteps, int(rate_scale * pixel * timesteps / 255.0 + 0.5))


def deterministic_rate_timestamps(count: int, timesteps: int) -> tuple[int, ...]:
    """Place ``count`` spikes at the centers of equal temporal partitions."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not 0 <= count <= timesteps:
        raise ValueError("count must be between zero and timesteps")
    if count == 0:
        return ()
    return tuple(((2 * index + 1) * timesteps) // (2 * count) for index in range(count))


@lru_cache(maxsize=64)
def deterministic_rate_lut(
    timesteps: int, rate_scale: float
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical spike timestamps for every unsigned 8-bit value."""
    return tuple(
        deterministic_rate_timestamps(
            deterministic_rate_count(pixel, timesteps, rate_scale), timesteps
        )
        for pixel in range(256)
    )


def encode_deterministic_rate_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    rate_scale: float = 1.0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Encode intensity as a deterministic, uniformly distributed spike count."""
    normalized = _validate_pixels(pixels, expected_pixels)
    timestamp_lut = deterministic_rate_lut(timesteps, rate_scale)
    events = [
        SpikeEvent(timestamp, neuron_id)
        for neuron_id, pixel in enumerate(normalized)
        for timestamp in timestamp_lut[pixel]
    ]
    return _sorted(events)


def encode_top_k_rank_order_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    top_k: int = 128,
    encoding_threshold: int = 0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Emit at most ``top_k`` active pixels, strongest and earliest first.

    Equal intensities are ranked by neuron ID.  Rank is spread over the complete
    horizon; multiple ranks can share a timestamp when ``top_k > timesteps``.
    """
    normalized = _validate_pixels(pixels, expected_pixels)
    threshold = _validate_threshold(encoding_threshold)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(
        (
            (pixel, neuron_id)
            for neuron_id, pixel in enumerate(normalized)
            if pixel > threshold
        ),
        key=lambda item: (-item[0], item[1]),
    )[: int(top_k)]
    count = len(ranked)
    if count == 0:
        return ()
    events = [
        SpikeEvent(
            0 if count == 1 else rank * (timesteps - 1) // (count - 1),
            neuron_id,
        )
        for rank, (_, neuron_id) in enumerate(ranked)
    ]
    return _sorted(events)


def burst_count(pixel: int, maximum_spikes: int, threshold: int = 0) -> int:
    """Map one pixel to a burst length in ``[0, maximum_spikes]``."""
    if not 0 <= pixel <= 255:
        raise ValueError(f"pixel={pixel} is outside unsigned 8-bit range")
    threshold = _validate_threshold(threshold)
    if maximum_spikes <= 0:
        raise ValueError("burst_max_spikes must be positive")
    if pixel <= threshold:
        return 0
    active_range = 255 - threshold
    return 1 + ((pixel - threshold - 1) * maximum_spikes) // active_range


@lru_cache(maxsize=64)
def burst_lut(
    timesteps: int,
    maximum_spikes: int,
    interval: int,
    threshold: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Return canonical hybrid latency/burst timestamps for all pixel values."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if interval <= 0:
        raise ValueError("burst_interval must be positive")
    if maximum_spikes <= 0:
        raise ValueError("burst_max_spikes must be positive")
    if (maximum_spikes - 1) * interval >= timesteps:
        raise ValueError("burst does not fit within the timestep horizon")
    threshold = _validate_threshold(threshold)
    result: list[tuple[int, ...]] = []
    for pixel in range(256):
        count = burst_count(pixel, maximum_spikes, threshold)
        if count == 0:
            result.append(())
            continue
        latest_start = timesteps - 1 - (count - 1) * interval
        active_range = 255 - threshold
        first = ((255 - pixel) * latest_start) // active_range
        result.append(tuple(first + index * interval for index in range(count)))
    return tuple(result)


def encode_burst_events(
    pixels: Sequence[int],
    *,
    timesteps: int = 16,
    burst_max_spikes: int = 4,
    burst_interval: int = 1,
    encoding_threshold: int = 0,
    expected_pixels: int | None = 784,
) -> tuple[SpikeEvent, ...]:
    """Encode active pixels using intensity-dependent latency and burst length."""
    normalized = _validate_pixels(pixels, expected_pixels)
    timestamp_lut = burst_lut(
        timesteps,
        burst_max_spikes,
        burst_interval,
        encoding_threshold,
    )
    return _sorted(
        [
            SpikeEvent(timestamp, neuron_id)
            for neuron_id, pixel in enumerate(normalized)
            for timestamp in timestamp_lut[pixel]
        ]
    )


def encode_events(
    pixels: Sequence[int],
    *,
    input_encoding: str,
    timesteps: int = 16,
    expected_pixels: int | None = 784,
    rate_scale: float = 1.0,
    rate_seed: int = 0,
    sample_index: int = 0,
    encoding_threshold: int = 0,
    latency_gamma: float = 2.0,
    top_k: int = 128,
    burst_max_spikes: int = 4,
    burst_interval: int = 1,
) -> tuple[SpikeEvent, ...]:
    """Dispatch to a named version-independent input encoding."""
    common = {"timesteps": timesteps, "expected_pixels": expected_pixels}
    if input_encoding == "latency":
        return encode_latency_events(pixels, **common)
    if input_encoding == "rate":
        return encode_rate_events(
            pixels,
            rate_scale=rate_scale,
            seed=rate_seed,
            sample_index=sample_index,
            **common,
        )
    if input_encoding == "sparse-latency":
        return encode_sparse_latency_events(
            pixels, encoding_threshold=encoding_threshold, **common
        )
    if input_encoding == "nonlinear-latency":
        return encode_nonlinear_latency_events(
            pixels, latency_gamma=latency_gamma, **common
        )
    if input_encoding == "deterministic-rate":
        return encode_deterministic_rate_events(
            pixels, rate_scale=rate_scale, **common
        )
    if input_encoding == "top-k-rank-order":
        return encode_top_k_rank_order_events(
            pixels,
            top_k=top_k,
            encoding_threshold=encoding_threshold,
            **common,
        )
    if input_encoding == "burst":
        return encode_burst_events(
            pixels,
            burst_max_spikes=burst_max_spikes,
            burst_interval=burst_interval,
            encoding_threshold=encoding_threshold,
            **common,
        )
    raise ValueError(f"unsupported input encoding {input_encoding!r}")
