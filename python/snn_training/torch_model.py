"""PyTorch QAT model whose forward pass mirrors the integer RTL equations."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn

from snn_reference import IzhikevichConfig, NeuronModel, NumericConfig, SNNConfig
from snn_training.encoding import (
    INPUT_ENCODINGS,
    burst_lut,
    deterministic_rate_lut,
    nonlinear_latency_lut,
)
from snn_training.hh_qat import HHQATDynamics, HHQATState


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance: Tensor) -> Tensor:
        ctx.save_for_backward(distance)
        return (distance >= 0).to(distance.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor]:
        (distance,) = ctx.saved_tensors
        return (gradient / (1.0 + distance.abs()).square(),)


def _ste_round_clamp_int8(values: Tensor) -> Tensor:
    clamped = values.clamp(-128.0, 127.0)
    rounded = clamped.round()
    return clamped + (rounded - clamped).detach()


def _ste_signed_wrap(values: Tensor, width: int) -> Tensor:
    modulus = float(1 << width)
    half = float(1 << (width - 1))
    wrapped = torch.remainder(values + half, modulus) - half
    return values + (wrapped - values).detach()


def _ste_arithmetic_shift(values: Tensor, shift: int) -> Tensor:
    scaled = values / float(1 << shift)
    shifted = torch.floor(scaled)
    return scaled + (shifted - scaled).detach()


def _ste_trunc_div(values: Tensor, denominator: int) -> Tensor:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    scaled = values / float(denominator)
    truncated = torch.trunc(scaled)
    return scaled + (truncated - scaled).detach()


def latency_spikes(images: Tensor, timesteps: int) -> Tensor:
    """Encode ``[batch, 1, 28, 28]`` or flattened images into ``[time,batch,784]``."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    timestamps = ((255 - pixels_u8) * (timesteps - 1)) // 255
    time_axis = torch.arange(timesteps, device=images.device).view(-1, 1, 1)
    return (timestamps.unsqueeze(0) == time_axis).to(images.dtype)


def sparse_latency_spikes(
    images: Tensor, timesteps: int, *, encoding_threshold: int = 0
) -> Tensor:
    """Linearly latency-encode only pixels above ``encoding_threshold``."""
    if not 0 <= encoding_threshold <= 255:
        raise ValueError("encoding_threshold must be in [0, 255]")
    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    spikes = latency_spikes(images, timesteps)
    return spikes * (pixels_u8 > encoding_threshold).unsqueeze(0).to(images.dtype)


def _timestamp_lut_spikes(
    images: Tensor, timesteps: int, timestamp_lut: tuple[int, ...]
) -> Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    lookup = torch.tensor(timestamp_lut, dtype=torch.int64, device=images.device)
    timestamps = lookup[pixels_u8]
    time_axis = torch.arange(timesteps, device=images.device).view(-1, 1, 1)
    return (timestamps.unsqueeze(0) == time_axis).to(images.dtype)


def nonlinear_latency_spikes(
    images: Tensor, timesteps: int, *, latency_gamma: float = 2.0
) -> Tensor:
    """Gamma latency-encode using the scalar encoder's canonical lookup table."""
    return _timestamp_lut_spikes(
        images, timesteps, nonlinear_latency_lut(timesteps, latency_gamma)
    )


def _event_lut_spikes(
    images: Tensor,
    timesteps: int,
    event_lut: tuple[tuple[int, ...], ...],
) -> Tensor:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    pattern = torch.tensor(
        _event_pattern_values(event_lut, timesteps),
        dtype=images.dtype,
        device=images.device,
    )
    return pattern[pixels_u8].permute(2, 0, 1)


@lru_cache(maxsize=64)
def _event_pattern_values(
    event_lut: tuple[tuple[int, ...], ...], timesteps: int
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if timestamp in timestamps else 0.0 for timestamp in range(timesteps))
        for timestamps in event_lut
    )


def deterministic_rate_spikes(
    images: Tensor, timesteps: int, *, rate_scale: float = 1.0
) -> Tensor:
    """Deterministically rate-encode intensities with uniformly spaced spikes."""
    return _event_lut_spikes(
        images, timesteps, deterministic_rate_lut(timesteps, rate_scale)
    )


def top_k_rank_order_spikes(
    images: Tensor,
    timesteps: int,
    *,
    top_k: int = 128,
    encoding_threshold: int = 0,
) -> Tensor:
    """Encode the strongest active pixels by their stable rank order."""
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= encoding_threshold <= 255:
        raise ValueError("encoding_threshold must be in [0, 255]")
    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0).to(torch.int64)
    order = torch.argsort(pixels_u8, dim=1, descending=True, stable=True)
    result = torch.zeros(
        (timesteps, images.shape[0], pixels.shape[1]),
        dtype=images.dtype,
        device=images.device,
    )
    for batch_index in range(images.shape[0]):
        ranked = order[batch_index, : min(top_k, pixels.shape[1])]
        ranked = ranked[pixels_u8[batch_index, ranked] > encoding_threshold]
        count = int(ranked.numel())
        if count == 0:
            continue
        if count == 1:
            timestamps = torch.zeros(1, dtype=torch.int64, device=images.device)
        else:
            ranks = torch.arange(count, dtype=torch.int64, device=images.device)
            timestamps = ranks * (timesteps - 1) // (count - 1)
        result[timestamps, batch_index, ranked] = 1.0
    return result


def burst_spikes(
    images: Tensor,
    timesteps: int,
    *,
    burst_max_spikes: int = 4,
    burst_interval: int = 1,
    encoding_threshold: int = 0,
) -> Tensor:
    """Hybrid latency/burst encoding backed by the scalar canonical LUT."""
    return _event_lut_spikes(
        images,
        timesteps,
        burst_lut(
            timesteps,
            burst_max_spikes,
            burst_interval,
            encoding_threshold,
        ),
    )


def rate_spikes(
    images: Tensor,
    timesteps: int,
    *,
    rate_scale: float = 1.0,
    seed: int = 0,
    sample_offset: int = 0,
) -> Tensor:
    """Bernoulli rate-encode images into ``[time, batch, inputs]`` spikes.

    The counter-based random samples match :func:`encode_rate_events` and are
    independent of PyTorch's global RNG and the chosen batch partitioning.
    """
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not math.isfinite(rate_scale) or rate_scale < 0.0:
        raise ValueError("rate_scale must be finite and non-negative")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")

    pixels = images.reshape(images.shape[0], -1)
    pixels_u8 = (pixels * 255.0).round().clamp(0.0, 255.0)
    probabilities = (pixels_u8 * (float(rate_scale) / 255.0)).clamp(0.0, 1.0)
    mask = (1 << 31) - 1
    device = images.device
    sample_ids = torch.arange(
        sample_offset + 1,
        sample_offset + images.shape[0] + 1,
        dtype=torch.int64,
        device=device,
    ).view(1, -1, 1)
    timestamps = torch.arange(
        1, timesteps + 1, dtype=torch.int64, device=device
    ).view(-1, 1, 1)
    neuron_ids = torch.arange(
        1, pixels.shape[1] + 1, dtype=torch.int64, device=device
    ).view(1, 1, -1)
    random_values = (
        ((int(seed) * 1_103_515_245) & mask)
        ^ ((sample_ids * 73_856_093) & mask)
        ^ ((timestamps * 19_349_663) & mask)
        ^ ((neuron_ids * 83_492_791) & mask)
    )
    random_values ^= (random_values << 13) & mask
    random_values ^= random_values >> 17
    random_values ^= (random_values << 5) & mask
    uniforms = (random_values & mask).to(torch.float64) / float(1 << 31)
    return (uniforms < probabilities.unsqueeze(0)).to(images.dtype)


class HardwareQATSNN(nn.Module):
    """Two-layer, bias-free SNN with fake-quantized INT8 weights."""

    def __init__(
        self,
        config: SNNConfig,
        timesteps: int = 16,
        bptt_window: int = 0,
        *,
        input_encoding: str = "latency",
        rate_scale: float = 1.0,
        rate_seed: int = 0,
        encoding_threshold: int = 0,
        latency_gamma: float = 2.0,
        top_k: int = 128,
        burst_max_spikes: int = 4,
        burst_interval: int = 1,
        hh_internal_steps: int = 100,
        hh_surrogate_width_mv: float = 5.0,
        hh_checkpoint_interval: int = 0,
        hh_backend: str = "eager",
    ) -> None:
        super().__init__()
        if hh_backend not in ("eager", "compiled"):
            raise ValueError("HH backend must be eager or compiled")
        if hh_backend == "compiled" and NeuronModel.HODGKIN_HUXLEY not in (
            config.hidden_neuron_model, config.output_neuron_model
        ):
            raise ValueError("the compiled HH backend requires an HH population")
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if bptt_window < 0:
            raise ValueError("bptt_window must be non-negative")
        if input_encoding not in INPUT_ENCODINGS:
            raise ValueError(f"input_encoding must be one of {INPUT_ENCODINGS}")
        if not math.isfinite(rate_scale) or rate_scale < 0.0:
            raise ValueError("rate_scale must be finite and non-negative")
        if not 0 <= encoding_threshold <= 255:
            raise ValueError("encoding_threshold must be in [0, 255]")
        if not math.isfinite(latency_gamma) or latency_gamma <= 0.0:
            raise ValueError("latency_gamma must be finite and positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if burst_max_spikes <= 0 or burst_interval <= 0:
            raise ValueError("burst parameters must be positive")
        if input_encoding == "burst" and (burst_max_spikes - 1) * burst_interval >= timesteps:
            raise ValueError("burst does not fit within the timestep horizon")
        self.config = config
        self.timesteps = timesteps
        self.bptt_window = bptt_window
        self.input_encoding = input_encoding
        self.rate_scale = float(rate_scale)
        self.rate_seed = int(rate_seed)
        self.encoding_threshold = int(encoding_threshold)
        self.latency_gamma = float(latency_gamma)
        self.top_k = int(top_k)
        self.burst_max_spikes = int(burst_max_spikes)
        self.burst_interval = int(burst_interval)
        self.hh_dynamics = (
            HHQATDynamics(
                internal_steps=hh_internal_steps,
                surrogate_width_mv=hh_surrogate_width_mv,
                checkpoint_interval=hh_checkpoint_interval,
                backend=hh_backend,
            )
            if NeuronModel.HODGKIN_HUXLEY
            in (config.hidden_neuron_model, config.output_neuron_model)
            else None
        )
        self.weight_l0 = nn.Parameter(
            torch.empty(config.input_neurons, config.hidden_neurons)
        )
        self.weight_l1 = nn.Parameter(
            torch.empty(config.hidden_neurons, config.output_neurons)
        )
        # A moderate initial spread is required to produce some output spikes
        # at the RTL threshold of 100; near-zero initialization leaves the
        # spike-count loss flat for many quantization steps.
        nn.init.normal_(self.weight_l0, mean=0.0, std=12.0)
        nn.init.normal_(self.weight_l1, mean=0.0, std=12.0)
        self._last_forward_diagnostics: dict[str, Tensor] = {}
        self._collect_hh_diagnostics = False

    def quantized_weights(self) -> tuple[Tensor, Tensor]:
        return (
            self.weight_l0.detach().round().clamp(-128, 127).to(torch.int8),
            self.weight_l1.detach().round().clamp(-128, 127).to(torch.int8),
        )

    def load_quantized_weights(self, layer0, layer1) -> None:
        """Initialize trainable tensors from INT8-compatible weight matrices."""
        values0 = torch.as_tensor(
            layer0, dtype=self.weight_l0.dtype, device=self.weight_l0.device
        )
        values1 = torch.as_tensor(
            layer1, dtype=self.weight_l1.dtype, device=self.weight_l1.device
        )
        if (
            values0.shape != self.weight_l0.shape
            or values1.shape != self.weight_l1.shape
        ):
            raise ValueError("initial weight shapes do not match the QAT topology")
        with torch.no_grad():
            self.weight_l0.copy_(values0)
            self.weight_l1.copy_(values1)

    def set_rate_seed(self, seed: int) -> None:
        """Select the deterministic Bernoulli realization for later forwards."""
        self.rate_seed = int(seed)

    def forward_diagnostics(self) -> dict[str, Tensor]:
        """Return detached activity recorded by the most recent forward pass."""
        return self._last_forward_diagnostics

    def set_hh_diagnostics(self, enabled: bool) -> None:
        self._collect_hh_diagnostics = bool(enabled)

    def input_spikes(self, images: Tensor, *, sample_offset: int = 0) -> Tensor:
        """Encode a batch using this model's configured input contract."""
        if self.input_encoding == "latency":
            return latency_spikes(images, self.timesteps)
        if self.input_encoding == "rate":
            return rate_spikes(
                images,
                self.timesteps,
                rate_scale=self.rate_scale,
                seed=self.rate_seed,
                sample_offset=sample_offset,
            )
        if self.input_encoding == "sparse-latency":
            return sparse_latency_spikes(
                images,
                self.timesteps,
                encoding_threshold=self.encoding_threshold,
            )
        if self.input_encoding == "nonlinear-latency":
            return nonlinear_latency_spikes(
                images, self.timesteps, latency_gamma=self.latency_gamma
            )
        if self.input_encoding == "deterministic-rate":
            return deterministic_rate_spikes(
                images, self.timesteps, rate_scale=self.rate_scale
            )
        if self.input_encoding == "top-k-rank-order":
            return top_k_rank_order_spikes(
                images,
                self.timesteps,
                top_k=self.top_k,
                encoding_threshold=self.encoding_threshold,
            )
        return burst_spikes(
            images,
            self.timesteps,
            burst_max_spikes=self.burst_max_spikes,
            burst_interval=self.burst_interval,
            encoding_threshold=self.encoding_threshold,
        )

    @staticmethod
    def _lif_step(
        membrane: Tensor,
        recovery: Tensor,
        current: Tensor,
        numeric: NumericConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        current = _ste_signed_wrap(current, numeric.current_width)
        leak = _ste_arithmetic_shift(membrane, numeric.leak_shift)
        next_membrane = _ste_signed_wrap(
            membrane - leak + current, numeric.mem_width
        )
        normalized_distance = (next_membrane - numeric.threshold) / max(
            1.0, float(abs(numeric.threshold))
        )
        spikes = _SurrogateSpike.apply(normalized_distance)
        membrane = next_membrane * (1.0 - spikes) + numeric.reset_potential * spikes
        return membrane, recovery, spikes

    @staticmethod
    def _izhikevich_step(
        membrane: Tensor,
        recovery: Tensor,
        current: Tensor,
        numeric: NumericConfig,
        config: IzhikevichConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale = 1 << config.frac_bits
        current = _ste_signed_wrap(current, numeric.current_width)
        current_q = current * float(scale)

        def voltage_half_step(voltage: Tensor) -> Tensor:
            quadratic = _ste_trunc_div(voltage.square(), 25 * scale)
            derivative = (
                quadratic
                + 5.0 * voltage
                + float(140 * scale)
                - recovery
                + current_q
            )
            return _ste_signed_wrap(
                voltage + _ste_trunc_div(derivative, 2), numeric.mem_width
            )

        membrane = voltage_half_step(membrane)
        membrane = voltage_half_step(membrane)
        b_voltage = _ste_trunc_div(
            membrane * float(config.b_numerator), config.b_denominator
        )
        recovery_delta = _ste_trunc_div(
            (b_voltage - recovery) * float(config.a_numerator),
            config.a_denominator,
        )
        recovery = _ste_signed_wrap(
            recovery + recovery_delta, numeric.mem_width
        )

        threshold_q = float(config.spike_threshold * scale)
        normalized_distance = (membrane - threshold_q) / max(
            1.0, abs(threshold_q)
        )
        spikes = _SurrogateSpike.apply(normalized_distance)
        reset_q = float(config.reset_voltage * scale)
        increment_q = float(config.recovery_increment * scale)
        membrane = membrane * (1.0 - spikes) + reset_q * spikes
        spiking_recovery = _ste_signed_wrap(
            recovery + increment_q, numeric.mem_width
        )
        recovery = recovery * (1.0 - spikes) + spiking_recovery * spikes
        return membrane, recovery, spikes

    @classmethod
    def _neuron_step(
        cls,
        model: NeuronModel,
        membrane: Tensor,
        recovery: Tensor,
        current: Tensor,
        numeric: NumericConfig,
        izhikevich: IzhikevichConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if model == NeuronModel.LIF:
            return cls._lif_step(membrane, recovery, current, numeric)
        if model == NeuronModel.IZHIKEVICH:
            return cls._izhikevich_step(
                membrane, recovery, current, numeric, izhikevich
            )
        raise NotImplementedError(f"QAT neuron model {model.name} is not implemented yet")

    @staticmethod
    def _initial_state(
        images: Tensor,
        size: int,
        model: NeuronModel,
        numeric: NumericConfig,
        izhikevich: IzhikevichConfig,
    ) -> tuple[Tensor, Tensor]:
        if model == NeuronModel.LIF:
            membrane_value = float(numeric.reset_potential)
            recovery_value = 0.0
        elif model == NeuronModel.IZHIKEVICH:
            scale = 1 << izhikevich.frac_bits
            membrane_value = float(izhikevich.reset_voltage * scale)
            recovery_value = float(
                int(
                    izhikevich.b_numerator
                    * izhikevich.reset_voltage
                    * scale
                    / izhikevich.b_denominator
                )
            )
        else:
            raise NotImplementedError(
                f"QAT neuron model {model.name} is not implemented yet"
            )
        shape = (images.shape[0], size)
        return (
            images.new_full(shape, membrane_value),
            images.new_full(shape, recovery_value),
        )

    def _population_initial_state(
        self,
        images: Tensor,
        size: int,
        model: NeuronModel,
        numeric: NumericConfig,
        izhikevich: IzhikevichConfig,
    ) -> tuple[Tensor, Tensor] | HHQATState:
        if model == NeuronModel.HODGKIN_HUXLEY:
            if self.hh_dynamics is None:
                raise RuntimeError("HH dynamics were not initialized")
            return self.hh_dynamics.initial_state(images, size)
        return self._initial_state(images, size, model, numeric, izhikevich)

    def _population_step(
        self,
        model: NeuronModel,
        state: tuple[Tensor, Tensor] | HHQATState,
        current: Tensor,
        numeric: NumericConfig,
        izhikevich: IzhikevichConfig,
    ) -> tuple[tuple[Tensor, Tensor] | HHQATState, Tensor]:
        if model == NeuronModel.HODGKIN_HUXLEY:
            if self.hh_dynamics is None or not isinstance(state, HHQATState):
                raise RuntimeError("invalid HH QAT state")
            return self.hh_dynamics(state, current)
        if isinstance(state, HHQATState):
            raise RuntimeError("HH state supplied to a non-HH population")
        membrane, recovery, spikes = self._neuron_step(
            model, state[0], state[1], current, numeric, izhikevich
        )
        return (membrane, recovery), spikes

    def forward(self, images: Tensor, *, sample_offset: int = 0) -> Tensor:
        if self.hh_dynamics is not None:
            self.hh_dynamics.begin_diagnostics(self._collect_hh_diagnostics)
        spikes_in = self.input_spikes(images, sample_offset=sample_offset)
        if spikes_in.shape[2] != self.config.input_neurons:
            raise ValueError("input image size does not match configured input neurons")

        batch = images.shape[0]
        numeric = self.config.numeric
        hidden_state = self._population_initial_state(
            images,
            self.config.hidden_neurons,
            self.config.hidden_neuron_model,
            numeric,
            self.config.hidden_izhikevich,
        )
        output_state = self._population_initial_state(
            images,
            self.config.output_neurons,
            self.config.output_neuron_model,
            numeric,
            self.config.output_izhikevich,
        )
        class_counts = images.new_zeros((batch, self.config.output_neurons))
        weight_l0 = _ste_round_clamp_int8(self.weight_l0)
        weight_l1 = _ste_round_clamp_int8(self.weight_l1)
        hidden_activity: list[Tensor] = []
        output_activity: list[Tensor] = []

        for timestamp in range(self.timesteps):
            hidden_current = spikes_in[timestamp] @ weight_l0
            hidden_state, hidden_spikes = self._population_step(
                self.config.hidden_neuron_model,
                hidden_state,
                hidden_current,
                numeric,
                self.config.hidden_izhikevich,
            )
            hidden_activity.append(hidden_spikes.detach().sum())
            output_current = hidden_spikes.to(weight_l1.dtype) @ weight_l1
            output_state, output_spikes = self._population_step(
                self.config.output_neuron_model,
                output_state,
                output_current,
                numeric,
                self.config.output_izhikevich,
            )
            output_activity.append(output_spikes.detach().sum())
            class_counts = class_counts + output_spikes
            if (
                self.bptt_window > 0
                and (timestamp + 1) % self.bptt_window == 0
                and timestamp + 1 < self.timesteps
            ):
                hidden_state = (
                    hidden_state.detach()
                    if isinstance(hidden_state, HHQATState)
                    else (hidden_state[0].detach(), hidden_state[1].detach())
                )
                output_state = (
                    output_state.detach()
                    if isinstance(output_state, HHQATState)
                    else (output_state[0].detach(), output_state[1].detach())
                )
        self._last_forward_diagnostics = {
            "hidden_spikes_by_timestep": torch.stack(hidden_activity),
            "output_spikes_by_timestep": torch.stack(output_activity),
        }
        if isinstance(hidden_state, HHQATState):
            self._last_forward_diagnostics.update(
                {
                    "hh_voltage_min": hidden_state.voltage.detach().min(),
                    "hh_voltage_max": hidden_state.voltage.detach().max(),
                    "hh_gate_min": torch.stack(
                        (
                            hidden_state.sodium_activation.detach().min(),
                            hidden_state.sodium_inactivation.detach().min(),
                            hidden_state.potassium_activation.detach().min(),
                        )
                    ).min(),
                    "hh_gate_max": torch.stack(
                        (
                            hidden_state.sodium_activation.detach().max(),
                            hidden_state.sodium_inactivation.detach().max(),
                            hidden_state.potassium_activation.detach().max(),
                        )
                    ).max(),
                }
            )
        if self.hh_dynamics is not None:
            self._last_forward_diagnostics.update(self.hh_dynamics.diagnostics())
        return class_counts
