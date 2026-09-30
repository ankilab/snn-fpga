"""Versioned, checksummed INT8 model package used by host and RTL tooling."""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

from snn_reference import (
    IzhikevichConfig,
    NeuronModel,
    NumericConfig,
    SNNConfig,
    SpikeEvent,
    TwoLayerSNN,
)
from snn_training.encoding import (
    INPUT_ENCODINGS,
    deterministic_rate_count,
    encode_events,
)

FORMAT_NAME = "snn-int8-v2"
LEGACY_FORMAT_NAME = "snn-int8-v1"
LAYOUT = "pre-major-post-minor"


def _validate_matrix(
    matrix: Sequence[Sequence[int]], rows: int, columns: int, name: str
) -> tuple[tuple[int, ...], ...]:
    if len(matrix) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    normalized: list[tuple[int, ...]] = []
    for row_index, row in enumerate(matrix):
        if len(row) != columns:
            raise ValueError(f"{name}[{row_index}] must contain {columns} entries")
        values = tuple(int(value) for value in row)
        if any(value < -128 or value > 127 for value in values):
            raise ValueError(f"{name}[{row_index}] contains a value outside INT8")
        normalized.append(values)
    return tuple(normalized)


def _encode_int8(matrix: Sequence[Sequence[int]]) -> bytes:
    return bytes(value & 0xFF for row in matrix for value in row)


def _decode_int8(data: bytes, rows: int, columns: int) -> tuple[tuple[int, ...], ...]:
    if len(data) != rows * columns:
        raise ValueError("weight blob size does not match manifest shape")
    values = [value - 256 if value >= 128 else value for value in data]
    return tuple(
        tuple(values[row * columns : (row + 1) * columns]) for row in range(rows)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _neuron_manifest(
    model: NeuronModel, izhikevich: IzhikevichConfig
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model.name.lower(),
        "model_id": int(model),
    }
    if model == NeuronModel.IZHIKEVICH:
        entry["parameters"] = asdict(izhikevich)
    return entry


def _load_neuron_manifest(
    entry: dict[str, Any], layer_name: str
) -> tuple[NeuronModel, IzhikevichConfig]:
    try:
        model = NeuronModel(int(entry["model_id"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {layer_name} neuron model ID") from error
    if entry.get("model") != model.name.lower():
        raise ValueError(f"inconsistent {layer_name} neuron model name")
    if model not in (
        NeuronModel.LIF,
        NeuronModel.IZHIKEVICH,
        NeuronModel.HODGKIN_HUXLEY,
    ):
        raise ValueError(f"unsupported {layer_name} neuron model {model.name}")
    parameters = entry.get("parameters", {})
    if model == NeuronModel.IZHIKEVICH:
        expected_parameters = {field.name for field in fields(IzhikevichConfig)}
        if not isinstance(parameters, dict) or set(parameters) != expected_parameters:
            raise ValueError(
                f"invalid {layer_name} Izhikevich parameter set"
            )
        try:
            config = IzhikevichConfig(
                **{key: int(value) for key, value in parameters.items()}
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid {layer_name} Izhikevich parameters"
            ) from error
    else:
        if parameters:
            raise ValueError(f"unexpected {layer_name} neuron parameters")
        config = IzhikevichConfig()
    return model, config


@dataclass(frozen=True)
class ExportedModel:
    config: SNNConfig
    timesteps: int
    input_hidden_weights: tuple[tuple[int, ...], ...]
    hidden_output_weights: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any]
    input_encoding: str = "latency"
    rate_scale: float = 1.0
    rate_seed: int = 0
    encoding_threshold: int = 0
    latency_gamma: float = 2.0
    top_k: int = 128
    burst_max_spikes: int = 4
    burst_interval: int = 1

    def reference_model(self) -> TwoLayerSNN:
        return TwoLayerSNN(
            self.config,
            self.input_hidden_weights,
            self.hidden_output_weights,
        )

    def encode_events(
        self, pixels: Sequence[int], *, sample_index: int = 0
    ) -> tuple[SpikeEvent, ...]:
        """Encode one input according to the package manifest contract."""
        return encode_events(
            pixels,
            input_encoding=self.input_encoding,
            timesteps=self.timesteps,
            expected_pixels=self.config.input_neurons,
            rate_scale=self.rate_scale,
            rate_seed=self.rate_seed,
            sample_index=sample_index,
            encoding_threshold=self.encoding_threshold,
            latency_gamma=self.latency_gamma,
            top_k=self.top_k,
            burst_max_spikes=self.burst_max_spikes,
            burst_interval=self.burst_interval,
        )


def _encoding_manifest(
    input_encoding: str,
    *,
    input_neurons: int,
    timesteps: int,
    rate_scale: float,
    rate_seed: int,
    encoding_threshold: int,
    latency_gamma: float,
    top_k: int,
    burst_max_spikes: int,
    burst_interval: int,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "timesteps": timesteps,
        "pixel_range": [0, 255],
        "event_order": "timestamp-then-neuron-id",
    }
    if input_encoding == "latency":
        return {
            **common,
            "type": "fixed-count-latency-v1",
            "events_per_image": input_neurons,
            "timestamp_formula": "((255 - pixel) * (timesteps - 1)) // 255",
        }
    if input_encoding == "rate":
        return {
            **common,
            "type": "bernoulli-rate-v1",
            "events_per_image": None,
            "maximum_events_per_image": input_neurons * timesteps,
            "probability_formula": "min(1, rate_scale * pixel / 255)",
            "rate_scale": float(rate_scale),
            "seed": int(rate_seed),
            "seed_policy": "counter-based-sample-index-v1",
        }
    if input_encoding == "sparse-latency":
        return {
            **common,
            "type": "sparse-latency-v1",
            "events_per_image": None,
            "maximum_events_per_image": input_neurons,
            "encoding_threshold": encoding_threshold,
            "active_rule": "pixel > encoding_threshold",
            "timestamp_formula": "((255 - pixel) * (timesteps - 1)) // 255",
        }
    if input_encoding == "nonlinear-latency":
        return {
            **common,
            "type": "nonlinear-latency-v1",
            "events_per_image": input_neurons,
            "latency_gamma": float(latency_gamma),
            "timestamp_formula": "floor((1 - (pixel / 255) ** latency_gamma) * (timesteps - 1))",
            "lookup_table_policy": "canonical-u8-host-lut-v1",
        }
    if input_encoding == "deterministic-rate":
        return {
            **common,
            "type": "deterministic-rate-v1",
            "events_per_image": None,
            "maximum_events_per_image": input_neurons
            * deterministic_rate_count(255, timesteps, rate_scale),
            "rate_scale": float(rate_scale),
            "count_formula": "min(timesteps, round_half_up(rate_scale * pixel * timesteps / 255))",
            "temporal_distribution": "centered-equal-partitions-v1",
        }
    if input_encoding == "top-k-rank-order":
        return {
            **common,
            "type": "top-k-rank-order-v1",
            "events_per_image": None,
            "maximum_events_per_image": min(top_k, input_neurons),
            "top_k": top_k,
            "encoding_threshold": encoding_threshold,
            "rank_policy": "pixel-descending-neuron-id-ascending-v1",
            "temporal_distribution": "rank-spans-horizon-v1",
        }
    return {
        **common,
        "type": "burst-v1",
        "events_per_image": None,
        "maximum_events_per_image": input_neurons * burst_max_spikes,
        "encoding_threshold": encoding_threshold,
        "burst_max_spikes": burst_max_spikes,
        "burst_interval": burst_interval,
        "count_policy": "linear-active-intensity-v1",
        "first_spike_policy": "inverse-intensity-with-burst-fit-v1",
    }


def export_model(
    output_dir: str | Path,
    config: SNNConfig,
    timesteps: int,
    input_hidden_weights: Sequence[Sequence[int]],
    hidden_output_weights: Sequence[Sequence[int]],
    *,
    metadata: dict[str, Any] | None = None,
    input_encoding: str = "latency",
    rate_scale: float = 1.0,
    rate_seed: int = 0,
    encoding_threshold: int = 0,
    latency_gamma: float = 2.0,
    top_k: int = 128,
    burst_max_spikes: int = 4,
    burst_interval: int = 1,
) -> Path:
    """Write a model directory and return the manifest path."""
    if config.numeric.weight_width != 8:
        raise ValueError("snn-int8-v2 requires weight_width == 8")
    for model in (config.hidden_neuron_model, config.output_neuron_model):
        if model not in (
            NeuronModel.LIF,
            NeuronModel.IZHIKEVICH,
            NeuronModel.HODGKIN_HUXLEY,
        ):
            raise ValueError(f"snn-int8-v2 does not support {model.name}")
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if input_encoding not in INPUT_ENCODINGS:
        raise ValueError(f"input_encoding must be one of {INPUT_ENCODINGS}")
    if not isinstance(rate_scale, (int, float)) or not 0.0 <= rate_scale < float("inf"):
        raise ValueError("rate_scale must be finite and non-negative")
    if not 0 <= encoding_threshold <= 255:
        raise ValueError("encoding_threshold must be in [0, 255]")
    if not isinstance(latency_gamma, (int, float)) or not 0.0 < latency_gamma < float("inf"):
        raise ValueError("latency_gamma must be finite and positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if burst_max_spikes <= 0 or burst_interval <= 0:
        raise ValueError("burst parameters must be positive")
    if input_encoding == "burst" and (burst_max_spikes - 1) * burst_interval >= timesteps:
        raise ValueError("burst does not fit within the timestep horizon")

    layer0 = _validate_matrix(
        input_hidden_weights,
        config.input_neurons,
        config.hidden_neurons,
        "input_hidden_weights",
    )
    layer1 = _validate_matrix(
        hidden_output_weights,
        config.hidden_neurons,
        config.output_neurons,
        "hidden_output_weights",
    )
    blobs = {
        "weights_l0.bin": _encode_int8(layer0),
        "weights_l1.bin": _encode_int8(layer1),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, data in blobs.items():
        (destination / filename).write_bytes(data)

    numeric = config.numeric
    manifest = {
        "format": FORMAT_NAME,
        "topology": {
            "input_neurons": config.input_neurons,
            "hidden_neurons": config.hidden_neurons,
            "output_neurons": config.output_neurons,
        },
        "numeric": {
            "mem_width": numeric.mem_width,
            "current_width": numeric.current_width,
            "weight_width": numeric.weight_width,
            "count_width": numeric.count_width,
            "time_width": numeric.time_width,
            "threshold": numeric.threshold,
            "reset_potential": numeric.reset_potential,
            "leak_shift": numeric.leak_shift,
        },
        "neurons": {
            "hidden": _neuron_manifest(
                config.hidden_neuron_model, config.hidden_izhikevich
            ),
            "output": _neuron_manifest(
                config.output_neuron_model, config.output_izhikevich
            ),
        },
        "encoding": _encoding_manifest(
            input_encoding,
            input_neurons=config.input_neurons,
            timesteps=timesteps,
            rate_scale=rate_scale,
            rate_seed=rate_seed,
            encoding_threshold=encoding_threshold,
            latency_gamma=latency_gamma,
            top_k=top_k,
            burst_max_spikes=burst_max_spikes,
            burst_interval=burst_interval,
        ),
        "weights": [
            {
                "layer": 0,
                "file": "weights_l0.bin",
                "shape": [config.input_neurons, config.hidden_neurons],
                "layout": LAYOUT,
                "dtype": "int8-twos-complement",
                "sha256": _sha256(blobs["weights_l0.bin"]),
                "crc32": f"{zlib.crc32(blobs['weights_l0.bin']) & 0xFFFFFFFF:08x}",
            },
            {
                "layer": 1,
                "file": "weights_l1.bin",
                "shape": [config.hidden_neurons, config.output_neurons],
                "layout": LAYOUT,
                "dtype": "int8-twos-complement",
                "sha256": _sha256(blobs["weights_l1.bin"]),
                "crc32": f"{zlib.crc32(blobs['weights_l1.bin']) & 0xFFFFFFFF:08x}",
            },
        ],
        "upload": {
            "weight_stream_order": ["weights_l0.bin", "weights_l1.bin"],
            "weight_write_count": len(blobs["weights_l0.bin"]) + len(blobs["weights_l1.bin"]),
            "weight_crc32": f"{zlib.crc32(blobs['weights_l0.bin'] + blobs['weights_l1.bin']) & 0xFFFFFFFF:08x}",
        },
        "metadata": metadata or {},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_exported_model(model_dir: str | Path) -> ExportedModel:
    source = Path(model_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    format_name = manifest.get("format")
    if format_name not in (FORMAT_NAME, LEGACY_FORMAT_NAME):
        raise ValueError("unsupported model format")

    topology = manifest["topology"]
    numeric_data = manifest["numeric"]
    if format_name == FORMAT_NAME:
        neurons = manifest.get("neurons")
        if not isinstance(neurons, dict):
            raise ValueError("snn-int8-v2 manifest is missing neuron configuration")
        hidden_model, hidden_izhikevich = _load_neuron_manifest(
            neurons.get("hidden", {}), "hidden"
        )
        output_model, output_izhikevich = _load_neuron_manifest(
            neurons.get("output", {}), "output"
        )
    else:
        hidden_model = NeuronModel.LIF
        output_model = NeuronModel.LIF
        hidden_izhikevich = IzhikevichConfig()
        output_izhikevich = IzhikevichConfig()

    config = SNNConfig(
        input_neurons=int(topology["input_neurons"]),
        hidden_neurons=int(topology["hidden_neurons"]),
        output_neurons=int(topology["output_neurons"]),
        numeric=NumericConfig(**{key: int(value) for key, value in numeric_data.items()}),
        hidden_neuron_model=hidden_model,
        output_neuron_model=output_model,
        hidden_izhikevich=hidden_izhikevich,
        output_izhikevich=output_izhikevich,
    )
    encoding = manifest["encoding"]
    encoding_type = encoding.get("type")
    try:
        timesteps = int(encoding["timesteps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid encoding timestep horizon") from error
    if timesteps <= 0:
        raise ValueError("encoding timestep horizon must be positive")
    rate_scale = 1.0
    rate_seed = 0
    encoding_threshold = 0
    latency_gamma = 2.0
    top_k = 128
    burst_max_spikes = 4
    burst_interval = 1

    def require_variable_capacity(expected: int, name: str) -> None:
        if encoding.get("events_per_image") is not None:
            raise ValueError(f"{name} encoding must declare a variable event count")
        if int(encoding.get("maximum_events_per_image", -1)) != expected:
            raise ValueError(f"{name} encoding maximum event count is inconsistent")

    def read_threshold() -> int:
        try:
            value = int(encoding["encoding_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid encoding threshold") from error
        if not 0 <= value <= 255:
            raise ValueError("encoding threshold must be in [0, 255]")
        return value

    if encoding_type == "fixed-count-latency-v1":
        input_encoding = "latency"
        if int(encoding.get("events_per_image", -1)) != config.input_neurons:
            raise ValueError("fixed-count latency event count does not match topology")
    elif encoding_type == "bernoulli-rate-v1":
        input_encoding = "rate"
        try:
            rate_scale = float(encoding["rate_scale"])
            rate_seed = int(encoding["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid rate encoding parameters") from error
        if not 0.0 <= rate_scale < float("inf"):
            raise ValueError("rate_scale must be finite and non-negative")
        require_variable_capacity(config.input_neurons * timesteps, "rate")
        if encoding.get("seed_policy") != "counter-based-sample-index-v1":
            raise ValueError("unsupported rate encoding seed policy")
    elif encoding_type == "sparse-latency-v1":
        input_encoding = "sparse-latency"
        encoding_threshold = read_threshold()
        require_variable_capacity(config.input_neurons, "sparse latency")
    elif encoding_type == "nonlinear-latency-v1":
        input_encoding = "nonlinear-latency"
        try:
            latency_gamma = float(encoding["latency_gamma"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid nonlinear latency gamma") from error
        if not 0.0 < latency_gamma < float("inf"):
            raise ValueError("nonlinear latency gamma must be finite and positive")
        if int(encoding.get("events_per_image", -1)) != config.input_neurons:
            raise ValueError("nonlinear latency event count does not match topology")
        if encoding.get("lookup_table_policy") != "canonical-u8-host-lut-v1":
            raise ValueError("unsupported nonlinear latency lookup policy")
    elif encoding_type == "deterministic-rate-v1":
        input_encoding = "deterministic-rate"
        try:
            rate_scale = float(encoding["rate_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid deterministic rate scale") from error
        if not 0.0 <= rate_scale < float("inf"):
            raise ValueError("rate_scale must be finite and non-negative")
        require_variable_capacity(
            config.input_neurons
            * deterministic_rate_count(255, timesteps, rate_scale),
            "deterministic rate",
        )
        if encoding.get("temporal_distribution") != "centered-equal-partitions-v1":
            raise ValueError("unsupported deterministic rate distribution")
    elif encoding_type == "top-k-rank-order-v1":
        input_encoding = "top-k-rank-order"
        encoding_threshold = read_threshold()
        try:
            top_k = int(encoding["top_k"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid top-k rank-order limit") from error
        if top_k <= 0:
            raise ValueError("top-k rank-order limit must be positive")
        require_variable_capacity(
            min(top_k, config.input_neurons), "top-k rank-order"
        )
        if encoding.get("rank_policy") != "pixel-descending-neuron-id-ascending-v1":
            raise ValueError("unsupported top-k rank policy")
    elif encoding_type == "burst-v1":
        input_encoding = "burst"
        encoding_threshold = read_threshold()
        try:
            burst_max_spikes = int(encoding["burst_max_spikes"])
            burst_interval = int(encoding["burst_interval"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid burst encoding parameters") from error
        if burst_max_spikes <= 0 or burst_interval <= 0:
            raise ValueError("burst encoding parameters must be positive")
        if (burst_max_spikes - 1) * burst_interval >= timesteps:
            raise ValueError("burst does not fit within the timestep horizon")
        require_variable_capacity(
            config.input_neurons * burst_max_spikes, "burst"
        )
    else:
        raise ValueError("unsupported input encoding")

    matrices: list[tuple[tuple[int, ...], ...]] = []
    expected_shapes = [
        (config.input_neurons, config.hidden_neurons),
        (config.hidden_neurons, config.output_neurons),
    ]
    weights = sorted(manifest["weights"], key=lambda item: int(item["layer"]))
    if len(weights) != 2:
        raise ValueError("model must contain exactly two weight blobs")
    for entry, expected_shape in zip(weights, expected_shapes, strict=True):
        if entry["layout"] != LAYOUT or entry["dtype"] != "int8-twos-complement":
            raise ValueError("unsupported weight layout or data type")
        shape = tuple(int(value) for value in entry["shape"])
        if shape != expected_shape:
            raise ValueError("weight shape does not match topology")
        data = (source / entry["file"]).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {entry['file']}")
        if int(entry["crc32"], 16) != (zlib.crc32(data) & 0xFFFFFFFF):
            raise ValueError(f"CRC32 mismatch for {entry['file']}")
        matrices.append(_decode_int8(data, *shape))

    return ExportedModel(
        config=config,
        timesteps=timesteps,
        input_hidden_weights=matrices[0],
        hidden_output_weights=matrices[1],
        metadata=dict(manifest.get("metadata", {})),
        input_encoding=input_encoding,
        rate_scale=rate_scale,
        rate_seed=rate_seed,
        encoding_threshold=encoding_threshold,
        latency_gamma=latency_gamma,
        top_k=top_k,
        burst_max_spikes=burst_max_spikes,
        burst_interval=burst_interval,
    )
