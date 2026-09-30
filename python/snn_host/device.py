"""Linux MMIO driver and checked deployment protocol for the SNN accelerator."""

from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from snn_reference import (
    HodgkinHuxleyFixedConfig,
    IzhikevichConfig,
    NeuronModel,
    SpikeEvent,
)
from snn_training.export import ExportedModel, load_exported_model
from snn_training.encoding import VARIABLE_EVENT_ENCODINGS

from .registers import (
    ControlBits,
    Register,
    SNN_ADDRESS_RANGE,
    StatusBits,
    event_stream_crc32,
    pack_weight_cursor,
    weight_stream_crc32,
)

CORE_ID = 0x534E4E31
SUPPORTED_INTERFACE_MAJOR = 1


class SNNHostError(RuntimeError):
    """Base class for host/deployment failures."""


class HardwareMismatchError(SNNHostError):
    """The model package does not match the instantiated RTL."""


class AcceleratorError(SNNHostError):
    """The accelerator reported a sticky protocol error."""


class InferenceTimeoutError(SNNHostError):
    """Inference did not complete within the requested deadline."""


class RegisterIO(Protocol):
    def read32(self, offset: int) -> int: ...

    def write32(self, offset: int, value: int) -> None: ...






def _signed32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - (1 << 32) if value & (1 << 31) else value


@dataclass(frozen=True)
class HardwareInfo:
    core_id: int
    interface_version: int
    input_neurons: int
    hidden_neurons: int
    output_neurons: int
    event_capacity: int
    weight_width: int
    mem_width: int
    current_width: int
    count_width: int
    time_width: int
    leak_shift: int
    threshold: int
    reset_potential: int
    hidden_neuron_model: int
    output_neuron_model: int
    hidden_executor_count: int
    output_executor_count: int
    izh_frac_bits: int
    izh_a_numerator: int
    izh_a_denominator: int
    izh_b_numerator: int
    izh_b_denominator: int
    izh_reset_voltage: int
    izh_recovery_increment: int
    izh_spike_threshold: int
    hh_voltage_width: int
    hh_voltage_frac_bits: int
    hh_gate_width: int
    hh_gate_frac_bits: int
    hh_rate_width: int
    hh_rate_frac_bits: int
    hh_lut_step_frac_bits: int
    hh_lut_min_voltage: int
    hh_lut_max_voltage: int
    hh_internal_steps: int
    hh_spike_threshold: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceTelemetry:
    prediction: int | None
    cycles: int
    consumed_input_events: int
    emitted_hidden_spikes: int
    emitted_output_spikes: int
    synapse_phase_cycles: int | None = None
    hidden_update_phase_cycles: int | None = None
    hidden_routing_phase_cycles: int | None = None
    output_update_phase_cycles: int | None = None
    control_overhead_phase_cycles: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


@dataclass(frozen=True)
class ModelPackage:
    directory: Path
    exported: ExportedModel
    layer0: bytes
    layer1: bytes
    expected_weight_count: int
    expected_weight_crc32: int
    events_per_image: int | None
    maximum_events_per_image: int

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> ModelPackage:
        source = Path(directory).resolve()
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        weights = sorted(manifest.get("weights", []), key=lambda item: int(item["layer"]))
        if len(weights) != 2 or [int(item["layer"]) for item in weights] != [0, 1]:
            raise ValueError("manifest must contain weight layers 0 and 1")

        blobs: list[bytes] = []
        filenames: list[str] = []
        for item in weights:
            filename = str(item["file"])
            candidate = (source / filename).resolve()
            try:
                candidate.relative_to(source)
            except ValueError as error:
                raise ValueError("weight file escapes the model directory") from error
            filenames.append(filename)
            blobs.append(candidate.read_bytes())

        upload = manifest.get("upload", {})
        if upload.get("weight_stream_order") != filenames:
            raise ValueError("manifest weight stream order does not match weight entries")
        combined = blobs[0] + blobs[1]
        expected_count = int(upload["weight_write_count"])
        expected_crc = int(str(upload["weight_crc32"]), 16)
        if expected_count != len(combined):
            raise ValueError("manifest upload weight count is inconsistent")
        if expected_crc != weight_stream_crc32(combined):
            raise ValueError("manifest upload weight CRC32 is inconsistent")

        exported = load_exported_model(source)
        declared_events = manifest["encoding"].get("events_per_image")
        events_per_image = (
            None if declared_events is None else int(declared_events)
        )
        maximum_events_per_image = int(
            manifest["encoding"].get(
                "maximum_events_per_image", events_per_image
            )
        )
        return cls(
            directory=source,
            exported=exported,
            layer0=blobs[0],
            layer1=blobs[1],
            expected_weight_count=expected_count,
            expected_weight_crc32=expected_crc,
            events_per_image=events_per_image,
            maximum_events_per_image=maximum_events_per_image,
        )


ProgressCallback = Callable[[int, int], None]


class SNNDevice:
    RESULT_SETTLE_SECONDS = 0.001

    ERROR_NAMES = {
        1: "bad register",
        2: "not ready",
        3: "bad index",
        4: "bad write strobe",
        5: "bad inference configuration",
    }

    def __init__(self, registers: RegisterIO) -> None:
        self.registers = registers

    def _read(self, register: Register) -> int:
        return self.registers.read32(int(register)) & 0xFFFF_FFFF

    def _read64(self, low: Register, high: Register) -> int:
        low_value = self._read(low)
        high_value = self._read(high)
        return (high_value << 32) | low_value

    def _write(self, register: Register, value: int) -> None:
        self.registers.write32(int(register), value & 0xFFFF_FFFF)

    def status(self) -> StatusBits:
        return StatusBits(self._read(Register.STATUS))

    def _raise_if_error(self, status: StatusBits | None = None) -> None:
        status = self.status() if status is None else status
        if status & StatusBits.ERROR:
            code = self._read(Register.ERROR) & 0xFF
            description = self.ERROR_NAMES.get(code, "unknown error")
            raise AcceleratorError(f"accelerator error {code}: {description}")

    def probe(self) -> HardwareInfo:
        core_id = self._read(Register.ID)
        interface_version = self._read(Register.VERSION)
        if core_id != CORE_ID:
            raise HardwareMismatchError(
                f"unexpected core ID 0x{core_id:08x}; expected 0x{CORE_ID:08x}"
            )
        if interface_version >> 16 != SUPPORTED_INTERFACE_MAJOR:
            raise HardwareMismatchError(
                f"unsupported interface version 0x{interface_version:08x}"
            )
        dimensions0 = self._read(Register.DIMENSIONS_0)
        dimensions1 = self._read(Register.DIMENSIONS_1)
        numeric0 = self._read(Register.NUMERIC_0)
        numeric1 = self._read(Register.NUMERIC_1)
        if (interface_version & 0xFFFF) >= 1:
            neuron_models = self._read(Register.NEURON_MODELS)
            izh_ab = self._read(Register.IZH_AB)
            hidden_neuron_model = neuron_models & 0xFF
            output_neuron_model = (neuron_models >> 8) & 0xFF
            izh_frac_bits = (neuron_models >> 24) & 0xFF
            izh_a_numerator = izh_ab & 0xFF
            izh_a_denominator = (izh_ab >> 8) & 0xFF
            izh_b_numerator = (izh_ab >> 16) & 0xFF
            izh_b_denominator = (izh_ab >> 24) & 0xFF
            izh_reset_voltage = _signed32(self._read(Register.IZH_RESET_VOLTAGE))
            izh_recovery_increment = _signed32(
                self._read(Register.IZH_RECOVERY_INCREMENT)
            )
            izh_spike_threshold = _signed32(
                self._read(Register.IZH_SPIKE_THRESHOLD)
            )
            try:
                hardware_models = (
                    NeuronModel(hidden_neuron_model),
                    NeuronModel(output_neuron_model),
                )
            except ValueError as error:
                raise HardwareMismatchError(
                    "hardware reports an unknown neuron model"
                ) from error
        else:
            defaults = IzhikevichConfig()
            hidden_neuron_model = int(NeuronModel.LIF)
            output_neuron_model = int(NeuronModel.LIF)
            izh_frac_bits = defaults.frac_bits
            izh_a_numerator = defaults.a_numerator
            izh_a_denominator = defaults.a_denominator
            izh_b_numerator = defaults.b_numerator
            izh_b_denominator = defaults.b_denominator
            izh_reset_voltage = defaults.reset_voltage
            izh_recovery_increment = defaults.recovery_increment
            izh_spike_threshold = defaults.spike_threshold
        hh_defaults = HodgkinHuxleyFixedConfig()
        if (interface_version & 0xFFFF) >= 5:
            executor_counts = self._read(Register.EXECUTOR_COUNTS)
            hidden_executor_count = (executor_counts >> 16) & 0xFFFF
            output_executor_count = executor_counts & 0xFFFF
        else:
            hidden_executor_count = 0
            output_executor_count = 0
        if (interface_version & 0xFFFF) >= 2:
            hh_q_format = self._read(Register.HH_Q_FORMAT)
            hh_widths_steps = self._read(Register.HH_WIDTHS_STEPS)
            hh_lut_range = self._read(Register.HH_LUT_RANGE)
            hh_voltage_frac_bits = hh_q_format & 0xFF
            hh_gate_frac_bits = (hh_q_format >> 8) & 0xFF
            hh_rate_frac_bits = (hh_q_format >> 16) & 0xFF
            hh_lut_step_frac_bits = (hh_q_format >> 24) & 0xFF
            hh_voltage_width = hh_widths_steps & 0xFF
            hh_gate_width = (hh_widths_steps >> 8) & 0xFF
            hh_rate_width = (hh_widths_steps >> 16) & 0xFF
            hh_internal_steps = (hh_widths_steps >> 24) & 0xFF
            hh_lut_min_voltage = _signed32(hh_lut_range & 0xFFFF)
            if hh_lut_min_voltage & 0x8000:
                hh_lut_min_voltage -= 1 << 16
            hh_lut_max_voltage = (hh_lut_range >> 16) & 0xFFFF
            if hh_lut_max_voltage & 0x8000:
                hh_lut_max_voltage -= 1 << 16
            hh_spike_threshold = _signed32(self._read(Register.HH_SPIKE_THRESHOLD))
        else:
            if (
                hidden_neuron_model == int(NeuronModel.HODGKIN_HUXLEY)
                or output_neuron_model == int(NeuronModel.HODGKIN_HUXLEY)
            ):
                raise HardwareMismatchError(
                    "Hodgkin-Huxley hardware requires interface version 1.2"
                )
            hh_voltage_width = hh_defaults.voltage_width
            hh_voltage_frac_bits = hh_defaults.voltage_frac_bits
            hh_gate_width = hh_defaults.gate_width
            hh_gate_frac_bits = hh_defaults.gate_frac_bits
            hh_rate_width = hh_defaults.rate_width
            hh_rate_frac_bits = hh_defaults.rate_frac_bits
            hh_lut_step_frac_bits = hh_defaults.lut_step_frac_bits
            hh_lut_min_voltage = hh_defaults.lut_min_voltage
            hh_lut_max_voltage = hh_defaults.lut_max_voltage
            hh_internal_steps = hh_defaults.internal_steps
            hh_spike_threshold = hh_defaults.spike_threshold
        info = HardwareInfo(
            core_id=core_id,
            interface_version=interface_version,
            input_neurons=dimensions0 & 0xFFFF,
            hidden_neurons=(dimensions0 >> 16) & 0xFFFF,
            output_neurons=dimensions1 & 0xFFFF,
            event_capacity=(dimensions1 >> 16) & 0xFFFF,
            weight_width=numeric0 & 0xFF,
            mem_width=(numeric0 >> 8) & 0xFF,
            current_width=(numeric0 >> 16) & 0xFF,
            count_width=(numeric0 >> 24) & 0xFF,
            time_width=numeric1 & 0xFF,
            leak_shift=(numeric1 >> 8) & 0xFF,
            threshold=_signed32(self._read(Register.THRESHOLD)),
            reset_potential=_signed32(self._read(Register.RESET_POTENTIAL)),
            hidden_neuron_model=hidden_neuron_model,
            output_neuron_model=output_neuron_model,
            hidden_executor_count=hidden_executor_count,
            output_executor_count=output_executor_count,
            izh_frac_bits=izh_frac_bits,
            izh_a_numerator=izh_a_numerator,
            izh_a_denominator=izh_a_denominator,
            izh_b_numerator=izh_b_numerator,
            izh_b_denominator=izh_b_denominator,
            izh_reset_voltage=izh_reset_voltage,
            izh_recovery_increment=izh_recovery_increment,
            izh_spike_threshold=izh_spike_threshold,
            hh_voltage_width=hh_voltage_width,
            hh_voltage_frac_bits=hh_voltage_frac_bits,
            hh_gate_width=hh_gate_width,
            hh_gate_frac_bits=hh_gate_frac_bits,
            hh_rate_width=hh_rate_width,
            hh_rate_frac_bits=hh_rate_frac_bits,
            hh_lut_step_frac_bits=hh_lut_step_frac_bits,
            hh_lut_min_voltage=hh_lut_min_voltage,
            hh_lut_max_voltage=hh_lut_max_voltage,
            hh_internal_steps=hh_internal_steps,
            hh_spike_threshold=hh_spike_threshold,
        )
        self._raise_if_error()
        return info

    def validate_model(
        self, package: ModelPackage, info: HardwareInfo | None = None
    ) -> HardwareInfo:
        info = self.probe() if info is None else info
        config = package.exported.config
        numeric = config.numeric
        expected = {
            "input_neurons": config.input_neurons,
            "hidden_neurons": config.hidden_neurons,
            "output_neurons": config.output_neurons,
            "weight_width": numeric.weight_width,
            "mem_width": numeric.mem_width,
            "current_width": numeric.current_width,
            "count_width": numeric.count_width,
            "time_width": numeric.time_width,
            "leak_shift": numeric.leak_shift,
            "threshold": numeric.threshold,
            "reset_potential": numeric.reset_potential,
            "hidden_neuron_model": int(config.hidden_neuron_model),
            "output_neuron_model": int(config.output_neuron_model),
        }
        mismatches = [
            f"{name}: hardware={getattr(info, name)}, model={value}"
            for name, value in expected.items()
            if getattr(info, name) != value
        ]
        if mismatches:
            raise HardwareMismatchError(
                "model/hardware mismatch: " + "; ".join(mismatches)
            )
        if info.event_capacity < package.maximum_events_per_image:
            raise HardwareMismatchError(
                "model/hardware mismatch: event_capacity: "
                f"hardware={info.event_capacity}, required={package.maximum_events_per_image}"
            )
        interface_minor = info.interface_version & 0xFFFF
        if (
            package.exported.input_encoding in VARIABLE_EVENT_ENCODINGS
            and interface_minor < 3
        ):
            raise HardwareMismatchError(
                f"{package.exported.input_encoding} coding requires accelerator "
                "interface version 1.3 or newer"
            )
        selected_izhikevich = []
        if config.hidden_neuron_model == NeuronModel.IZHIKEVICH:
            selected_izhikevich.append(("hidden", config.hidden_izhikevich))
        if config.output_neuron_model == NeuronModel.IZHIKEVICH:
            selected_izhikevich.append(("output", config.output_izhikevich))
        hardware_izhikevich = {
            "frac_bits": info.izh_frac_bits,
            "a_numerator": info.izh_a_numerator,
            "a_denominator": info.izh_a_denominator,
            "b_numerator": info.izh_b_numerator,
            "b_denominator": info.izh_b_denominator,
            "reset_voltage": info.izh_reset_voltage,
            "recovery_increment": info.izh_recovery_increment,
            "spike_threshold": info.izh_spike_threshold,
        }
        parameter_mismatches = [
            f"{layer} Izhikevich {name}: hardware={hardware_izhikevich[name]}, "
            f"model={getattr(parameters, name)}"
            for layer, parameters in selected_izhikevich
            for name in hardware_izhikevich
            if hardware_izhikevich[name] != getattr(parameters, name)
        ]
        if parameter_mismatches:
            raise HardwareMismatchError(
                "model/hardware mismatch: " + "; ".join(parameter_mismatches)
            )
        if (
            config.hidden_neuron_model == NeuronModel.HODGKIN_HUXLEY
            or config.output_neuron_model == NeuronModel.HODGKIN_HUXLEY
        ):
            hh = HodgkinHuxleyFixedConfig()
            hh_expected = {
                "voltage_width": hh.voltage_width,
                "voltage_frac_bits": hh.voltage_frac_bits,
                "gate_width": hh.gate_width,
                "gate_frac_bits": hh.gate_frac_bits,
                "rate_width": hh.rate_width,
                "rate_frac_bits": hh.rate_frac_bits,
                "lut_step_frac_bits": hh.lut_step_frac_bits,
                "lut_min_voltage": hh.lut_min_voltage,
                "lut_max_voltage": hh.lut_max_voltage,
                "internal_steps": hh.internal_steps,
                "spike_threshold": hh.spike_threshold,
            }
            hh_mismatches = [
                f"Hodgkin-Huxley {name}: hardware={getattr(info, 'hh_' + name)}, "
                f"model={value}"
                for name, value in hh_expected.items()
                if getattr(info, "hh_" + name) != value
            ]
            if hh_mismatches:
                raise HardwareMismatchError(
                    "model/hardware mismatch: " + "; ".join(hh_mismatches)
                )
        return info

    def clear(self) -> None:
        self._write(Register.CONTROL, int(ControlBits.CLEAR_STATUS_AND_UPLOAD_CHECKS))
        self._raise_if_error(self.status())

    def _require_idle(self, ready_bit: StatusBits | None = None) -> None:
        status = self.status()
        self._raise_if_error(status)
        if not status & StatusBits.IDLE or status & StatusBits.BUSY:
            raise AcceleratorError("accelerator is not idle")
        if ready_bit is not None and not status & ready_bit:
            raise AcceleratorError(f"accelerator is missing ready bit {ready_bit.name}")

    def upload_model(
        self,
        package: ModelPackage,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.validate_model(package)
        self._require_idle(StatusBits.WEIGHT_READY)
        self.clear()
        total = package.expected_weight_count
        written = 0
        for layer, blob in enumerate((package.layer0, package.layer1)):
            self._write(Register.WEIGHT_CURSOR, pack_weight_cursor(layer, 0, 0))
            for value in blob:
                self._write(Register.WEIGHT_DATA, value)
                written += 1
                if progress is not None and (written % 8192 == 0 or written == total):
                    progress(written, total)
            self._raise_if_error()

        actual_count = self._read(Register.WEIGHT_WRITE_COUNT)
        actual_crc = self._read(Register.WEIGHT_CRC32)
        if actual_count != package.expected_weight_count:
            raise AcceleratorError(
                f"weight count mismatch: hardware={actual_count}, "
                f"expected={package.expected_weight_count}"
            )
        if actual_crc != package.expected_weight_crc32:
            raise AcceleratorError(
                f"weight CRC mismatch: hardware=0x{actual_crc:08x}, "
                f"expected=0x{package.expected_weight_crc32:08x}"
            )

    def upload_events(
        self,
        events: Sequence[SpikeEvent],
        info: HardwareInfo,
        *,
        timesteps: int | None = None,
    ) -> None:
        interface_minor = info.interface_version & 0xFFFF
        if interface_minor < 3:
            if len(events) != info.event_capacity:
                raise ValueError(
                    f"expected {info.event_capacity} events, got {len(events)}"
                )
        elif len(events) > info.event_capacity:
            raise ValueError(
                f"event stream has {len(events)} events but hardware capacity is "
                f"{info.event_capacity}"
            )
        if timesteps is None:
            if not events:
                raise ValueError("empty event streams require explicit timesteps")
            timesteps = events[-1].timestamp + 1
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        previous: tuple[int, int] | None = None
        max_timestamp = (1 << info.time_width) - 1
        for event in events:
            key = (event.timestamp, event.neuron_id)
            if previous is not None and key < previous:
                raise ValueError("events must be sorted by timestamp and neuron ID")
            if not 0 <= event.timestamp <= max_timestamp:
                raise ValueError("event timestamp does not fit hardware time width")
            if event.timestamp >= timesteps:
                raise ValueError("event timestamp lies outside the inference horizon")
            if not 0 <= event.neuron_id < info.input_neurons:
                raise ValueError("event neuron ID is outside the input layer")
            previous = key

        self._require_idle(StatusBits.EVENT_READY)
        if interface_minor >= 3:
            self._write(Register.INFERENCE_EVENT_COUNT, len(events))
            self._write(Register.TIMESTEP_COUNT, timesteps)
        self._write(Register.EVENT_CURSOR, 0)
        for event in events:
            self._write(Register.EVENT_TIMESTAMP, event.timestamp)
            self._write(Register.EVENT_NEURON, event.neuron_id)
        self._raise_if_error()

        actual_count = self._read(Register.EVENT_WRITE_COUNT)
        expected_crc = event_stream_crc32(events)
        actual_crc = self._read(Register.EVENT_CRC32)
        if actual_count != len(events):
            raise AcceleratorError(
                f"event count mismatch: hardware={actual_count}, expected={len(events)}"
            )
        if actual_crc != expected_crc:
            raise AcceleratorError(
                f"event CRC mismatch: hardware=0x{actual_crc:08x}, "
                f"expected=0x{expected_crc:08x}"
            )

    def infer(
        self,
        events: Sequence[SpikeEvent],
        *,
        timesteps: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.001,
    ) -> int | None:
        info = self.prepare_inference(events, timesteps=timesteps)
        return self._start_and_wait(
            info, timeout=timeout, poll_interval=poll_interval
        )

    def prepare_inference(
        self,
        events: Sequence[SpikeEvent],
        *,
        timesteps: int | None = None,
    ) -> HardwareInfo:
        """Upload one event stream and leave it ready for one or more starts."""

        info = self.probe()
        self._require_idle()
        self.clear()
        self.upload_events(events, info, timesteps=timesteps)
        return info

    def _start_and_wait(
        self,
        info: HardwareInfo,
        *,
        timeout: float,
        poll_interval: float,
    ) -> int | None:
        if timeout <= 0 or poll_interval < 0:
            raise ValueError("timeout must be positive and poll interval non-negative")
        self._require_idle()
        self._write(Register.CONTROL, int(ControlBits.START))

        deadline = time.monotonic() + timeout
        start_acknowledged = False
        result_settle_deadline: float | None = None
        while True:
            status = self.status()
            self._raise_if_error(status)
            if not start_acknowledged:
                start_acknowledged = bool(status & StatusBits.BUSY) or not bool(
                    status & StatusBits.DONE
                )
            if start_acknowledged and status & StatusBits.DONE:
                if status & StatusBits.RESULT_VALID:
                    result = self._read(Register.RESULT)
                    if result >= info.output_neurons:
                        raise AcceleratorError(f"result class {result} is outside output layer")
                    return result
                # DONE and RESULT_VALID are independent sticky bits in interface
                # 1.4.  The prediction selector, registered output interface and
                # AXI wrapper form a pipeline after core completion.  Allow that
                # pipeline and the PS-to-PL bridge to settle before treating a
                # missing result as a genuine no-prediction outcome.
                now = time.monotonic()
                if result_settle_deadline is None:
                    result_settle_deadline = min(
                        now + self.RESULT_SETTLE_SECONDS, deadline
                    )
                elif now >= result_settle_deadline:
                    return None
            else:
                result_settle_deadline = None
            if time.monotonic() >= deadline:
                raise InferenceTimeoutError(f"inference timed out after {timeout:.3f} s")
            time.sleep(poll_interval)

    def replay(
        self,
        info: HardwareInfo | None = None,
        *,
        timeout: float = 5.0,
        poll_interval: float = 0.001,
    ) -> InferenceTelemetry:
        """Rerun the prepared event memory and return interface-1.4 telemetry."""

        info = self.probe() if info is None else info
        if (info.interface_version & 0xFFFF) < 4:
            raise HardwareMismatchError(
                "inference replay telemetry requires accelerator interface "
                "version 1.4 or newer"
            )
        prediction = self._start_and_wait(
            info, timeout=timeout, poll_interval=poll_interval
        )
        cycles = self._read64(
            Register.INFERENCE_CYCLES_LO, Register.INFERENCE_CYCLES_HI
        )
        phase_cycles: dict[str, int | None] = {
            "synapse_phase_cycles": None,
            "hidden_update_phase_cycles": None,
            "hidden_routing_phase_cycles": None,
            "output_update_phase_cycles": None,
            "control_overhead_phase_cycles": None,
        }
        if (info.interface_version & 0xFFFF) >= 6:
            phase_cycles = {
                "synapse_phase_cycles": self._read64(
                    Register.PHASE_SYNAPSE_CYCLES_LO,
                    Register.PHASE_SYNAPSE_CYCLES_HI,
                ),
                "hidden_update_phase_cycles": self._read64(
                    Register.PHASE_HIDDEN_UPDATE_CYCLES_LO,
                    Register.PHASE_HIDDEN_UPDATE_CYCLES_HI,
                ),
                "hidden_routing_phase_cycles": self._read64(
                    Register.PHASE_HIDDEN_ROUTING_CYCLES_LO,
                    Register.PHASE_HIDDEN_ROUTING_CYCLES_HI,
                ),
                "output_update_phase_cycles": self._read64(
                    Register.PHASE_OUTPUT_UPDATE_CYCLES_LO,
                    Register.PHASE_OUTPUT_UPDATE_CYCLES_HI,
                ),
                "control_overhead_phase_cycles": self._read64(
                    Register.PHASE_CONTROL_OVERHEAD_CYCLES_LO,
                    Register.PHASE_CONTROL_OVERHEAD_CYCLES_HI,
                ),
            }
            if sum(value for value in phase_cycles.values() if value is not None) != cycles:
                raise HardwareMismatchError(
                    "phase-cycle telemetry does not sum to inference cycles"
                )
        return InferenceTelemetry(
            prediction=prediction,
            cycles=cycles,
            consumed_input_events=self._read(Register.CONSUMED_INPUT_EVENTS),
            emitted_hidden_spikes=self._read(Register.EMITTED_HIDDEN_SPIKES),
            emitted_output_spikes=self._read(Register.EMITTED_OUTPUT_SPIKES),
            **phase_cycles,
        )
