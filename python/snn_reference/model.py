"""Bit-accurate functional model of the fixed two-layer RTL accelerator.

The model intentionally uses only Python integers and explicit width wrapping.
It is a functional reference, not a cycle-accurate model of RTL handshakes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

from .hodgkin_huxley_fixed import HodgkinHuxleyFixedNeuron


def signed_wrap(value: int, width: int) -> int:
    """Return *value* interpreted as a signed two's-complement *width*-bit value."""
    if width <= 0:
        raise ValueError("width must be positive")
    mask = (1 << width) - 1
    wrapped = value & mask
    sign_bit = 1 << (width - 1)
    return wrapped - (1 << width) if wrapped & sign_bit else wrapped


def signed_overflow(value: int, width: int) -> bool:
    """Return whether *value* lies outside signed *width*-bit storage."""
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    return value < minimum or value > maximum


def unsigned_wrap(value: int, width: int) -> int:
    """Return *value* truncated to an unsigned *width*-bit value."""
    if width <= 0:
        raise ValueError("width must be positive")
    return value & ((1 << width) - 1)


def _check_signed(value: int, width: int, name: str) -> None:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}={value} does not fit signed {width}-bit storage")


class NeuronModel(IntEnum):
    """Stable neuron-model identifiers shared with the RTL parameters."""

    LIF = 0
    IZHIKEVICH = 1
    HODGKIN_HUXLEY = 2


@dataclass(frozen=True)
class NumericConfig:
    mem_width: int = 24
    current_width: int = 24
    weight_width: int = 8
    count_width: int = 16
    time_width: int = 32
    threshold: int = 100
    reset_potential: int = 0
    leak_shift: int = 3

    def __post_init__(self) -> None:
        for name in ("mem_width", "current_width", "weight_width", "count_width", "time_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mem_width != self.current_width:
            raise ValueError("RTL requires mem_width == current_width")
        if self.leak_shift < 0:
            raise ValueError("leak_shift must be non-negative")
        _check_signed(self.threshold, self.mem_width, "threshold")
        _check_signed(self.reset_potential, self.mem_width, "reset_potential")


@dataclass(frozen=True)
class IzhikevichConfig:
    """Rational, fixed-point parameters for a regular-spiking neuron."""

    frac_bits: int = 8
    a_numerator: int = 1
    a_denominator: int = 50
    b_numerator: int = 1
    b_denominator: int = 5
    reset_voltage: int = -65
    recovery_increment: int = 8
    spike_threshold: int = 30

    def validate_for_width(self, width: int) -> None:
        if not 0 <= self.frac_bits < width - 1:
            raise ValueError("Izhikevich frac_bits must leave a sign and integer bit")
        if self.a_denominator <= 0 or self.b_denominator <= 0:
            raise ValueError("Izhikevich denominators must be positive")
        if self.a_numerator < 0 or self.b_numerator < 0:
            raise ValueError("Izhikevich numerators must be non-negative")
        scale = 1 << self.frac_bits
        for name in ("reset_voltage", "recovery_increment", "spike_threshold"):
            _check_signed(getattr(self, name) * scale, width, f"Izhikevich {name}")


@dataclass(frozen=True)
class SNNConfig:
    input_neurons: int = 784
    hidden_neurons: int = 256
    output_neurons: int = 10
    numeric: NumericConfig = NumericConfig()
    hidden_neuron_model: NeuronModel = NeuronModel.LIF
    output_neuron_model: NeuronModel = NeuronModel.LIF
    hidden_izhikevich: IzhikevichConfig = IzhikevichConfig()
    output_izhikevich: IzhikevichConfig = IzhikevichConfig()

    def __post_init__(self) -> None:
        for name in ("input_neurons", "hidden_neurons", "output_neurons"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("hidden_neuron_model", "output_neuron_model"):
            try:
                model = NeuronModel(getattr(self, name))
            except ValueError as error:
                raise ValueError(f"unsupported {name}") from error
            object.__setattr__(self, name, model)
        if self.hidden_neuron_model == NeuronModel.IZHIKEVICH:
            self.hidden_izhikevich.validate_for_width(self.numeric.mem_width)
        if self.output_neuron_model == NeuronModel.IZHIKEVICH:
            self.output_izhikevich.validate_for_width(self.numeric.mem_width)
        if (
            self.hidden_neuron_model == NeuronModel.HODGKIN_HUXLEY
            or self.output_neuron_model == NeuronModel.HODGKIN_HUXLEY
        ) and (self.numeric.mem_width != 24 or self.numeric.current_width != 24):
            raise ValueError(
                "Hodgkin-Huxley requires 24-bit membrane and current storage"
            )


@dataclass(frozen=True, order=True)
class SpikeEvent:
    timestamp: int
    neuron_id: int


@dataclass(frozen=True)
class NeuronStatePoint:
    """Observed integer state; offset is within a network update (0..1).

    Equal offsets preserve arithmetic order, including pre-reset and reset.
    No interpolated states or HH gates are retained.
    """

    offset: float
    voltage: int
    recovery: int | None
    phase: str
    spike: bool = False


@dataclass(frozen=True)
class TimestepTrace:
    timestamp: int
    input_events: tuple[int, ...]
    hidden_current: tuple[int, ...]
    hidden_membrane: tuple[int, ...]
    hidden_spikes: tuple[bool, ...]
    output_current: tuple[int, ...]
    output_membrane: tuple[int, ...]
    output_spikes: tuple[bool, ...]
    class_counts: tuple[int, ...]
    hidden_recovery: tuple[int, ...] | None = None
    output_recovery: tuple[int, ...] | None = None
    hidden_current_wraps: int = 0
    hidden_membrane_wraps: int = 0
    hidden_recovery_wraps: int = 0
    output_current_wraps: int = 0
    output_membrane_wraps: int = 0
    output_recovery_wraps: int = 0
    hidden_state_points: tuple[tuple[NeuronStatePoint, ...], ...] = ()
    output_state_points: tuple[tuple[NeuronStatePoint, ...], ...] = ()


@dataclass(frozen=True)
class InferenceResult:
    prediction: int | None
    class_counts: tuple[int, ...]
    hidden_membrane: tuple[int, ...]
    output_membrane: tuple[int, ...]
    timesteps: tuple[TimestepTrace, ...]
    hidden_recovery: tuple[int, ...] | None = None
    output_recovery: tuple[int, ...] | None = None


def _normalize_matrix(
    values: Sequence[Sequence[int]],
    rows: int,
    columns: int,
    width: int,
    name: str,
) -> tuple[tuple[int, ...], ...]:
    if len(values) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {len(values)}")
    normalized: list[tuple[int, ...]] = []
    for row_index, row in enumerate(values):
        if len(row) != columns:
            raise ValueError(
                f"{name}[{row_index}] must have {columns} entries, got {len(row)}"
            )
        for column_index, value in enumerate(row):
            _check_signed(value, width, f"{name}[{row_index}][{column_index}]")
        normalized.append(tuple(row))
    return tuple(normalized)


class _NeuronPopulation(ABC):
    """Model-independent population contract used by the network runner."""

    membrane: list[int]
    recovery: list[int] | None = None
    last_membrane_wraps: int = 0
    last_recovery_wraps: int = 0
    detailed_trace: bool = False

    def _begin_trace(self) -> None:
        self.state_points: list[list[NeuronStatePoint]] = []
        if self.detailed_trace:
            self.state_points = [[] for _ in self.membrane]
            for index, voltage in enumerate(self.membrane):
                self._record(index, 0, voltage,
                             self.recovery[index] if self.recovery is not None else None,
                             "start")

    def _record(self, index: int, offset: float, voltage: int,
                recovery: int | None, phase: str, spike: bool = False) -> None:
        if self.detailed_trace:
            self.state_points[index].append(
                NeuronStatePoint(offset, voltage, recovery, phase, spike)
            )

    @abstractmethod
    def step(self, currents: Sequence[int]) -> tuple[bool, ...]:
        """Advance every neuron by one network timestep."""


class _LIFPopulation(_NeuronPopulation):
    def __init__(self, size: int, numeric: NumericConfig) -> None:
        self._numeric = numeric
        self.membrane = [numeric.reset_potential] * size

    def step(self, currents: Sequence[int]) -> tuple[bool, ...]:
        if len(currents) != len(self.membrane):
            raise ValueError("current vector has the wrong size")

        spikes: list[bool] = []
        self._begin_trace()
        self.last_membrane_wraps = 0
        self.last_recovery_wraps = 0
        for index, current in enumerate(currents):
            membrane = self.membrane[index]
            # Python's signed right shift matches SystemVerilog >>>: negative
            # values are sign-extended and rounded toward negative infinity.
            leak = membrane >> self._numeric.leak_shift
            unwrapped_membrane = membrane - leak + current
            self.last_membrane_wraps += signed_overflow(
                unwrapped_membrane, self._numeric.mem_width
            )
            next_membrane = signed_wrap(unwrapped_membrane, self._numeric.mem_width)
            spike = next_membrane >= self._numeric.threshold
            self._record(index, 1, next_membrane, None, "threshold", spike)
            self.membrane[index] = (
                self._numeric.reset_potential if spike else next_membrane
            )
            spikes.append(spike)
            if spike:
                self._record(index, 1, self.membrane[index], None, "reset")
        return tuple(spikes)


def _trunc_div(numerator: int, denominator: int) -> int:
    """Signed integer division truncated toward zero, matching SystemVerilog."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


class _IzhikevichPopulation(_NeuronPopulation):
    def __init__(
        self, size: int, numeric: NumericConfig, config: IzhikevichConfig
    ) -> None:
        self._numeric = numeric
        self._config = config
        self._scale = 1 << config.frac_bits
        initial_voltage = config.reset_voltage * self._scale
        initial_recovery = _trunc_div(
            config.b_numerator * initial_voltage, config.b_denominator
        )
        self.membrane = [initial_voltage] * size
        self.recovery = [initial_recovery] * size

    def _voltage_half_step(
        self, voltage: int, recovery: int, current: int
    ) -> tuple[int, bool]:
        # 0.04 is represented exactly as 1/25. All terms remain in Q format.
        quadratic = _trunc_div(voltage * voltage, 25 * self._scale)
        derivative = (
            quadratic
            + 5 * voltage
            + 140 * self._scale
            - recovery
            + current * self._scale
        )
        unwrapped_voltage = voltage + _trunc_div(derivative, 2)
        return (
            signed_wrap(unwrapped_voltage, self._numeric.mem_width),
            signed_overflow(unwrapped_voltage, self._numeric.mem_width),
        )

    def step(self, currents: Sequence[int]) -> tuple[bool, ...]:
        if len(currents) != len(self.membrane):
            raise ValueError("current vector has the wrong size")

        spikes: list[bool] = []
        self.last_membrane_wraps = 0
        self.last_recovery_wraps = 0
        config = self._config
        self._begin_trace()
        for index, current in enumerate(currents):
            voltage, wrapped = self._voltage_half_step(
                self.membrane[index], self.recovery[index], current
            )
            self.last_membrane_wraps += wrapped
            self._record(index, 0.5, voltage, self.recovery[index], "half-step")
            voltage, wrapped = self._voltage_half_step(
                voltage, self.recovery[index], current
            )
            self.last_membrane_wraps += wrapped
            self._record(index, 1, voltage, self.recovery[index], "voltage-step")
            b_voltage = _trunc_div(
                config.b_numerator * voltage, config.b_denominator
            )
            recovery_delta = _trunc_div(
                config.a_numerator * (b_voltage - self.recovery[index]),
                config.a_denominator,
            )
            unwrapped_recovery = self.recovery[index] + recovery_delta
            self.last_recovery_wraps += signed_overflow(
                unwrapped_recovery, self._numeric.mem_width
            )
            recovery = signed_wrap(unwrapped_recovery, self._numeric.mem_width)
            spike = voltage >= config.spike_threshold * self._scale
            self._record(index, 1, voltage, recovery, "threshold", spike)
            if spike:
                voltage = config.reset_voltage * self._scale
                unwrapped_recovery = (
                    recovery + config.recovery_increment * self._scale
                )
                self.last_recovery_wraps += signed_overflow(
                    unwrapped_recovery, self._numeric.mem_width
                )
                recovery = signed_wrap(unwrapped_recovery, self._numeric.mem_width)
                self._record(index, 1, voltage, recovery, "reset")
            self.membrane[index] = voltage
            self.recovery[index] = recovery
            spikes.append(spike)
        return tuple(spikes)


class _HodgkinHuxleyPopulation(_NeuronPopulation):
    """Population wrapper around the bit-exact single-neuron HH contract."""

    def __init__(self, size: int) -> None:
        self._neurons = [HodgkinHuxleyFixedNeuron() for _ in range(size)]
        self.membrane = [neuron.state.voltage for neuron in self._neurons]

    def step(self, currents: Sequence[int]) -> tuple[bool, ...]:
        if len(currents) != len(self._neurons):
            raise ValueError("current vector has the wrong size")
        self.last_membrane_wraps = 0
        self.last_recovery_wraps = 0
        self._begin_trace()
        results = []
        for index, (neuron, current) in enumerate(zip(self._neurons, currents, strict=True)):
            observer = None
            if self.detailed_trace:
                def observer(substep, voltage, spike, index=index, neuron=neuron):
                    self._record(index, (substep + 1) / neuron.config.internal_steps,
                                 voltage, None, "substep", spike)
            results.append(neuron.step(int(current), voltage_observer=observer))
        self.membrane = [result.state.voltage for result in results]
        return tuple(result.spike for result in results)


def _create_population(
    model: NeuronModel,
    size: int,
    numeric: NumericConfig,
    izhikevich: IzhikevichConfig,
) -> _NeuronPopulation:
    if model == NeuronModel.LIF:
        return _LIFPopulation(size, numeric)
    if model == NeuronModel.IZHIKEVICH:
        return _IzhikevichPopulation(size, numeric, izhikevich)
    if model == NeuronModel.HODGKIN_HUXLEY:
        return _HodgkinHuxleyPopulation(size)
    raise NotImplementedError(f"neuron model {model.name} is not implemented yet")


class TwoLayerSNN:
    """Functional model of ``snn_top`` with pre-major weight matrices.

    ``input_hidden_weights[pre][post]`` represents the weight from an input
    neuron to a hidden neuron. ``hidden_output_weights`` uses the same layout.
    A new call to :meth:`infer` resets transient neuron and class-counter state,
    matching ``start_inference`` in RTL; the stored weights remain unchanged.
    """

    def __init__(
        self,
        config: SNNConfig,
        input_hidden_weights: Sequence[Sequence[int]],
        hidden_output_weights: Sequence[Sequence[int]],
    ) -> None:
        self.config = config
        numeric = config.numeric
        self.input_hidden_weights = _normalize_matrix(
            input_hidden_weights,
            config.input_neurons,
            config.hidden_neurons,
            numeric.weight_width,
            "input_hidden_weights",
        )
        self.hidden_output_weights = _normalize_matrix(
            hidden_output_weights,
            config.hidden_neurons,
            config.output_neurons,
            numeric.weight_width,
            "hidden_output_weights",
        )

    def _validate_events(self, events: Iterable[SpikeEvent]) -> tuple[SpikeEvent, ...]:
        normalized = tuple(events)
        previous_timestamp = -1
        maximum_timestamp = (1 << self.config.numeric.time_width) - 1
        for index, event in enumerate(normalized):
            if not 0 <= event.timestamp <= maximum_timestamp:
                raise ValueError(f"events[{index}].timestamp is outside the configured width")
            if event.timestamp < previous_timestamp:
                raise ValueError("events must be sorted by nondecreasing timestamp")
            if not 0 <= event.neuron_id < self.config.input_neurons:
                raise ValueError(f"events[{index}].neuron_id is out of range")
            previous_timestamp = event.timestamp
        return normalized

    @staticmethod
    def select_prediction(class_counts: Sequence[int]) -> int | None:
        """Match RTL argmax: no all-zero result; ties select the lowest index."""
        if not class_counts or not any(class_counts):
            return None
        maximum = class_counts[0]
        prediction = 0
        for index in range(1, len(class_counts)):
            if class_counts[index] > maximum:
                maximum = class_counts[index]
                prediction = index
        return prediction

    def infer(
        self,
        events: Iterable[SpikeEvent],
        *,
        trace: bool = False,
        detailed_state_trace: bool = False,
        timesteps: int | None = None,
    ) -> InferenceResult:
        if detailed_state_trace and not trace:
            raise ValueError("detailed_state_trace requires trace=True")
        event_stream = self._validate_events(events)
        if timesteps is None:
            if not event_stream:
                raise ValueError("a non-empty stream or explicit timesteps are required")
            inference_timesteps = event_stream[-1].timestamp + 1
        else:
            if timesteps <= 0:
                raise ValueError("timesteps must be positive")
            if event_stream and event_stream[-1].timestamp >= timesteps:
                raise ValueError("event timestamp lies outside the inference horizon")
            inference_timesteps = timesteps
        numeric = self.config.numeric
        hidden = _create_population(
            self.config.hidden_neuron_model,
            self.config.hidden_neurons,
            numeric,
            self.config.hidden_izhikevich,
        )
        output = _create_population(
            self.config.output_neuron_model,
            self.config.output_neurons,
            numeric,
            self.config.output_izhikevich,
        )
        class_counts = [0] * self.config.output_neurons
        hidden.detailed_trace = output.detailed_trace = detailed_state_trace
        traces: list[TimestepTrace] = []
        event_index = 0

        # Without an explicit horizon this mirrors the fixed-count RTL and
        # stops after the final event. Software rate coding supplies a horizon
        # so empty and trailing-silent timesteps are still evaluated.
        for timestamp in range(inference_timesteps):
            input_ids: list[int] = []
            while (
                event_index < len(event_stream)
                and event_stream[event_index].timestamp == timestamp
            ):
                input_ids.append(event_stream[event_index].neuron_id)
                event_index += 1

            hidden_current = [0] * self.config.hidden_neurons
            hidden_current_wraps = 0
            for neuron_id in input_ids:
                for post_index, weight in enumerate(
                    self.input_hidden_weights[neuron_id]
                ):
                    unwrapped_current = hidden_current[post_index] + weight
                    hidden_current_wraps += signed_overflow(
                        unwrapped_current, numeric.current_width
                    )
                    hidden_current[post_index] = signed_wrap(
                        unwrapped_current, numeric.current_width
                    )

            hidden_spikes = hidden.step(hidden_current)

            output_current = [0] * self.config.output_neurons
            output_current_wraps = 0
            for hidden_index, spike in enumerate(hidden_spikes):
                if spike:
                    for post_index, weight in enumerate(
                        self.hidden_output_weights[hidden_index]
                    ):
                        unwrapped_current = output_current[post_index] + weight
                        output_current_wraps += signed_overflow(
                            unwrapped_current, numeric.current_width
                        )
                        output_current[post_index] = signed_wrap(
                            unwrapped_current, numeric.current_width
                        )

            output_spikes = output.step(output_current)
            for class_index, spike in enumerate(output_spikes):
                if spike:
                    class_counts[class_index] = unsigned_wrap(
                        class_counts[class_index] + 1,
                        numeric.count_width,
                    )

            if trace:
                traces.append(
                    TimestepTrace(
                        timestamp=timestamp,
                        input_events=tuple(input_ids),
                        hidden_current=tuple(hidden_current),
                        hidden_membrane=tuple(hidden.membrane),
                        hidden_spikes=hidden_spikes,
                        output_current=tuple(output_current),
                        output_membrane=tuple(output.membrane),
                        output_spikes=output_spikes,
                        class_counts=tuple(class_counts),
                        hidden_recovery=(
                            tuple(hidden.recovery)
                            if hidden.recovery is not None
                            else None
                        ),
                        output_recovery=(
                            tuple(output.recovery)
                            if output.recovery is not None
                            else None
                        ),
                        hidden_current_wraps=hidden_current_wraps,
                        hidden_membrane_wraps=hidden.last_membrane_wraps,
                        hidden_recovery_wraps=hidden.last_recovery_wraps,
                        output_current_wraps=output_current_wraps,
                        output_membrane_wraps=output.last_membrane_wraps,
                        output_recovery_wraps=output.last_recovery_wraps,
                        hidden_state_points=tuple(tuple(p) for p in hidden.state_points),
                        output_state_points=tuple(tuple(p) for p in output.state_points),
                    )
                )

        counts = tuple(class_counts)
        return InferenceResult(
            prediction=self.select_prediction(counts),
            class_counts=counts,
            hidden_membrane=tuple(hidden.membrane),
            output_membrane=tuple(output.membrane),
            timesteps=tuple(traces),
            hidden_recovery=(
                tuple(hidden.recovery) if hidden.recovery is not None else None
            ),
            output_recovery=(
                tuple(output.recovery) if output.recovery is not None else None
            ),
        )
