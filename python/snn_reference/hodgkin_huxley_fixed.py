"""Bit-exact fixed-point contract for the Hodgkin-Huxley RTL core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .hodgkin_huxley import HodgkinHuxleyState, rate_constants, resting_state


def _round_div(value: int, divisor: int) -> int:
    """Round to nearest integer, with half-way cases away from zero."""
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if value >= 0:
        return (value + divisor // 2) // divisor
    return -((-value + divisor // 2) // divisor)


def _signed_limits(width: int) -> tuple[int, int]:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _saturate(value: int, minimum: int, maximum: int) -> int:
    return min(max(value, minimum), maximum)


@dataclass(frozen=True)
class HodgkinHuxleyFixedConfig:
    """Synthesis constants shared by the Python model, ROMs and RTL."""

    voltage_width: int = 24
    voltage_frac_bits: int = 12
    gate_width: int = 20
    gate_frac_bits: int = 18
    rate_width: int = 24
    rate_frac_bits: int = 18
    lut_min_voltage: int = -32
    lut_max_voltage: int = 128
    lut_step_frac_bits: int = 2  # one address every 0.25 mV
    internal_steps: int = 100
    spike_threshold: int = 20

    def __post_init__(self) -> None:
        if self.voltage_frac_bits <= self.lut_step_frac_bits:
            raise ValueError("voltage precision must exceed LUT address precision")
        if self.gate_frac_bits >= self.gate_width:
            raise ValueError("gate Q format cannot represent one")
        if self.rate_frac_bits >= self.rate_width - 1:
            raise ValueError("rate Q format needs a signed integer range")
        if self.lut_min_voltage >= self.lut_max_voltage:
            raise ValueError("invalid LUT voltage range")
        if self.internal_steps <= 0:
            raise ValueError("internal_steps must be positive")

    @property
    def voltage_scale(self) -> int:
        return 1 << self.voltage_frac_bits

    @property
    def gate_scale(self) -> int:
        return 1 << self.gate_frac_bits

    @property
    def rate_scale(self) -> int:
        return 1 << self.rate_frac_bits

    @property
    def lut_size(self) -> int:
        return (
            (self.lut_max_voltage - self.lut_min_voltage)
            * (1 << self.lut_step_frac_bits)
            + 1
        )


@dataclass(frozen=True)
class HodgkinHuxleyFixedState:
    """Raw integer state: voltage Q12 and gates Q18 by default."""

    voltage: int
    sodium_activation: int
    sodium_inactivation: int
    potassium_activation: int


@dataclass(frozen=True)
class HodgkinHuxleyFixedResult:
    state: HodgkinHuxleyFixedState
    spike: bool
    spike_substep: int | None


RateTables = tuple[tuple[int, ...], ...]


def build_rate_tables(
    config: HodgkinHuxleyFixedConfig = HodgkinHuxleyFixedConfig(),
) -> RateTables:
    """Quantize the six original HH rates onto the synthesis LUT grid."""
    tables: list[list[int]] = [[] for _ in range(6)]
    rate_min, rate_max = _signed_limits(config.rate_width)
    address_scale = 1 << config.lut_step_frac_bits
    for address in range(config.lut_size):
        voltage = config.lut_min_voltage + address / address_scale
        for table, rate in zip(tables, rate_constants(voltage), strict=True):
            # All six rates are non-negative, so this is nearest with half up.
            raw = int(rate * config.rate_scale + 0.5)
            table.append(_saturate(raw, rate_min, rate_max))
    return tuple(tuple(table) for table in tables)


def quantize_state(
    state: HodgkinHuxleyState,
    config: HodgkinHuxleyFixedConfig = HodgkinHuxleyFixedConfig(),
) -> HodgkinHuxleyFixedState:
    voltage_min, voltage_max = _signed_limits(config.voltage_width)
    return HodgkinHuxleyFixedState(
        voltage=_saturate(round(state.voltage * config.voltage_scale), voltage_min, voltage_max),
        sodium_activation=_saturate(round(state.sodium_activation * config.gate_scale), 0, config.gate_scale),
        sodium_inactivation=_saturate(round(state.sodium_inactivation * config.gate_scale), 0, config.gate_scale),
        potassium_activation=_saturate(round(state.potassium_activation * config.gate_scale), 0, config.gate_scale),
    )


def dequantize_state(
    state: HodgkinHuxleyFixedState,
    config: HodgkinHuxleyFixedConfig = HodgkinHuxleyFixedConfig(),
) -> HodgkinHuxleyState:
    return HodgkinHuxleyState(
        voltage=state.voltage / config.voltage_scale,
        sodium_activation=state.sodium_activation / config.gate_scale,
        sodium_inactivation=state.sodium_inactivation / config.gate_scale,
        potassium_activation=state.potassium_activation / config.gate_scale,
    )


class HodgkinHuxleyFixedNeuron:
    """Executable, bit-exact specification of one 1 ms RTL update."""

    def __init__(
        self,
        config: HodgkinHuxleyFixedConfig = HodgkinHuxleyFixedConfig(),
        tables: RateTables | None = None,
    ) -> None:
        self.config = config
        self.tables = build_rate_tables(config) if tables is None else tables
        if len(self.tables) != 6 or any(len(table) != config.lut_size for table in self.tables):
            raise ValueError("rate tables do not match the fixed-point configuration")
        self.state = quantize_state(resting_state(), config)

    def clear(self) -> None:
        self.state = quantize_state(resting_state(), self.config)

    def _lookup(self, voltage: int) -> tuple[int, ...]:
        config = self.config
        shift = config.voltage_frac_bits - config.lut_step_frac_bits
        grid_voltage = voltage >> shift
        minimum = config.lut_min_voltage << config.lut_step_frac_bits
        address = _saturate(grid_voltage - minimum, 0, config.lut_size - 1)
        return tuple(table[address] for table in self.tables)

    def _mul_gate(self, left: int, right: int) -> int:
        return _round_div(left * right, self.config.gate_scale)

    def _mul_rate_gate(self, rate: int, gate: int) -> int:
        # Both defaults use Q18; retain the general conversion explicitly.
        product = rate * gate
        return _round_div(product, self.config.rate_scale)

    def _substep(self, current: int) -> tuple[HodgkinHuxleyFixedState, bool]:
        config = self.config
        old = self.state
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._lookup(old.voltage)
        one = config.gate_scale

        m2 = self._mul_gate(old.sodium_activation, old.sodium_activation)
        m3 = self._mul_gate(m2, old.sodium_activation)
        m3h = self._mul_gate(m3, old.sodium_inactivation)
        n2 = self._mul_gate(old.potassium_activation, old.potassium_activation)
        n4 = self._mul_gate(n2, n2)

        sodium = 120 * _round_div(
            m3h * (old.voltage - 115 * config.voltage_scale), config.gate_scale
        )
        potassium = 36 * _round_div(
            n4 * (old.voltage + 12 * config.voltage_scale), config.gate_scale
        )
        leak = _round_div(
            3 * (old.voltage - round(10.613 * config.voltage_scale)), 10
        )
        voltage_derivative = current * config.voltage_scale - sodium - potassium - leak

        dm = self._mul_rate_gate(alpha_m, one - old.sodium_activation) - self._mul_rate_gate(beta_m, old.sodium_activation)
        dh = self._mul_rate_gate(alpha_h, one - old.sodium_inactivation) - self._mul_rate_gate(beta_h, old.sodium_inactivation)
        dn = self._mul_rate_gate(alpha_n, one - old.potassium_activation) - self._mul_rate_gate(beta_n, old.potassium_activation)

        voltage_min, voltage_max = _signed_limits(config.voltage_width)
        next_state = HodgkinHuxleyFixedState(
            voltage=_saturate(old.voltage + _round_div(voltage_derivative, config.internal_steps), voltage_min, voltage_max),
            sodium_activation=_saturate(old.sodium_activation + _round_div(dm, config.internal_steps), 0, one),
            sodium_inactivation=_saturate(old.sodium_inactivation + _round_div(dh, config.internal_steps), 0, one),
            potassium_activation=_saturate(old.potassium_activation + _round_div(dn, config.internal_steps), 0, one),
        )
        threshold = config.spike_threshold * config.voltage_scale
        spike = old.voltage < threshold <= next_state.voltage
        self.state = next_state
        return next_state, spike

    def step(self, current: int, *,
             voltage_observer: Callable[[int, int, bool], None] | None = None
             ) -> HodgkinHuxleyFixedResult:
        if not isinstance(current, int):
            raise TypeError("fixed-point current must be an integer in uA/cm^2")
        first_spike: int | None = None
        for index in range(self.config.internal_steps):
            _, spike = self._substep(current)
            if voltage_observer is not None:
                voltage_observer(index, self.state.voltage, spike)
            if spike and first_spike is None:
                first_spike = index
        return HodgkinHuxleyFixedResult(self.state, first_spike is not None, first_spike)
