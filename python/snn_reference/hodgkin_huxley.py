"""Float64 reference for the original squid-axon Hodgkin-Huxley dynamics.

The implementation uses a depolarization-positive, rest-relative voltage. It
is the sign-transformed equivalent of the 1952 paper's convention, in which
depolarization is negative. Voltage is in mV, time in ms, current density in
uA/cm^2, capacitance in uF/cm^2 and conductance in mS/cm^2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


@dataclass(frozen=True)
class HodgkinHuxleyConfig:
    capacitance: float = 1.0
    sodium_conductance: float = 120.0
    potassium_conductance: float = 36.0
    leak_conductance: float = 0.3
    sodium_reversal: float = 115.0
    potassium_reversal: float = -12.0
    leak_reversal: float = 10.613
    temperature_celsius: float = 6.3
    q10: float = 3.0
    spike_threshold: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "capacitance",
            "sodium_conductance",
            "potassium_conductance",
            "leak_conductance",
            "q10",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "sodium_reversal",
            "potassium_reversal",
            "leak_reversal",
            "temperature_celsius",
            "spike_threshold",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    @property
    def rate_scale(self) -> float:
        return self.q10 ** ((self.temperature_celsius - 6.3) / 10.0)


@dataclass(frozen=True)
class HodgkinHuxleyState:
    voltage: float
    sodium_activation: float
    sodium_inactivation: float
    potassium_activation: float


@dataclass(frozen=True)
class HodgkinHuxleyTrace:
    times: tuple[float, ...]
    states: tuple[HodgkinHuxleyState, ...]
    spike_times: tuple[float, ...]


def _vtrap(value: float, scale: float) -> float:
    """Return value/expm1(value/scale), including its removable singularity."""
    ratio = value / scale
    if abs(ratio) < 1.0e-7:
        # Series around zero avoids cancellation and is accurate through x^2.
        return scale * (1.0 - ratio / 2.0 + ratio * ratio / 12.0)
    return value / math.expm1(ratio)


def rate_constants(voltage: float) -> tuple[float, float, float, float, float, float]:
    """Return alpha_m, beta_m, alpha_h, beta_h, alpha_n and beta_n."""
    alpha_m = 0.1 * _vtrap(25.0 - voltage, 10.0)
    beta_m = 4.0 * math.exp(-voltage / 18.0)
    alpha_h = 0.07 * math.exp(-voltage / 20.0)
    beta_h = 1.0 / (math.exp((30.0 - voltage) / 10.0) + 1.0)
    alpha_n = 0.01 * _vtrap(10.0 - voltage, 10.0)
    beta_n = 0.125 * math.exp(-voltage / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def resting_state(config: HodgkinHuxleyConfig = HodgkinHuxleyConfig()) -> HodgkinHuxleyState:
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = rate_constants(0.0)
    return HodgkinHuxleyState(
        voltage=0.0,
        sodium_activation=alpha_m / (alpha_m + beta_m),
        sodium_inactivation=alpha_h / (alpha_h + beta_h),
        potassium_activation=alpha_n / (alpha_n + beta_n),
    )


def derivatives(
    state: HodgkinHuxleyState,
    current: float,
    config: HodgkinHuxleyConfig = HodgkinHuxleyConfig(),
) -> HodgkinHuxleyState:
    if not math.isfinite(current):
        raise ValueError("current must be finite")
    voltage = state.voltage
    m = state.sodium_activation
    h = state.sodium_inactivation
    n = state.potassium_activation
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = rate_constants(voltage)
    sodium_current = (
        config.sodium_conductance * m**3 * h * (voltage - config.sodium_reversal)
    )
    potassium_current = (
        config.potassium_conductance * n**4 * (voltage - config.potassium_reversal)
    )
    leak_current = config.leak_conductance * (voltage - config.leak_reversal)
    rate_scale = config.rate_scale
    return HodgkinHuxleyState(
        voltage=(current - sodium_current - potassium_current - leak_current)
        / config.capacitance,
        sodium_activation=rate_scale * (alpha_m * (1.0 - m) - beta_m * m),
        sodium_inactivation=rate_scale * (alpha_h * (1.0 - h) - beta_h * h),
        potassium_activation=rate_scale * (alpha_n * (1.0 - n) - beta_n * n),
    )


def _combine(
    state: HodgkinHuxleyState,
    derivative: HodgkinHuxleyState,
    factor: float,
) -> HodgkinHuxleyState:
    return HodgkinHuxleyState(
        voltage=state.voltage + factor * derivative.voltage,
        sodium_activation=(
            state.sodium_activation + factor * derivative.sodium_activation
        ),
        sodium_inactivation=(
            state.sodium_inactivation + factor * derivative.sodium_inactivation
        ),
        potassium_activation=(
            state.potassium_activation + factor * derivative.potassium_activation
        ),
    )


def rk4_step(
    state: HodgkinHuxleyState,
    current: float,
    dt: float,
    config: HodgkinHuxleyConfig = HodgkinHuxleyConfig(),
) -> HodgkinHuxleyState:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    k1 = derivatives(state, current, config)
    k2 = derivatives(_combine(state, k1, dt / 2.0), current, config)
    k3 = derivatives(_combine(state, k2, dt / 2.0), current, config)
    k4 = derivatives(_combine(state, k3, dt), current, config)
    return HodgkinHuxleyState(
        voltage=state.voltage
        + dt * (k1.voltage + 2.0 * k2.voltage + 2.0 * k3.voltage + k4.voltage)
        / 6.0,
        sodium_activation=state.sodium_activation
        + dt
        * (
            k1.sodium_activation
            + 2.0 * k2.sodium_activation
            + 2.0 * k3.sodium_activation
            + k4.sodium_activation
        )
        / 6.0,
        sodium_inactivation=state.sodium_inactivation
        + dt
        * (
            k1.sodium_inactivation
            + 2.0 * k2.sodium_inactivation
            + 2.0 * k3.sodium_inactivation
            + k4.sodium_inactivation
        )
        / 6.0,
        potassium_activation=state.potassium_activation
        + dt
        * (
            k1.potassium_activation
            + 2.0 * k2.potassium_activation
            + 2.0 * k3.potassium_activation
            + k4.potassium_activation
        )
        / 6.0,
    )


def simulate(
    duration: float,
    dt: float,
    current: Callable[[float], float],
    *,
    config: HodgkinHuxleyConfig = HodgkinHuxleyConfig(),
    initial_state: HodgkinHuxleyState | None = None,
    state_transform: Callable[[HodgkinHuxleyState], HodgkinHuxleyState] | None = None,
) -> HodgkinHuxleyTrace:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    steps_float = duration / dt
    steps = round(steps_float)
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("duration must be an integer multiple of dt")
    state = resting_state(config) if initial_state is None else initial_state
    times = [0.0]
    states = [state]
    spikes: list[float] = []
    for index in range(steps):
        time = index * dt
        next_state = rk4_step(state, float(current(time + dt / 2.0)), dt, config)
        if state_transform is not None:
            next_state = state_transform(next_state)
        if state.voltage < config.spike_threshold <= next_state.voltage:
            fraction = (config.spike_threshold - state.voltage) / (
                next_state.voltage - state.voltage
            )
            spikes.append(time + fraction * dt)
        state = next_state
        times.append((index + 1) * dt)
        states.append(state)
    return HodgkinHuxleyTrace(tuple(times), tuple(states), tuple(spikes))
