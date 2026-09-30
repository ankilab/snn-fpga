"""Differentiable, fixed-point-faithful Hodgkin-Huxley QAT primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from snn_reference.hodgkin_huxley import resting_state
from snn_reference.hodgkin_huxley_fixed import (
    HodgkinHuxleyFixedConfig,
    build_rate_tables,
    quantize_state,
)


def _ste_round_div(values: Tensor, divisor: int) -> Tensor:
    """Round like RTL in the forward pass and differentiate the division."""
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    scaled = values / float(divisor)
    rounded = torch.sign(values) * torch.floor(values.abs() / float(divisor) + 0.5)
    return scaled + (rounded - scaled).detach()


def _ste_saturate(values: Tensor, minimum: int, maximum: int) -> Tensor:
    """Saturate in the forward pass while retaining an identity STE."""
    saturated = values.clamp(float(minimum), float(maximum))
    return values + (saturated - values).detach()


def _ste_signed_wrap(values: Tensor, width: int) -> Tensor:
    modulus = float(1 << width)
    half = float(1 << (width - 1))
    wrapped = torch.remainder(values + half, modulus) - half
    return values + (wrapped - values).detach()


class _CrossingSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance: Tensor, crossing: Tensor) -> Tensor:
        ctx.save_for_backward(distance)
        return crossing.to(distance.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        (distance,) = ctx.saved_tensors
        return gradient / (1.0 + distance.abs()).square(), None


@dataclass
class HHQATState:
    """Raw fixed-point population state stored as Float64 integer values."""

    voltage: Tensor
    sodium_activation: Tensor
    sodium_inactivation: Tensor
    potassium_activation: Tensor

    def detach(self) -> "HHQATState":
        return HHQATState(
            self.voltage.detach(),
            self.sodium_activation.detach(),
            self.sodium_inactivation.detach(),
            self.potassium_activation.detach(),
        )


class HHQATDynamics(nn.Module):
    """Vectorized HH network-timestep update with an exact forward contract.

    Stored states and arithmetic use Float64 so every integer intermediate in
    the current Q12/Q18 contract is represented exactly. Discrete operations
    expose straight-through or interpolated gradients only in the backward
    pass.
    """

    def __init__(
        self,
        *,
        internal_steps: int = 100,
        surrogate_width_mv: float = 5.0,
        checkpoint_interval: int = 0,
        backend: str = "eager",
    ) -> None:
        super().__init__()
        if internal_steps <= 0:
            raise ValueError("HH internal_steps must be positive")
        if surrogate_width_mv <= 0.0:
            raise ValueError("HH surrogate_width_mv must be positive")
        if checkpoint_interval < 0:
            raise ValueError("HH checkpoint_interval must be non-negative")
        if backend not in ("eager", "compiled"):
            raise ValueError("HH backend must be eager or compiled")
        self.config = HodgkinHuxleyFixedConfig(internal_steps=internal_steps)
        self.surrogate_width_mv = float(surrogate_width_mv)
        self.checkpoint_interval = int(checkpoint_interval)
        self.backend = backend
        self._compiled_substep = None
        self.register_buffer(
            "rate_tables",
            torch.tensor(build_rate_tables(self.config), dtype=torch.float64),
            persistent=False,
        )
        initial = quantize_state(resting_state(), self.config)
        self._initial_values = (
            initial.voltage,
            initial.sodium_activation,
            initial.sodium_inactivation,
            initial.potassium_activation,
        )
        self._collect_diagnostics = False
        self._diagnostics: dict[str, Tensor] = {}

    def begin_diagnostics(self, enabled: bool) -> None:
        self._collect_diagnostics = bool(enabled)
        self._diagnostics = {}

    def diagnostics(self) -> dict[str, Tensor]:
        return self._diagnostics

    def _record_count(self, name: str, mask: Tensor) -> None:
        if not self._collect_diagnostics:
            return
        value = mask.detach().sum()
        self._diagnostics[name] = self._diagnostics.get(
            name, torch.zeros_like(value)
        ) + value

    def initial_state(self, reference: Tensor, size: int) -> HHQATState:
        shape = (reference.shape[0], size)
        values = [
            torch.full(
                shape,
                float(value),
                dtype=torch.float64,
                device=reference.device,
            )
            for value in self._initial_values
        ]
        return HHQATState(*values)

    def _lookup_rates(self, voltage: Tensor) -> tuple[Tensor, ...]:
        config = self.config
        address_scale = 1 << config.lut_step_frac_bits
        address_position = (
            voltage / float(1 << (config.voltage_frac_bits - config.lut_step_frac_bits))
            - float(config.lut_min_voltage * address_scale)
        )
        if self._collect_diagnostics:
            self._record_count(
                "lut_clamps",
                (address_position < 0.0)
                | (address_position > float(config.lut_size - 1)),
            )
        exact_address = torch.floor(address_position).clamp(
            0, config.lut_size - 1
        )
        exact_index = exact_address.to(torch.long)

        # The exact floor-indexed value is used forward. Linear interpolation
        # supplies the otherwise missing voltage derivative in backward.
        soft_position = address_position.clamp(0.0, float(config.lut_size - 1))
        lower = torch.floor(soft_position).to(torch.long)
        upper = (lower + 1).clamp(max=config.lut_size - 1)
        fraction = soft_position - lower.to(soft_position.dtype)
        rates: list[Tensor] = []
        for table in self.rate_tables:
            exact = table[exact_index]
            soft = table[lower] + fraction * (table[upper] - table[lower])
            rates.append(soft + (exact - soft).detach())
        return tuple(rates)

    def _substep_tensors(
        self,
        voltage: Tensor,
        sodium_activation: Tensor,
        sodium_inactivation: Tensor,
        potassium_activation: Tensor,
        current: Tensor,
        spike: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        config = self.config
        gate_scale = config.gate_scale
        voltage_scale = config.voltage_scale
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._lookup_rates(
            voltage
        )

        m2 = _ste_round_div(sodium_activation * sodium_activation, gate_scale)
        m3 = _ste_round_div(m2 * sodium_activation, gate_scale)
        m3h = _ste_round_div(m3 * sodium_inactivation, gate_scale)
        n2 = _ste_round_div(potassium_activation * potassium_activation, gate_scale)
        n4 = _ste_round_div(n2 * n2, gate_scale)

        sodium = 120.0 * _ste_round_div(
            m3h * (voltage - 115.0 * voltage_scale), gate_scale
        )
        potassium = 36.0 * _ste_round_div(
            n4 * (voltage + 12.0 * voltage_scale), gate_scale
        )
        leak = _ste_round_div(
            3.0 * (voltage - float(round(10.613 * voltage_scale))), 10
        )
        derivative = current * voltage_scale - sodium - potassium - leak

        dm = _ste_round_div(
            alpha_m * (gate_scale - sodium_activation), config.rate_scale
        ) - _ste_round_div(beta_m * sodium_activation, config.rate_scale)
        dh = _ste_round_div(
            alpha_h * (gate_scale - sodium_inactivation), config.rate_scale
        ) - _ste_round_div(beta_h * sodium_inactivation, config.rate_scale)
        dn = _ste_round_div(
            alpha_n * (gate_scale - potassium_activation), config.rate_scale
        ) - _ste_round_div(beta_n * potassium_activation, config.rate_scale)

        voltage_min = -(1 << (config.voltage_width - 1))
        voltage_max = (1 << (config.voltage_width - 1)) - 1
        raw_voltage = voltage + _ste_round_div(derivative, config.internal_steps)
        raw_m = sodium_activation + _ste_round_div(dm, config.internal_steps)
        raw_h = sodium_inactivation + _ste_round_div(dh, config.internal_steps)
        raw_n = potassium_activation + _ste_round_div(dn, config.internal_steps)
        if self._collect_diagnostics:
            self._record_count(
                "voltage_saturations",
                (raw_voltage < voltage_min) | (raw_voltage > voltage_max),
            )
            self._record_count(
                "gate_saturations",
                ((raw_m < 0) | (raw_m > gate_scale)).to(torch.int64)
                + ((raw_h < 0) | (raw_h > gate_scale)).to(torch.int64)
                + ((raw_n < 0) | (raw_n > gate_scale)).to(torch.int64),
            )
        next_voltage = _ste_saturate(
            raw_voltage,
            voltage_min,
            voltage_max,
        )
        next_m = _ste_saturate(
            raw_m,
            0,
            gate_scale,
        )
        next_h = _ste_saturate(
            raw_h,
            0,
            gate_scale,
        )
        next_n = _ste_saturate(
            raw_n,
            0,
            gate_scale,
        )

        threshold = float(config.spike_threshold * voltage_scale)
        exact_crossing = (voltage < threshold) & (next_voltage >= threshold)
        distance = (next_voltage - threshold) / (
            self.surrogate_width_mv * voltage_scale
        )
        crossing = _CrossingSurrogate.apply(distance, exact_crossing)
        spike = 1.0 - (1.0 - spike) * (1.0 - crossing)
        return next_voltage, next_m, next_h, next_n, spike

    def forward(self, state: HHQATState, current: Tensor) -> tuple[HHQATState, Tensor]:
        substep = self._substep_tensors
        if self.backend == "compiled":
            if current.device.type != "cuda":
                raise ValueError("the compiled HH backend requires CUDA")
            # Keep diagnostic collection eager: its accumulating counters are
            # deliberately outside the pure compiled arithmetic. Compile one
            # Euler substep, not the 100-step recurrence, to bound compile cost.
            # Dynamic shapes share kernels across layers and partial batches.
            if not self._collect_diagnostics:
                if self._compiled_substep is None:
                    self._compiled_substep = torch.compile(
                        self._substep_tensors, fullgraph=True, dynamic=True,
                    )
                substep = self._compiled_substep
        current64 = _ste_signed_wrap(current, 24).to(torch.float64)
        tensors = (
            state.voltage,
            state.sodium_activation,
            state.sodium_inactivation,
            state.potassium_activation,
            torch.zeros_like(state.voltage),
        )
        interval = self.checkpoint_interval or self.config.internal_steps
        for start in range(0, self.config.internal_steps, interval):
            steps = min(interval, self.config.internal_steps - start)

            def segment(*values: Tensor, count: int = steps):
                result = values
                for _ in range(count):
                    result = substep(*result[:4], current64, result[4])
                return result

            use_checkpoint = (
                self.training
                and self.checkpoint_interval > 0
                and torch.is_grad_enabled()
                and any(value.requires_grad for value in (*tensors, current64))
            )
            tensors = (
                checkpoint(segment, *tensors, use_reentrant=False)
                if use_checkpoint
                else segment(*tensors)
            )
        return HHQATState(*tensors[:4]), tensors[4]
