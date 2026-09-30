"""Model-independent diagnostics for bit-exact inference and QAT gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Mapping, Sequence

from snn_reference import InferenceResult, SpikeEvent, TwoLayerSNN


@dataclass
class _RunningStats:
    count: int = 0
    minimum: int | None = None
    maximum: int | None = None
    total: int = 0
    absolute_total: int = 0

    def add(self, values: Sequence[int]) -> None:
        if not values:
            return
        self.count += len(values)
        local_minimum = min(values)
        local_maximum = max(values)
        self.minimum = (
            local_minimum if self.minimum is None else min(self.minimum, local_minimum)
        )
        self.maximum = (
            local_maximum if self.maximum is None else max(self.maximum, local_maximum)
        )
        self.total += sum(values)
        self.absolute_total += sum(abs(value) for value in values)

    def summary(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count if self.count else None,
            "mean_abs": self.absolute_total / self.count if self.count else None,
        }


class _LayerAccumulator:
    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self.current = _RunningStats()
        self.membrane = _RunningStats()
        self.recovery = _RunningStats()
        self.has_recovery = False
        self.spikes_total = 0
        self.spikes_by_neuron = [0] * neurons
        self.spikes_by_timestep: list[int] = []
        self.current_wraps = 0
        self.membrane_wraps = 0
        self.recovery_wraps = 0
        self.current_wraps_by_timestep: list[int] = []
        self.membrane_wraps_by_timestep: list[int] = []
        self.recovery_wraps_by_timestep: list[int] = []

    def add(
        self,
        *,
        timestamp: int,
        current: Sequence[int],
        membrane: Sequence[int],
        recovery: Sequence[int] | None,
        spikes: Sequence[bool],
        current_wraps: int,
        membrane_wraps: int,
        recovery_wraps: int,
    ) -> None:
        self.current.add(current)
        self.membrane.add(membrane)
        if recovery is not None:
            self.has_recovery = True
            self.recovery.add(recovery)
        while len(self.spikes_by_timestep) <= timestamp:
            self.spikes_by_timestep.append(0)
            self.current_wraps_by_timestep.append(0)
            self.membrane_wraps_by_timestep.append(0)
            self.recovery_wraps_by_timestep.append(0)
        timestep_spikes = sum(spikes)
        self.spikes_by_timestep[timestamp] += timestep_spikes
        self.spikes_total += timestep_spikes
        for index, spike in enumerate(spikes):
            self.spikes_by_neuron[index] += int(spike)
        self.current_wraps += current_wraps
        self.membrane_wraps += membrane_wraps
        self.recovery_wraps += recovery_wraps
        self.current_wraps_by_timestep[timestamp] += current_wraps
        self.membrane_wraps_by_timestep[timestamp] += membrane_wraps
        self.recovery_wraps_by_timestep[timestamp] += recovery_wraps

    def summary(self, samples: int) -> dict[str, object]:
        active = sum(count > 0 for count in self.spikes_by_neuron)
        return {
            "neurons": self.neurons,
            "current": self.current.summary(),
            "membrane": self.membrane.summary(),
            "recovery": self.recovery.summary() if self.has_recovery else None,
            "spikes_total": self.spikes_total,
            "spikes_per_sample": self.spikes_total / samples if samples else 0.0,
            "active_neurons": active,
            "silent_neurons": self.neurons - active,
            "active_neuron_fraction": active / self.neurons,
            "spikes_by_neuron": self.spikes_by_neuron,
            "spikes_by_timestep": self.spikes_by_timestep,
            "wraps": {
                "current": self.current_wraps,
                "membrane": self.membrane_wraps,
                "recovery": self.recovery_wraps,
            },
            "wraps_by_timestep": {
                "current": self.current_wraps_by_timestep,
                "membrane": self.membrane_wraps_by_timestep,
                "recovery": self.recovery_wraps_by_timestep,
            },
        }


def _weight_summary(values: Sequence[Sequence[int]]) -> dict[str, int | float | None]:
    flattened = [value for row in values for value in row]
    if not flattened:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "standard_deviation": None,
            "zeros": 0,
            "saturated_int8": 0,
            "saturated_int8_fraction": 0.0,
        }
    mean = sum(flattened) / len(flattened)
    variance = sum((value - mean) ** 2 for value in flattened) / len(flattened)
    saturated = sum(value in (-128, 127) for value in flattened)
    return {
        "count": len(flattened),
        "min": min(flattened),
        "max": max(flattened),
        "mean": mean,
        "standard_deviation": math.sqrt(variance),
        "zeros": sum(value == 0 for value in flattened),
        "saturated_int8": saturated,
        "saturated_int8_fraction": saturated / len(flattened),
    }


class ReferenceDiagnostics:
    """Aggregate comparable diagnostics from traced bit-exact inferences."""

    def __init__(self, model: TwoLayerSNN) -> None:
        self.model = model
        config = model.config
        self.samples = 0
        self.correct = 0
        self.no_prediction = 0
        self.output_ties = 0
        self.margin = _RunningStats()
        self.predicted_classes = [0] * config.output_neurons
        self.confusion_matrix = [
            [0] * config.output_neurons for _ in range(config.output_neurons)
        ]
        self.hidden = _LayerAccumulator(config.hidden_neurons)
        self.output = _LayerAccumulator(config.output_neurons)

    def add(self, result: InferenceResult, label: int) -> None:
        config = self.model.config
        if not 0 <= label < config.output_neurons:
            raise ValueError("label is outside the configured output range")
        if not result.timesteps:
            raise ValueError("diagnostics require infer(..., trace=True)")

        self.samples += 1
        prediction = result.prediction
        if prediction is None:
            self.no_prediction += 1
        else:
            self.predicted_classes[prediction] += 1
            self.confusion_matrix[label][prediction] += 1
            self.correct += prediction == label

        ordered_counts = sorted(result.class_counts, reverse=True)
        maximum = ordered_counts[0]
        if maximum > 0:
            self.output_ties += sum(value == maximum for value in result.class_counts) > 1
            runner_up = ordered_counts[1] if len(ordered_counts) > 1 else 0
            self.margin.add((maximum - runner_up,))

        for timestep in result.timesteps:
            self.hidden.add(
                timestamp=timestep.timestamp,
                current=timestep.hidden_current,
                membrane=timestep.hidden_membrane,
                recovery=timestep.hidden_recovery,
                spikes=timestep.hidden_spikes,
                current_wraps=timestep.hidden_current_wraps,
                membrane_wraps=timestep.hidden_membrane_wraps,
                recovery_wraps=timestep.hidden_recovery_wraps,
            )
            self.output.add(
                timestamp=timestep.timestamp,
                current=timestep.output_current,
                membrane=timestep.output_membrane,
                recovery=timestep.output_recovery,
                spikes=timestep.output_spikes,
                current_wraps=timestep.output_current_wraps,
                membrane_wraps=timestep.output_membrane_wraps,
                recovery_wraps=timestep.output_recovery_wraps,
            )

    def summary(self) -> dict[str, object]:
        config = self.model.config
        return {
            "schema": "snn-diagnostics-v1",
            "samples": self.samples,
            "accuracy": self.correct / self.samples if self.samples else 0.0,
            "correct": self.correct,
            "no_prediction": self.no_prediction,
            "no_prediction_rate": (
                self.no_prediction / self.samples if self.samples else 0.0
            ),
            "output_ties": self.output_ties,
            "output_tie_rate": self.output_ties / self.samples if self.samples else 0.0,
            "output_margin": self.margin.summary(),
            "predicted_classes": self.predicted_classes,
            "confusion_matrix": self.confusion_matrix,
            "models": {
                "hidden": config.hidden_neuron_model.name.lower(),
                "output": config.output_neuron_model.name.lower(),
            },
            "timesteps_configured": None,
            "layers": {
                "hidden": self.hidden.summary(self.samples),
                "output": self.output.summary(self.samples),
            },
            "weights": {
                "input_hidden": _weight_summary(self.model.input_hidden_weights),
                "hidden_output": _weight_summary(self.model.hidden_output_weights),
            },
        }


def diagnose_reference(
    model: TwoLayerSNN,
    examples: Iterable[tuple[Iterable[SpikeEvent], int]],
    *,
    timesteps: int | None = None,
) -> dict[str, object]:
    """Run and summarize labelled examples through the bit-exact reference."""
    diagnostics = ReferenceDiagnostics(model)
    iterator = iter(examples)
    infer_batch = getattr(model, "infer_batch", None)
    if infer_batch is None:
        for events, label in iterator:
            diagnostics.add(
                model.infer(events, trace=True, timesteps=timesteps), int(label)
            )
        return diagnostics.summary()

    while chunk := tuple(islice(iterator, 64)):
        results = infer_batch(
            tuple(events for events, _ in chunk), trace=True, timesteps=timesteps
        )
        for result, (_, label) in zip(results, chunk, strict=True):
            diagnostics.add(result, int(label))
    return diagnostics.summary()


class GradientAccumulator:
    """Accumulate per-parameter gradient health metrics across optimizer steps."""

    def __init__(self) -> None:
        self._values: dict[str, dict[str, float | int]] = {}

    def add(self, parameters: Mapping[str, object]) -> None:
        for name, parameter in parameters.items():
            gradient = getattr(parameter, "grad", None)
            if gradient is None:
                continue
            import torch

            values = gradient.detach()
            finite = torch.isfinite(values)
            finite_values = values[finite]
            record = self._values.setdefault(
                name,
                {
                    "batches": 0,
                    "l2_norm_total": 0.0,
                    "absolute_total": 0.0,
                    "finite_elements": 0,
                    "zero_elements": 0,
                    "nonfinite_elements": 0,
                },
            )
            record["batches"] += 1
            record["nonfinite_elements"] += int((~finite).sum().item())
            if finite_values.numel():
                record["l2_norm_total"] += float(
                    torch.linalg.vector_norm(finite_values.to(torch.float64))
                )
                record["absolute_total"] += float(finite_values.abs().sum().item())
                record["finite_elements"] += finite_values.numel()
                record["zero_elements"] += int((finite_values == 0).sum().item())

    def summary(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, values in self._values.items():
            batches = int(values["batches"])
            finite_elements = int(values["finite_elements"])
            result[name] = {
                "batches": batches,
                "l2_norm_mean": float(values["l2_norm_total"]) / batches,
                "mean_abs": (
                    float(values["absolute_total"]) / finite_elements
                    if finite_elements
                    else 0.0
                ),
                "zero_fraction": (
                    int(values["zero_elements"]) / finite_elements
                    if finite_elements
                    else 0.0
                ),
                "nonfinite_elements": int(values["nonfinite_elements"]),
            }
        return result
