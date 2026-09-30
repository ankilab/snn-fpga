"""Command-line training entry point for the hardware-oriented MNIST SNN."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from snn_reference import NeuronModel, NumericConfig, SNNConfig
from snn_training.diagnostics import GradientAccumulator
from snn_training.encoding import INPUT_ENCODINGS
from snn_training.export import export_model, load_exported_model
from snn_training.torch_model import HardwareQATSNN


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mnist_int8"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="planned resumable stop after this completed epoch",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="reserve this many final training examples for checkpoint selection",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop after this many non-improving epochs; zero disables",
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument(
        "--lr-reduce-patience",
        type=int,
        default=0,
        help="ReduceLROnPlateau patience; zero disables the scheduler",
    )
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="clip the global gradient L2 norm; zero disables clipping",
    )
    parser.add_argument(
        "--layer0-column-mean-penalty",
        type=float,
        default=0.0,
        help="L2 penalty on layer-0 column means to reduce background current",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initial-state",
        type=Path,
        help="optional training_state.pt used to initialize both weight matrices",
    )
    initialization.add_argument(
        "--initial-model-dir",
        type=Path,
        help="optional exported INT8 package used to initialize both weight matrices",
    )
    initialization.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="resume an interrupted run from an epoch checkpoint",
    )
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument(
        "--input-encoding",
        choices=INPUT_ENCODINGS,
        default="latency",
        help="input spike encoding used during QAT",
    )
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Bernoulli probability multiplier for rate encoding",
    )
    parser.add_argument(
        "--rate-seed",
        type=int,
        help="base rate-encoding seed; defaults to --seed",
    )
    parser.add_argument(
        "--encoding-threshold",
        type=int,
        default=0,
        help="minimum active pixel for sparse latency, top-k and burst encoding",
    )
    parser.add_argument(
        "--latency-gamma",
        type=float,
        default=2.0,
        help="positive gamma exponent for nonlinear latency encoding",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=128,
        help="maximum active pixels emitted by top-k rank-order encoding",
    )
    parser.add_argument(
        "--burst-max-spikes",
        type=int,
        default=4,
        help="maximum spikes emitted per active pixel by burst encoding",
    )
    parser.add_argument(
        "--burst-interval",
        type=int,
        default=1,
        help="timesteps between spikes in a burst",
    )
    parser.add_argument(
        "--bptt-window",
        type=int,
        default=0,
        help="detach recurrent state after this many timesteps; zero uses full BPTT",
    )
    parser.add_argument("--hidden-neurons", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--leak-shift", type=int, default=3)
    parser.add_argument(
        "--hidden-neuron-model",
        choices=("lif", "izhikevich", "hodgkin_huxley"),
        default="lif",
    )
    parser.add_argument(
        "--output-neuron-model",
        choices=("lif", "izhikevich", "hodgkin_huxley"),
        default="lif",
    )
    parser.add_argument(
        "--hh-surrogate-width-mv",
        type=float,
        default=5.0,
        help="voltage scale of the HH threshold-crossing surrogate gradient",
    )
    parser.add_argument(
        "--hh-backend",
        choices=("eager", "compiled"),
        default="eager",
        help="compiled fuses HH Euler substeps with torch.compile on CUDA; first use compiles",
    )
    parser.add_argument(
        "--hh-checkpoint-interval",
        type=int,
        default=10,
        help="recompute HH substeps in backward in chunks; zero disables",
    )
    parser.add_argument(
        "--freeze-readout-epochs",
        type=int,
        default=0,
        help="train only layer 0 for this many initial epochs",
    )
    parser.add_argument(
        "--minimum-finetune-epochs",
        type=int,
        default=1,
        help="minimum joint-training epochs before early stopping is allowed",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="epoch checkpoint path; defaults to OUTPUT_DIR/checkpoint.pt",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--training-observability", action="store_true",
                        help="evaluate the training split, time phases and retain every epoch's weights")
    parser.add_argument("--skip-final-test", action="store_true",
                        help="do not inspect test data for an intermediate training stage")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _limit(dataset, maximum: int | None):
    return dataset if maximum is None else Subset(dataset, range(min(maximum, len(dataset))))


def _metrics_detailed(
    model: HardwareQATSNN, loader: DataLoader, device: torch.device,
    *, detailed_hh: bool = True,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    cross_entropy_sum = 0.0
    no_prediction = 0
    ties = 0
    hidden_spikes = 0.0
    output_spikes = 0.0
    hh_voltage_min: float | None = None
    hh_voltage_max: float | None = None
    hh_gate_min: float | None = None
    hh_gate_max: float | None = None
    hh_lut_clamps = 0
    hh_voltage_saturations = 0
    hh_gate_saturations = 0
    model.set_hh_diagnostics(detailed_hh)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            counts = model(images, sample_offset=total)
            cross_entropy_sum += float(nn.functional.cross_entropy(counts, labels, reduction="sum").item())
            valid = counts.ne(0).any(dim=1)
            predictions = counts.argmax(dim=1)
            maxima = counts.max(dim=1, keepdim=True).values
            ties += int(((counts == maxima).sum(dim=1) > 1).sum().item())
            correct += int(((predictions == labels) & valid).sum().item())
            no_prediction += int((~valid).sum().item())
            total += labels.numel()
            diagnostics = model.forward_diagnostics()
            hidden_spikes += float(
                diagnostics["hidden_spikes_by_timestep"].sum().item()
            )
            output_spikes += float(
                diagnostics["output_spikes_by_timestep"].sum().item()
            )
            if detailed_hh and "hh_voltage_min" in diagnostics:
                local_voltage_min = float(diagnostics["hh_voltage_min"].item())
                local_voltage_max = float(diagnostics["hh_voltage_max"].item())
                local_gate_min = float(diagnostics["hh_gate_min"].item())
                local_gate_max = float(diagnostics["hh_gate_max"].item())
                hh_voltage_min = (
                    local_voltage_min
                    if hh_voltage_min is None
                    else min(hh_voltage_min, local_voltage_min)
                )
                hh_voltage_max = (
                    local_voltage_max
                    if hh_voltage_max is None
                    else max(hh_voltage_max, local_voltage_max)
                )
                hh_gate_min = (
                    local_gate_min
                    if hh_gate_min is None
                    else min(hh_gate_min, local_gate_min)
                )
                hh_gate_max = (
                    local_gate_max
                    if hh_gate_max is None
                    else max(hh_gate_max, local_gate_max)
                )
                hh_lut_clamps += int(diagnostics["lut_clamps"].item())
                hh_voltage_saturations += int(
                    diagnostics["voltage_saturations"].item()
                )
                hh_gate_saturations += int(
                    diagnostics["gate_saturations"].item()
                )
    model.set_hh_diagnostics(False)
    if not total:
        return {"accuracy": 0.0, "no_prediction_rate": 0.0, "tie_rate": 0.0}
    result = {
        "accuracy": correct / total,
        "cross_entropy": cross_entropy_sum / total,
        "samples": total,
        "no_prediction_rate": no_prediction / total,
        "tie_rate": ties / total,
        "hidden_spikes_per_image": hidden_spikes / total,
        "output_spikes_per_image": output_spikes / total,
    }
    if hh_voltage_min is not None:
        result.update(
            {
                "hh_voltage_q12_min": hh_voltage_min,
                "hh_voltage_q12_max": hh_voltage_max,
                "hh_gate_q18_min": hh_gate_min,
                "hh_gate_q18_max": hh_gate_max,
                "hh_lut_clamps": float(hh_lut_clamps),
                "hh_voltage_saturations": float(hh_voltage_saturations),
                "hh_gate_saturations": float(hh_gate_saturations),
            }
        )
    return result


def _weight_diagnostics(
    model: HardwareQATSNN,
    initial: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, current, baseline in zip(
        ("layer0", "layer1"), model.quantized_weights(), initial, strict=True
    ):
        values = current.cpu()
        result[name] = {
            "at_int8_min": int((values == -128).sum().item()),
            "at_int8_max": int((values == 127).sum().item()),
            "changed_from_initial": int((values != baseline).sum().item()),
        }
    return result


def _metrics(
    model: HardwareQATSNN, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    metrics = _metrics_detailed(model, loader, device)
    return metrics["accuracy"], metrics["no_prediction_rate"]


def layer0_column_mean_penalty(model: HardwareQATSNN) -> torch.Tensor:
    """Penalize systematic per-hidden-neuron input-weight offsets."""
    return model.weight_l0.mean(dim=0).square().mean()


def _training_phase(epoch: int, freeze_readout_epochs: int) -> str:
    return "frozen_readout" if epoch <= freeze_readout_epochs else "joint"


def _early_stopping_eligible(
    epoch: int, freeze_readout_epochs: int, minimum_finetune_epochs: int
) -> bool:
    return epoch >= freeze_readout_epochs + minimum_finetune_epochs


def _planned_stop_reached(epoch: int, stop_after_epoch: int | None) -> bool:
    return stop_after_epoch is not None and epoch >= stop_after_epoch


def _step_plateau_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    optimizer: torch.optim.Optimizer,
    metric: float,
    checkpoint: "BestCheckpoint",
) -> tuple[float, bool]:
    """Step the scheduler and give a reduced LR fresh stopping patience."""
    previous_lr = float(optimizer.param_groups[0]["lr"])
    scheduler.step(metric)
    next_lr = float(optimizer.param_groups[0]["lr"])
    reduced = next_lr < previous_lr
    if reduced:
        checkpoint.reset_patience()
    return next_lr, reduced


@dataclass
class BestCheckpoint:
    """Track and restore the best validation-accuracy state in memory."""

    minimum_improvement: float = 0.0
    best_accuracy: float = float("-inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    state: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module, epoch: int, accuracy: float) -> bool:
        if accuracy <= self.best_accuracy + self.minimum_improvement:
            self.epochs_without_improvement += 1
            return False
        self.best_accuracy = accuracy
        self.best_epoch = epoch
        self.epochs_without_improvement = 0
        self.state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        if self.state is None:
            raise RuntimeError("no best checkpoint was recorded")
        model.load_state_dict(self.state)

    def should_stop(self, patience: int) -> bool:
        return patience > 0 and self.epochs_without_improvement >= patience

    def reset_patience(self) -> None:
        self.epochs_without_improvement = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_improvement": self.minimum_improvement,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "BestCheckpoint":
        result = cls(float(values["minimum_improvement"]))
        result.best_accuracy = float(values["best_accuracy"])
        result.best_epoch = int(values["best_epoch"])
        result.epochs_without_improvement = int(
            values["epochs_without_improvement"]
        )
        result.state = values["state"]  # type: ignore[assignment]
        return result


def _training_contract(args: argparse.Namespace, config: SNNConfig) -> dict[str, object]:
    contract = {
        "topology": [config.input_neurons, config.hidden_neurons, config.output_neurons],
        "hidden_neuron_model": int(config.hidden_neuron_model),
        "output_neuron_model": int(config.output_neuron_model),
        "numeric": vars(config.numeric),
        "timesteps": args.timesteps,
        "input_encoding": args.input_encoding,
        "rate_scale": args.rate_scale,
        "rate_seed": args.seed if args.rate_seed is None else args.rate_seed,
        "encoding_threshold": args.encoding_threshold,
        "latency_gamma": args.latency_gamma,
        "top_k": args.top_k,
        "burst_max_spikes": args.burst_max_spikes,
        "burst_interval": args.burst_interval,
        "bptt_window": args.bptt_window,
        "freeze_readout_epochs": args.freeze_readout_epochs,
        "minimum_finetune_epochs": args.minimum_finetune_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_improvement": args.minimum_improvement,
        "lr_reduce_patience": args.lr_reduce_patience,
        "lr_reduce_factor": args.lr_reduce_factor,
        "gradient_clip_norm": args.gradient_clip_norm,
        "layer0_column_mean_penalty": args.layer0_column_mean_penalty,
        "hh_surrogate_width_mv": args.hh_surrogate_width_mv,
        "hh_checkpoint_interval": args.hh_checkpoint_interval,
        "validation_samples": args.validation_samples,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
    }
    # Preserve compatibility with existing eager checkpoints. Compiled runs
    # record the execution backend so resume cannot silently switch numerics.
    if getattr(args, "hh_backend", "eager") != "eager":
        contract["hh_backend"] = args.hh_backend
    if getattr(args, "training_observability", False):
        contract["training_observability"] = "full-train-evaluation-and-epoch-weights-v1"
    if getattr(args, "skip_final_test", False):
        contract["skip_final_test"] = True
    return contract


def _validate_initial_model(initial, config: SNNConfig, args: argparse.Namespace) -> None:
    if initial.config != config:
        raise ValueError("initial model topology, neuron or numeric contract mismatch")
    if initial.timesteps != args.timesteps:
        raise ValueError("initial model timestep horizon mismatch")
    if initial.input_encoding != args.input_encoding:
        raise ValueError("initial model encoding mismatch")
    expected_rate_seed = args.seed if args.rate_seed is None else args.rate_seed
    expected = (
        args.rate_scale,
        expected_rate_seed,
        getattr(args, "encoding_threshold", 0),
        getattr(args, "latency_gamma", 2.0),
        getattr(args, "top_k", 128),
        getattr(args, "burst_max_spikes", 4),
        getattr(args, "burst_interval", 1),
    )
    actual = (
        initial.rate_scale,
        initial.rate_seed,
        initial.encoding_threshold,
        initial.latency_gamma,
        initial.top_k,
        initial.burst_max_spikes,
        initial.burst_interval,
    )
    relevant = {
        "rate": (0, 1),
        "sparse-latency": (2,),
        "nonlinear-latency": (3,),
        "deterministic-rate": (0,),
        "top-k-rank-order": (2, 4),
        "burst": (2, 5, 6),
    }.get(args.input_encoding, ())
    if any(actual[index] != expected[index] for index in relevant):
        raise ValueError("initial model input-encoding parameter mismatch")


def _atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _synchronized_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _save_observation(output: Path, record: dict) -> None:
    destination = output / "epoch_metrics" / f"epoch_{record['epoch']:04d}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _save_epoch_weights(output: Path, epoch: int, model: nn.Module, contract: dict) -> None:
    _atomic_torch_save({"schema": "snn-epoch-weights-v1", "epoch": epoch,
                        "contract": contract,
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}},
                       output / "epoch_weights" / f"epoch_{epoch:04d}.pt")


def _save_epoch_checkpoint(
    destination: Path,
    *,
    contract: dict[str, object],
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best: BestCheckpoint,
    history: list[dict[str, object]],
    generator: torch.Generator,
    initial_quantized_weights: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _atomic_torch_save(
        {
            "schema": "snn-qat-training-checkpoint-v1",
            "contract": contract,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best": best.to_dict(),
            "history": history,
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "data_loader_generator_state": generator.get_state(),
            "initial_quantized_weights": initial_quantized_weights,
        },
        destination,
    )


def _restore_cuda_rng_states(states: list[torch.Tensor]) -> None:
    """Restore CUDA RNG states after a checkpoint was mapped to any device."""
    if torch.cuda.is_available() and states:
        torch.cuda.set_rng_state_all([state.cpu() for state in states])


@torch.no_grad()
def _clip_grad_norm_float64(parameters, maximum_norm: float) -> float:
    """Clip using a Float64 norm so large finite Float32 gradients remain usable."""
    gradients = [
        parameter.grad
        for parameter in parameters
        if getattr(parameter, "grad", None) is not None
    ]
    squared_norm = sum(
        float(gradient.detach().to(torch.float64).square().sum().item())
        for gradient in gradients
    )
    total_norm = squared_norm**0.5
    if not torch.isfinite(torch.tensor(total_norm, dtype=torch.float64)):
        raise RuntimeError("gradient norm is non-finite")
    coefficient = min(1.0, maximum_norm / (total_norm + 1.0e-12))
    for gradient in gradients:
        gradient.mul_(coefficient)
    return total_norm


def main() -> None:
    args = _arguments()
    if args.epochs <= 0 or args.batch_size <= 0 or args.timesteps <= 0:
        raise ValueError("epochs, batch-size and timesteps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.early_stopping_patience < 0 or args.lr_reduce_patience < 0:
        raise ValueError("patience values must be non-negative")
    if args.bptt_window < 0:
        raise ValueError("bptt-window must be non-negative")
    if not torch.isfinite(torch.tensor(args.rate_scale)) or args.rate_scale < 0.0:
        raise ValueError("rate-scale must be finite and non-negative")
    if not 0 <= args.encoding_threshold <= 255:
        raise ValueError("encoding-threshold must be in [0, 255]")
    if not torch.isfinite(torch.tensor(args.latency_gamma)) or args.latency_gamma <= 0.0:
        raise ValueError("latency-gamma must be finite and positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.burst_max_spikes <= 0 or args.burst_interval <= 0:
        raise ValueError("burst parameters must be positive")
    if (
        args.input_encoding == "burst"
        and (args.burst_max_spikes - 1) * args.burst_interval >= args.timesteps
    ):
        raise ValueError("burst does not fit within the timestep horizon")
    if args.minimum_improvement < 0:
        raise ValueError("minimum-improvement must be non-negative")
    if args.gradient_clip_norm < 0:
        raise ValueError("gradient-clip-norm must be non-negative")
    if args.validation_samples < 0 or args.layer0_column_mean_penalty < 0:
        raise ValueError("validation samples and penalties must be non-negative")
    if args.hh_surrogate_width_mv <= 0.0 or args.hh_checkpoint_interval < 0:
        raise ValueError(
            "HH surrogate width must be positive and checkpoint interval non-negative"
        )
    if args.freeze_readout_epochs < 0:
        raise ValueError("freeze-readout-epochs must be non-negative")
    if args.minimum_finetune_epochs < 0:
        raise ValueError("minimum-finetune-epochs must be non-negative")
    if args.epochs < args.freeze_readout_epochs + args.minimum_finetune_epochs:
        raise ValueError(
            "epochs must cover freeze-readout-epochs plus minimum-finetune-epochs"
        )
    if (
        args.stop_after_epoch is not None
        and not 1 <= args.stop_after_epoch <= args.epochs
    ):
        raise ValueError("stop-after-epoch must be between one and epochs")
    if not 0.0 < args.lr_reduce_factor < 1.0:
        raise ValueError("lr-reduce-factor must lie between zero and one")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.hh_backend == "compiled" and device.type != "cuda":
        raise ValueError("the compiled HH backend requires CUDA")
    if args.hh_backend == "compiled" and "hodgkin_huxley" not in (
        args.hidden_neuron_model, args.output_neuron_model
    ):
        raise ValueError("the compiled HH backend requires an HH population")

    transform = transforms.ToTensor()
    full_train_data = _limit(
        datasets.MNIST(args.data_dir, train=True, transform=transform, download=args.download),
        args.max_train_samples,
    )
    if args.validation_samples >= len(full_train_data):
        raise ValueError("validation-samples must be smaller than training data")
    if args.validation_samples:
        split = len(full_train_data) - args.validation_samples
        train_data = Subset(full_train_data, range(split))
        validation_data = Subset(full_train_data, range(split, len(full_train_data)))
    else:
        train_data = full_train_data
        validation_data = None
    test_data = _limit(
        datasets.MNIST(args.data_dir, train=False, transform=transform, download=args.download),
        args.max_test_samples,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    # A separate generator keeps the extra evaluation from advancing the training RNG.
    train_evaluation_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, generator=torch.Generator().manual_seed(0),
    ) if args.training_observability else None
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    validation_loader = (
        DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if validation_data is not None
        else None
    )
    selection_loader = validation_loader or test_loader
    selection_name = "validation" if validation_loader is not None else "test"

    numeric = NumericConfig(
        mem_width=24,
        current_width=24,
        weight_width=8,
        count_width=16,
        time_width=32,
        threshold=args.threshold,
        reset_potential=0,
        leak_shift=args.leak_shift,
    )
    model_ids = {
        "lif": NeuronModel.LIF,
        "izhikevich": NeuronModel.IZHIKEVICH,
        "hodgkin_huxley": NeuronModel.HODGKIN_HUXLEY,
    }
    config = SNNConfig(
        784,
        args.hidden_neurons,
        10,
        numeric,
        hidden_neuron_model=model_ids[args.hidden_neuron_model],
        output_neuron_model=model_ids[args.output_neuron_model],
    )
    rate_seed = args.seed if args.rate_seed is None else args.rate_seed
    model = HardwareQATSNN(
        config,
        args.timesteps,
        bptt_window=args.bptt_window,
        input_encoding=args.input_encoding,
        rate_scale=args.rate_scale,
        rate_seed=rate_seed,
        encoding_threshold=args.encoding_threshold,
        latency_gamma=args.latency_gamma,
        top_k=args.top_k,
        burst_max_spikes=args.burst_max_spikes,
        burst_interval=args.burst_interval,
        hh_surrogate_width_mv=args.hh_surrogate_width_mv,
        hh_checkpoint_interval=args.hh_checkpoint_interval,
        hh_backend=args.hh_backend,
    ).to(device)
    if args.initial_state is not None:
        initial_state = torch.load(
            args.initial_state,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(initial_state)
    elif args.initial_model_dir is not None:
        initial_model = load_exported_model(args.initial_model_dir)
        _validate_initial_model(initial_model, config, args)
        model.load_quantized_weights(
            initial_model.input_hidden_weights,
            initial_model.hidden_output_weights,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_reduce_factor,
            patience=args.lr_reduce_patience,
        )
        if args.lr_reduce_patience > 0
        else None
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    checkpoint = BestCheckpoint(args.minimum_improvement)
    contract = _training_contract(args, config)
    epoch_checkpoint = args.checkpoint_path or args.output_dir / "checkpoint.pt"
    initial_quantized_weights = tuple(
        weight.cpu().clone() for weight in model.quantized_weights()
    )
    start_epoch = 1

    if args.resume_checkpoint is not None:
        resumed = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if resumed.get("schema") != "snn-qat-training-checkpoint-v1":
            raise ValueError("unsupported training checkpoint schema")
        if resumed.get("contract") != contract:
            raise ValueError("resume checkpoint training contract mismatch")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        if scheduler is not None:
            if resumed["scheduler"] is None:
                raise ValueError("resume checkpoint is missing scheduler state")
            scheduler.load_state_dict(resumed["scheduler"])
        elif resumed["scheduler"] is not None:
            raise ValueError("resume checkpoint unexpectedly contains scheduler state")
        checkpoint = BestCheckpoint.from_dict(resumed["best"])
        history = list(resumed["history"])
        if args.training_observability:
            for index, record in enumerate(history):
                sidecar = args.output_dir / "epoch_metrics" / f"epoch_{record['epoch']:04d}.json"
                if sidecar.exists():
                    history[index] = json.loads(sidecar.read_text(encoding="utf-8"))
        random.setstate(resumed["python_random_state"])
        torch.set_rng_state(resumed["torch_rng_state"].cpu())
        _restore_cuda_rng_states(resumed["cuda_rng_states"])
        generator.set_state(resumed["data_loader_generator_state"].cpu())
        initial_quantized_weights = tuple(
            value.cpu() for value in resumed["initial_quantized_weights"]
        )
        start_epoch = int(resumed["epoch"]) + 1
        print(
            json.dumps(
                {"resumed_from": str(args.resume_checkpoint), "start_epoch": start_epoch},
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        baseline_start = _synchronized_time(device)
        model.set_rate_seed(rate_seed)
        baseline = _metrics_detailed(model, selection_loader, device)
        checkpoint.update(model, 0, baseline["accuracy"])
        baseline_record: dict[str, object] = {
            "epoch": 0,
            "phase": "baseline",
            selection_name: baseline,
            "best_checkpoint": True,
        }
        if args.training_observability:
            baseline_record["train_evaluation"] = _metrics_detailed(
                model, train_evaluation_loader, device, detailed_hh=False)
            _save_epoch_weights(args.output_dir, 0, model, contract)
            baseline_record["baseline_wall_seconds"] = _synchronized_time(device) - baseline_start
            _save_observation(args.output_dir, baseline_record)
        history.append(baseline_record)
        print(json.dumps(baseline_record, sort_keys=True), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = _synchronized_time(device)
        epoch_started_utc = datetime.now(timezone.utc).isoformat()
        phase = _training_phase(epoch, args.freeze_readout_epochs)
        if epoch == args.freeze_readout_epochs + 1 and args.freeze_readout_epochs:
            checkpoint.reset_patience()
            print(json.dumps({"epoch": epoch, "phase_transition": "joint"}), flush=True)
        model.weight_l1.requires_grad_(epoch > args.freeze_readout_epochs)
        training_rate_seed = rate_seed + epoch - 1
        model.set_rate_seed(training_rate_seed)
        model.train()
        total_loss = 0.0
        total_cross_entropy = 0.0
        total_regularization = 0.0
        online_correct = 0
        examples = 0
        gradients = GradientAccumulator()
        clipped_gradients = GradientAccumulator()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            counts = model(images, sample_offset=examples)
            classification_loss = criterion(counts, labels)
            regularization = layer0_column_mean_penalty(model)
            loss = (
                classification_loss
                + args.layer0_column_mean_penalty * regularization
            )
            if args.training_observability:
                total_cross_entropy += float(classification_loss.item()) * labels.numel()
                total_regularization += float(regularization.item()) * labels.numel()
                online_correct += int(((counts.argmax(dim=1) == labels) & counts.ne(0).any(dim=1)).sum().item())
            loss.backward()
            gradients.add(dict(model.named_parameters()))
            if args.gradient_clip_norm > 0:
                _clip_grad_norm_float64(model.parameters(), args.gradient_clip_norm)
                clipped_gradients.add(dict(model.named_parameters()))
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            examples += labels.numel()

        training_end = _synchronized_time(device)
        model.set_rate_seed(rate_seed)
        selection_metrics = _metrics_detailed(model, selection_loader, device)
        validation_end = _synchronized_time(device)
        selection_accuracy = selection_metrics["accuracy"]
        record: dict[str, object] = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": total_loss / max(1, examples),
            f"{selection_name}_accuracy": selection_accuracy,
            selection_name: selection_metrics,
            "layer0_column_mean_penalty": float(
                layer0_column_mean_penalty(model).detach().item()
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "readout_frozen": not model.weight_l1.requires_grad,
            "input_encoding": args.input_encoding,
            "rate_seed": training_rate_seed if args.input_encoding == "rate" else None,
            "gradients": gradients.summary(),
            "weights": _weight_diagnostics(model, initial_quantized_weights),
        }
        if args.training_observability:
            record.update({
                "train_cross_entropy": total_cross_entropy / max(1, examples),
                "train_regularization": total_regularization / max(1, examples),
                "train_regularization_weighted": args.layer0_column_mean_penalty * total_regularization / max(1, examples),
                "train_accuracy_online": online_correct / max(1, examples),
                "train_samples_online": examples,
                "optimizer_steps": len(train_loader),
                "train_evaluation": _metrics_detailed(model, train_evaluation_loader, device, detailed_hh=False),
                "epoch_started_utc": epoch_started_utc,
                "train_seconds": training_end - epoch_start,
                "validation_seconds": validation_end - training_end,
                "evaluation_rate_seed": rate_seed,
            })
        evaluation_end = _synchronized_time(device)
        if args.gradient_clip_norm > 0:
            record["gradients_after_clip"] = clipped_gradients.summary()
        record["best_checkpoint"] = checkpoint.update(
            model, epoch, selection_accuracy
        )
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        learning_rate_reduced = False
        if scheduler is not None and phase == "joint":
            next_learning_rate, learning_rate_reduced = _step_plateau_scheduler(
                scheduler,
                optimizer,
                selection_accuracy,
                checkpoint,
            )
        record["next_learning_rate"] = next_learning_rate
        record["learning_rate_reduced"] = learning_rate_reduced
        history.append(record)
        eligible_for_early_stop = _early_stopping_eligible(
            epoch,
            args.freeze_readout_epochs,
            args.minimum_finetune_epochs,
        )
        checkpoint_start = _synchronized_time(device)
        if args.training_observability:
            _save_epoch_weights(args.output_dir, epoch, model, contract)
        _save_epoch_checkpoint(
            epoch_checkpoint,
            contract=contract,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best=checkpoint,
            history=history,
            generator=generator,
            initial_quantized_weights=initial_quantized_weights,
        )
        if args.training_observability:
            checkpoint_end = _synchronized_time(device)
            record.update({
                "train_evaluation_seconds": evaluation_end - validation_end,
                "checkpoint_seconds": checkpoint_end - checkpoint_start,
                "epoch_wall_seconds": checkpoint_end - epoch_start,
                "epoch_finished_utc": datetime.now(timezone.utc).isoformat(),
                "cumulative_epoch_wall_seconds": sum(r.get("epoch_wall_seconds", 0.0) for r in history[:-1]) + checkpoint_end - epoch_start,
                "timing_scope": "CUDA-synchronized; includes epoch weights and recovery checkpoint; excludes metrics JSON/log write; first epoch may include compilation",
            })
            _save_observation(args.output_dir, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if _planned_stop_reached(epoch, args.stop_after_epoch):
            print(
                f"planned stop after epoch {epoch}; resume from {epoch_checkpoint}",
                flush=True,
            )
            if args.training_observability and epoch < args.epochs:
                # Pilot/maintenance pauses must not inspect test data or export a
                # partial run as a completed campaign candidate.
                return
            break
        if eligible_for_early_stop and checkpoint.should_stop(
            args.early_stopping_patience
        ):
            print(
                f"early stopping after epoch {epoch}; best epoch={checkpoint.best_epoch}",
                flush=True,
            )
            break

    checkpoint.restore(model)
    model.set_rate_seed(rate_seed)
    final_test = ({"not_evaluated": True} if args.skip_final_test
                  else _metrics_detailed(model, test_loader, device))
    weight_l0, weight_l1 = model.quantized_weights()
    changed_l0 = int((weight_l0.cpu() != initial_quantized_weights[0]).sum().item())
    changed_l1 = int((weight_l1.cpu() != initial_quantized_weights[1]).sum().item())
    metadata = {
        "training": {
            "seed": args.seed,
            "input_encoding": args.input_encoding,
            "rate_scale": (
                args.rate_scale
                if args.input_encoding in ("rate", "deterministic-rate")
                else None
            ),
            "rate_seed": rate_seed if args.input_encoding == "rate" else None,
            "rate_seed_policy": (
                "base-plus-epoch-minus-one; evaluation-uses-base"
                if args.input_encoding == "rate"
                else None
            ),
            "encoding_parameters": {
                "encoding_threshold": args.encoding_threshold,
                "latency_gamma": args.latency_gamma,
                "top_k": args.top_k,
                "burst_max_spikes": args.burst_max_spikes,
                "burst_interval": args.burst_interval,
                "rate_scale": args.rate_scale,
            },
            "epochs": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": checkpoint.best_epoch,
            f"best_{selection_name}_accuracy": checkpoint.best_accuracy,
            "final_test": final_test,
            "final_test_accuracy": final_test.get("accuracy"),
            "final_test_no_prediction_rate": final_test.get("no_prediction_rate"),
            "training_observability": args.training_observability,
            "completed_training_epochs": sum(r["epoch"] > 0 for r in history),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "bptt_window": args.bptt_window,
            "early_stopping_patience": args.early_stopping_patience,
            "minimum_improvement": args.minimum_improvement,
            "lr_reduce_patience": args.lr_reduce_patience,
            "lr_reduce_factor": args.lr_reduce_factor,
            "gradient_clip_norm": args.gradient_clip_norm,
            "layer0_column_mean_penalty": args.layer0_column_mean_penalty,
            "initial_state": str(args.initial_state) if args.initial_state else None,
            "initial_model_dir": (
                str(args.initial_model_dir) if args.initial_model_dir else None
            ),
            "hh_surrogate_width_mv": args.hh_surrogate_width_mv,
            "hh_checkpoint_interval": args.hh_checkpoint_interval,
            "hh_backend": args.hh_backend,
            "hh_internal_steps": 100,
            "freeze_readout_epochs": args.freeze_readout_epochs,
            "minimum_finetune_epochs": args.minimum_finetune_epochs,
            "resume_checkpoint": (
                str(args.resume_checkpoint) if args.resume_checkpoint else None
            ),
            "epoch_checkpoint": str(epoch_checkpoint),
            "changed_int8_weights": {"layer0": changed_l0, "layer1": changed_l1},
            "train_samples": len(train_data),
            "validation_samples": len(validation_data) if validation_data else 0,
            "test_samples": len(test_data),
            "torch_version": torch.__version__,
            "history": history,
        }
    }
    manifest = export_model(
        args.output_dir,
        config,
        args.timesteps,
        weight_l0.cpu().tolist(),
        weight_l1.cpu().tolist(),
        metadata=metadata,
        input_encoding=args.input_encoding,
        rate_scale=args.rate_scale,
        rate_seed=rate_seed,
        encoding_threshold=args.encoding_threshold,
        latency_gamma=args.latency_gamma,
        top_k=args.top_k,
        burst_max_spikes=args.burst_max_spikes,
        burst_interval=args.burst_interval,
    )
    torch.save(model.state_dict(), args.output_dir / "training_state.pt")
    print(f"exported {manifest}")


if __name__ == "__main__":
    main()
