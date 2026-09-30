"""Command-line deployment and MNIST inference utility for the Alveo U50 SNN."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


from .device import (
    ModelPackage,
    SNNDevice,
    SNNHostError,
)
from .mnist import iter_mnist
from .xrt import XrtRegisterIO


def _integer(value: str) -> int:
    return int(value, 0)


def _distribution(values: list[float | int]) -> dict[str, float | int]:
    """Return a stable descriptive summary for a non-empty measurement series."""

    if not values:
        raise ValueError("cannot summarize an empty measurement series")
    ordered = sorted(values)

    def percentile(percent: int) -> float | int:
        # Nearest-rank percentiles keep the reported value in the observed sample.
        rank = max(1, (percent * len(ordered) + 99) // 100)
        return ordered[rank - 1]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p05": percentile(5),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xrt-xclbin", type=Path, required=True,
                        help="Alveo XCLBIN to load through XRT")
    parser.add_argument("--xrt-device", type=int, default=0)
    parser.add_argument(
        "--xrt-kernel", default="snn_accelerator:{snn_accelerator_1}"
    )
    parser.add_argument("--xrt-server", default="xrt-register-server")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe", help="read and validate the hardware identity")
    upload = subparsers.add_parser("upload-model", help="upload and verify weights")
    upload.add_argument("model_dir", type=Path)

    infer = subparsers.add_parser("infer", help="classify one raw 8-bit image")
    infer.add_argument("model_dir", type=Path)
    infer.add_argument("image", type=Path, help="raw row-major uint8 pixels")
    infer.add_argument("--reuse-weights", action="store_true")

    mnist = subparsers.add_parser("mnist", help="evaluate MNIST IDX test files")
    mnist.add_argument("model_dir", type=Path)
    mnist.add_argument("images", type=Path)
    mnist.add_argument("labels", type=Path)
    mnist.add_argument("--samples", type=int, default=100)
    mnist.add_argument("--reuse-weights", action="store_true")

    qualify = subparsers.add_parser(
        "qualify", help="compare hardware and bit-exact reference predictions"
    )
    qualify.add_argument("model_dir", type=Path)
    qualify.add_argument("images", type=Path)
    qualify.add_argument("labels", type=Path)
    qualify.add_argument("--samples", type=int, default=10)
    qualify.add_argument("--reuse-weights", action="store_true")

    return parser


def _open_registers(args: argparse.Namespace) -> XrtRegisterIO:
    return XrtRegisterIO(args.xrt_xclbin, device=args.xrt_device,
                         kernel=args.xrt_kernel, server=args.xrt_server)


def _progress(written: int, total: int) -> None:
    print(f"weights={written}/{total}", file=sys.stderr)


def _upload(device: SNNDevice, package: ModelPackage, reuse: bool) -> None:
    device.validate_model(package)
    if not reuse:
        device.upload_model(package, progress=_progress)


def _events(package: ModelPackage, pixels: bytes, *, sample_index: int = 0):
    return package.exported.encode_events(pixels, sample_index=sample_index)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with _open_registers(args) as registers:
            device = SNNDevice(registers)
            if args.command == "probe":
                print(json.dumps(device.probe().to_dict(), indent=2, sort_keys=True))
                return 0

            package = ModelPackage.load(args.model_dir)
            if args.command == "upload-model":
                device.upload_model(package, progress=_progress)
                print(
                    json.dumps(
                        {
                            "weight_crc32": f"{package.expected_weight_crc32:08x}",
                            "weight_write_count": package.expected_weight_count,
                        },
                        sort_keys=True,
                    )
                )
                return 0


            _upload(device, package, args.reuse_weights)

            if args.command == "infer":
                pixels = args.image.read_bytes()
                events = _events(package, pixels)
                started = time.perf_counter()
                prediction = device.infer(
                    events,
                    timesteps=package.exported.timesteps,
                    timeout=args.timeout,
                    poll_interval=args.poll_interval,
                )
                print(
                    json.dumps(
                        {
                            "latency_ms": (time.perf_counter() - started) * 1000.0,
                            "prediction": prediction,
                        },
                        sort_keys=True,
                    )
                )
                return 0

            if args.samples <= 0:
                raise ValueError("samples must be positive")
            correct = 0
            no_prediction = 0
            reference_correct = 0
            reference_mismatches = 0
            latencies: list[float] = []
            processed = 0
            reference = (
                package.exported.reference_model()
                if args.command == "qualify"
                else None
            )
            for label, pixels in iter_mnist(args.images, args.labels, limit=args.samples):
                events = _events(package, pixels, sample_index=processed)
                started = time.perf_counter()
                prediction = device.infer(
                    events,
                    timesteps=package.exported.timesteps,
                    timeout=args.timeout,
                    poll_interval=args.poll_interval,
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                processed += 1
                correct += prediction == label
                no_prediction += prediction is None
                reference_prediction = None
                if reference is not None:
                    reference_prediction = reference.infer(
                        events, timesteps=package.exported.timesteps
                    ).prediction
                    reference_correct += reference_prediction == label
                    reference_mismatches += prediction != reference_prediction
                print(
                    json.dumps(
                        {
                            "index": processed - 1,
                            "label": label,
                            "prediction": prediction,
                            **(
                                {
                                    "reference_prediction": reference_prediction,
                                    "matches_reference": prediction
                                    == reference_prediction,
                                }
                                if reference is not None
                                else {}
                            ),
                        }
                    )
                )
            if processed == 0:
                raise ValueError("MNIST dataset contains no samples")
            print(
                json.dumps(
                    {
                        "accuracy": correct / processed,
                        "latency_ms_mean": statistics.fmean(latencies),
                        "latency_ms_median": statistics.median(latencies),
                        "no_prediction": no_prediction,
                        "samples": processed,
                        **(
                            {
                                "reference_accuracy": reference_correct / processed,
                                "reference_mismatches": reference_mismatches,
                            }
                            if reference is not None
                            else {}
                        ),
                    },
                    sort_keys=True,
                )
            )
            if reference_mismatches:
                raise SNNHostError(
                    f"{reference_mismatches} hardware/reference prediction mismatch(es)"
                )
            return 0
    except (OSError, ValueError, SNNHostError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
