"""Evaluate an exported INT8 model through the bit-accurate Python reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from snn_training.export import load_exported_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    exported = load_exported_model(args.model_dir)
    model = exported.reference_model()
    dataset = datasets.MNIST(
        args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=args.download,
    )
    correct = 0
    no_prediction = 0
    samples = min(args.samples, len(dataset))
    for index in range(samples):
        image, label = dataset[index]
        pixels = (image.reshape(-1) * 255.0).round().to(torch.uint8).tolist()
        events = exported.encode_events(pixels, sample_index=index)
        prediction = model.infer(events, timesteps=exported.timesteps).prediction
        no_prediction += prediction is None
        correct += prediction == int(label)

    print(f"samples={samples}")
    print(f"input_encoding={exported.input_encoding}")
    print(f"accuracy={correct / samples:.6f}")
    print(f"no_prediction={no_prediction}")


if __name__ == "__main__":
    main()
