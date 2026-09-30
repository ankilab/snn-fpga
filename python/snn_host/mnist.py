"""Dependency-free reader for the canonical MNIST IDX files."""

from __future__ import annotations

import gzip
import struct
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO


def _open(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def iter_mnist(
    images_path: str | Path,
    labels_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(label, pixels)`` pairs from uncompressed or gzip IDX files."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with ExitStack() as stack:
        images = stack.enter_context(_open(Path(images_path)))
        labels = stack.enter_context(_open(Path(labels_path)))
        image_header = images.read(16)
        label_header = labels.read(8)
        if len(image_header) != 16 or len(label_header) != 8:
            raise ValueError("truncated MNIST IDX header")
        image_magic, image_count, rows, columns = struct.unpack(">IIII", image_header)
        label_magic, label_count = struct.unpack(">II", label_header)
        if image_magic != 2051 or label_magic != 2049:
            raise ValueError("invalid MNIST IDX magic number")
        if image_count != label_count:
            raise ValueError("MNIST image and label counts differ")
        count = image_count if limit is None else min(image_count, limit)
        image_size = rows * columns
        for _ in range(count):
            pixels = images.read(image_size)
            label = labels.read(1)
            if len(pixels) != image_size or len(label) != 1:
                raise ValueError("truncated MNIST IDX payload")
            yield label[0], pixels
