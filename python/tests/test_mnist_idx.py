import gzip
import struct
import tempfile
import unittest
from pathlib import Path

from snn_host.mnist import iter_mnist


class MnistIdxTests(unittest.TestCase):
    def test_reads_gzip_idx_and_honors_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images.gz"
            labels = root / "labels.gz"
            with gzip.open(images, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 2, 2, 2))
                stream.write(bytes([0, 1, 2, 3, 4, 5, 6, 7]))
            with gzip.open(labels, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 2))
                stream.write(bytes([9, 8]))

            self.assertEqual(list(iter_mnist(images, labels, limit=1)), [(9, bytes([0, 1, 2, 3]))])

    def test_rejects_invalid_magic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.write_bytes(struct.pack(">IIII", 0, 1, 1, 1) + b"\0")
            labels.write_bytes(struct.pack(">II", 2049, 1) + b"\0")
            with self.assertRaisesRegex(ValueError, "magic"):
                list(iter_mnist(images, labels))


if __name__ == "__main__":
    unittest.main()
