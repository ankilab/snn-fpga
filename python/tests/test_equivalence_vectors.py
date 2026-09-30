import tempfile
import unittest
from pathlib import Path

from snn_verification.generate_vectors import generate


class EquivalenceVectorTests(unittest.TestCase):
    def test_tie_and_no_prediction_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tie = generate(output, "tie", 1)
            self.assertEqual(tie["prediction"], 0)
            self.assertEqual(tie["class_counts"][0], tie["class_counts"][1])
            self.assertGreater(tie["class_counts"][0], 0)

            no_output = generate(output, "no_output", 1)
            self.assertIsNone(no_output["prediction"])
            self.assertEqual(no_output["class_counts"], [0, 0])

    def test_random_generation_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = generate(Path(left), "random", 42)
            second = generate(Path(right), "random", 42)
            self.assertEqual(first, second)
            self.assertEqual(
                (Path(left) / "weights_l0.mem").read_bytes(),
                (Path(right) / "weights_l0.mem").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
