"""Generate small-network vectors and expected state from the gold model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from snn_reference import NumericConfig, SNNConfig, SpikeEvent, TwoLayerSNN

INPUT_NEURONS = 4
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 2
NUM_EVENTS = 8
MEM_WIDTH = 16
COUNT_WIDTH = 8
TIME_WIDTH = 8
THRESHOLD = 50
LEAK_SHIFT = 2


def _zeros(rows: int, columns: int) -> list[list[int]]:
    return [[0 for _ in range(columns)] for _ in range(rows)]


def _scenario(
    name: str, seed: int
) -> tuple[list[list[int]], list[list[int]], list[SpikeEvent]]:
    layer0 = _zeros(INPUT_NEURONS, HIDDEN_NEURONS)
    layer1 = _zeros(HIDDEN_NEURONS, OUTPUT_NEURONS)

    if name == "tie":
        layer0[0][0] = 10
        layer1[0] = [60, 60]
        events = [SpikeEvent(0, 0) for _ in range(NUM_EVENTS)]
    elif name == "no_output":
        layer0[0] = [7, -5, 3]
        events = [SpikeEvent(index // 2, index % INPUT_NEURONS) for index in range(NUM_EVENTS)]
    elif name == "negative_gap":
        layer0[0] = [-32, 45, -12]
        layer0[1] = [11, -27, 19]
        layer0[2] = [-8, 16, 31]
        layer0[3] = [5, -9, -22]
        layer1[0] = [-40, 63]
        layer1[1] = [52, -35]
        layer1[2] = [24, 41]
        times = [0, 0, 1, 1, 3, 3, 4, 4]
        ids = [0, 1, 2, 3, 0, 2, 1, 3]
        events = [SpikeEvent(timestamp, neuron_id) for timestamp, neuron_id in zip(times, ids)]
    elif name == "random":
        rng = random.Random(seed)
        layer0 = [
            [rng.randint(-40, 70) for _ in range(HIDDEN_NEURONS)]
            for _ in range(INPUT_NEURONS)
        ]
        layer1 = [
            [rng.randint(-50, 80) for _ in range(OUTPUT_NEURONS)]
            for _ in range(HIDDEN_NEURONS)
        ]
        times = sorted([rng.randint(0, 4) for _ in range(NUM_EVENTS - 1)] + [4])
        events = [
            SpikeEvent(timestamp, rng.randrange(INPUT_NEURONS)) for timestamp in times
        ]
    else:
        raise ValueError(f"unknown scenario: {name}")
    return layer0, layer1, events


def _hex(value: int, width: int) -> str:
    digits = (width + 3) // 4
    return f"{value & ((1 << width) - 1):0{digits}x}"


def _write_values(path: Path, values, width: int) -> None:
    path.write_text("".join(f"{_hex(int(value), width)}\n" for value in values), encoding="ascii")


def generate(output_dir: Path, scenario: str, seed: int) -> dict:
    layer0, layer1, events = _scenario(scenario, seed)
    config = SNNConfig(
        INPUT_NEURONS,
        HIDDEN_NEURONS,
        OUTPUT_NEURONS,
        NumericConfig(
            mem_width=MEM_WIDTH,
            current_width=MEM_WIDTH,
            weight_width=8,
            count_width=COUNT_WIDTH,
            time_width=TIME_WIDTH,
            threshold=THRESHOLD,
            reset_potential=0,
            leak_shift=LEAK_SHIFT,
        ),
    )
    result = TwoLayerSNN(config, layer0, layer1).infer(events)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_values(output_dir / "weights_l0.mem", (v for row in layer0 for v in row), 8)
    _write_values(output_dir / "weights_l1.mem", (v for row in layer1 for v in row), 8)
    _write_values(output_dir / "event_times.mem", (event.timestamp for event in events), TIME_WIDTH)
    _write_values(output_dir / "event_ids.mem", (event.neuron_id for event in events), 8)
    _write_values(output_dir / "expected_hidden.mem", result.hidden_membrane, MEM_WIDTH)
    _write_values(output_dir / "expected_output.mem", result.output_membrane, MEM_WIDTH)
    _write_values(output_dir / "expected_counts.mem", result.class_counts, COUNT_WIDTH)
    _write_values(output_dir / "expected_prediction_valid.mem", [result.prediction is not None], 1)
    _write_values(output_dir / "expected_prediction.mem", [result.prediction or 0], 8)

    metadata = {
        "scenario": scenario,
        "seed": seed,
        "events": [[event.timestamp, event.neuron_id] for event in events],
        "prediction": result.prediction,
        "class_counts": list(result.class_counts),
        "hidden_membrane": list(result.hidden_membrane),
        "output_membrane": list(result.output_membrane),
    }
    (output_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("random", "tie", "no_output", "negative_gap"), required=True
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    metadata = generate(args.output, args.scenario, args.seed)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
