#!/usr/bin/env python3
"""Validate the final summary record of an SNN qualification JSONL file."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any


def read_summary(path: Path) -> tuple[dict[str, Any], int]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        records.append(value)
    if not records:
        raise ValueError("result file is empty")
    return records[-1], len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gate", "full"))
    parser.add_argument("result_jsonl", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--expected-accuracy", type=Decimal)
    parser.add_argument("--expected-no-prediction", type=int)
    args = parser.parse_args()

    mismatches: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    record_count = 0
    try:
        summary, record_count = read_summary(args.result_jsonl)
        expected_records = args.samples + 1
        if record_count != expected_records:
            mismatches["record_count"] = {
                "expected": expected_records,
                "actual": record_count,
            }
        if summary.get("samples") != args.samples:
            mismatches["samples"] = {
                "expected": args.samples,
                "actual": summary.get("samples"),
            }
        if args.stage == "gate":
            if summary.get("reference_mismatches") != 0:
                mismatches["reference_mismatches"] = {
                    "expected": 0,
                    "actual": summary.get("reference_mismatches"),
                }
            if summary.get("accuracy") != summary.get("reference_accuracy"):
                mismatches["accuracy"] = {
                    "expected": summary.get("reference_accuracy"),
                    "actual": summary.get("accuracy"),
                }
        elif args.expected_accuracy is None:
            mismatches["expected_accuracy"] = {"expected": "argument", "actual": None}
        else:
            try:
                actual_accuracy = Decimal(str(summary["accuracy"]))
            except (KeyError, InvalidOperation) as error:
                raise ValueError("summary has no valid accuracy") from error
            if actual_accuracy != args.expected_accuracy:
                mismatches["accuracy"] = {
                    "expected": str(args.expected_accuracy),
                    "actual": str(actual_accuracy),
                }
            if (
                args.expected_no_prediction is not None
                and summary.get("no_prediction") != args.expected_no_prediction
            ):
                mismatches["no_prediction"] = {
                    "expected": args.expected_no_prediction,
                    "actual": summary.get("no_prediction"),
                }
    except (OSError, ValueError) as error:
        mismatches["result_file"] = {"expected": "valid JSONL", "actual": str(error)}

    result = {
        "schema": "snn-u50-image-qualification-validation-v1",
        "stage": args.stage,
        "status": "passed" if not mismatches else "failed",
        "result_jsonl": str(args.result_jsonl),
        "record_count": record_count,
        "summary": summary,
        "mismatches": mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
