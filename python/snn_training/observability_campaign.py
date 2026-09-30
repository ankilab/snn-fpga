"""Frozen, resumable training campaigns with complete epoch evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

from snn_training.campaign_support import _atomic_json, _freeze_plan, _source_fingerprint, _utc_now
from snn_training.campaign_support import gpu_snapshot, wait_for_gpu
from snn_training.export import load_exported_model


def build_jobs(matrix: dict, root: Path, data: Path) -> list[dict]:
    jobs = []
    for seed in matrix["training_seeds"]:
        for candidate in matrix["candidates"]:
            stages = ["base", "final"] if "base_epochs" in candidate else ["final"]
            parent = None
            for stage in stages:
                identifier = f"{candidate['id']}__s{seed}" + ("__base" if stage == "base" else "")
                output = root / "models" / identifier
                recipe = {k: candidate[k] for k in ("epochs", "batch_size", "learning_rate", "bptt_window",
                                                    "early_stopping_patience", "lr_reduce_patience", "hh_backend")}
                if stage == "base":
                    recipe.update(epochs=candidate["base_epochs"], learning_rate=candidate["base_learning_rate"], lr_reduce_patience=1)
                options = {"data_dir": str(data), "output_dir": str(output),
                           "seed": seed, "rate_seed": matrix["rate_seed"],
                           "validation_samples": matrix["validation_samples"],
                           "hidden_neurons": matrix["hidden_neurons"], "timesteps": matrix["timesteps"],
                           "hidden_neuron_model": candidate["dynamics"], "output_neuron_model": candidate["dynamics"],
                           "input_encoding": candidate["encoding"], "device": "cuda", "num_workers": 0,
                           "gradient_clip_norm": 1.0, "minimum_finetune_epochs": 4,
                           **candidate["parameters"], **recipe}
                if parent:
                    options.update(initial_state=str(root / "models" / parent / "training_state.pt"),
                                   layer0_column_mean_penalty=candidate["regularization"])
                arguments = [item for k, v in options.items() for item in ("--" + k.replace("_", "-"), str(v))]
                arguments.append("--training-observability")
                if stage == "base":
                    arguments.append("--skip-final-test")
                jobs.append({"id": identifier, "candidate": candidate["id"], "seed": seed,
                             "stage": stage, "parent": parent, "output": str(output), "arguments": arguments,
                             "epoch_cap": recipe["epochs"]})
                parent = identifier
    return jobs


def verify_history(output: Path, *, completed: bool) -> dict:
    records = [json.loads(p.read_text()) for p in sorted((output / "epoch_metrics").glob("epoch_*.json"))]
    if not records or [r["epoch"] for r in records] != list(range(len(records))):
        raise ValueError(f"missing epoch evidence: {output}")
    for record in records:
        if not (output / "epoch_weights" / f"epoch_{record['epoch']:04d}.pt").is_file():
            raise ValueError("missing epoch weights")
        for split in ("validation", "train_evaluation"):
            for key in ("accuracy", "cross_entropy"):
                if not math.isfinite(record[split][key]):
                    raise ValueError("nonfinite evaluation")
        if record["epoch"]:
            for key in ("train_loss", "train_cross_entropy", "train_accuracy_online", "train_regularization_weighted",
                        "train_seconds", "validation_seconds", "train_evaluation_seconds", "checkpoint_seconds", "epoch_wall_seconds"):
                if not math.isfinite(record[key]) or record[key] < 0:
                    raise ValueError(f"invalid {key}")
            if abs(record["train_loss"] - record["train_cross_entropy"] - record["train_regularization_weighted"]) > 1e-5:
                raise ValueError("objective decomposition mismatch")
    result = {"epochs": len(records) - 1, "epoch_wall_seconds": [r["epoch_wall_seconds"] for r in records[1:]],
              "total_epoch_seconds": sum(r["epoch_wall_seconds"] for r in records[1:])}
    if completed:
        exported = load_exported_model(output)
        training = exported.metadata["training"]
        if training["history"] != records:
            raise ValueError("manifest and metrics differ")
        result.update(best_epoch=training["best_epoch"], validation_accuracy=training["best_validation_accuracy"],
                      test=training["final_test"], manifest=str(output / "manifest.json"))
    return result


def run_job(job: dict, args, matrix: dict, source_hash: str, *, pilot: bool = False) -> dict:
    if _source_fingerprint(args.repo_root) != source_hash:
        raise ValueError("source changed after plan freeze")
    output = Path(job["output"])
    if shutil.disk_usage(args.root).free < matrix.get("minimum_free_disk_gb", 0) * 1024**3:
        raise RuntimeError("insufficient free disk space")
    if (output / "manifest.json").exists() and not pilot:
        result = verify_history(output, completed=True)
        if matrix.get("require_full_epochs") and result["epochs"] != job["epoch_cap"]:
            raise ValueError("completed fit does not contain the required epoch count")
        return result
    wait_for_gpu(args.root, matrix)
    command = [sys.executable, "-m", "snn_training.train_mnist", *job["arguments"]]
    if (output / "checkpoint.pt").exists():
        command.extend(("--resume-checkpoint", str(output / "checkpoint.pt")))
    if pilot:
        command.extend(("--stop-after-epoch", "2"))
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{job['id']}.log").open("a", encoding="utf-8") as log:
        log.write(f"\n=== {_utc_now()} START {json.dumps(command)} ===\n")
        log.flush()
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                snapshot = gpu_snapshot()
                snapshot["job_id"] = job["id"]
                with (args.root / "gpu_telemetry.jsonl").open("a") as stream:
                    stream.write(json.dumps(snapshot) + "\n")
                if time.monotonic() - start > matrix["job_timeout_seconds"]:
                    raise TimeoutError(job["id"])
                if snapshot["temperature_c"] >= 85:
                    raise RuntimeError("GPU temperature reached 85 C")
                time.sleep(15)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            raise RuntimeError(f"trainer {job['id']} exited {process.returncode}")
        log.write(f"=== {_utc_now()} END elapsed_seconds={time.monotonic() - start:.3f} ===\n")
    result = verify_history(output, completed=not pilot)
    if not pilot and matrix.get("require_full_epochs") and result["epochs"] != job["epoch_cap"]:
        raise ValueError("fit did not complete the required epoch count")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("prepare", "pilot", "run"), required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        matrix = json.loads(args.matrix.read_text())
        if matrix.get("require_full_epochs") and any(c["early_stopping_patience"] != 0 for c in matrix["candidates"]):
            raise ValueError("full-epoch campaigns must disable early stopping")
        source_hash = _source_fingerprint(args.repo_root)
        jobs = build_jobs(matrix, args.root, args.data_dir)
        plan = {"schema": matrix["schema"], "matrix": matrix, "source_sha256": source_hash, "jobs": jobs}
        _freeze_plan(args.root / "plan.json", plan)
        data_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((args.data_dir / "MNIST" / "raw").glob("*-ubyte"))}
        if len(data_hashes) != 4:
            raise ValueError("four unpacked MNIST IDX files are required")
        _freeze_plan(args.root / "dataset.json", {"sha256": data_hashes, "train_indices": [0, 55000],
                                                  "validation_indices": [55000, 60000], "test_indices": [0, 10000]})
        results = {}
        try:
            if args.mode == "prepare":
                print(json.dumps({"status": "prepared", "jobs": len(jobs), "epochs": sum(j["epoch_cap"] for j in jobs), "source_sha256": source_hash}), flush=True)
                return
            if args.mode == "pilot":
                pilot_path = args.root / "pilot.json"
                if pilot_path.exists():
                    print(pilot_path.read_text(), flush=True)
                    return
                result = run_job(jobs[0], args, matrix, source_hash, pilot=True)
                if result["epochs"] != 2:
                    raise ValueError("pilot must complete two epochs")
                result["pilot_job"] = jobs[0]["id"]
                result["projection_note"] = "Pilot timing applies only to this candidate; other dynamics and encodings can differ"
                result["status"] = "passed"
                _atomic_json(pilot_path, result)
                print(json.dumps(result), flush=True)
                return
            if not (args.root / "pilot.json").exists() or json.loads((args.root / "pilot.json").read_text()).get("status") != "passed":
                raise ValueError("validated full-size pilot required before campaign")
            for job in jobs:
                _atomic_json(args.root / "status.json", {"status": "running", "job": job["id"], "completed": len(results), "total": len(jobs), "utc": _utc_now()})
                results[job["id"]] = {**job, **run_job(job, args, matrix, source_hash)}
                _atomic_json(args.root / "results.json", results)
                print(f"COMPLETED {job['id']} ({len(results)}/{len(jobs)})", flush=True)
                time.sleep(matrix["cooldown_seconds"])
            summary = {}
            for candidate in matrix["candidates"]:
                fits = [r for r in results.values() if r["candidate"] == candidate["id"] and r["stage"] == "final"]
                scores = [r["test"]["accuracy"] for r in fits]
                summary[candidate["id"]] = {"seeds": [r["seed"] for r in fits], "test_accuracy_mean": statistics.mean(scores),
                                            "test_accuracy_sample_sd": statistics.stdev(scores) if len(scores) > 1 else None, "fits": fits}
            _atomic_json(args.root / "summary.json", summary)
            _atomic_json(args.root / "status.json", {"status": "complete", "completed": len(results), "total": len(jobs), "utc": _utc_now()})
        except Exception as exc:
            _atomic_json(args.root / "status.json", {"status": "failed", "error": str(exc), "completed": len(results), "utc": _utc_now()})
            raise


if __name__ == "__main__":
    main()
