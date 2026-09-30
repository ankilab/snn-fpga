"""Portable helpers for the 50-epoch campaign (GPU scheduling requires Linux/NVIDIA)."""
from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import subprocess
import time

def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)

def _signature(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _freeze_plan(path: Path, plan: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError(f"campaign plan changed: {path}")
        return
    _atomic_json(path, plan)

def _source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((repo_root / "python" / "snn_training").glob("*.py"))
    paths.extend(sorted((repo_root / "python" / "snn_reference").rglob("*.py")))
    paths.append(repo_root / "python" / "requirements-training.txt")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"campaign source file is missing: {path}")
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()

def gpu_snapshot() -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()[0]
    temperature, utilization, memory, power = [float(v.strip()) for v in output.split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()
    return {"utc": _utc_now(), "temperature_c": temperature, "utilization_percent": utilization,
            "memory_mib": memory, "power_w": power,
            # Preflight initializes this supervisor's CUDA context; it does not train.
            "compute_pids": [pid.strip() for pid in processes.splitlines()
                             if pid.strip() and pid.strip() != str(os.getpid())]}

def wait_for_gpu(root: Path, matrix: dict) -> None:
    while True:
        snapshot = gpu_snapshot()
        with (root / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        if not snapshot["compute_pids"] and snapshot["temperature_c"] <= matrix["maximum_start_temperature_c"]:
            return
        print(f"[{_utc_now()}] waiting for free/cool GPU: {snapshot}", flush=True)
        time.sleep(30)
