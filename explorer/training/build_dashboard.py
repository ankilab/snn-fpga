"""Build an offline training explorer from explicit training sources; no model execution."""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODEL_NAMES = {"lif": "LIF", "izhikevich": "IZH", "hodgkin_huxley": "HH"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm_history(history):
    rows = []
    for r in history:
        if "epoch" not in r:
            continue
        val = r.get("validation", {})
        train = r.get("train_evaluation", {})
        grads = r.get("gradients", {})
        weights = r.get("weights", {})
        rows.append({
            "epoch": r["epoch"], "phase": r.get("phase"), "best": r.get("best_checkpoint", False),
            "valAcc": val.get("accuracy", r.get("validation_accuracy")), "valCe": val.get("cross_entropy"),
            "trainLoss": r.get("train_loss"), "trainCe": r.get("train_cross_entropy"),
            "trainReg": r.get("train_regularization_weighted"), "trainAcc": train.get("accuracy"),
            "trainEvalCe": train.get("cross_entropy"), "onlineAcc": r.get("train_accuracy_online"),
            "lr": r.get("learning_rate"), "lrReduced": r.get("learning_rate_reduced", False),
            "ties": val.get("tie_rate", val.get("output_tie_rate")),
            "silent": val.get("no_prediction_rate", r.get("validation_no_prediction_rate")),
            "hiddenSpikes": val.get("hidden_spikes_per_image"), "outputSpikes": val.get("output_spikes_per_image"),
            "grad0": grads.get("weight_l0", {}).get("l2_norm_mean"),
            "grad1": grads.get("weight_l1", grads.get("weight", {})).get("l2_norm_mean"),
            "changed0": weights.get("layer0", {}).get("changed_from_initial"),
            "changed1": weights.get("layer1", {}).get("changed_from_initial"),
            "lutClamps": val.get("hh_lut_clamps"),
            "trainSeconds": r.get("train_seconds"), "valSeconds": r.get("validation_seconds"),
            "trainEvalSeconds": r.get("train_evaluation_seconds"), "saveSeconds": r.get("checkpoint_seconds"),
            "epochSeconds": r.get("epoch_wall_seconds"), "cumulativeSeconds": r.get("cumulative_epoch_wall_seconds"),
            "finishedUtc": r.get("epoch_finished_utc"),
        })
    return sorted(rows, key=lambda r: r["epoch"])


def variant(family, params):
    if family in ("rate", "deterministic-rate"):
        return "scale " + str(params.get("rate_scale", "?"))
    if family == "burst":
        return f"b{params.get('burst_max_spikes', '?')} / i{params.get('burst_interval', '?')}"
    if family == "sparse-latency":
        return "threshold " + str(params.get("encoding_threshold", "?"))
    if family == "nonlinear-latency":
        return "γ " + str(params.get("latency_gamma", "?"))
    if family == "top-k-rank-order":
        return "k " + str(params.get("top_k", "?"))
    return "default"


def manifest_run(path, campaign, label):
    model = read(path)
    metadata = model.get("metadata", {})
    if metadata.get("derived_package"):
        raise ValueError(f"untrained transfer must not be imported: {path}")
    readout = False
    training = metadata["hh_readout_training" if readout else "training"]
    hidden = MODEL_NAMES[model["neurons"]["hidden"]["model"]]
    output = MODEL_NAMES[model["neurons"]["output"]["model"]]
    encoding = training.get("input_encoding", model["encoding"]["type"].removesuffix("-v1"))
    if encoding == "latency-single-spike":
        encoding = "latency"
    params = {**model["encoding"], **training.get("encoding_parameters", {})}
    parent_path = training.get("initial_state")
    parent_name = parent_path.replace("\\", "/").split("/")[-2] if parent_path else None
    run = {
        "id": campaign + ":" + path.parent.name, "job": path.parent.name,
        "campaign": campaign, "campaignLabel": label, "dynamics": hidden + "/" + output,
        "family": encoding, "variant": variant(encoding, params), "seed": training.get("seed"),
        "timesteps": model["encoding"]["timesteps"],
        "method": "Readout" if readout else ("Resumed" if parent_path else "Direct"),
        "stage": "Base" if path.parent.name.endswith("__base") else ("Readout" if readout else "Final"),
        "parent": campaign + ":" + parent_name if parent_name else None,
        "status": "complete", "bestEpoch": training.get("best_epoch"),
        "test": training.get("final_test"), "history": norm_history(training["history"]),
        "specs": {k: v for k, v in training.items() if k not in ("history", "final_test")},
        "topology": model["topology"], "numeric": model["numeric"],
        "source": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "regularization": training.get("layer0_column_mean_penalty", 0),
    }
    return run


def build_data():
    campaign = "full_encoding_50"
    source = ROOT / "results/full-encoding-50epoch-2026-09-11"
    runs = []
    for path in sorted((source / "models").glob("*/manifest.json")):
        run = manifest_run(path, campaign, "50 epochs · Seed 14")
        original = read(path)["metadata"]["training"]
        records = [read(p) for p in sorted((path.parent / "epoch_metrics").glob("epoch_*.json"))]
        if records != original["history"] or [r["epoch"] for r in records] != list(range(51)):
            raise ValueError(f"Manifest/epoch evidence mismatch: {path}")
        run["lastEpoch"] = 50
        run["bestVal"] = original["best_validation_accuracy"]
        run["trainSamples"] = 55000
        run["valSamples"] = 5000
        # Local scheduler paths have no meaning in the public explorer.
        run["specs"] = {k:v for k,v in run["specs"].items()
                        if not isinstance(v,str) or not (v.startswith("/") or ":\\" in v)}
        runs.append(run)
    if len(runs) != 66 or len({r["id"] for r in runs}) != 66:
        raise ValueError("Expected exactly 66 unique completed fits")
    return {"schema":"training-explorer-v1", "builtUtc":datetime.now(timezone.utc).isoformat(),
            "importedUtc":None, "liveStatus":{"status":"complete","completed":66,"total":66},
            "runs":runs, "notes":["66 fits, each with 50 epochs plus baseline; one training seed.",
                                   "Model selection uses validation; test results are final endpoints only."]}


def main():
    data = build_data()
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (HERE / "training_data.json").write_text(text + "\n", encoding="utf-8")
    html = (HERE / "dashboard.template.html").read_text(encoding="utf-8")
    html = html.replace("__DATA__", text.replace("<", "\\u003c")).replace("__SCRIPT__", (HERE / "dashboard.js").read_text(encoding="utf-8"))
    (HERE / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"stages": len(data["runs"]), "with_history": sum(bool(r["history"]) for r in data["runs"]),
                      "epoch_records": sum(len(r["history"]) for r in data["runs"]), "html_bytes": len(html.encode())}))


if __name__ == "__main__":
    main()
