"""Compare one RT-DETR crop pilot with its frozen baseline on validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

METRICS = (
    "map_50_95",
    "map_50",
    "map_small",
    "recall",
    "missed_defect_rate",
    "false_positives_per_image",
)


def _checkpoint_record(path: Path, model: str) -> dict:
    metadata_path = path / "aeroinspect_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    architecture = metadata.get("architecture", metadata.get("model_name"))
    if architecture != model:
        raise ValueError(f"Checkpoint at {path} is {architecture!r}, not {model!r}.")
    if metadata.get("labels") != ["defect"]:
        raise ValueError(f"Checkpoint at {path} is not the binary defect detector.")
    metrics = metadata.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint at {path} has no saved validation metrics.")
    missing = [key for key in METRICS if not isinstance(metrics.get(key), (int, float))]
    if missing:
        raise ValueError(f"Checkpoint at {path} is missing numeric metrics: {missing}")
    return {
        "checkpoint": str(path),
        "best_epoch": int(metadata["epoch"]),
        "metrics": {key: float(metrics[key]) for key in METRICS},
    }


def compare_records(baseline: dict, pilot: dict) -> dict:
    """Rank by COCO mAP50-95, then recall, and expose every important delta."""
    baseline_key = (
        baseline["metrics"]["map_50_95"],
        baseline["metrics"]["recall"],
    )
    pilot_key = (pilot["metrics"]["map_50_95"], pilot["metrics"]["recall"])
    recommended = "crop_pilot" if pilot_key > baseline_key else "baseline"
    deltas = {
        key: pilot["metrics"][key] - baseline["metrics"][key] for key in METRICS
    }
    warnings = []
    if deltas["map_small"] < 0:
        warnings.append("Crop pilot reduced small-defect AP.")
    if deltas["false_positives_per_image"] > 0:
        warnings.append("Crop pilot increased false positives per image.")
    if deltas["missed_defect_rate"] > 0:
        warnings.append("Crop pilot increased the missed-defect rate.")
    return {
        "split": "validation",
        "recommended": recommended,
        "selection_rule": "highest mAP50-95, then recall",
        "baseline": baseline,
        "crop_pilot": pilot,
        "pilot_minus_baseline": deltas,
        "warnings": warnings,
        "final_test_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("rt_detr", "rt_detr_v2"))
    parser.add_argument("--baseline-config", default="config.yaml")
    parser.add_argument("--pilot-config", default=None)
    args = parser.parse_args()

    baseline_config = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
    pilot_config_path = Path(
        args.pilot_config or f"data/{args.model}_slice_pilot.yaml"
    )
    pilot_config = yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    pilot_definition = pilot_config.get("accuracy_pilot", {})
    if pilot_definition.get("model") != args.model:
        parser.error("Pilot configuration is missing or belongs to another model.")

    baseline_path = Path(
        baseline_config["aircraft"]["transformer_candidates"][args.model]["checkpoint"]
    )
    pilot_path = Path(
        pilot_config["aircraft"]["transformer_candidates"][args.model]["checkpoint"]
    )
    result = compare_records(
        _checkpoint_record(baseline_path, args.model),
        _checkpoint_record(pilot_path, args.model),
    )
    result["model"] = args.model
    destination = Path(pilot_config["paths"]["reports"]) / "accuracy_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Comparison saved to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
