"""Choose an operating threshold on validation predictions subject to explicit targets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.evaluation import detection_prf


def operating_point(ground_truth, predictions, image_count, min_recall, max_fp_per_image):
    if image_count < 1 or not 0 <= min_recall <= 1 or max_fp_per_image < 0:
        raise ValueError("Invalid operating targets.")
    thresholds = sorted({0.0, 1.0, *np.quantile(
        [float(p["score"]) for p in predictions], np.linspace(0, 1, 201)
    )}) if predictions else [1.0]
    rows = []
    for threshold in thresholds:
        metrics, _, _ = detection_prf(ground_truth, [p for p in predictions if p["score"] >= threshold])
        rows.append({"threshold": float(threshold), "recall": metrics["recall"],
                     "precision": metrics["precision"], "false_positives_per_image": metrics["fp"]/image_count})
    feasible = [r for r in rows if r["recall"] >= min_recall and r["false_positives_per_image"] <= max_fp_per_image]
    return {"split": "validation", "min_recall": min_recall, "max_fp_per_image": max_fp_per_image,
            "feasible": bool(feasible), "operating_point": max(feasible, key=lambda r: (r["precision"], r["recall"])) if feasible else None,
            "curve": rows, "note": "FP/image is not a false-positive rate on confirmed-normal images."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", default="data/processed/aircraft/validation.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--min-recall", type=float, required=True)
    parser.add_argument("--max-fp-per-image", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.validation).name != "validation.json" or "validation" not in Path(args.predictions).parts:
        parser.error("Use validation.json and predictions from reports/validation; never test predictions.")
    payload = json.loads(Path(args.validation).read_text())
    predictions = json.loads(Path(args.predictions).read_text())
    result = operating_point(payload["annotations"], predictions, len(payload["images"]), args.min_recall, args.max_fp_per_image)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
