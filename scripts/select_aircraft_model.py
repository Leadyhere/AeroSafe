"""Select the aircraft candidate using current validation results only."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import load_config


def select_candidates(results):
    if not results or any(item.get("split") != "validation" for item in results.values()):
        raise ValueError("Model selection requires validation-only result records.")
    if len({item["dataset_identity"] for item in results.values()}) != 1:
        raise ValueError("Candidates were evaluated on different datasets.")
    if len({json.dumps(item.get("tiling", {}), sort_keys=True) for item in results.values()}) != 1:
        raise ValueError("Candidates used different inference policies.")
    ranking = sorted(results, key=lambda name: (
        results[name]["metrics"]["map_50_95"], results[name]["metrics"]["recall"]
    ), reverse=True)
    return {"split": "validation", "recommended": ranking[0], "ranking": ranking,
            "selection_rule": "mAP50:95 then recall", "candidates": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["paths"]["reports"]) / "validation"
    results = {}
    for name in args.models:
        folder = root / ("baselines/faster_rcnn" if name == "faster_rcnn" else f"aircraft_transformers/{name}")
        results[name] = json.loads((folder / "validation_record.json").read_text())
    selection = select_candidates(results)
    destination = root.parent / "model_selection.json"
    destination.write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
