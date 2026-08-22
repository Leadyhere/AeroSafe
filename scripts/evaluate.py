"""Evaluate frozen AeroInspect checkpoints on prepared final test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_aircraft import collect_predictions
from src import load_config
from src.aircraft_model import AircraftDetector
from src.baselines import FasterRCNNBaseline, PatchCoreBaseline
from src.data import AeBADDataset, load_aircraft_source, records_to_coco
from src.engine_model import MaskedMultiScaleReconstruction
from src.evaluation import evaluate_aircraft_predictions, evaluate_engine_predictions


def main() -> int:
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("aircraft", "engine", "faster-rcnn", "patchcore", "external-aircraft", "all"),
        default="all",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    reports = Path(config["paths"]["reports"])
    if args.target in {"aircraft", "all"}:
        test_json = Path(config["paths"]["processed"]) / "aircraft" / "test.json"
        detector = AircraftDetector.load(config["aircraft"]["checkpoint"])
        predictions, latencies = collect_predictions(detector, test_json)
        evaluate_aircraft_predictions(
            test_json,
            predictions,
            reports,
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
    if args.target in {"faster-rcnn", "all"}:
        test_json = Path(config["paths"]["processed"]) / "aircraft" / "test.json"
        detector = FasterRCNNBaseline.load(config["baselines"]["faster_rcnn"]["checkpoint"])
        predictions, latencies = collect_predictions(detector, test_json)
        evaluate_aircraft_predictions(
            test_json,
            predictions,
            reports / "baselines/faster_rcnn",
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
    if args.target in {"engine", "all"}:
        model = MaskedMultiScaleReconstruction.load_checkpoint(config["engine"]["checkpoint"])
        if model.pixel_threshold is None or model.anomaly_threshold is None:
            parser.error("MMR checkpoint lacks validation-derived thresholds.")
        dataset = AeBADDataset(
            config["paths"]["aebad"], "test", image_size=int(config["engine"]["image_size"])
        )
        loader = DataLoader(dataset, batch_size=int(config["engine"]["batch_size"]), shuffle=False)
        labels, scores, masks, maps, domains, latencies = [], [], [], [], [], []
        for batch in loader:
            predictions = model.predict_tensor(batch["image"], mask_ratio=0.0, passes=1)
            labels.extend(np.asarray(batch["is_anomaly"]).astype(int).tolist())
            masks.extend(np.asarray(batch["mask"]).squeeze(1))
            domains.extend(batch["domain"])
            for prediction in predictions:
                scores.append(prediction["anomaly_score"])
                maps.append(prediction["anomaly_map"])
                latencies.append(prediction["inference_time_ms"])
        evaluate_engine_predictions(
            labels, scores, np.asarray(masks), np.asarray(maps), domains, reports,
            latencies_ms=latencies,
            pixel_threshold=float(model.pixel_threshold),
            anomaly_threshold=float(model.anomaly_threshold),
        )
    if args.target in {"patchcore", "all"}:
        model = PatchCoreBaseline.load(config["baselines"]["patchcore"]["checkpoint"])
        if model.pixel_threshold is None or model.anomaly_threshold is None:
            parser.error("PatchCore checkpoint lacks validation-derived thresholds.")
        dataset = AeBADDataset(
            config["paths"]["aebad"], "test", image_size=int(config["engine"]["image_size"])
        )
        loader = DataLoader(
            dataset, batch_size=int(config["baselines"]["patchcore"]["batch_size"]), shuffle=False
        )
        labels, scores, masks, maps, domains, latencies = [], [], [], [], [], []
        for batch in loader:
            predictions = model.predict_tensor(batch["image"])
            labels.extend(np.asarray(batch["is_anomaly"]).astype(int).tolist())
            masks.extend(np.asarray(batch["mask"]).squeeze(1))
            domains.extend(batch["domain"])
            for prediction in predictions:
                scores.append(prediction["anomaly_score"])
                maps.append(prediction["anomaly_map"])
                latencies.append(prediction["inference_time_ms"])
        evaluate_engine_predictions(
            labels,
            scores,
            np.asarray(masks),
            np.asarray(maps),
            domains,
            reports / "baselines/patchcore",
            latencies_ms=latencies,
            pixel_threshold=float(model.pixel_threshold),
            anomaly_threshold=float(model.anomaly_threshold),
        )
    if args.target == "external-aircraft":
        detector = AircraftDetector.load(config["aircraft"]["checkpoint"])
        records = load_aircraft_source(
            config["paths"]["external_iisc"], "IISc external generalization",
            config["aircraft_labels"]
        )
        external_root = reports / "external_generalization"
        external_json = external_root / "iisc_external_test.json"
        records_to_coco(records, detector.labels, external_json)
        predictions, latencies = collect_predictions(detector, external_json)
        evaluate_aircraft_predictions(
            external_json,
            predictions,
            external_root,
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
    if args.target == "all":
        metric_files = {
            "deformable_detr": reports / "aircraft_metrics.json",
            "faster_rcnn": reports / "baselines/faster_rcnn/aircraft_metrics.json",
            "mmr": reports / "engine_metrics.json",
            "patchcore": reports / "baselines/patchcore/engine_metrics.json",
        }
        comparison = {
            model_name: json.loads(path.read_text(encoding="utf-8"))
            for model_name, path in metric_files.items()
        }
        (reports / "model_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
