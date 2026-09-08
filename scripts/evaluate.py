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
from src.experiment_identity import aircraft_data_identity


def write_validation_record(folder, annotation_file, detector, config):
    folder = Path(folder)
    record = {
        "split": "validation",
        "dataset_identity": aircraft_data_identity([annotation_file]),
        "model_version": detector.version,
        "tiling": config["aircraft"].get("tiled_inference", {}),
        "metrics": json.loads((folder / "aircraft_metrics.json").read_text()),
    }
    (folder / "validation_record.json").write_text(json.dumps(record, indent=2))


def metric_rank_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    return -1.0 if value is None else float(value)


def main() -> int:
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=(
            "aircraft",
            "aircraft-transformers",
            "engine",
            "engine-bladesynth",
            "faster-rcnn",
            "patchcore",
            "external-aircraft",
            "all",
        ),
        default="aircraft-transformers",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--final-test", action="store_true",
                        help="Explicitly unlock final/external testing after model selection is frozen.")
    args = parser.parse_args()
    config = load_config(args.config)
    if (args.split == "test" or args.target == "external-aircraft") and not args.final_test:
        parser.error("Final test is locked. Freeze validation model selection, then pass --final-test --split test.")
    if args.split == "validation" and args.target in {"engine", "engine-bladesynth", "patchcore", "all"}:
        parser.error("Engine validation contains normals only: calibrate during training; AP/AUROC model selection needs a separate labeled development set.")
    reports = Path(config["paths"]["reports"])
    if args.split == "validation":
        reports = reports / "validation"
    if args.target == "aircraft":
        test_json = Path(config["paths"]["processed"]) / "aircraft" / f"{args.split}.json"
        detector = AircraftDetector.load(config["aircraft"]["checkpoint"])
        predictions, latencies = collect_predictions(detector, test_json, config=config)
        evaluate_aircraft_predictions(
            test_json,
            predictions,
            reports,
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
    if args.target in {"aircraft-transformers", "all"}:
        test_json = Path(config["paths"]["processed"]) / "aircraft" / f"{args.split}.json"
        for candidate_name, candidate in config["aircraft"].get(
            "transformer_candidates", {}
        ).items():
            detector = AircraftDetector.load(candidate["checkpoint"])
            predictions, latencies = collect_predictions(detector, test_json, config=config)
            evaluate_aircraft_predictions(
                test_json,
                predictions,
                reports / "aircraft_transformers" / candidate_name,
                latencies_ms=latencies,
                confidence_threshold=detector.confidence_threshold,
            )
            if args.split == "validation":
                write_validation_record(reports / "aircraft_transformers" / candidate_name,
                                        test_json, detector, config)
    if args.target in {"faster-rcnn", "all"}:
        test_json = Path(config["paths"]["processed"]) / "aircraft" / f"{args.split}.json"
        detector = FasterRCNNBaseline.load(config["baselines"]["faster_rcnn"]["checkpoint"])
        predictions, latencies = collect_predictions(detector, test_json, config=config)
        evaluate_aircraft_predictions(
            test_json,
            predictions,
            reports / "baselines/faster_rcnn",
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
        if args.split == "validation":
            write_validation_record(reports / "baselines/faster_rcnn", test_json, detector, config)
    if args.target in {"engine", "engine-bladesynth", "all"}:
        variants = []
        if args.target in {"engine", "all"}:
            variants.append(("mmr_real", config["engine"]["checkpoint"], reports))
        if args.target in {"engine-bladesynth", "all"}:
            variants.append(
                (
                    "mmr_bladesynth",
                    config["engine"]["bladesynth_checkpoint"],
                    reports / "experiments/bladesynth_mmr",
                )
            )
        dataset = AeBADDataset(
            config["paths"]["aebad"], "test", image_size=int(config["engine"]["image_size"])
        )
        loader = DataLoader(
            dataset, batch_size=int(config["engine"]["batch_size"]), shuffle=False
        )
        for variant_name, checkpoint, output_dir in variants:
            model = MaskedMultiScaleReconstruction.load_checkpoint(checkpoint)
            if model.pixel_threshold is None or model.anomaly_threshold is None:
                parser.error(f"{variant_name} checkpoint lacks validation-derived thresholds.")
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
                labels,
                scores,
                np.asarray(masks),
                np.asarray(maps),
                domains,
                output_dir,
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
        if detector.labels == ["defect"]:
            from src.data import collapse_aircraft_annotations
            collapse_aircraft_annotations(records)
        records_to_coco(records, detector.labels, external_json)
        predictions, latencies = collect_predictions(detector, external_json, config=config)
        evaluate_aircraft_predictions(
            external_json,
            predictions,
            external_root,
            latencies_ms=latencies,
            confidence_threshold=detector.confidence_threshold,
        )
    if args.target == "all":
        metric_files = {
            "faster_rcnn": reports / "baselines/faster_rcnn/aircraft_metrics.json",
            "mmr": reports / "engine_metrics.json",
            "mmr_bladesynth": reports / "experiments/bladesynth_mmr/engine_metrics.json",
            "patchcore": reports / "baselines/patchcore/engine_metrics.json",
        }
        metric_files.update(
            {
                candidate_name: reports
                / "aircraft_transformers"
                / candidate_name
                / "aircraft_metrics.json"
                for candidate_name in config["aircraft"].get(
                    "transformer_candidates", {}
                )
            }
        )
        comparison = {
            model_name: json.loads(path.read_text(encoding="utf-8"))
            for model_name, path in metric_files.items()
        }
        (reports / "model_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        aircraft_names = [
            *config["aircraft"].get("transformer_candidates", {}).keys(),
            "faster_rcnn",
        ]
        engine_names = ["mmr", "mmr_bladesynth", "patchcore"]
        aircraft_ranking = sorted(
            aircraft_names,
            key=lambda name: (
                metric_rank_value(comparison[name], "map_50_95"),
                metric_rank_value(comparison[name], "recall"),
            ),
            reverse=True,
        )
        engine_ranking = sorted(
            engine_names,
            key=lambda name: (
                metric_rank_value(comparison[name], "image_average_precision"),
                metric_rank_value(comparison[name], "image_auroc"),
                metric_rank_value(comparison[name], "aupro"),
            ),
            reverse=True,
        )
        selection = {
            "aircraft": {
                "ranking": aircraft_ranking,
                "recommended": None,
                "selection_rule": "highest mAP@50:95, then recall",
            },
            "engine": {
                "ranking": engine_ranking,
                "recommended": None,
                "selection_rule": "highest image average precision, then image AUROC, then AUPRO",
            },
            "split": "test",
            "warning": "Descriptive test rankings only. Never use these to select or tune models.",
        }
        (reports / "test_comparison.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
