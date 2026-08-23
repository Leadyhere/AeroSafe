"""Fit and evaluate the PatchCore engine anomaly comparison baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.baselines import PatchCoreBaseline
from src.data import AeBADDataset, split_aebad_training_paths
from src.evaluation import evaluate_engine_predictions
from src.training_monitor import (
    create_tensorboard_writer,
    finish_tensorboard,
    log_numeric_metrics,
)


def predict_dataset(model, loader):
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
    return labels, scores, np.asarray(masks), np.asarray(maps), domains, latencies


def main() -> int:
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    engine = config["engine"]
    baseline = config["baselines"]["patchcore"]
    train_paths, validation_paths = split_aebad_training_paths(
        config["paths"]["aebad"],
        float(engine["validation_fraction"]),
        int(config["training"]["seed"]),
    )
    if args.mode == "smoke":
        train_paths, validation_paths = train_paths[:2], validation_paths[:2]
    train_dataset = AeBADDataset(
        config["paths"]["aebad"], "train", image_size=int(engine["image_size"]), paths=train_paths
    )
    validation_dataset = AeBADDataset(
        config["paths"]["aebad"], "validation", image_size=int(engine["image_size"]), paths=validation_paths
    )
    batch_size = 1 if args.mode == "smoke" else int(baseline["batch_size"])
    workers = 0 if args.mode == "smoke" else int(config["training"]["num_workers"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = PatchCoreBaseline(
        image_size=int(engine["image_size"]),
        coreset_fraction=float(baseline["coreset_fraction"]),
        max_patches=int(baseline["max_patches"]),
        projection_dim=int(baseline["projection_dim"]),
        pretrained=True,
    )
    writer, _ = create_tensorboard_writer(config, "patchcore", args.mode)

    def report_fit_progress(completed_batches, total_batches, retained_patches):
        percent = 100.0 * completed_batches / max(1, total_batches)
        print(
            f"PatchCore feature extraction {completed_batches}/{total_batches} "
            f"({percent:.1f}%); retained patches={retained_patches}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("progress/percent", percent, completed_batches)
            writer.add_scalar("fit/retained_patches", retained_patches, completed_batches)

    model.fit(
        train_loader,
        smoke=args.mode == "smoke",
        progress_callback=report_fit_progress,
    )
    if writer is not None:
        writer.add_scalar("fit/memory_bank_patches", len(model.memory_bank), 1)

    validation_scores, validation_pixels = [], []
    for batch in validation_loader:
        for prediction in model.predict_tensor(batch["image"]):
            validation_scores.append(prediction["anomaly_score"])
            validation_pixels.append(np.asarray(prediction["anomaly_map"]).reshape(-1))
    if not validation_scores:
        raise RuntimeError("PatchCore validation calibration produced no scores.")
    quantile = float(engine["normal_threshold_quantile"])
    anomaly_threshold = float(np.quantile(validation_scores, quantile))
    pixel_threshold = float(np.quantile(np.concatenate(validation_pixels), quantile))
    log_numeric_metrics(
        writer,
        "validation_calibration",
        {
            "anomaly_threshold": anomaly_threshold,
            "pixel_threshold": pixel_threshold,
            "normal_samples": len(validation_scores),
            "score_mean": float(np.mean(validation_scores)),
            "score_std": float(np.std(validation_scores)),
        },
        1,
    )
    checkpoint_root = Path(config["paths"]["checkpoints"])
    checkpoint_path = (
        checkpoint_root / "smoke/patchcore_smoke.pt"
        if args.mode == "smoke"
        else Path(baseline["checkpoint"])
    )
    metadata = {
        "version": "fitted",
        "anomaly_threshold": anomaly_threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_source": "same held-out AeBAD-S normal split as MMR",
        "threshold_quantile": quantile,
        "normal_training_samples": len(train_dataset),
        "normal_validation_samples": len(validation_dataset),
        "memory_bank_patches": len(model.memory_bank),
        "config": config,
    }
    model.save(checkpoint_path, metadata)

    if args.mode == "full":
        test_dataset = AeBADDataset(
            config["paths"]["aebad"], "test", image_size=int(engine["image_size"])
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        result = predict_dataset(model, test_loader)
        evaluation_metrics = evaluate_engine_predictions(
            *result[:5],
            Path(config["paths"]["reports"]) / "baselines/patchcore",
            latencies_ms=result[5],
            pixel_threshold=pixel_threshold,
            anomaly_threshold=anomaly_threshold,
        )
        log_numeric_metrics(writer, "test", evaluation_metrics, 1)
    finish_tensorboard(writer)
    print(
        f"Completed {args.mode} PatchCore fitting with {len(model.memory_bank)} coreset patches."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
