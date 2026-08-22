"""Train MMR on normal AeBAD-S images and calibrate thresholds on held-out normal validation data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import AeBADDataset, split_aebad_training_paths
from src.engine_model import MaskedMultiScaleReconstruction
from src.evaluation import evaluate_engine_predictions


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    engine = config["engine"]
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    train_paths, validation_paths = split_aebad_training_paths(
        config["paths"]["aebad"], float(engine["validation_fraction"]), seed
    )
    if args.mode == "smoke":
        train_paths, validation_paths = train_paths[:2], validation_paths[:2]
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                (int(engine["image_size"]), int(engine["image_size"])), scale=(0.7, 1.0), antialias=True
            ),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = AeBADDataset(
        config["paths"]["aebad"], "train", image_size=int(engine["image_size"]),
        paths=train_paths, transform=train_transform
    )
    validation_dataset = AeBADDataset(
        config["paths"]["aebad"], "validation", image_size=int(engine["image_size"]), paths=validation_paths
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if args.mode == "smoke" else int(engine["batch_size"]),
        shuffle=True,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=int(engine["batch_size"]), shuffle=False)
    model = MaskedMultiScaleReconstruction(
        image_size=int(engine["image_size"]),
        teacher_backbone=engine["teacher_backbone"],
        mae_backbone=engine["mae_backbone"],
        pretrained=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(engine["learning_rate"]),
        weight_decay=float(engine["weight_decay"]),
    )
    epochs = 1 if args.mode == "smoke" else int(engine["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    amp_enabled = bool(config["training"]["mixed_precision"]) and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_root = Path(config["paths"]["checkpoints"])
    training_state = checkpoint_root / (
        "smoke/engine_training_state.pt" if args.mode == "smoke" else "engine_training_state.pt"
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(images, mask_ratio=float(engine["mask_ratio"]))
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if args.mode == "smoke":
                break
        scheduler.step()
        if (epoch + 1) % int(config["training"].get("checkpoint_every", 1)) == 0:
            training_state.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "metadata": {"config": config, "thresholds_calibrated": False},
                },
                training_state,
            )
    # Calibration uses held-out training normals only, never the official final test set.
    validation_scores, validation_pixels = [], []
    for batch in validation_loader:
        predictions = model.predict_tensor(batch["image"], mask_ratio=0.0, passes=1)
        validation_scores.extend(item["anomaly_score"] for item in predictions)
        validation_pixels.extend(np.asarray(item["anomaly_map"]).reshape(-1) for item in predictions)
    if not validation_scores:
        raise RuntimeError("MMR validation calibration produced no scores.")
    quantile = float(engine["normal_threshold_quantile"])
    anomaly_threshold = float(np.quantile(validation_scores, quantile))
    pixel_threshold = float(np.quantile(np.concatenate(validation_pixels), quantile))
    metadata = {
        "version": f"epoch-{epochs}",
        "epoch": epochs,
        "anomaly_threshold": anomaly_threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_source": "held-out AeBAD-S training normals",
        "threshold_quantile": quantile,
        "inference_masks": int(engine["inference_masks"]),
        "validation_calibration": {
            "normal_samples": len(validation_scores),
            "score_mean": float(np.mean(validation_scores)),
            "score_std": float(np.std(validation_scores)),
            "score_min": float(np.min(validation_scores)),
            "score_max": float(np.max(validation_scores)),
        },
        "dataset_report": (
            json.loads((Path(config["paths"]["reports"]) / "dataset_report.json").read_text(encoding="utf-8"))
            if (Path(config["paths"]["reports"]) / "dataset_report.json").is_file() else None
        ),
        "config": config,
    }
    checkpoint_destination = (
        checkpoint_root / "smoke/engine_best.pt"
        if args.mode == "smoke"
        else Path(engine["checkpoint"])
    )
    model.save_checkpoint(checkpoint_destination, metadata)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epochs - 1,
            "metadata": metadata,
        },
        training_state,
    )
    if args.mode == "full":
        test_dataset = AeBADDataset(
            config["paths"]["aebad"], "test", image_size=int(engine["image_size"])
        )
        test_loader = DataLoader(test_dataset, batch_size=int(engine["batch_size"]), shuffle=False)
        labels, scores, masks, maps, domains, latencies = [], [], [], [], [], []
        for batch in test_loader:
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
            config["paths"]["reports"],
            latencies_ms=latencies,
            pixel_threshold=pixel_threshold,
        )
    print(
        f"Completed {args.mode} MMR training; thresholds calibrated from held-out normal validation data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
