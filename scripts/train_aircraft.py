"""Fine-tune Deformable DETR on prepared ASDD + cleaned aircraftsurface1 data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.aircraft_model import AircraftDetector, collate_detection_batch
from src.data import AircraftDetectionDataset, detection_sampling_weights
from src.evaluation import evaluate_aircraft_predictions
from src.preprocessing import build_aircraft_augmentation, load_image
from src.training_monitor import (
    create_tensorboard_writer,
    finish_tensorboard,
    log_epoch_progress,
    log_numeric_metrics,
)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_predictions(detector, annotation_file: Path, limit: int | None = None):
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    category_ids = [item["id"] for item in sorted(payload["categories"], key=lambda item: item["id"])]
    predictions, latencies = [], []
    for item in payload["images"][:limit]:
        # COCO AP needs the ranked prediction set; the deployment threshold is applied only in the app.
        detections = detector.predict(load_image(item["file_name"]), threshold=0.001)
        latencies.append(detector.last_inference_time_ms)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            predictions.append(
                {
                    "image_id": item["id"],
                    "category_id": category_ids[detector.label2id[detection["class"]]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": detection["confidence"],
                }
            )
    return predictions, latencies


def resolve_epoch_window(
    *, start_epoch: int, total_epochs: int, stop_after_epoch: int | None
) -> range:
    """Return the absolute, zero-based epoch range for this invocation.

    ``stop_after_epoch`` is intentionally an absolute completed-epoch count,
    rather than a per-run count. This makes a saved checkpoint safe to resume
    across short Kaggle sessions without silently repeating epochs.
    """
    if start_epoch < 0 or total_epochs < 1:
        raise ValueError("Epoch bounds must be non-negative with at least one total epoch.")
    if start_epoch >= total_epochs:
        return range(start_epoch, start_epoch)
    if stop_after_epoch is None:
        return range(start_epoch, total_epochs)
    if not 1 <= stop_after_epoch <= total_epochs:
        raise ValueError(
            f"--stop-after-epoch must be between 1 and {total_epochs}, got {stop_after_epoch}."
        )
    if stop_after_epoch <= start_epoch:
        raise ValueError(
            f"The resumed checkpoint already completed epoch {start_epoch}; "
            f"--stop-after-epoch must be greater than {start_epoch}."
        )
    return range(start_epoch, stop_after_epoch)


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Finish cleanly after this absolute completed epoch (1-based). "
            "Use it to create a resumable Kaggle training chunk."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    processed = Path(config["paths"]["processed"]) / "aircraft"
    train_file, validation_file = processed / "train.json", processed / "validation.json"
    auxiliary_file = Path(config["paths"]["processed"]) / "aircraft_auxiliary/agdd_train.json"
    if not train_file.is_file() or not validation_file.is_file():
        parser.error("Prepared COCO files are missing. Run scripts/prepare_data.py first.")
    if not auxiliary_file.is_file():
        parser.error("Prepared AGDD auxiliary file is missing. Run scripts/prepare_data.py first.")
    categories = sorted(json.loads(train_file.read_text())["categories"], key=lambda item: item["id"])
    labels = [item["name"] for item in categories]
    detector = AircraftDetector(
        labels,
        pretrained_model=config["aircraft"]["pretrained_model"],
        confidence_threshold=float(config["aircraft"]["confidence_threshold"]),
    )
    train_dataset = AircraftDetectionDataset(
        train_file,
        detector.processor,
        transform=build_aircraft_augmentation(
            int(config["aircraft"]["image_size"]),
            float(config["aircraft"].get("small_defect_crop_probability", 0.35)),
        ),
    )
    auxiliary_dataset = AircraftDetectionDataset(
        auxiliary_file,
        detector.processor,
        transform=build_aircraft_augmentation(
            int(config["aircraft"]["image_size"]),
            float(config["aircraft"].get("small_defect_crop_probability", 0.35)),
        ),
    )
    train_sampler = None
    auxiliary_sampler = None
    if args.mode == "smoke":
        train_dataset = Subset(train_dataset, range(min(2, len(train_dataset))))
        auxiliary_dataset = Subset(auxiliary_dataset, range(min(2, len(auxiliary_dataset))))
    else:
        max_sampling_weight = float(
            config["dataset"].get("class_balance_max_sampling_weight", 4.0)
        )
        train_weights = detection_sampling_weights(train_file, max_sampling_weight)
        auxiliary_weights = detection_sampling_weights(auxiliary_file, max_sampling_weight)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        auxiliary_sampler = WeightedRandomSampler(
            auxiliary_weights, len(auxiliary_weights), replacement=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if args.mode == "smoke" else int(config["aircraft"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        collate_fn=partial(collate_detection_batch, processor=detector.processor),
    )
    auxiliary_loader = DataLoader(
        auxiliary_dataset,
        batch_size=1 if args.mode == "smoke" else int(config["aircraft"]["batch_size"]),
        shuffle=auxiliary_sampler is None,
        sampler=auxiliary_sampler,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        collate_fn=partial(collate_detection_batch, processor=detector.processor),
    )
    backbone, remaining = [], []
    for name, parameter in detector.model.named_parameters():
        (backbone if "backbone" in name else remaining).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": remaining, "lr": float(config["aircraft"]["learning_rate"])},
            {"params": backbone, "lr": float(config["aircraft"]["backbone_learning_rate"])},
        ],
        weight_decay=float(config["aircraft"]["weight_decay"]),
    )
    main_epochs = 1 if args.mode == "smoke" else int(config["aircraft"]["epochs"])
    auxiliary_epochs = (
        1 if args.mode == "smoke" else int(config["aircraft"]["auxiliary_pretrain_epochs"])
    )
    total_epochs = auxiliary_epochs + main_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    amp_enabled = bool(config["training"]["mixed_precision"]) and detector.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_map = -1.0
    global_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=detector.device, weights_only=False)
        detector.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_map = float(state.get("best_map", -1.0))
        global_step = int(state.get("global_step", 0))
    try:
        epoch_window = resolve_epoch_window(
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            stop_after_epoch=args.stop_after_epoch,
        )
    except ValueError as exc:
        parser.error(str(exc))
    accumulation = 1 if args.mode == "smoke" else int(config["aircraft"]["gradient_accumulation"])
    checkpoint_root = Path(config["paths"]["checkpoints"])
    reports_root = Path(config["paths"]["reports"])
    checkpoint_destination = (
        checkpoint_root / "smoke" / "aircraft_best"
        if args.mode == "smoke"
        else Path(config["aircraft"]["checkpoint"])
    )
    training_state = checkpoint_root / (
        "smoke/aircraft_training_state.pt" if args.mode == "smoke" else "aircraft_training_state.pt"
    )
    reports = reports_root / "smoke/deformable_detr" if args.mode == "smoke" else reports_root
    dataset_report_path = reports / "dataset_report.json"
    dataset_report = (
        json.loads(dataset_report_path.read_text(encoding="utf-8"))
        if dataset_report_path.is_file() else None
    )
    writer, _ = create_tensorboard_writer(config, "deformable_detr", args.mode)
    for epoch in epoch_window:
        phase = "agdd_pretraining" if epoch < auxiliary_epochs else "aircraft_skin_finetuning"
        loader = auxiliary_loader if phase == "agdd_pretraining" else train_loader
        detector.model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_batches = 0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(detector.device)
            pixel_mask = batch["pixel_mask"].to(detector.device)
            targets = [{key: value.to(detector.device) for key, value in label.items()} for label in batch["labels"]]
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = detector.training_forward(pixel_values, pixel_mask, targets)
                raw_loss = outputs.loss
                loss = raw_loss / accumulation
            loss_value = float(raw_loss.detach())
            epoch_loss += loss_value
            epoch_batches += 1
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/batch_loss", loss_value, global_step)
                for loss_name, loss_component in (getattr(outputs, "loss_dict", {}) or {}).items():
                    writer.add_scalar(
                        f"train/loss_components/{loss_name}",
                        float(loss_component.detach()),
                        global_step,
                    )
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(detector.model.parameters(), float(config["aircraft"]["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if args.mode == "smoke":
                break
        scheduler.step()
        average_epoch_loss = epoch_loss / max(1, epoch_batches)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", average_epoch_loss, epoch + 1)
        log_epoch_progress(
            writer,
            epoch_index=epoch,
            total_epochs=total_epochs,
            phase=phase,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        metrics = None
        if phase == "aircraft_skin_finetuning":
            predictions, latencies = collect_predictions(
                detector, validation_file, 2 if args.mode == "smoke" else None
            )
            metrics = evaluate_aircraft_predictions(
                validation_file,
                predictions,
                reports,
                latencies_ms=latencies,
                calibrate_threshold=True,
            )
            log_numeric_metrics(writer, "validation", metrics, epoch + 1)
        metadata = {
            "version": f"epoch-{epoch + 1}",
            "epoch": epoch + 1,
            "validation_metrics": metrics,
            "dataset_report": dataset_report,
            "config": config,
            "training_phases": {
                "agdd_epochs": auxiliary_epochs,
                "aircraft_skin_epochs": main_epochs,
                "imdd_detector_usage": False,
            },
        }
        if metrics is not None and metrics["map_50_95"] >= best_map:
            best_map = metrics["map_50_95"]
            detector.confidence_threshold = float(metrics["recommended_confidence_threshold"])
            detector.save(checkpoint_destination, metadata)
        training_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": detector.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "global_step": global_step,
                "metadata": metadata,
            },
            training_state,
        )
        if writer is not None:
            writer.add_scalar("validation/best_map_50_95", best_map, epoch + 1)
            writer.flush()
    finish_tensorboard(writer)
    if epoch_window.stop < total_epochs:
        print(
            "Completed a resumable training chunk through "
            f"epoch {epoch_window.stop}/{total_epochs}; resume with: "
            f"--resume {training_state} --stop-after-epoch {total_epochs}"
        )
    else:
        print(f"Completed {args.mode} Deformable DETR training; best validation mAP={best_map:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
