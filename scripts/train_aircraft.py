"""Fine-tune one configured transformer detector on prepared aircraft data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
from src.experiment_identity import aircraft_data_identity, validate_resume_identity
from src.preprocessing import build_aircraft_augmentation, load_image
from src.training_artifacts import artifact_output_path, atomic_torch_save, create_training_archive
from src.training_chunks import SessionTimeGuard, boundary_for_part, training_boundaries
from src.training_monitor import (
    create_tensorboard_writer,
    finish_tensorboard,
    log_epoch_progress,
    log_numeric_metrics,
)

SESSION_STARTED = time.monotonic()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_predictions(detector, annotation_file: Path, limit: int | None = None, *, config=None):
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    expected_labels = [c["name"] for c in sorted(payload["categories"], key=lambda c: c["id"])]
    if list(detector.labels) != expected_labels:
        raise ValueError("Checkpoint label taxonomy differs from this dataset. Do not use old 7-class weights for binary evaluation.")
    category_ids = [item["id"] for item in sorted(payload["categories"], key=lambda item: item["id"])]
    predictions, latencies = [], []
    for item in payload["images"][:limit]:
        # COCO AP needs the ranked prediction set; the deployment threshold is applied only in the app.
        image = load_image(item["file_name"])
        tiling = (config or {}).get("aircraft", {}).get("tiled_inference", {})
        if tiling.get("enabled", False) and max(image.size) >= tiling.get("min_image_size", 768):
            from src.inference import predict_aircraft_tiled
            detections, elapsed = predict_aircraft_tiled(
                detector, image, threshold=0.001, tile_size=tiling.get("tile_size", 512),
                overlap=tiling.get("overlap", 128), nms_iou=tiling.get("nms_iou", 0.45)
            )
        else:
            detections = detector.predict(image, threshold=0.001)
            elapsed = detector.last_inference_time_ms
        latencies.append(elapsed)
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


def warmup_cosine_multiplier(epoch: int, *, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warm-up followed by cosine decay, expressed as an LR multiplier."""
    if total_epochs < 1 or warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("Warm-up must be non-negative and smaller than total epochs.")
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(1, warmup_epochs)
    # Epoch indices are zero-based: reach zero only AFTER the final update.
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--transformer",
        default=None,
        help="Transformer candidate key from aircraft.transformer_candidates.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--quarter", type=int, default=None)
    parser.add_argument("--max-session-hours", type=float, default=None)
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
    aircraft_config = config["aircraft"]
    candidates = aircraft_config.get("transformer_candidates", {})
    candidate_name = args.transformer or aircraft_config.get(
        "primary_transformer", "deformable_detr"
    )
    if candidate_name not in candidates:
        parser.error(
            f"Unknown transformer {candidate_name!r}; choose one of {sorted(candidates)}."
        )
    candidate_config = candidates[candidate_name]
    candidate_image_size = int(candidate_config.get("image_size", aircraft_config["image_size"]))
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    processed = Path(config["paths"]["processed"]) / "aircraft"
    train_file, validation_file = processed / "train.json", processed / "validation.json"
    if aircraft_config.get("training_annotations_override"):
        train_file = Path(aircraft_config["training_annotations_override"])
    auxiliary_file = Path(config["paths"]["processed"]) / "aircraft_auxiliary/agdd_train.json"
    if not train_file.is_file() or not validation_file.is_file():
        parser.error("Prepared COCO files are missing. Run scripts/prepare_data.py first.")
    if not auxiliary_file.is_file():
        parser.error("Prepared AGDD auxiliary file is missing. Run scripts/prepare_data.py first.")
    categories = sorted(json.loads(train_file.read_text())["categories"], key=lambda item: item["id"])
    labels = [item["name"] for item in categories]
    detector = AircraftDetector(
        labels,
        pretrained_model=candidate_config["pretrained_model"],
        architecture=candidate_name,
        image_size=candidate_image_size,
        confidence_threshold=float(aircraft_config["confidence_threshold"]),
    )
    train_dataset = AircraftDetectionDataset(
        train_file,
        detector.processor,
        transform=build_aircraft_augmentation(
            candidate_image_size,
            float(aircraft_config.get("small_defect_crop_probability", 0.35)),
        ),
    )
    auxiliary_dataset = AircraftDetectionDataset(
        auxiliary_file,
        detector.processor,
        transform=build_aircraft_augmentation(
            candidate_image_size,
            float(aircraft_config.get("small_defect_crop_probability", 0.35)),
        ),
    )
    train_sampler = None
    auxiliary_sampler = None
    smoke_batches = 3 if candidate_name == "deformable_detr" else 1
    if args.mode == "smoke":
        smoke_samples = max(2, smoke_batches)
        train_dataset = Subset(train_dataset, range(min(smoke_samples, len(train_dataset))))
        auxiliary_dataset = Subset(auxiliary_dataset, range(min(smoke_samples, len(auxiliary_dataset))))
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
        batch_size=1 if args.mode == "smoke" else int(aircraft_config["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        collate_fn=partial(collate_detection_batch, processor=detector.processor),
    )
    auxiliary_loader = DataLoader(
        auxiliary_dataset,
        batch_size=1 if args.mode == "smoke" else int(aircraft_config["batch_size"]),
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
            {"params": remaining, "lr": float(aircraft_config["learning_rate"])},
            {"params": backbone, "lr": float(aircraft_config["backbone_learning_rate"])},
        ],
        weight_decay=float(aircraft_config["weight_decay"]),
    )
    main_epochs = 1 if args.mode == "smoke" else int(
        candidate_config.get("epochs", aircraft_config["epochs"])
    )
    auxiliary_epochs = (
        1 if args.mode == "smoke" else int(aircraft_config["auxiliary_pretrain_epochs"])
    )
    total_epochs = auxiliary_epochs + main_epochs
    training_config = config["training"]
    boundaries = training_boundaries(
        main_epochs,
        auxiliary_epochs=auxiliary_epochs,
        quarter_count=int(training_config.get("quarter_count", 4)),
        one_go_max_epochs=int(training_config.get("one_go_max_epochs", 1)),
    )
    if args.quarter is not None and args.stop_after_epoch is not None:
        parser.error("Use either --quarter or --stop-after-epoch, not both.")
    try:
        planned_stop_epoch = (
            args.stop_after_epoch
            if args.stop_after_epoch is not None
            else boundary_for_part(boundaries, args.quarter)
        )
    except ValueError as exc:
        parser.error(str(exc))
    warmup_epochs = 0 if args.mode == "smoke" else int(aircraft_config.get("warmup_epochs", 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_multiplier(
            epoch, warmup_epochs=warmup_epochs, total_epochs=total_epochs
        ),
    )
    amp_enabled = (bool(config["training"]["mixed_precision"])
                   and detector.device.type == "cuda" and candidate_name != "deformable_detr")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_root = Path(config["paths"]["checkpoints"])
    reports_root = Path(config["paths"]["reports"])
    checkpoint_destination = (
        checkpoint_root / "smoke" / "aircraft_candidates" / candidate_name
        if args.mode == "smoke"
        else Path(candidate_config["checkpoint"])
    )
    training_state = checkpoint_root / (
        f"smoke/aircraft_candidates/{candidate_name}_training_state.pt"
        if args.mode == "smoke"
        else f"aircraft_candidates/{candidate_name}_training_state.pt"
    )
    reports = reports_root / (
        f"smoke/aircraft_transformers/{candidate_name}"
        if args.mode == "smoke"
        else f"aircraft_transformers/{candidate_name}"
    )
    start_epoch = 0
    best_map = -1.0
    global_step = 0
    resume_path = Path(args.resume) if args.resume else None
    experiment_identity = {
        "revision": 2, "architecture": candidate_name, "seed": seed,
        "data": aircraft_data_identity([train_file, validation_file, auxiliary_file]),
        "recipe": {key: value for key, value in aircraft_config.items()
                   if key not in {"checkpoint", "transformer_candidates", "training_annotations_override"}},
        "candidate": {key: value for key, value in candidate_config.items() if key != "checkpoint"},
    }
    if candidate_name == "deformable_detr":
        experiment_identity["loading_policy"] = "explicit_cpu_native_fp32_v1"
    if resume_path is None and args.quarter is not None and training_state.is_file():
        resume_path = training_state
    if args.quarter is not None and args.quarter > 1 and resume_path is None:
        parser.error(f"Part {args.quarter} requires the previous state at {training_state}.")
    if resume_path is not None:
        state = torch.load(resume_path, map_location=detector.device, weights_only=False)
        validate_resume_identity(state, experiment_identity)
        resumed_architecture = state.get("architecture")
        if resumed_architecture is not None and resumed_architecture != candidate_name:
            parser.error(
                f"Resume state is for {resumed_architecture!r}, not {candidate_name!r}."
            )
        detector.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_map = float(state.get("best_map", -1.0))
        global_step = int(state.get("global_step", 0))
    if args.quarter is not None and args.quarter > 1:
        required_completed = boundaries[args.quarter - 2]
        if start_epoch < required_completed:
            parser.error(
                f"Part {args.quarter - 1} is incomplete: state has {start_epoch} completed "
                f"epochs but requires {required_completed}."
            )
    try:
        epoch_window = resolve_epoch_window(
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            stop_after_epoch=planned_stop_epoch,
        )
    except ValueError as exc:
        parser.error(str(exc))
    accumulation = 1 if args.mode == "smoke" else int(aircraft_config["gradient_accumulation"])
    dataset_report_path = reports_root / "dataset_report.json"
    dataset_report = (
        json.loads(dataset_report_path.read_text(encoding="utf-8"))
        if dataset_report_path.is_file() else None
    )
    writer, _ = create_tensorboard_writer(config, f"aircraft_{candidate_name}", args.mode)
    time_guard = SessionTimeGuard(
        max_session_hours=float(
            args.max_session_hours or training_config.get("max_session_hours", 11.0)
        ),
        packaging_reserve_minutes=float(
            training_config.get("artifact_reserve_minutes", 45)
        ),
        started_at=SESSION_STARTED,
    )
    validation_interval = int(aircraft_config.get("validation_interval", 5))
    completed_epochs = start_epoch
    stopped_for_time = False
    for epoch in epoch_window:
        epoch_started = time.monotonic()
        phase = "agdd_pretraining" if epoch < auxiliary_epochs else "aircraft_skin_finetuning"
        loader = auxiliary_loader if phase == "agdd_pretraining" else train_loader
        detector.model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_batches = 0
        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(detector.device)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(detector.device)
            targets = [{key: value.to(detector.device) for key, value in label.items()} for label in batch["labels"]]
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = detector.training_forward(pixel_values, pixel_mask, targets)
                raw_loss = outputs.loss
                # The final accumulation group can contain fewer batches.
                group_start = (step // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                loss = raw_loss / group_size
            loss_value = float(raw_loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite loss before backward: epoch={epoch + 1}, batch={step + 1}, phase={phase}."
                )
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
                torch.nn.utils.clip_grad_norm_(
                    detector.model.parameters(), float(aircraft_config["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if args.mode == "smoke" and step + 1 >= smoke_batches:
                break
        scheduler.step()
        time_guard.record_epoch(time.monotonic() - epoch_started)
        completed_epochs = epoch + 1
        stopped_for_time = time_guard.should_stop_before_next_epoch(
            completed_epochs, epoch_window.stop
        )
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
        fine_tune_epoch = completed_epochs - auxiliary_epochs
        if phase == "aircraft_skin_finetuning" and (
            args.mode == "smoke"
            or fine_tune_epoch % validation_interval == 0
            or completed_epochs == epoch_window.stop
            or stopped_for_time
        ):
            predictions, latencies = collect_predictions(
                detector, validation_file, 2 if args.mode == "smoke" else None, config=config
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
            "experiment_identity": experiment_identity,
            "architecture": candidate_name,
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
        atomic_torch_save(
            {
                "model": detector.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "global_step": global_step,
                "metadata": metadata,
                "architecture": candidate_name,
                "epoch_seconds": time_guard.epoch_seconds,
            },
            training_state,
        )
        if writer is not None:
            writer.add_scalar("validation/best_map_50_95", best_map, epoch + 1)
            writer.flush()
        if stopped_for_time:
            print("Stopping before the next epoch to preserve Kaggle packaging time.", flush=True)
            break
    finish_tensorboard(writer)
    if args.mode == "full":
        artifact_root = Path(config["paths"].get("artifacts", "artifacts"))
        artifact_path = artifact_output_path(
            artifact_root,
            candidate_name,
            completed_epochs,
            total_epochs,
            part=args.quarter,
            time_limited=stopped_for_time,
        )
        artifact = create_training_archive(
            PROJECT_ROOT,
            candidate_name,
            artifact_path,
            [
                training_state,
                *([checkpoint_destination] if checkpoint_destination.exists() else []),
                reports,
                dataset_report_path,
                processed,
                Path(config["paths"]["processed"]) / "aircraft_auxiliary",
                Path(args.config),
            ],
            run_metadata={
                "completed_epochs": completed_epochs,
                "total_epochs": total_epochs,
                "part": args.quarter,
                "time_limited": stopped_for_time,
                "resume_state": str(training_state),
            },
        )
        print(f"Downloadable artifact: {artifact['archive']['path']}")
        print(f"SHA-256 file: {artifact['archive']['checksum_file']}")
    if completed_epochs < total_epochs:
        print(
            "Completed a resumable training chunk through "
            f"epoch {completed_epochs}/{total_epochs}; rerun the same part or resume with "
            f"--resume {training_state}."
        )
    else:
        print(
            f"Completed {args.mode} {candidate_name} training; "
            f"best validation mAP={best_map:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
