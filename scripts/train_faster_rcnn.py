"""Train the Faster R-CNN ResNet50-FPN aircraft comparison baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_aircraft import collect_predictions, resolve_epoch_window
from src import load_config
from src.baselines import FasterRCNNBaseline, FasterRCNNDataset, collate_faster_rcnn
from src.data import detection_sampling_weights
from src.evaluation import evaluate_aircraft_predictions
from src.preprocessing import build_aircraft_augmentation
from src.training_artifacts import artifact_output_path, atomic_torch_save, create_training_archive
from src.training_chunks import SessionTimeGuard, boundary_for_part, training_boundaries
from src.training_monitor import (
    create_tensorboard_writer,
    finish_tensorboard,
    log_epoch_progress,
    log_numeric_metrics,
)

SESSION_STARTED = time.monotonic()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--quarter", type=int, default=None)
    parser.add_argument("--max-session-hours", type=float, default=None)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Finish cleanly after this absolute completed epoch (1-based).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    baseline = config["baselines"]["faster_rcnn"]
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processed = Path(config["paths"]["processed"]) / "aircraft"
    train_file = processed / "train.json"
    validation_file = processed / "validation.json"
    auxiliary_file = Path(config["paths"]["processed"]) / "aircraft_auxiliary/agdd_train.json"
    if not train_file.is_file() or not validation_file.is_file():
        parser.error("Prepared aircraft splits are missing. Run scripts/prepare_data.py first.")
    if not auxiliary_file.is_file():
        parser.error("Prepared AGDD auxiliary split is missing. Run scripts/prepare_data.py first.")
    categories = sorted(json.loads(train_file.read_text(encoding="utf-8"))["categories"], key=lambda x: x["id"])
    model = FasterRCNNBaseline(
        [item["name"] for item in categories],
        pretrained=True,
        confidence_threshold=float(config["aircraft"]["confidence_threshold"]),
    )
    augmentation = build_aircraft_augmentation(
        int(config["aircraft"]["image_size"]),
        float(config["aircraft"].get("small_defect_crop_probability", 0.35)),
    )
    dataset = FasterRCNNDataset(train_file, train=True, transform=augmentation)
    auxiliary_dataset = FasterRCNNDataset(auxiliary_file, train=True, transform=augmentation)
    train_sampler = None
    auxiliary_sampler = None
    if args.mode == "smoke":
        dataset = Subset(dataset, range(min(2, len(dataset))))
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
        dataset,
        batch_size=1 if args.mode == "smoke" else int(baseline["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        collate_fn=collate_faster_rcnn,
    )
    auxiliary_loader = DataLoader(
        auxiliary_dataset,
        batch_size=1 if args.mode == "smoke" else int(baseline["batch_size"]),
        shuffle=auxiliary_sampler is None,
        sampler=auxiliary_sampler,
        num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        collate_fn=collate_faster_rcnn,
    )
    parameters = [parameter for parameter in model.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(baseline["learning_rate"]),
        momentum=float(baseline["momentum"]),
        weight_decay=float(baseline["weight_decay"]),
    )
    main_epochs = 1 if args.mode == "smoke" else int(baseline["epochs"])
    auxiliary_epochs = (
        1 if args.mode == "smoke" else int(config["aircraft"]["auxiliary_pretrain_epochs"])
    )
    total_epochs = auxiliary_epochs + main_epochs
    training_config = config["training"]
    boundaries = training_boundaries(
        main_epochs,
        auxiliary_epochs=auxiliary_epochs,
        quarter_count=int(training_config.get("quarter_count", 4)),
        one_go_max_epochs=int(training_config.get("one_go_max_epochs", 20)),
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_epochs))
    amp_enabled = bool(config["training"]["mixed_precision"]) and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_root = Path(config["paths"]["checkpoints"])
    checkpoint_path = (
        output_root / "smoke" / "faster_rcnn_smoke.pt"
        if args.mode == "smoke"
        else Path(baseline["checkpoint"])
    )
    state_path = output_root / ("smoke/faster_rcnn_state.pt" if args.mode == "smoke" else "faster_rcnn_training_state.pt")
    report_dir = Path(config["paths"]["reports"]) / (
        "smoke/faster_rcnn" if args.mode == "smoke" else "baselines/faster_rcnn"
    )
    start_epoch, best_map, global_step = 0, -1.0, 0
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.quarter is not None and state_path.is_file():
        resume_path = state_path
    if args.quarter is not None and args.quarter > 1 and resume_path is None:
        parser.error(f"Part {args.quarter} requires the previous state at {state_path}.")
    if resume_path is not None:
        state = torch.load(resume_path, map_location=model.device, weights_only=False)
        model.model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
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

    writer, _ = create_tensorboard_writer(config, "faster_rcnn", args.mode)
    time_guard = SessionTimeGuard(
        max_session_hours=float(
            args.max_session_hours or training_config.get("max_session_hours", 11.0)
        ),
        packaging_reserve_minutes=float(
            training_config.get("artifact_reserve_minutes", 45)
        ),
        started_at=SESSION_STARTED,
    )
    validation_interval = int(baseline.get("validation_interval", 5))
    completed_epochs = start_epoch
    stopped_for_time = False
    for epoch in epoch_window:
        epoch_started = time.monotonic()
        phase = "agdd_pretraining" if epoch < auxiliary_epochs else "aircraft_skin_finetuning"
        loader = auxiliary_loader if phase == "agdd_pretraining" else train_loader
        model.model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for images, targets in loader:
            images = [image.to(model.device) for image in images]
            targets = [{key: value.to(model.device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = model.model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Faster R-CNN loss: {float(loss.detach())}")
            loss_value = float(loss.detach())
            epoch_loss += loss_value
            epoch_batches += 1
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/batch_loss", loss_value, global_step)
                for loss_name, loss_component in losses.items():
                    writer.add_scalar(
                        f"train/loss_components/{loss_name}",
                        float(loss_component.detach()),
                        global_step,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            if args.mode == "smoke":
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
                model, validation_file, limit=2 if args.mode == "smoke" else None
            )
            metrics = evaluate_aircraft_predictions(
                validation_file,
                predictions,
                report_dir,
                latencies_ms=latencies,
                calibrate_threshold=True,
            )
            log_numeric_metrics(writer, "validation", metrics, epoch + 1)
        metadata = {
            "version": f"epoch-{epoch + 1}",
            "epoch": epoch + 1,
            "validation_metrics": metrics,
            "config": config,
            "comparison_split": "same prepared split as Deformable DETR",
            "training_phases": {
                "agdd_epochs": auxiliary_epochs,
                "aircraft_skin_epochs": main_epochs,
                "imdd_detector_usage": False,
            },
        }
        if metrics is not None and metrics["map_50_95"] >= best_map:
            best_map = float(metrics["map_50_95"])
            model.confidence_threshold = float(metrics["recommended_confidence_threshold"])
            model.save(checkpoint_path, metadata)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {
                "state_dict": model.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "global_step": global_step,
                "epoch_seconds": time_guard.epoch_seconds,
            },
            state_path,
        )
        if writer is not None:
            writer.add_scalar("validation/best_map_50_95", best_map, epoch + 1)
            writer.flush()
        if stopped_for_time:
            print("Stopping before the next epoch to preserve Kaggle packaging time.", flush=True)
            break
    finish_tensorboard(writer)
    if args.mode == "full":
        artifact_path = artifact_output_path(
            config["paths"].get("artifacts", "artifacts"),
            "faster_rcnn",
            completed_epochs,
            total_epochs,
            part=args.quarter,
            time_limited=stopped_for_time,
        )
        artifact = create_training_archive(
            PROJECT_ROOT,
            "faster_rcnn",
            artifact_path,
            [
                state_path,
                *([checkpoint_path] if checkpoint_path.exists() else []),
                report_dir,
                Path(config["paths"]["reports"]) / "dataset_report.json",
                processed,
                Path(config["paths"]["processed"]) / "aircraft_auxiliary",
                Path(args.config),
            ],
            run_metadata={
                "completed_epochs": completed_epochs,
                "total_epochs": total_epochs,
                "part": args.quarter,
                "time_limited": stopped_for_time,
                "resume_state": str(state_path),
            },
        )
        print(f"Downloadable artifact: {artifact['archive']['path']}")
        print(f"SHA-256 file: {artifact['archive']['checksum_file']}")
    if completed_epochs < total_epochs:
        print(
            "Completed a resumable Faster R-CNN chunk through "
            f"epoch {completed_epochs}/{total_epochs}; rerun the same part or resume from "
            f"{state_path}."
        )
    else:
        print(f"Completed {args.mode} Faster R-CNN training; best validation mAP={best_map:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
