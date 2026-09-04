"""Train the real-only MMR or a separately reported BladeSynth auxiliary experiment."""

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

from scripts.train_aircraft import resolve_epoch_window, warmup_cosine_multiplier
from src import load_config
from src.data import (
    AeBADDataset,
    NormalImageDataset,
    find_bladesynth_normal_paths,
    sample_aebad_v_training_paths,
    split_aebad_training_paths,
)
from src.engine_model import MaskedMultiScaleReconstruction
from src.evaluation import evaluate_engine_predictions
from src.training_artifacts import artifact_output_path, create_training_archive
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


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--variant",
        choices=("real", "bladesynth"),
        default="real",
        help="real = AeBAD-S only; bladesynth = separate synthetic-auxiliary experiment",
    )
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
    engine = config["engine"]
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    train_paths, validation_paths = split_aebad_training_paths(
        config["paths"]["aebad"], float(engine["validation_fraction"]), seed
    )
    video_paths: list[str] = []
    synthetic_normal_paths: list[str] = []
    auxiliary_paths: list[str] = []
    if args.variant == "bladesynth":
        video_paths = sample_aebad_v_training_paths(
            config["paths"]["aebad"], int(engine["aebad_v_frame_stride"])
        )
        synthetic_normal_paths = find_bladesynth_normal_paths(config["paths"]["bladesynth"])
        expected_bladesynth_normals = int(
            engine.get("bladesynth_expected_images_per_class", 2500)
        )
        if len(synthetic_normal_paths) != expected_bladesynth_normals:
            parser.error(
                f"BladeSynth Normal has {len(synthetic_normal_paths)} images; expected "
                f"{expected_bladesynth_normals}. Run dataset preparation for details."
            )
        auxiliary_paths = [*video_paths, *synthetic_normal_paths]
        random.Random(seed).shuffle(auxiliary_paths)
    if args.mode == "smoke":
        train_paths, validation_paths = train_paths[:2], validation_paths[:2]
        auxiliary_paths = auxiliary_paths[:2]
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
    auxiliary_loader = None
    if auxiliary_paths:
        auxiliary_dataset = NormalImageDataset(
            auxiliary_paths, image_size=int(engine["image_size"]), transform=train_transform
        )
        auxiliary_loader = DataLoader(
            auxiliary_dataset,
            batch_size=1 if args.mode == "smoke" else int(engine["batch_size"]),
            shuffle=True,
            num_workers=0 if args.mode == "smoke" else int(config["training"]["num_workers"]),
        )
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
    main_epochs = 1 if args.mode == "smoke" else int(engine["epochs"])
    auxiliary_epochs = (
        (1 if args.mode == "smoke" else int(engine["auxiliary_pretrain_epochs"]))
        if args.variant == "bladesynth"
        else 0
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
    warmup_epochs = 0 if args.mode == "smoke" else int(engine.get("warmup_epochs", 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_multiplier(
            epoch, warmup_epochs=warmup_epochs, total_epochs=total_epochs
        ),
    )
    amp_enabled = bool(config["training"]["mixed_precision"]) and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_root = Path(config["paths"]["checkpoints"])
    variant_suffix = "" if args.variant == "real" else "_bladesynth"
    training_state = checkpoint_root / (
        f"smoke/engine{variant_suffix}_training_state.pt"
        if args.mode == "smoke"
        else f"engine{variant_suffix}_training_state.pt"
    )
    checkpoint_destination = (
        checkpoint_root / f"smoke/engine{variant_suffix}_best.pt"
        if args.mode == "smoke"
        else Path(
            engine["checkpoint"]
            if args.variant == "real"
            else engine["bladesynth_checkpoint"]
        )
    )
    evaluation_root = (
        Path(config["paths"]["reports"])
        if args.variant == "real"
        else Path(config["paths"]["reports"]) / "experiments/bladesynth_mmr"
    )
    artifact_report_sources = (
        [evaluation_root]
        if args.variant == "bladesynth"
        else [
            evaluation_root / "engine_metrics.json",
            evaluation_root / "engine_domain_metrics.csv",
            evaluation_root / "engine_examples",
        ]
    )
    start_epoch = 0
    global_step = 0
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.quarter is not None and training_state.is_file():
        resume_path = training_state
    if args.quarter is not None and args.quarter > 1 and resume_path is None:
        parser.error(f"Part {args.quarter} requires the previous state at {training_state}.")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
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
    model_log_name = "mmr_real" if args.variant == "real" else "mmr_bladesynth"
    writer, _ = create_tensorboard_writer(config, model_log_name, args.mode)
    time_guard = SessionTimeGuard(
        max_session_hours=float(
            args.max_session_hours or training_config.get("max_session_hours", 11.0)
        ),
        packaging_reserve_minutes=float(
            training_config.get("artifact_reserve_minutes", 45)
        ),
        started_at=SESSION_STARTED,
    )
    completed_epochs = start_epoch
    stopped_for_time = False
    for epoch in epoch_window:
        epoch_started = time.monotonic()
        phase = "auxiliary_pretraining" if epoch < auxiliary_epochs else "aebad_s_finetuning"
        active_loader = auxiliary_loader if phase == "auxiliary_pretraining" else train_loader
        if active_loader is None:
            raise RuntimeError("Auxiliary MMR phase was selected without auxiliary data.")
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in active_loader:
            images = batch["image"].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(images, mask_ratio=float(engine["mask_ratio"]))
            loss_value = float(output["loss"].detach())
            epoch_loss += loss_value
            epoch_batches += 1
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/batch_loss", loss_value, global_step)
            scaler.scale(output["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        if (
            (epoch + 1) % int(config["training"].get("checkpoint_every", 1)) == 0
            or epoch + 1 == epoch_window.stop
            or stopped_for_time
        ):
            training_state.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "metadata": {
                        "config": config,
                        "thresholds_calibrated": False,
                        "epoch_seconds": time_guard.epoch_seconds,
                    },
                },
                training_state,
            )
        if writer is not None:
            writer.flush()
        if stopped_for_time:
            print("Stopping before the next epoch to preserve Kaggle packaging time.", flush=True)
            break
    if completed_epochs < total_epochs:
        finish_tensorboard(writer)
        artifact_path = artifact_output_path(
            config["paths"].get("artifacts", "artifacts"),
            model_log_name,
            completed_epochs,
            total_epochs,
            part=args.quarter,
            time_limited=stopped_for_time,
        )
        artifact = create_training_archive(
            PROJECT_ROOT,
            model_log_name,
            artifact_path,
            [
                training_state,
                *([checkpoint_destination] if completed_epochs >= total_epochs else []),
                Path(config["paths"]["reports"]) / "dataset_report.json",
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
        print(
            f"Completed a resumable MMR variant={args.variant} chunk through "
            f"epoch {completed_epochs}/{total_epochs}; rerun the same part or resume from "
            f"{training_state}."
        )
        return 0
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
    calibration_metrics = {
        "anomaly_threshold": anomaly_threshold,
        "pixel_threshold": pixel_threshold,
        "normal_samples": len(validation_scores),
        "score_mean": float(np.mean(validation_scores)),
        "score_std": float(np.std(validation_scores)),
        "score_min": float(np.min(validation_scores)),
        "score_max": float(np.max(validation_scores)),
    }
    log_numeric_metrics(writer, "validation_calibration", calibration_metrics, total_epochs)
    metadata = {
        "version": f"epoch-{total_epochs}",
        "epoch": total_epochs,
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
        "training_phases": {
            "variant": args.variant,
            "auxiliary_epochs": auxiliary_epochs,
            "aebad_s_epochs": main_epochs,
            "aebad_v_sampled_normals": len(video_paths),
            "bladesynth_normals": len(synthetic_normal_paths),
            "synthetic_anomalies_used_as_normal": False,
        },
        "dataset_report": (
            json.loads((Path(config["paths"]["reports"]) / "dataset_report.json").read_text(encoding="utf-8"))
            if (Path(config["paths"]["reports"]) / "dataset_report.json").is_file() else None
        ),
        "config": config,
    }
    model.save_checkpoint(checkpoint_destination, metadata)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": total_epochs - 1,
            "global_step": global_step,
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
        evaluation_metrics = evaluate_engine_predictions(
            labels,
            scores,
            np.asarray(masks),
            np.asarray(maps),
            domains,
            evaluation_root,
            latencies_ms=latencies,
            pixel_threshold=pixel_threshold,
            anomaly_threshold=anomaly_threshold,
        )
        log_numeric_metrics(writer, "test", evaluation_metrics, total_epochs)
    finish_tensorboard(writer)
    if args.mode == "full":
        artifact_path = artifact_output_path(
            config["paths"].get("artifacts", "artifacts"),
            model_log_name,
            completed_epochs,
            total_epochs,
            part=args.quarter,
        )
        artifact = create_training_archive(
            PROJECT_ROOT,
            model_log_name,
            artifact_path,
            [
                training_state,
                checkpoint_destination,
                *artifact_report_sources,
                Path(config["paths"]["reports"]) / "dataset_report.json",
                Path(args.config),
            ],
            run_metadata={
                "completed_epochs": completed_epochs,
                "total_epochs": total_epochs,
                "part": args.quarter,
                "time_limited": False,
                "resume_state": str(training_state),
            },
        )
        print(f"Downloadable artifact: {artifact['archive']['path']}")
        print(f"SHA-256 file: {artifact['archive']['checksum_file']}")
    print(
        f"Completed {args.mode} MMR variant={args.variant}; thresholds calibrated from "
        "held-out AeBAD-S normal validation data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
