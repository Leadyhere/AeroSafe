"""Prepare an isolated 10-epoch full-image plus slice experiment for RT-DETR."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("rt_detr", "rt_detr_v2"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, choices=(512, 640), default=512)
    parser.add_argument("--max-per-image", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")

    source_config = Path(args.config)
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    candidate = config["aircraft"]["transformer_candidates"][args.model]
    baseline = Path(candidate["checkpoint"])
    metadata_path = baseline / "aeroinspect_metadata.json"
    if not metadata_path.is_file():
        parser.error(
            f"Finished baseline checkpoint not found at {baseline}. "
            "Finish or restore its 23-epoch run first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("architecture") != args.model or metadata.get("labels") != ["defect"]:
        parser.error("Baseline architecture/labels are wrong; old 7-class weights are forbidden.")
    import torch

    baseline_state = (
        Path(config["paths"]["checkpoints"])
        / "aircraft_candidates"
        / f"{args.model}_training_state.pt"
    )
    expected_baseline_epochs = int(config["aircraft"]["auxiliary_pretrain_epochs"]) + int(
        candidate.get("epochs", config["aircraft"]["epochs"])
    )
    if not baseline_state.is_file():
        parser.error(f"Baseline training state is missing: {baseline_state}")
    completed_baseline_epochs = int(
        torch.load(baseline_state, map_location="cpu", weights_only=False)["epoch"]
    ) + 1
    if completed_baseline_epochs < expected_baseline_epochs:
        parser.error(
            f"Baseline is only at epoch {completed_baseline_epochs}/{expected_baseline_epochs}. "
            "Finish it before starting the accuracy pilot."
        )

    processed = Path(config["paths"]["processed"])
    original_train = processed / "aircraft/train.json"
    validation = processed / "aircraft/validation.json"
    auxiliary = processed / "aircraft_auxiliary/agdd_train.json"
    for required in (original_train, validation, auxiliary):
        if not required.is_file():
            parser.error(f"Prepared data missing: {required}")

    experiment = f"{args.model}_slice_pilot"
    slices = processed / "aircraft" / experiment
    sliced_train = slices / "train.json"
    if slices.exists() and not sliced_train.is_file():
        parser.error(
            f"Incomplete slice directory exists: {slices}. Remove only that directory, "
            "then rerun this preparation command."
        )
    if not sliced_train.is_file():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts/add_training_slices.py"),
                "--input",
                str(original_train),
                "--output",
                str(slices),
                "--size",
                str(args.size),
                "--max-per-image",
                str(args.max_per_image),
            ],
            check=True,
        )

    # Separate all mutable outputs so this experiment cannot overwrite the baseline.
    config["paths"].update(
        checkpoints=f"checkpoints/{experiment}",
        reports=f"reports/{experiment}",
        artifacts=f"artifacts/{experiment}",
    )
    config["training"]["tensorboard_log_dir"] = None
    config["aircraft"].update(
        training_annotations_override=str(sliced_train),
        epochs=args.epochs,
        auxiliary_pretrain_epochs=0,
        warmup_epochs=1,
        validation_interval=1,
        learning_rate=0.00001,
        backbone_learning_rate=0.000001,
    )
    selected = config["aircraft"]["transformer_candidates"][args.model]
    selected["epochs"] = args.epochs
    selected["checkpoint"] = f"checkpoints/{experiment}/best"

    output_config = Path("data") / f"{experiment}.yaml"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(f"MODEL={args.model}")
    print(f"BASELINE={baseline}")
    print(f"PILOT_CONFIG={output_config}")
    print(f"PILOT_TRAIN={sliced_train}")
    print(f"PILOT_ARTIFACTS=artifacts/{experiment}")
    print("PASS: baseline preserved; validation/test unchanged; pilot optimizer will be fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
