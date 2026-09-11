"""Plan and run Kaggle-safe AeroInspect training sessions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.training_chunks import training_boundaries

MODEL_ORDER = (
    "rt_detr_v2",
    "rt_detr",
    "deformable_detr",
    "mmr_real",
    "mmr_bladesynth",
    "faster_rcnn",
    "patchcore",
)

MODEL_CATALOG = {
    "rt_detr_v2": {
        "family": "Transformer",
        "use": "Primary aircraft defect localizer",
        "dataset": "AGDD boxes, then binary ASDD + aircraftsurface1",
        "t4_time": "8-16 h total; usually 1-2 sessions",
    },
    "rt_detr": {
        "family": "Transformer",
        "use": "Second real-time aircraft transformer",
        "dataset": "AGDD boxes, then binary ASDD + aircraftsurface1",
        "t4_time": "8-16 h total; usually 1-2 sessions",
    },
    "deformable_detr": {
        "family": "Transformer",
        "use": "Multi-scale aircraft transformer comparison",
        "dataset": "AGDD boxes, then binary ASDD + aircraftsurface1",
        "t4_time": "10-20 h total; usually 1-2 sessions",
    },
    "mmr_real": {
        "family": "Transformer hybrid",
        "use": "Primary engine anomaly detector and heatmap model",
        "dataset": "AeBAD-S normal train; held-out normals calibrate thresholds",
        "t4_time": "6-10 h total; about 1.5-2.5 h per quarter",
    },
    "mmr_bladesynth": {
        "family": "Transformer-hybrid experiment",
        "use": "Tests whether extra normal-domain pretraining helps MMR",
        "dataset": "AeBAD-V normals + BladeSynth Normal, then AeBAD-S normals",
        "t4_time": "8-14 h total; about 2-4 h per quarter",
    },
    "faster_rcnn": {
        "family": "CNN baseline",
        "use": "Checks whether transformers really improve aircraft detection",
        "dataset": "AGDD boxes, then binary ASDD + aircraftsurface1",
        "t4_time": "8-16 h total; usually 1-2 sessions",
    },
    "patchcore": {
        "family": "CNN baseline",
        "use": "Training-free-style engine anomaly comparison",
        "dataset": "AeBAD-S normal train only",
        "t4_time": "0.5-1.5 h; one session",
    },
}


def run(command: list[str]) -> None:
    print(f"\n[AeroInspect] Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def printable_command(command: list[str]) -> str:
    return " ".join(["python", *command[1:]])


def model_boundaries(config: dict[str, Any], model: str) -> list[int]:
    training = config["training"]
    common = {
        "quarter_count": int(training.get("quarter_count", 4)),
        "one_go_max_epochs": int(training.get("one_go_max_epochs", 1)),
    }
    if model in {"rt_detr_v2", "rt_detr", "deformable_detr"}:
        candidate = config["aircraft"]["transformer_candidates"][model]
        return training_boundaries(
            int(candidate.get("epochs", config["aircraft"]["epochs"])),
            auxiliary_epochs=int(config["aircraft"]["auxiliary_pretrain_epochs"]),
            **common,
        )
    if model == "faster_rcnn":
        return training_boundaries(
            int(config["baselines"]["faster_rcnn"]["epochs"]),
            auxiliary_epochs=int(config["aircraft"]["auxiliary_pretrain_epochs"]),
            **common,
        )
    if model in {"mmr_real", "mmr_bladesynth"}:
        auxiliary = int(config["engine"]["auxiliary_pretrain_epochs"]) if model == "mmr_bladesynth" else 0
        return training_boundaries(
            int(config["engine"]["epochs"]),
            auxiliary_epochs=auxiliary,
            **common,
        )
    if model == "patchcore":
        return [1]
    raise ValueError(f"Unknown model: {model}")


def training_command(
    model: str,
    config_path: str,
    *,
    mode: str,
    part: int | None = None,
    max_session_hours: float | None = None,
) -> list[str]:
    command = [sys.executable]
    if model in {"rt_detr_v2", "rt_detr", "deformable_detr"}:
        command += [
            "scripts/train_aircraft.py",
            "--mode",
            mode,
            "--transformer",
            model,
            "--config",
            config_path,
        ]
    elif model == "faster_rcnn":
        command += ["scripts/train_faster_rcnn.py", "--mode", mode, "--config", config_path]
    elif model in {"mmr_real", "mmr_bladesynth"}:
        variant = "real" if model == "mmr_real" else "bladesynth"
        command += [
            "scripts/train_engine.py",
            "--mode",
            mode,
            "--variant",
            variant,
            "--config",
            config_path,
        ]
    elif model == "patchcore":
        command += ["scripts/train_patchcore.py", "--mode", mode, "--config", config_path]
    else:
        raise ValueError(f"Unknown model: {model}")
    if mode == "full" and model != "patchcore":
        command += ["--quarter", str(part or 1)]
        if max_session_hours is not None:
            command += ["--max-session-hours", str(max_session_hours)]
    return command


def print_plan(config: dict[str, Any], config_path: str) -> None:
    print("AeroInspect Kaggle plan: 7 runs (5 transformer-based, 2 baselines)\n")
    for model in MODEL_ORDER:
        details = MODEL_CATALOG[model]
        boundaries = model_boundaries(config, model)
        print(f"{model}: {details['family']} | {details['use']}")
        print(f"  data: {details['dataset']}")
        print(f"  estimated Kaggle T4 time: {details['t4_time']}")
        if model == "patchcore":
            print(
                "  command: "
                + printable_command(training_command(model, config_path, mode="full", part=1))
            )
        else:
            print(f"  completed-epoch boundaries: {boundaries}")
            for part in range(1, len(boundaries) + 1):
                print(
                    f"  part {part}: "
                    + printable_command(
                        training_command(model, config_path, mode="full", part=part)
                    )
                )
        print()
    print(
        "Each command writes a verified archive and .sha256 file under artifacts/. "
        "If the time guard stops early, rerun the same command; it resumes automatically."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("plan", "prepare", "smoke", "train", "evaluate"),
        default="plan",
    )
    parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    parser.add_argument("--quarter", type=int, default=None)
    parser.add_argument("--max-session-hours", type=float, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage == "plan":
        print_plan(config, args.config)
        return 0
    if args.stage == "prepare":
        run([sys.executable, "scripts/prepare_data.py", "--config", args.config])
        return 0
    if args.stage == "smoke":
        selected = [args.model] if args.model else list(MODEL_ORDER)
        for model in selected:
            run(training_command(model, args.config, mode="smoke"))
        return 0
    if args.stage == "train":
        if args.model is None:
            parser.error("--stage train requires --model so one Kaggle session trains one model.")
        boundaries = model_boundaries(config, args.model)
        part = args.quarter or 1
        if not 1 <= part <= len(boundaries):
            parser.error(
                f"{args.model} has {len(boundaries)} training part(s); got --quarter {part}."
            )
        run(
            training_command(
                args.model,
                args.config,
                mode="full",
                part=part,
                max_session_hours=args.max_session_hours,
            )
        )
        return 0
    run([sys.executable, "scripts/evaluate.py", "--target", "aircraft-transformers", "--config", args.config])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
