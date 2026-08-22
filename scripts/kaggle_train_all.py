"""Run four primary AeroInspect models plus the separate BladeSynth MMR experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n[AeroInspect] Running: {printable}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def training_commands(mode: str, config: str) -> list[list[str]]:
    python = sys.executable
    return [
        [python, "scripts/train_aircraft.py", "--mode", mode, "--config", config],
        [python, "scripts/train_faster_rcnn.py", "--mode", mode, "--config", config],
        [
            python,
            "scripts/train_engine.py",
            "--mode",
            mode,
            "--variant",
            "real",
            "--config",
            config,
        ],
        [python, "scripts/train_patchcore.py", "--mode", mode, "--config", config],
        [
            python,
            "scripts/train_engine.py",
            "--mode",
            mode,
            "--variant",
            "bladesynth",
            "--config",
            config,
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "smoke", "full", "evaluate", "all"),
        default="all",
        help=(
            "all = prepare, smoke-test four primary models plus the BladeSynth MMR experiment, "
            "full training, then final evaluation"
        ),
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    commands: list[list[str]] = []
    if args.stage in {"prepare", "all"}:
        commands.append([sys.executable, "scripts/prepare_data.py", "--config", args.config])
    if args.stage in {"smoke", "all"}:
        commands.extend(training_commands("smoke", args.config))
    if args.stage in {"full", "all"}:
        commands.extend(training_commands("full", args.config))
    if args.stage in {"evaluate", "all"}:
        commands.append(
            [sys.executable, "scripts/evaluate.py", "--target", "all", "--config", args.config]
        )
    for command in commands:
        run(command)
    print("\n[AeroInspect] Requested workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
