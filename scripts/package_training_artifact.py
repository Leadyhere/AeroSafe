"""Create and verify a portable, resumable model-training archive."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_artifacts import create_training_archive

MODEL_CHOICES = (
    "deformable_detr",
    "rt_detr",
    "rt_detr_v2",
    "faster_rcnn",
    "mmr_real",
    "mmr_bladesynth",
    "patchcore",
)


def artifact_sources(project: Path, model: str, include_training_state: bool) -> list[Path]:
    checkpoint_root = project / "checkpoints"
    candidate_root = checkpoint_root / "aircraft_candidates"
    if model in {"deformable_detr", "rt_detr", "rt_detr_v2"}:
        modern_checkpoint = candidate_root / model
        sources = [modern_checkpoint if modern_checkpoint.exists() else checkpoint_root / "aircraft_best"]
        if include_training_state:
            modern_state = candidate_root / f"{model}_training_state.pt"
            sources.append(
                modern_state if modern_state.exists() else checkpoint_root / "aircraft_training_state.pt"
            )
    elif model == "faster_rcnn":
        sources = [checkpoint_root / "faster_rcnn_best.pt"]
        if include_training_state:
            sources.append(checkpoint_root / "faster_rcnn_training_state.pt")
    elif model == "mmr_real":
        sources = [checkpoint_root / "engine_best.pt"]
        if include_training_state:
            sources.append(checkpoint_root / "engine_training_state.pt")
    elif model == "mmr_bladesynth":
        sources = [checkpoint_root / "engine_bladesynth_best.pt"]
        if include_training_state:
            sources.append(checkpoint_root / "engine_bladesynth_training_state.pt")
    elif model == "patchcore":
        sources = [checkpoint_root / "patchcore_best.pt"]
    else:
        raise ValueError(f"Unsupported model: {model}")
    sources.append(project / "reports")
    config_path = project / "config.yaml"
    sources.append(config_path if config_path.exists() else project / "config.kaggle.yaml")
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required artifact inputs are missing: {missing}")
    return sources


def create_verified_archive(
    project: str | Path,
    model: str,
    output: str | Path,
    *,
    include_training_state: bool = True,
) -> dict[str, Any]:
    project_path = Path(project).resolve()
    manifest = create_training_archive(
        project_path,
        model,
        output,
        artifact_sources(project_path, model, include_training_state),
        run_metadata={"includes_training_state": include_training_state},
    )
    manifest["includes_training_state"] = include_training_state
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(PROJECT_ROOT))
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Exclude optimizer/scheduler state; the result cannot resume training.",
    )
    args = parser.parse_args()
    try:
        manifest = create_verified_archive(
            args.project,
            args.model,
            args.output,
            include_training_state=not args.deployment_only,
        )
    except (FileNotFoundError, RuntimeError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["archive"], indent=2))
    print("PASS: archive contents and SHA-256 checksums were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
