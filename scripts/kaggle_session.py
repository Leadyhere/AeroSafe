"""Small notebook helpers: data checks, bounded runs, resume, and download links."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kaggle_train_all import MODEL_ORDER, training_command
from src.training_artifacts import sha256_file

FRIENDS = {
    1: "rt_detr_v2", 2: "rt_detr", 3: "deformable_detr", 4: "faster_rcnn",
    5: "mmr_real", 6: "mmr_bladesynth", 7: "patchcore",
}
AIRCRAFT = {"rt_detr_v2", "rt_detr", "deformable_detr", "faster_rcnn"}
DATA = {
    "asdd": ("Dataset2.v2-defect-classification.coco", "aerosafe-asdd"),
    "aircraftsurface": ("aircraftsurface1.v1i.coco", "aerosafe-aircraftsurface"),
    "agdd": ("AGDD-main", "aerosafe-agdd"),
    "imdd_aircraft_images": ("0to4_aircraft_skin4000pics", "aerosafe-imdd"),
    "imdd_aircraft_csv": ("0-4aircraft4000.csv", None),
    "aebad": ("AeBAD_S", None),
    "bladesynth": ("Dataset_bladesynth", "aerosafe-bladesynth"),
}


def configure_data(model: str, input_root: str = "/kaggle/input") -> None:
    """Resolve unambiguous dataset roots; full content checks run in prepare()."""
    if model not in MODEL_ORDER:
        raise ValueError(f"Unknown model: {model}")
    required = (
        ["asdd", "aircraftsurface", "agdd", "imdd_aircraft_images", "imdd_aircraft_csv"]
        if model in AIRCRAFT else ["aebad"]
    )
    if model == "mmr_bladesynth":
        required.append("bladesynth")
    # Walk directory names only; avoid entering known large image datasets.
    candidates: dict[str, list[Path]] = {key: [] for key in required}
    names = {DATA[key][0].lower(): key for key in required}
    root = Path(input_root)
    for current, dirs, files in os.walk(root):
        if len(Path(current).relative_to(root).parts) > 8:
            dirs[:] = []
            continue
        for name in [*dirs, *files]:
            if name.lower() in names:
                candidates[names[name.lower()]].append(Path(current) / name)
                if name in dirs:
                    dirs.remove(name)
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    for key in required:
        matches = candidates[key]
        if not matches and DATA[key][1]:
            matches = [p for p in root.rglob(DATA[key][1]) if p.is_dir()]
        if len(matches) != 1:
            raise ValueError(
                f"{key}: expected one {DATA[key][0]}, found {len(matches)}. "
                "Attach the extracted dataset once; keep its original folder name."
            )
        selected = matches[0].parent if key == "aebad" else matches[0]
        config["paths"][key] = selected.as_posix()
        print(f"FOUND {key}: {selected}", flush=True)
    # Relative output paths keep archives portable between notebooks.
    for key, value in {"processed": "data/processed", "reports": "reports",
                       "checkpoints": "checkpoints", "artifacts": "artifacts"}.items():
        config["paths"][key] = value
    Path("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if model in AIRCRAFT:
        manifests = list(root.rglob("aerosafe_group_manifest.csv"))
        if len(manifests) > 1:
            raise ValueError("Attach exactly one reviewed grouping-manifest dataset version.")
        if manifests:
            config["paths"]["group_manifest"] = manifests[0].as_posix()
            Path("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"FOUND grouping manifest: {manifests[0]}")
    print("Dataset paths configured. Run prepare() to validate their contents.")


def remaining_seconds(started_at: float, limit_hours: float) -> float:
    return max(0.0, limit_hours * 3600 - (time.time() - started_at))


def bounded_command(command: list[str], started_at: float, limit_hours: float) -> None:
    """On Kaggle/Linux terminate the whole process group at the deadline."""
    if os.name != "posix":
        raise RuntimeError("This deadline wrapper is intended for Kaggle/Linux.")
    remaining = remaining_seconds(started_at, limit_hours)
    if remaining < 60:
        raise RuntimeError("Session budget exhausted. Save outputs and use a fresh session.")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "Run stopped at the notebook deadline or by interruption. "
            "Download the last completed checkpoint; the unfinished epoch must be repeated."
        ) from None
    if code:
        raise subprocess.CalledProcessError(code, command)


def prepare(model: str, started_at: float) -> None:
    """Check only the data used by the selected model, within the session clock."""
    bounded_command(
        [sys.executable, "scripts/kaggle_session.py", "validate", model], started_at, 9.5
    )


def validate(model: str) -> None:
    from src import load_config
    from src.data import (
        AeBADDataset,
        audit_bladesynth,
        prepare_datasets,
        sample_aebad_v_training_paths,
        split_aebad_training_paths,
    )

    config = load_config("config.yaml")
    if model in AIRCRAFT:
        prepare_datasets(config, scope="aircraft")
    else:
        train, val = split_aebad_training_paths(
            config["paths"]["aebad"], config["engine"]["validation_fraction"],
            config["dataset"].get("split_seed", config["training"]["seed"]),
        )
        test = AeBADDataset(config["paths"]["aebad"], "test")
        missing = [p for p in test.paths if not ({"normal", "good"} & set(p.parts))
                   and test._mask_path(p) is None]
        if missing:
            raise ValueError(f"Missing anomaly masks: {missing[:5]}")
        print(f"AeBAD: {len(train)} training normals, {len(val)} validation normals, "
              f"{len(test)} test images")
        if model == "mmr_bladesynth":
            sample_aebad_v_training_paths(config["paths"]["aebad"],
                                         config["engine"]["aebad_v_frame_stride"])
            audit_bladesynth(config["paths"]["bladesynth"],
                            config["engine"]["bladesynth_expected_images_per_class"])
    print("PASS: selected model's dataset checks completed.", flush=True)


def download_links(model: str) -> None:
    """Display completed archive links, or raw state links after an interrupted run."""
    from IPython.display import FileLink, display

    archives = sorted(Path("artifacts").glob(f"{model}_*.tar.gz"))
    complete = [p for p in archives if p.with_name(p.name + ".sha256").is_file()]
    for path in complete:
        print(f"Download archive ({path.stat().st_size / 1024**2:.1f} MiB):")
        display(FileLink(str(path)))
        display(FileLink(str(path) + ".sha256"))
    state = state_path(model)
    if state.is_file():
        print("Emergency/resume checkpoint: latest completed saved epoch")
        display(FileLink(str(state)))
        display(FileLink("config.yaml"))
    if not complete and not state.is_file():
        print("No completed checkpoint yet. Check the error above; do not start the next part.")


def state_path(model: str) -> Path:
    if model in AIRCRAFT - {"faster_rcnn"}:
        return Path(f"checkpoints/aircraft_candidates/{model}_training_state.pt")
    return Path("checkpoints") / {
        "faster_rcnn": "faster_rcnn_training_state.pt",
        "mmr_real": "engine_training_state.pt",
        "mmr_bladesynth": "engine_bladesynth_training_state.pt",
        "patchcore": "patchcore_best.pt",
    }[model]


def run_part(model: str, part: int, started_at: float) -> None:
    """One part per session; reserve time for the artifact and download cells."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before training.")
    budget = min(9.0, remaining_seconds(started_at, 9.5) / 3600)
    if budget < 1.5:
        raise RuntimeError("Less than 1.5 hours left in the training budget. Start a fresh session.")
    command = training_command(model, "config.yaml", mode="full", part=part,
                               max_session_hours=budget)
    try:
        bounded_command(command, started_at, 10.5)
    finally:
        download_links(model)


def restore(archive_path: str, model: str) -> None:
    """Restore only an explicitly selected, trusted archive from your own run."""
    path = Path(archive_path)
    checksum = path.with_name(path.name + ".sha256")
    if not checksum.is_file() or sha256_file(path) != checksum.read_text().split()[0]:
        raise ValueError("Archive checksum missing or incorrect. Attach the .sha256 file too.")
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile("artifact_manifest.json"))
        if manifest.get("model") != model:
            raise ValueError("This archive belongs to a different model.")
        # Check all members before writing; never restore config or source code.
        members = []
        root = Path.cwd().resolve()
        for member in archive.getmembers():
            destination = (root / member.name).resolve()
            if not destination.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if Path(member.name).parts[0] in {"checkpoints", "reports", "data"}:
                members.append(member)
        archive.extractall(root, members=members, filter="data")
    print("Restored. Re-run dataset configuration and preparation for current Kaggle paths.")


if __name__ == "__main__":
    # Script execution needs the repository root for package imports.
    validate(sys.argv[2])
