import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.kaggle_session import configure_data, remaining_seconds
from src.training_artifacts import atomic_torch_save


def test_failed_checkpoint_save_preserves_completed_epoch(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    atomic_torch_save({"epoch": 4}, path)

    def interrupted(payload, destination):
        Path(destination).write_bytes(b"incomplete")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError):
        atomic_torch_save({"epoch": 5}, path)
    assert torch.load(path, weights_only=True)["epoch"] == 4


def test_clock_accounts_for_setup_time(monkeypatch):
    monkeypatch.setattr("scripts.kaggle_session.time.time", lambda: 7200)
    assert remaining_seconds(0, 10.5) == 8.5 * 3600
    assert remaining_seconds(0, 1) == 0


def test_ambiguous_dataset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text("paths: {}")
    (tmp_path / "a" / "AeBAD_S").mkdir(parents=True)
    (tmp_path / "b" / "AeBAD_S").mkdir(parents=True)
    with pytest.raises(ValueError, match="found 2"):
        configure_data("mmr_real", str(tmp_path))


def test_agdd_nested_inside_imdd_is_discovered(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text("paths: {}")
    inputs = tmp_path / "input"
    imdd = inputs / "0to4_aircraft_skin4000pics"
    nested = imdd / "AGDD-main" / "AGDD-main"
    nested.mkdir(parents=True)
    (inputs / "Dataset2.v2-defect-classification.coco").mkdir()
    (inputs / "aircraftsurface1.v1i.coco").mkdir()
    (inputs / "0-4aircraft4000.csv").write_text("fixture")
    configure_data("deformable_detr", str(inputs))
    config = yaml.safe_load(Path("config.yaml").read_text())
    assert Path(config["paths"]["agdd"]) == nested.parent
    assert Path(config["paths"]["imdd_aircraft_images"]) == imdd


def test_deadline_terminates_worker_group(monkeypatch):
    import scripts.kaggle_session as session

    waits = []
    signals = []

    def wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("train", timeout)
        return -15

    monkeypatch.setattr(session, "os", SimpleNamespace(
        name="posix", killpg=lambda pid, sig: signals.append((pid, sig))
    ))
    monkeypatch.setattr(session, "remaining_seconds", lambda start, limit: 120)
    monkeypatch.setattr(session.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(
        pid=123, wait=wait
    ))
    with pytest.raises(RuntimeError, match="deadline"):
        session.bounded_command(["train"], 0, 10.5)
    assert waits == [120, 15]
    assert signals == [(123, session.signal.SIGTERM)]
