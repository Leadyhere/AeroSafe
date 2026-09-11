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


@pytest.mark.parametrize("model", ["rt_detr_v2", "rt_detr", "deformable_detr", "faster_rcnn", "mmr_real", "mmr_bladesynth"])
def test_explicit_cycle_target_and_resume(tmp_path, monkeypatch, model):
    import scripts.kaggle_session as session

    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text("{}")
    monkeypatch.setattr(session, "model_boundaries", lambda config, model: [23])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(session, "remaining_seconds", lambda start, limit: 8 * 3600)
    calls = []
    downloads = []
    monkeypatch.setattr(session, "bounded_command", lambda *args: calls.append(args))
    monkeypatch.setattr(session, "download_links", downloads.append)
    session.run_until(model, 12, 123)
    command, start, deadline = calls[-1]
    assert "--quarter" not in command
    assert command[command.index("--stop-after-epoch") + 1] == "12"
    assert "--resume" not in command
    assert (start, deadline) == (123, 10.5)
    assert downloads == [model]
    with pytest.raises(FileNotFoundError, match="Restore"):
        session.run_until(model, 23, 123, resume_required=True)
    state = session.state_path(model)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.touch()
    session.run_until(model, 23, 123, resume_required=True)
    command = calls[-1][0]
    assert command[command.index("--resume") + 1] == str(state)
    assert command[command.index("--stop-after-epoch") + 1] == "23"


def test_explicit_cycle_shows_downloads_on_failure(tmp_path, monkeypatch):
    import scripts.kaggle_session as session

    monkeypatch.chdir(tmp_path)
    Path("config.yaml").write_text("{}")
    monkeypatch.setattr(session, "model_boundaries", lambda config, model: [23])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(session, "remaining_seconds", lambda start, limit: 8 * 3600)
    downloads = []
    monkeypatch.setattr(session, "download_links", downloads.append)
    def fail(*args):
        raise RuntimeError("deadline")
    monkeypatch.setattr(session, "bounded_command", fail)
    with pytest.raises(RuntimeError, match="deadline"):
        session.run_until("rt_detr_v2", 12, 0)
    assert downloads == ["rt_detr_v2"]


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
