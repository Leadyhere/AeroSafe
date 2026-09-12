import json

import pytest

from scripts.train_aircraft import initialization_identity


def checkpoint(tmp_path, *, architecture="rt_detr", labels=None):
    labels = ["defect"] if labels is None else labels
    (tmp_path / "aeroinspect_metadata.json").write_text(
        json.dumps({"architecture": architecture, "labels": labels})
    )
    (tmp_path / "model.safetensors").write_bytes(b"trained-weights")
    return tmp_path


def test_initialization_identity_is_stable_and_validates_binary_checkpoint(tmp_path):
    path = checkpoint(tmp_path)
    first = initialization_identity(path, architecture="rt_detr", labels=["defect"])
    second = initialization_identity(path, architecture="rt_detr", labels=["defect"])
    assert first == second
    assert first["architecture"] == "rt_detr"
    assert set(first["weights"]) == {"model.safetensors"}


def test_initialization_identity_rejects_wrong_architecture(tmp_path):
    path = checkpoint(tmp_path, architecture="rt_detr_v2")
    with pytest.raises(ValueError, match="not 'rt_detr'"):
        initialization_identity(path, architecture="rt_detr", labels=["defect"])


def test_initialization_identity_rejects_old_multiclass_checkpoint(tmp_path):
    path = checkpoint(tmp_path, labels=["crack", "dent"])
    with pytest.raises(ValueError, match="old 7-class weights"):
        initialization_identity(path, architecture="rt_detr", labels=["defect"])
