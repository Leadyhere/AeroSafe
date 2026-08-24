import tarfile
from pathlib import Path

from scripts.package_training_artifact import create_verified_archive


def test_verified_faster_rcnn_archive_contains_resumable_state(tmp_path):
    project = tmp_path / "project"
    checkpoints = project / "checkpoints"
    reports = project / "reports"
    checkpoints.mkdir(parents=True)
    reports.mkdir()
    (checkpoints / "faster_rcnn_best.pt").write_bytes(b"best-model")
    (checkpoints / "faster_rcnn_training_state.pt").write_bytes(b"optimizer-state")
    (reports / "aircraft_metrics.json").write_text("{}", encoding="utf-8")
    (project / "config.kaggle.yaml").write_text("paths: {}\n", encoding="utf-8")
    output = tmp_path / "faster.tar.gz"

    manifest = create_verified_archive(project, "faster_rcnn", output)

    assert output.is_file()
    assert Path(manifest["archive"]["checksum_file"]).is_file()
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "checkpoints/faster_rcnn_best.pt" in names
    assert "checkpoints/faster_rcnn_training_state.pt" in names
    assert "reports/aircraft_metrics.json" in names
    assert "config.kaggle.yaml" in names
    assert "artifact_manifest.json" in names


def test_deployment_archive_excludes_training_state(tmp_path):
    project = tmp_path / "project"
    checkpoints = project / "checkpoints" / "aircraft_best"
    reports = project / "reports"
    checkpoints.mkdir(parents=True)
    reports.mkdir()
    (checkpoints / "model.safetensors").write_bytes(b"weights")
    (reports / "aircraft_metrics.json").write_text("{}", encoding="utf-8")
    (project / "config.kaggle.yaml").write_text("paths: {}\n", encoding="utf-8")
    output = tmp_path / "detr.tar.gz"

    create_verified_archive(
        project,
        "deformable_detr",
        output,
        include_training_state=False,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "checkpoints/aircraft_best/model.safetensors" in names
    assert "checkpoints/aircraft_training_state.pt" not in names
