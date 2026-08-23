import json

from src import data


def test_aircraft_scope_does_not_touch_engine_archives(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "audit_imdd_aircraft_subset", lambda *_: {"images": 4})
    monkeypatch.setattr(data, "prepare_aircraft_data", lambda _: {"train": 3})

    def engine_should_not_run(*_args, **_kwargs):
        raise AssertionError("aircraft preparation must not inspect engine data")

    monkeypatch.setattr(data, "split_aebad_training_paths", engine_should_not_run)
    report_path = tmp_path / "report.json"
    report = data.prepare_datasets(
        {"paths": {"imdd_aircraft_images": "images", "imdd_aircraft_csv": "labels.csv"}},
        report_path,
        scope="aircraft",
    )

    assert report == {
        "scope": "aircraft",
        "aircraft": {"train": 3},
        "aircraft_image_level_auxiliary": {"source": "IMDD aircraft subset", "images": 4},
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
