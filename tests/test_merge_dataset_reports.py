import json

from scripts.merge_dataset_reports import merge_reports


def test_merge_reports_combines_aircraft_and_engine_sections(tmp_path):
    aircraft = tmp_path / "aircraft.json"
    engine = tmp_path / "engine.json"
    output = tmp_path / "combined.json"
    aircraft.write_text(json.dumps({"scope": "aircraft", "aircraft": {"images": 10}}))
    engine.write_text(json.dumps({"scope": "engine", "engine": {"images": 20}}))

    report = merge_reports([aircraft, engine], output)

    assert report == {
        "aircraft": {"images": 10},
        "engine": {"images": 20},
        "scope": "all",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report
