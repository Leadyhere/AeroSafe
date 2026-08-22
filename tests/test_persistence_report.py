from __future__ import annotations

from pathlib import Path

from PIL import Image

pytest_plugins = []


def test_database_round_trip_and_zero_dashboard(tmp_path: Path) -> None:
    from src.database import dashboard_statistics, get_inspection, init_database, save_inspection

    sessions = init_database(tmp_path / "history.sqlite3")
    assert dashboard_statistics(sessions)["total_inspections"] == 0
    result = {
        "inspection_type": "aircraft",
        "status": "defect_detected",
        "defects": [{"class": "crack", "confidence": 0.9, "bbox": [1, 2, 30, 40]}],
        "model_name": "deformable_detr",
        "model_version": "test",
        "inference_time_ms": 4.2,
    }
    inspection_id = save_inspection(
        sessions, image_name="../unsafe.jpg", saved_image_path=None, result=result
    )
    saved = get_inspection(sessions, inspection_id)
    assert saved["image_name"] == "unsafe.jpg"
    assert saved["detections"][0]["class"] == "crack"
    assert dashboard_statistics(sessions)["aircraft_findings"] == 1


def test_html_report_escapes_user_content() -> None:
    from src.report import generate_html_report

    result = {
        "inspection_type": "engine",
        "status": "anomalous",
        "anomaly_score": 0.8,
        "affected_visible_area": 0.1,
        "quality": {"warnings": ["<script>alert(1)</script>"]},
        "model_name": "mmr",
        "model_version": "test",
        "inference_time_ms": 2.0,
    }
    report = generate_html_report(
        inspection_id=1, image_name='<img src=x onerror="bad">', result=result,
        visual=Image.new("RGB", (16, 16), "black")
    )
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;" in report
    assert "data:image/jpeg;base64," in report
