from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.inference import (
    ModelNotReadyError,
    class_aware_nms,
    inspect_aircraft,
    inspect_engine,
    predict_aircraft_tiled,
)
from src.preprocessing import ImageValidationError


class MockAircraft:
    model_name = "deformable_detr"
    version = "test"
    last_inference_time_ms = 3.5

    def __init__(self, detections):
        self.detections = detections

    def predict(self, image, threshold=None):
        return self.detections


class MockEngine:
    model_name = "mmr"
    version = "test"
    anomaly_threshold = 0.5
    pixel_threshold = 0.4

    def __init__(self, score):
        self.score = score

    def predict(self, image):
        anomaly_map = np.zeros((32, 32), dtype=np.float32)
        anomaly_map[8:16, 8:16] = 0.8
        return {"anomaly_score": self.score, "anomaly_map": anomaly_map, "inference_time_ms": 2.0}


def test_aircraft_no_defect_response() -> None:
    result = inspect_aircraft(Image.new("RGB", (160, 160), "gray"), MockAircraft([]))
    assert result["status"] == "no_defect_detected"
    assert result["defect_count"] == 0
    assert result["model_name"] == "deformable_detr"
    assert result["annotated_image"].size == (160, 160)


def test_aircraft_detection_response() -> None:
    detection = {"class": "crack", "confidence": 0.91, "bbox": [10, 11, 90, 92]}
    result = inspect_aircraft(Image.new("RGB", (160, 160), "gray"), MockAircraft([detection]))
    assert result["status"] == "defect_detected"
    assert result["defects"][0]["class"] == "crack"


def test_engine_anomaly_response_and_area() -> None:
    result = inspect_engine(Image.new("RGB", (160, 160), "gray"), MockEngine(0.8))
    assert result["status"] == "anomalous"
    assert result["affected_visible_area"] == pytest.approx(64 / 1024)
    assert result["heatmap"].size == (160, 160)


def test_engine_normal_response() -> None:
    assert inspect_engine(Image.new("RGB", (160, 160), "gray"), MockEngine(0.1))["status"] == "normal"


def test_invalid_and_unusable_images() -> None:
    with pytest.raises(ImageValidationError):
        inspect_aircraft(b"broken", MockAircraft([]))
    with pytest.raises(ImageValidationError, match="too small"):
        inspect_engine(Image.new("RGB", (8, 8), "gray"), MockEngine(0.1))


def test_missing_threshold_fails_closed() -> None:
    model = MockEngine(0.1)
    model.anomaly_threshold = None
    with pytest.raises(ModelNotReadyError, match="validation-derived"):
        inspect_engine(Image.new("RGB", (160, 160), "gray"), model)


def test_tiled_prediction_offsets_boxes_and_merges_overlap() -> None:
    class TileModel:
        last_inference_time_ms = 1.0

        def predict(self, image, threshold=None):
            self.last_inference_time_ms = 1.0
            return [{"class": "crack", "confidence": 0.8, "bbox": [1, 2, 10, 12]}]

    detections, elapsed = predict_aircraft_tiled(
        TileModel(), Image.new("RGB", (700, 512)), tile_size=512, overlap=128,
        include_full_frame=False,
    )
    assert len(detections) == 2
    assert detections[1]["bbox"][0] == 189.0
    assert elapsed == 2.0
    merged = class_aware_nms(
        [
            {"class": "crack", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"class": "crack", "confidence": 0.8, "bbox": [1, 1, 10, 10]},
            {"class": "dent", "confidence": 0.7, "bbox": [1, 1, 10, 10]},
        ],
        0.5,
    )
    assert len(merged) == 2
