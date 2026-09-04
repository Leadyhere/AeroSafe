from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate import metric_rank_value
from src.evaluation import (
    bbox_iou_xywh,
    compute_aupro,
    detection_calibration_error,
    detection_prf,
    safe_auroc,
    safe_average_precision,
    select_detection_threshold,
)


def test_metric_rank_value_preserves_real_zero():
    assert metric_rank_value({"score": 0.0}, "score") == 0.0
    assert metric_rank_value({"score": None}, "score") == -1.0


def test_bbox_iou_and_detection_prf() -> None:
    assert bbox_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    ground = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]
    predicted = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
    metrics, matrix, ids = detection_prf(ground, predicted)
    assert metrics["f1"] == 1.0
    assert matrix[0, 0] == 1
    assert ids == [1]


def test_undefined_auroc_is_none_and_nonfinite_rejected() -> None:
    assert safe_auroc([0, 0], [0.1, 0.2]) is None
    assert safe_average_precision([0, 0], [0.1, 0.2]) is None
    assert safe_average_precision([0, 1], [0.1, 0.9]) == 1.0
    with pytest.raises(ValueError, match="NaN"):
        safe_auroc([0, 1], [0.1, np.nan])


def test_aupro_shape_validation() -> None:
    with pytest.raises(ValueError, match="identical"):
        compute_aupro(np.zeros((1, 4, 4)), np.zeros((1, 3, 3)))


def test_detection_threshold_selection_and_calibration() -> None:
    ground = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 5, 5], "score": 0.1},
    ]
    selected = select_detection_threshold(ground, predictions)
    assert selected["f1"] == 1.0
    assert 0.1 < selected["threshold"] <= 0.8
    error = detection_calibration_error(ground, predictions, bins=5)
    assert error is not None and 0 <= error <= 1
