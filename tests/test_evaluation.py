from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import bbox_iou_xywh, compute_aupro, detection_prf, safe_auroc


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
    with pytest.raises(ValueError, match="NaN"):
        safe_auroc([0, 1], [0.1, np.nan])


def test_aupro_shape_validation() -> None:
    with pytest.raises(ValueError, match="identical"):
        compute_aupro(np.zeros((1, 4, 4)), np.zeros((1, 3, 3)))
