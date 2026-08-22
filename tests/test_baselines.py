import json

import torch
from PIL import Image

from src.baselines import FasterRCNNDataset, PatchCoreBaseline


def test_faster_rcnn_dataset_maps_coco_labels_and_boxes(tmp_path):
    image_path = tmp_path / "surface.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": str(image_path), "width": 20, "height": 10}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 10, "bbox": [2, 3, 5, 4], "area": 20}
                ],
                "categories": [{"id": 10, "name": "crack"}],
            }
        ),
        encoding="utf-8",
    )
    dataset = FasterRCNNDataset(annotation_path)
    image, target = dataset[0]
    assert image.shape == (3, 10, 20)
    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[2.0, 3.0, 7.0, 7.0]]


def test_patchcore_greedy_coreset_is_bounded_and_deterministic():
    embeddings = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    first = PatchCoreBaseline._greedy_coreset(embeddings, 5, 3)
    second = PatchCoreBaseline._greedy_coreset(embeddings, 5, 3)
    assert first.shape == (5, 4)
    assert torch.equal(first, second)


def test_patchcore_requires_fit_before_distance_query():
    baseline = PatchCoreBaseline.__new__(PatchCoreBaseline)
    baseline.memory_bank = None
    baseline.device = torch.device("cpu")
    try:
        baseline._nearest_distances(torch.zeros(2, 4))
    except RuntimeError as error:
        assert "Fit the baseline first" in str(error)
    else:
        raise AssertionError("Expected an unfitted PatchCore baseline to fail closed")
