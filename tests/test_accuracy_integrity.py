import csv
import json

import pytest
import torch

from scripts.add_training_slices import clipped_boxes
from scripts.calibrate_aircraft_threshold import operating_point
from scripts.select_aircraft_model import select_candidates
from src.baselines import PatchCoreBaseline
from src.data import AircraftImageRecord, load_group_manifest
from src.experiment_identity import aircraft_data_identity, validate_resume_identity
from src.preprocessing import sha256_file


def test_model_selection_rejects_test_metrics_and_mixed_splits():
    item = {"split": "test", "dataset_identity": "a", "metrics": {"map_50_95": .9, "recall": .9}}
    with pytest.raises(ValueError, match="validation-only"):
        select_candidates({"model": item})
    first = {**item, "split": "validation"}
    second = {**first, "dataset_identity": "b"}
    with pytest.raises(ValueError, match="different datasets"):
        select_candidates({"a": first, "b": second})
    assert select_candidates({"a": first})["recommended"] == "a"


def test_legacy_or_changed_recipe_resume_fails():
    with pytest.raises(ValueError, match="legacy"):
        validate_resume_identity({"metadata": {"config": {}}}, {"revision": 2})
    validate_resume_identity({"metadata": {"experiment_identity": {"revision": 2}}}, {"revision": 2})


def test_data_identity_is_portable_and_detects_box_changes(tmp_path):
    payload = {"categories": [{"id": 1, "name": "defect"}],
               "images": [{"id": 1, "file_name": "/old/a.png", "sha256": "same", "width": 10, "height": 10}],
               "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3]}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(payload))
    before = aircraft_data_identity([path])
    payload["images"][0]["file_name"] = "/kaggle/new/a.png"
    path.write_text(json.dumps(payload))
    assert aircraft_data_identity([path]) == before
    payload["annotations"][0]["bbox"][2] = 4
    path.write_text(json.dumps(payload))
    assert aircraft_data_identity([path]) != before


def test_manifest_hash_applies_to_all_copies_after_relocation(tmp_path):
    records = {}
    for name in ["a.png", "copy.png"]:
        path = tmp_path / name
        path.write_bytes(b"identical")
        records[str(path)] = AircraftImageRecord(str(path), 10, 10, "source")
    manifest = tmp_path / "groups.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "sha256", "aircraft_id", "split"])
        writer.writeheader()
        writer.writerow({"image": "old_name.png", "sha256": sha256_file(tmp_path / "a.png"),
                         "aircraft_id": "plane1", "split": "train"})
    assert load_group_manifest(manifest, records)["matched"] == 2
    assert all(r.fixed_split == "train" and "aircraft_id:plane1" in r.leakage_keys for r in records.values())


def test_patchcore_candidates_cover_early_and_late_images_equally():
    class Loader:
        dataset = range(4)

        def __iter__(self):
            for i in range(4):
                yield {"image": torch.full((1, 1, 3, 3), float(i))}

        def __len__(self):
            return 4

    model = PatchCoreBaseline.__new__(PatchCoreBaseline)
    model.max_patches, model.coreset_fraction, model.projection_dim = 8, 1.0, 1
    model.metadata = {}
    model._embeddings = lambda images: images
    model.fit(Loader())
    assert torch.bincount(model.memory_bank[:, 0].long()).tolist() == [2, 2, 2, 2]
    assert model.metadata["patch_sampling"]["all_training_images_visited"]


def test_slices_keep_visible_fragments_and_exclude_outside_boxes():
    annotations = [{"bbox": [8, 8, 5, 5]}, {"bbox": [20, 20, 2, 2]}]
    result = clipped_boxes(annotations, 0, 0, 10, 10)
    assert len(result) == 1
    assert result[0]["bbox"] == [8, 8, 2, 2]
    assert result[0]["area"] == 4


def test_unachievable_recall_does_not_invent_threshold():
    result = operating_point([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}], [], 1, .9, .1)
    assert not result["feasible"]
    assert result["operating_point"] is None
