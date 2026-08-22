from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import (
    AircraftAnnotation,
    AircraftImageRecord,
    DatasetConfigurationError,
    _grouped_split,
    load_aircraft_source,
    normalize_label,
    records_to_coco,
    split_aebad_training_paths,
)
from src.preprocessing import (
    ImageValidationError,
    analyze_duplicates,
    check_image_quality,
    load_image,
    perceptual_hashes,
    sha256_file,
)

LABELS = {
    "crack": ["crack"],
    "paint_damage": ["paint-off", "paint damage"],
    "missing_fastener": ["missing-head", "missing rivet"],
}


def save_image(path: Path, color=(80, 120, 160), size=(128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_invalid_dataset_path_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetConfigurationError, match="dataset directory is missing"):
        load_aircraft_source(tmp_path / "absent", "ASDD", LABELS)


def test_image_loading_and_quality(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    save_image(path)
    assert load_image(path).mode == "RGB"
    quality = check_image_quality(path)
    assert quality["usable"] is True
    assert quality["width"] == 128
    with pytest.raises(ImageValidationError):
        load_image(b"not an image")


def test_label_normalization() -> None:
    assert normalize_label(" Paint-Off ", LABELS) == "paint_damage"
    assert normalize_label("missing_rivet", LABELS) == "missing_fastener"
    assert normalize_label("invented", LABELS) is None


def test_hashing_and_duplicate_recognition(tmp_path: Path) -> None:
    original = tmp_path / "blade.png"
    exact = tmp_path / "blade_copy.png"
    resized = tmp_path / "blade_resized.png"
    save_image(original)
    exact.write_bytes(original.read_bytes())
    with Image.open(original) as image:
        image.resize((160, 160)).save(resized)
    assert sha256_file(original) == sha256_file(exact)
    assert perceptual_hashes(original)[0] == perceptual_hashes(resized)[0]
    result = analyze_duplicates([original, exact, resized], near_hamming=2)
    assert len(result.exact_pairs) == 1
    assert result.near_pairs
    assert len(result.groups) == 1


def test_coco_conversion_preserves_traceability(tmp_path: Path) -> None:
    image_path = tmp_path / "aircraft.png"
    save_image(image_path, size=(200, 100))
    record = AircraftImageRecord(
        str(image_path),
        200,
        100,
        "ASDD",
        [AircraftAnnotation([10, 20, 30, 40], "crack", "Crack", "ASDD")],
    )
    destination = tmp_path / "train.json"
    payload = records_to_coco([record], ["crack"], destination)
    assert payload["annotations"][0]["area"] == 1200
    assert payload["annotations"][0]["source"] == "ASDD"
    assert payload["annotations"][0]["original_class_name"] == "Crack"
    assert json.loads(destination.read_text())["categories"][0]["name"] == "crack"


def test_load_pascal_voc_and_clip_box(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "panel.jpg"
    save_image(image_path, size=(100, 80))
    annotation = tmp_path / "annotations" / "panel.xml"
    annotation.parent.mkdir()
    annotation.write_text(
        """<annotation><filename>panel.jpg</filename><object><name>paint-off</name>"
        "<bndbox><xmin>-5</xmin><ymin>10</ymin><xmax>120</xmax><ymax>70</ymax>"
        "</bndbox></object></annotation>""",
        encoding="utf-8",
    )
    records = load_aircraft_source(tmp_path, "ASDD", LABELS)
    assert len(records) == 1
    assert records[0].annotations[0].normalized_class == "paint_damage"
    assert records[0].annotations[0].bbox == [0.0, 10.0, 100.0, 60.0]


def test_numpy_nan_image_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="NaN"):
        load_image(np.full((10, 10), np.nan))


def test_duplicate_group_never_crosses_splits() -> None:
    paths = ["a", "a_aug", "b", "c", "d", "e"]
    splits = _grouped_split(
        paths, [["a", "a_aug"], ["b"], ["c"], ["d"], ["e"]],
        {"train": 0.5, "validation": 0.25, "test": 0.25}, seed=42
    )
    memberships = {path: split for split, items in splits.items() for path in items}
    assert memberships["a"] == memberships["a_aug"]
    assert set(memberships) == set(paths)


def test_official_aebad_parent_or_direct_root_is_accepted(tmp_path: Path) -> None:
    direct = tmp_path / "AeBAD_S"
    for relative in ["train/good/same", "train/good/view", "test/good/same", "ground_truth"]:
        (direct / relative).mkdir(parents=True, exist_ok=True)
    save_image(direct / "train/good/same/one.png")
    save_image(direct / "train/good/view/two.png")
    train, validation = split_aebad_training_paths(tmp_path, 0.5, seed=7)
    assert len(train) == len(validation) == 1
    train_direct, validation_direct = split_aebad_training_paths(direct, 0.5, seed=7)
    assert train == train_direct
    assert validation == validation_direct
