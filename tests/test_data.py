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
    _select_exact_duplicate_removals,
    audit_bladesynth,
    audit_coco_source_annotations,
    audit_imdd_aircraft_subset,
    detection_sampling_weights,
    find_bladesynth_normal_paths,
    load_agdd_source,
    load_aircraft_source,
    load_group_manifest,
    merge_leakage_groups,
    normalize_label,
    records_to_coco,
    sample_aebad_v_training_paths,
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


def test_detection_sampling_weights_upweight_rare_classes_without_overweighting_negatives(
    tmp_path: Path,
) -> None:
    payload = {
        "images": [{"id": index} for index in (1, 2, 3, 4)],
        "annotations": [
            {"image_id": 1, "category_id": 1},
            {"image_id": 2, "category_id": 1},
            {"image_id": 3, "category_id": 2},
        ],
    }
    path = tmp_path / "balanced.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    weights = detection_sampling_weights(path, max_weight=1.25)
    assert weights == [1.0, 1.0, 1.25, 1.0]


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


def test_all_roboflow_coco_split_files_are_loaded(tmp_path: Path) -> None:
    for offset, split in enumerate(("train", "valid", "test"), start=1):
        image_path = tmp_path / split / f"image_{offset}.jpg"
        save_image(image_path, size=(100, 80))
        payload = {
            "images": [{"id": offset, "file_name": image_path.name, "width": 100, "height": 80}],
            "annotations": [
                {"id": offset, "image_id": offset, "category_id": 1, "bbox": [1, 2, 10, 20]}
            ],
            "categories": [{"id": 1, "name": "crack"}],
        }
        (image_path.parent / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    records = load_aircraft_source(tmp_path, "Roboflow", LABELS)
    assert len(records) == 3
    audit = audit_coco_source_annotations(tmp_path, LABELS)
    assert audit["totals"]["images"] == 3
    assert audit["totals"]["missing_image_files"] == 0


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


def test_explicit_acquisition_groups_and_frozen_split_never_leak(tmp_path: Path) -> None:
    records = {}
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = str((tmp_path / name).resolve())
        records[path] = AircraftImageRecord(path, 100, 100, "source")
    manifest = tmp_path / "groups.csv"
    manifest.write_text(
        "image,aircraft_id,inspection_session,camera_id,split\n"
        "a.jpg,tail-1,visit-1,cam-a,test\n"
        "b.jpg,tail-1,visit-2,cam-b,test\n"
        "c.jpg,tail-2,visit-3,cam-c,train\n",
        encoding="utf-8",
    )
    report = load_group_manifest(manifest, records)
    assert report["matched"] == 3
    groups = merge_leakage_groups(list(records), [], records)
    fixed = {path: record.fixed_split for path, record in records.items() if record.fixed_split}
    splits = _grouped_split(
        list(records), groups, {"train": 0.7, "validation": 0.15, "test": 0.15}, 42,
        fixed_splits=fixed,
    )
    assert set(splits["test"]) == {
        str((tmp_path / "a.jpg").resolve()), str((tmp_path / "b.jpg").resolve())
    }
    assert splits["train"] == [str((tmp_path / "c.jpg").resolve())]


def test_exact_duplicate_prefers_specific_annotation_over_composite_label() -> None:
    broad = AircraftImageRecord(
        "asdd.jpg",
        100,
        100,
        "ASDD",
        [AircraftAnnotation([0, 0, 20, 20], "surface_damage", "composite", "ASDD")],
    )
    specific = AircraftImageRecord(
        "aircraftsurface.jpg",
        100,
        100,
        "aircraftsurface1",
        [AircraftAnnotation([0, 0, 20, 20], "dent", "Dent", "aircraftsurface1")],
    )
    records = {broad.path: broad, specific.path: specific}
    hashes = {
        broad.path: {"sha256": "same"},
        specific.path: {"sha256": "same"},
    }
    removed, conflicts = _select_exact_duplicate_removals(records, hashes)
    assert removed == {broad.path}
    assert conflicts == 1


def test_official_aebad_parent_or_direct_root_is_accepted(tmp_path: Path) -> None:
    direct = tmp_path / "AeBAD_S"
    for relative in ["train/good/same", "train/good/view", "test/good/same", "ground_truth"]:
        (direct / relative).mkdir(parents=True, exist_ok=True)
    save_image(direct / "train/good/same/one.png", color=(10, 20, 30))
    save_image(direct / "train/good/view/two.png", color=(40, 50, 60))
    train, validation = split_aebad_training_paths(tmp_path, 0.5, seed=7)
    assert len(train) == len(validation) == 1
    train_direct, validation_direct = split_aebad_training_paths(direct, 0.5, seed=7)
    assert train == train_direct
    assert validation == validation_direct


def test_aebad_split_removes_exact_copies_and_groups_repeated_names(tmp_path: Path) -> None:
    root = tmp_path / "AeBAD_S"
    (root / "test").mkdir(parents=True)
    save_image(root / "train/good/background/shared.png", color=(10, 20, 30))
    save_image(root / "train/good/view/shared.png", color=(30, 40, 50))
    save_image(root / "train/good/view/exact_copy.png", color=(10, 20, 30))
    save_image(root / "train/good/illumination/unique.png", color=(60, 70, 80))
    train, validation = split_aebad_training_paths(root, 0.5, seed=3)
    combined = train + validation
    assert len(combined) == 3
    shared_splits = {
        "train" if path in train else "validation"
        for path in combined
        if Path(path).name == "shared.png"
    }
    assert len(shared_splits) == 1


def test_agdd_rectangular_boxes_and_unmapped_spot(tmp_path: Path) -> None:
    data = tmp_path / "nested" / "data"
    for split in ("train", "val"):
        for modality in ("image", "images"):
            save_image(data / modality / split / f"{split}.png", size=(100, 80))
        label = data / "labels_rect" / split / f"{split}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("2 0.5 0.5 0.4 0.5\n3 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    records = load_agdd_source(tmp_path, LABELS)
    assert len(records["train"]) == len(records["validation"]) == 2
    assert records["train"][0].annotations[0].normalized_class == "crack"
    assert records["train"][0].annotations[0].bbox == [30.0, 20.0, 40.0, 40.0]
    assert len(records["train"][0].annotations) == 1


def test_imdd_audit_rejects_missing_images_and_records_no_boxes(tmp_path: Path) -> None:
    images = tmp_path / "imdd"
    save_image(images / "crack" / "one.jpg")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "Image Name,label,Categories,Description\none.jpg,0,crack,visible crack\n",
        encoding="utf-8",
    )
    report = audit_imdd_aircraft_subset(images, csv_path)
    assert report["images"] == 1
    assert report["has_localization_boxes"] is False
    assert report["used_for_detector_training"] is False
    csv_path.write_text(
        "Image Name,label,Categories,Description\nmissing.jpg,0,aircraft crack,visible crack\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetConfigurationError, match="missing or ambiguous"):
        audit_imdd_aircraft_subset(images, csv_path)


def test_aebad_v_sampling_is_per_video_and_bladesynth_uses_only_normal(tmp_path: Path) -> None:
    for relative in ["AeBAD_S/train", "AeBAD_S/test", "AeBAD_V/train/good/video1", "AeBAD_V/train/good/video2"]:
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for video in ("video1", "video2"):
        for index in range(5):
            save_image(tmp_path / "AeBAD_V/train/good" / video / f"{index:03}.png")
    save_image(tmp_path / "AeBAD_V/train/good/video1/._resource_fork.png")
    save_image(tmp_path / "AeBAD_V/train/good/video1/__MACOSX/metadata.png")
    sampled = sample_aebad_v_training_paths(tmp_path, stride=3)
    assert len(sampled) == 4
    save_image(tmp_path / "BladeSynth/Normal/images/normal.png")
    save_image(tmp_path / "BladeSynth/Normal/images/._resource_fork.png")
    save_image(tmp_path / "BladeSynth/Scratch/images/bad.png")
    save_image(tmp_path / "BladeSynth/Normal/masks/not_input.png")
    save_image(tmp_path / "BladeSynth/Dent/images/dent.png")
    save_image(tmp_path / "BladeSynth/Nick/images/nick.png")
    save_image(tmp_path / "BladeSynth/Corrosion/images/corrosion.png")
    normal = find_bladesynth_normal_paths(tmp_path / "BladeSynth")
    assert len(normal) == 1
    assert normal[0].endswith("normal.png")
    audit = audit_bladesynth(tmp_path / "BladeSynth", expected_images_per_class=1)
    assert audit["image_class_counts"] == {
        "corrosion": 1,
        "dent": 1,
        "nick": 1,
        "normal": 1,
        "scratch": 1,
    }
    assert audit["mask_files"] == 1
