"""Dataset adapters, annotation normalization, split integrity, and COCO export."""

from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .preprocessing import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageValidationError,
    analyze_duplicates,
    load_image,
    sha256_file,
)


class DatasetConfigurationError(RuntimeError):
    """Raised when a required real dataset is absent or structurally unsupported."""


def _is_dataset_image(path: Path) -> bool:
    """Return true only for real image files, excluding archive metadata artifacts."""
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not path.name.startswith("._")
        and not any(part.lower() == "__macosx" for part in path.parts)
    )


@dataclass
class AircraftAnnotation:
    bbox: list[float]  # COCO x, y, width, height
    normalized_class: str
    original_class: str
    source: str
    iscrowd: int = 0


@dataclass
class AircraftImageRecord:
    path: str
    width: int
    height: int
    source: str
    annotations: list[AircraftAnnotation] = field(default_factory=list)


def canonical_label(value: str) -> str:
    return re.sub(r"[\s_\-]+", " ", value.strip().lower())


def build_label_lookup(label_config: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for normalized, aliases in label_config.items():
        for alias in [normalized, *aliases]:
            key = canonical_label(alias)
            previous = lookup.get(key)
            if previous and previous != normalized:
                raise ValueError(f"Ambiguous aircraft label alias {alias!r}: {previous} vs {normalized}")
            lookup[key] = normalized
    return lookup


def normalize_label(value: str, label_config: Mapping[str, Sequence[str]]) -> str | None:
    return build_label_lookup(label_config).get(canonical_label(value))


def _valid_bbox(bbox: Sequence[float], width: int, height: int) -> list[float] | None:
    try:
        if len(bbox) != 4:
            return None
    except TypeError:
        return None
    try:
        x, y, w, h = map(float, bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in (x, y, w, h)):
        return None
    x = max(0.0, min(x, float(width)))
    y = max(0.0, min(y, float(height)))
    w = max(0.0, min(w, float(width) - x))
    h = max(0.0, min(h, float(height) - y))
    return [x, y, w, h] if w > 0 and h > 0 else None


def _find_image(root: Path, file_name: str, index: Mapping[str, list[Path]]) -> Path | None:
    direct = root / file_name
    if direct.is_file():
        return direct.resolve()
    candidates = index.get(Path(file_name).name.lower(), [])
    return candidates[0].resolve() if len(candidates) == 1 else None


def _image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if _is_dataset_image(path):
            index[path.name.lower()].append(path)
    return index


def _records_from_coco(
    root: Path,
    annotation_path: Path,
    source: str,
    labels: Mapping[str, Sequence[str]],
    image_index: Mapping[str, list[Path]] | None = None,
) -> list[AircraftImageRecord]:
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetConfigurationError(f"Invalid COCO JSON {annotation_path}: {exc}") from exc
    if not all(key in payload for key in ("images", "annotations", "categories")):
        return []
    category_names = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    index = image_index or _image_index(root)
    records: list[AircraftImageRecord] = []
    for item in payload["images"]:
        image_path = _find_image(annotation_path.parent, str(item["file_name"]), index)
        if image_path is None:
            continue
        try:
            image = load_image(image_path)
        except ImageValidationError:
            continue
        annotations: list[AircraftAnnotation] = []
        for annotation in grouped.get(int(item["id"]), []):
            original = category_names.get(int(annotation["category_id"]), "")
            normalized = normalize_label(original, labels)
            bbox = _valid_bbox(annotation.get("bbox", []), image.width, image.height)
            if normalized and bbox:
                annotations.append(
                    AircraftAnnotation(bbox, normalized, original, source, int(annotation.get("iscrowd", 0)))
                )
        records.append(AircraftImageRecord(str(image_path), image.width, image.height, source, annotations))
    return records


def _records_from_voc(
    root: Path, source: str, labels: Mapping[str, Sequence[str]]
) -> list[AircraftImageRecord]:
    index = _image_index(root)
    records: list[AircraftImageRecord] = []
    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except ET.ParseError:
            continue
        file_name = xml_root.findtext("path") or xml_root.findtext("filename")
        if not file_name:
            continue
        image_path = Path(file_name)
        if not image_path.is_file():
            image_path = _find_image(root, Path(file_name).name, index)  # type: ignore[assignment]
        if image_path is None or not Path(image_path).is_file():
            continue
        try:
            image = load_image(image_path)
        except ImageValidationError:
            continue
        annotations: list[AircraftAnnotation] = []
        for obj in xml_root.findall("object"):
            original = obj.findtext("name") or ""
            normalized = normalize_label(original, labels)
            box = obj.find("bndbox")
            if not normalized or box is None:
                continue
            try:
                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))
            except ValueError:
                continue
            bbox = _valid_bbox([xmin, ymin, xmax - xmin, ymax - ymin], image.width, image.height)
            if bbox:
                annotations.append(AircraftAnnotation(bbox, normalized, original, source))
        records.append(
            AircraftImageRecord(str(Path(image_path).resolve()), image.width, image.height, source, annotations)
        )
    return records


def _find_yolo_names(root: Path) -> list[str] | None:
    import yaml

    for candidate in [*root.glob("*.yaml"), *root.glob("*.yml")]:
        try:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        names = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    for candidate in (root / "classes.txt", root / "obj.names"):
        if candidate.is_file():
            return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def _candidate_yolo_label(image_path: Path, root: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    for marker in ("images", "JPEGImages"):
        if marker in parts:
            idx = parts.index(marker)
            candidates.append(Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt"))
    for path in candidates:
        if path.is_file() and path.resolve() != (root / "classes.txt").resolve():
            return path
    return None


def _records_from_yolo(
    root: Path, source: str, labels: Mapping[str, Sequence[str]]
) -> list[AircraftImageRecord]:
    names = _find_yolo_names(root)
    if not names:
        return []
    records: list[AircraftImageRecord] = []
    for image_path in sorted(path for paths in _image_index(root).values() for path in paths):
        label_path = _candidate_yolo_label(image_path, root)
        if label_path is None:
            continue
        try:
            image = load_image(image_path)
        except ImageValidationError:
            continue
        annotations: list[AircraftAnnotation] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(fields[0])
                cx, cy, bw, bh = map(float, fields[1:5])
            except ValueError:
                continue
            if class_id < 0 or class_id >= len(names):
                continue
            original = names[class_id]
            normalized = normalize_label(original, labels)
            bbox = _valid_bbox(
                [(cx - bw / 2) * image.width, (cy - bh / 2) * image.height, bw * image.width, bh * image.height],
                image.width,
                image.height,
            )
            if normalized and bbox:
                annotations.append(AircraftAnnotation(bbox, normalized, original, source))
        records.append(AircraftImageRecord(str(image_path.resolve()), image.width, image.height, source, annotations))
    return records


def _records_from_labelme(
    root: Path, source: str, labels: Mapping[str, Sequence[str]]
) -> list[AircraftImageRecord]:
    index = _image_index(root)
    records: list[AircraftImageRecord] = []
    for annotation_path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or "shapes" not in payload or "imagePath" not in payload:
            continue
        image_path = _find_image(annotation_path.parent, str(payload["imagePath"]), index)
        if image_path is None:
            continue
        try:
            image = load_image(image_path)
        except ImageValidationError:
            continue
        annotations: list[AircraftAnnotation] = []
        for shape in payload.get("shapes", []):
            original = str(shape.get("label", ""))
            normalized = normalize_label(original, labels)
            points = np.asarray(shape.get("points", []), dtype=float)
            if not normalized or points.ndim != 2 or len(points) < 2 or points.shape[1] != 2:
                continue
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            bbox = _valid_bbox([xmin, ymin, xmax - xmin, ymax - ymin], image.width, image.height)
            if bbox:
                annotations.append(AircraftAnnotation(bbox, normalized, original, source))
        records.append(AircraftImageRecord(str(image_path), image.width, image.height, source, annotations))
    return records


def load_aircraft_source(
    root: str | Path,
    source: str,
    label_config: Mapping[str, Sequence[str]],
) -> list[AircraftImageRecord]:
    """Load one aircraft dataset without guessing labels or inventing annotations."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetConfigurationError(
            f"{source} dataset directory is missing: {root}. Download the real dataset and update config.yaml."
        )
    json_candidates = sorted(root.rglob("*.json"))
    coco_records: list[AircraftImageRecord] = []
    image_index = _image_index(root) if json_candidates else {}
    for candidate in json_candidates:
        coco_records.extend(
            _records_from_coco(root, candidate, source, label_config, image_index=image_index)
        )
    if coco_records:
        # Roboflow exports one COCO file per train/valid/test directory. Load all of them.
        unique = {str(Path(record.path).resolve()): record for record in coco_records}
        return [unique[path] for path in sorted(unique)]
    records = _records_from_voc(root, source, label_config)
    if records:
        return records
    records = _records_from_yolo(root, source, label_config)
    if records:
        return records
    records = _records_from_labelme(root, source, label_config)
    if records:
        return records
    raise DatasetConfigurationError(
        f"No supported COCO, Pascal VOC, YOLO, or LabelMe annotations were found in {root}. "
        "AeroInspect will not infer labels from folder names."
    )


def audit_coco_source_annotations(
    root: str | Path,
    label_config: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Report raw COCO integrity issues that normalization later repairs or excludes."""
    root = Path(root)
    image_index = _image_index(root)
    files: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    total_raw_classes: Counter[str] = Counter()
    total_normalized_classes: Counter[str] = Counter()
    for annotation_path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not all(key in payload for key in ("images", "annotations", "categories")):
            continue
        categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
        images = {int(item["id"]): item for item in payload["images"]}
        annotation_counts: Counter[int] = Counter()
        raw_classes: Counter[str] = Counter()
        normalized_classes: Counter[str] = Counter()
        missing_image_references = 0
        missing_category_references = 0
        invalid_boxes = 0
        boxes_outside_declared_bounds = 0
        unmapped_annotations = 0
        for annotation in payload["annotations"]:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            annotation_counts[image_id] += 1
            image = images.get(image_id)
            original = categories.get(category_id)
            if image is None:
                missing_image_references += 1
                continue
            if original is None:
                missing_category_references += 1
                continue
            raw_classes[original] += 1
            normalized = normalize_label(original, label_config)
            if normalized is None:
                unmapped_annotations += 1
            else:
                normalized_classes[normalized] += 1
            bbox = annotation.get("bbox", [])
            try:
                numeric_bbox = list(map(float, bbox))
            except (TypeError, ValueError):
                numeric_bbox = []
            if (
                len(numeric_bbox) != 4
                or not all(np.isfinite(item) for item in numeric_bbox)
                or numeric_bbox[2] <= 0
                or numeric_bbox[3] <= 0
            ):
                invalid_boxes += 1
                continue
            x, y, width, height = numeric_bbox
            if (
                x < 0
                or y < 0
                or x + width > float(image["width"])
                or y + height > float(image["height"])
            ):
                boxes_outside_declared_bounds += 1
        missing_files = sum(
            _find_image(annotation_path.parent, str(image["file_name"]), image_index) is None
            for image in images.values()
        )
        category_name_counts = Counter(canonical_label(name) for name in categories.values())
        duplicate_category_names = sorted(
            name for name, count in category_name_counts.items() if count > 1
        )
        unused_categories = sorted(
            name for category_id, name in categories.items()
            if not any(int(item["category_id"]) == category_id for item in payload["annotations"])
        )
        summary = {
            "images": len(images),
            "annotations": len(payload["annotations"]),
            "unannotated_images": sum(annotation_counts[image_id] == 0 for image_id in images),
            "missing_image_files": missing_files,
            "missing_image_references": missing_image_references,
            "missing_category_references": missing_category_references,
            "invalid_boxes": invalid_boxes,
            "boxes_outside_declared_bounds_clipped_during_loading": boxes_outside_declared_bounds,
            "unmapped_annotations_excluded": unmapped_annotations,
            "duplicate_category_names": duplicate_category_names,
            "unused_categories": unused_categories,
            "raw_class_distribution": dict(sorted(raw_classes.items())),
            "normalized_class_distribution": dict(sorted(normalized_classes.items())),
        }
        relative = str(annotation_path.relative_to(root))
        files[relative] = summary
        for key in (
            "images",
            "annotations",
            "unannotated_images",
            "missing_image_files",
            "missing_image_references",
            "missing_category_references",
            "invalid_boxes",
            "boxes_outside_declared_bounds_clipped_during_loading",
            "unmapped_annotations_excluded",
        ):
            totals[key] += int(summary[key])
        total_raw_classes.update(raw_classes)
        total_normalized_classes.update(normalized_classes)
    return {
        "totals": dict(totals),
        "raw_class_distribution": dict(sorted(total_raw_classes.items())),
        "normalized_class_distribution": dict(sorted(total_normalized_classes.items())),
        "annotation_files": files,
    }


def load_agdd_source(
    root: str | Path,
    label_config: Mapping[str, Sequence[str]],
) -> dict[str, list[AircraftImageRecord]]:
    """Load AGDD's paired illumination images with rectangular YOLO boxes."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetConfigurationError(f"AGDD dataset directory is missing: {root}")
    label_roots = [path for path in root.rglob("labels_rect") if path.is_dir()]
    if len(label_roots) != 1:
        raise DatasetConfigurationError(
            f"Expected exactly one AGDD labels_rect directory under {root}, found {len(label_roots)}."
        )
    data_root = label_roots[0].parent
    class_names = ["contusion", "scratches", "crack", "spot"]
    result: dict[str, list[AircraftImageRecord]] = {"train": [], "validation": []}
    for source_split, output_split in (("train", "train"), ("val", "validation")):
        label_dir = data_root / "labels_rect" / source_split
        if not label_dir.is_dir():
            raise DatasetConfigurationError(f"AGDD label split is missing: {label_dir}")
        for modality in ("image", "images"):
            image_dir = data_root / modality / source_split
            if not image_dir.is_dir():
                raise DatasetConfigurationError(f"AGDD image split is missing: {image_dir}")
            for image_path in sorted(
                path for path in image_dir.iterdir()
                if _is_dataset_image(path)
            ):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise DatasetConfigurationError(f"AGDD rectangular label is missing: {label_path}")
                try:
                    image = load_image(image_path)
                except ImageValidationError as exc:
                    raise DatasetConfigurationError(f"Invalid AGDD image {image_path}: {exc}") from exc
                annotations: list[AircraftAnnotation] = []
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    fields = line.split()
                    if len(fields) != 5:
                        raise DatasetConfigurationError(
                            f"Invalid AGDD rectangular label at {label_path}:{line_number}"
                        )
                    try:
                        class_id = int(fields[0])
                        cx, cy, box_width, box_height = map(float, fields[1:])
                    except ValueError as exc:
                        raise DatasetConfigurationError(
                            f"Invalid AGDD numeric label at {label_path}:{line_number}"
                        ) from exc
                    if not 0 <= class_id < len(class_names):
                        raise DatasetConfigurationError(
                            f"Unknown AGDD class id {class_id} at {label_path}:{line_number}"
                        )
                    original = class_names[class_id]
                    normalized = normalize_label(original, label_config)
                    bbox = _valid_bbox(
                        [
                            (cx - box_width / 2) * image.width,
                            (cy - box_height / 2) * image.height,
                            box_width * image.width,
                            box_height * image.height,
                        ],
                        image.width,
                        image.height,
                    )
                    # AGDD "spot" has no defensible mapping to the aircraft-skin taxonomy.
                    if normalized and bbox:
                        annotations.append(
                            AircraftAnnotation(bbox, normalized, original, "AGDD")
                        )
                result[output_split].append(
                    AircraftImageRecord(
                        str(image_path.resolve()), image.width, image.height, "AGDD", annotations
                    )
                )
    return result


def audit_imdd_aircraft_subset(
    image_root: str | Path, csv_path: str | Path
) -> dict[str, Any]:
    """Validate IMDD aircraft image-level labels without inventing detector boxes."""
    image_root, csv_path = Path(image_root), Path(csv_path)
    if not image_root.is_dir():
        raise DatasetConfigurationError(f"IMDD aircraft image directory is missing: {image_root}")
    if not csv_path.is_file():
        raise DatasetConfigurationError(f"IMDD aircraft CSV is missing: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Image Name", "label", "Categories", "Description"}
    if not rows or not required.issubset(rows[0]):
        raise DatasetConfigurationError(
            f"IMDD CSV must contain {sorted(required)} and at least one data row: {csv_path}"
        )
    images = [
        path for path in image_root.rglob("*")
        if _is_dataset_image(path)
    ]
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        by_name[path.name.lower()].append(path)
    csv_names = [str(row["Image Name"]).strip().lower() for row in rows]
    missing = sorted({name for name in csv_names if len(by_name.get(name, [])) != 1})
    unreferenced = sorted(path.name for path in images if path.name.lower() not in set(csv_names))
    if missing:
        raise DatasetConfigurationError(
            f"IMDD CSV has {len(missing)} missing or ambiguous image references; first: {missing[:5]}"
        )
    class_counts = Counter(
        canonical_label(str(row["Categories"]).split(",")[-1]) for row in rows
    )
    rows_by_name = {str(row["Image Name"]).strip().lower(): row for row in rows}
    folder_label_mismatches = []
    category_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_categories: dict[str, set[str]] = defaultdict(set)
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    monochrome_images = 0
    black_border_images = 0
    for path in images:
        row = rows_by_name[path.name.lower()]
        category = canonical_label(str(row["Categories"]).split(",")[-1])
        label_id = str(row["label"]).strip()
        category_to_ids[category].add(label_id)
        id_to_categories[label_id].add(category)
        if canonical_label(path.parent.name) != category:
            folder_label_mismatches.append(path.name)
        try:
            image = load_image(path)
        except ImageValidationError as exc:
            raise DatasetConfigurationError(f"Invalid IMDD image {path}: {exc}") from exc
        pixels = np.asarray(image)
        dimensions[f"{image.width}x{image.height}"] += 1
        monochrome_images += int(
            np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
            and np.array_equal(pixels[:, :, 1], pixels[:, :, 2])
        )
        edge = np.concatenate(
            [pixels[0, :, 0], pixels[-1, :, 0], pixels[:, 0, 0], pixels[:, -1, 0]]
        )
        black_border_images += int(float(np.mean(edge <= 5)) > 0.10)
        exact_hashes[sha256_file(path)].append(path.name)
    exact_duplicate_groups = [group for group in exact_hashes.values() if len(group) > 1]
    if folder_label_mismatches:
        raise DatasetConfigurationError(
            f"IMDD has {len(folder_label_mismatches)} folder/CSV label mismatches; first: "
            f"{folder_label_mismatches[:5]}"
        )
    inconsistent_categories = {
        category: sorted(ids) for category, ids in category_to_ids.items() if len(ids) != 1
    }
    inconsistent_ids = {
        label_id: sorted(categories)
        for label_id, categories in id_to_categories.items()
        if len(categories) != 1
    }
    if inconsistent_categories or inconsistent_ids:
        raise DatasetConfigurationError(
            "IMDD numeric labels and category names are not one-to-one: "
            f"categories={inconsistent_categories}, ids={inconsistent_ids}"
        )
    smallest_class = min(class_counts.values())
    largest_class = max(class_counts.values())
    return {
        "images": len(images),
        "csv_rows": len(rows),
        "unreferenced_images": len(unreferenced),
        "class_distribution": dict(sorted(class_counts.items())),
        "class_imbalance_largest_to_smallest": round(largest_class / smallest_class, 3),
        "dimensions": dict(sorted(dimensions.items())),
        "monochrome_images": monochrome_images,
        "black_border_images_over_10_percent_edge": black_border_images,
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "folder_csv_label_mismatches": 0,
        "annotation_level": "image_classification_and_text",
        "has_localization_boxes": False,
        "used_for_detector_training": False,
        "exclusion_reason": (
            "The CSV has image-level labels/descriptions but no bounding boxes; detector boxes are never invented."
        ),
        "recommended_use": (
            "External image-level presence evaluation or manual box annotation; not localization training."
        ),
    }


def _grouped_split(
    paths: Sequence[str], groups: Sequence[Sequence[str]], ratios: Mapping[str, float], seed: int
) -> dict[str, list[str]]:
    names = ("train", "validation", "test")
    values = [float(ratios[name]) for name in names]
    if any(value < 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError("Dataset split ratios must be non-negative and sum to 1.0.")
    path_set = set(paths)
    clean_groups = [sorted(path_set.intersection(group)) for group in groups]
    clean_groups = [group for group in clean_groups if group]
    covered = {path for group in clean_groups for path in group}
    clean_groups.extend([[path]] for path in sorted(path_set - covered))
    random.Random(seed).shuffle(clean_groups)
    targets = dict(zip(names, (len(paths) * value for value in values)))
    result = {name: [] for name in names}
    # Largest groups first makes target counts and leakage control more stable.
    clean_groups.sort(key=len, reverse=True)
    for group in clean_groups:
        split = max(names, key=lambda name: targets[name] - len(result[name]))
        result[split].extend(group)
    return {key: sorted(value) for key, value in result.items()}


def records_to_coco(
    records: Sequence[AircraftImageRecord], categories: Sequence[str], output_path: str | Path
) -> dict[str, Any]:
    category_to_id = {name: index + 1 for index, name in enumerate(categories)}
    output: dict[str, Any] = {
        "info": {"description": "AeroInspect unified aircraft defect annotations"},
        "licenses": [],
        "categories": [
            {"id": category_to_id[name], "name": name, "supercategory": "aircraft_defect"}
            for name in categories
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    for image_id, record in enumerate(sorted(records, key=lambda item: item.path), start=1):
        output["images"].append(
            {
                "id": image_id,
                "file_name": record.path,
                "width": record.width,
                "height": record.height,
                "source": record.source,
            }
        )
        for annotation in record.annotations:
            if annotation.normalized_class not in category_to_id:
                continue
            x, y, width, height = annotation.bbox
            output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[annotation.normalized_class],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": annotation.iscrowd,
                    "source": annotation.source,
                    "original_class_name": annotation.original_class,
                }
            )
            annotation_id += 1
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _select_exact_duplicate_removals(
    records_by_path: Mapping[str, AircraftImageRecord],
    hashes: Mapping[str, Mapping[str, str]],
) -> tuple[set[str], int]:
    """Choose the best-labeled representative of each byte-identical image group."""
    exact_hash_groups: dict[str, list[str]] = defaultdict(list)
    for path, image_hashes in hashes.items():
        if path in records_by_path:
            exact_hash_groups[image_hashes["sha256"]].append(path)
    removed: set[str] = set()
    label_conflicts = 0
    for group in exact_hash_groups.values():
        if len(group) < 2:
            continue
        label_sets = {
            tuple(sorted({annotation.normalized_class for annotation in records_by_path[path].annotations}))
            for path in group
        }
        label_conflicts += int(len(label_sets) > 1)

        def annotation_quality(path: str) -> tuple[int, int, int, int, str]:
            record = records_by_path[path]
            labels_for_record = {annotation.normalized_class for annotation in record.annotations}
            specific = labels_for_record - {"surface_damage"}
            # Prefer specific classes and richer annotations. aircraftsurface1 is
            # the tie-breaker because ASDD's composite label is intentionally broad.
            source_preference = int(record.source == "aircraftsurface1")
            return len(specific), len(labels_for_record), len(record.annotations), source_preference, path

        preferred = max(group, key=annotation_quality)
        removed.update(path for path in group if path != preferred)
    return removed, label_conflicts


def detection_sampling_weights(
    annotation_file: str | Path,
    max_weight: float = 4.0,
) -> list[float]:
    """Build capped inverse-square-root image weights for imbalanced detector classes."""
    if max_weight < 1:
        raise ValueError("Detector maximum sampling weight must be at least one.")
    payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    image_categories: dict[int, set[int]] = defaultdict(set)
    for annotation in payload["annotations"]:
        image_categories[int(annotation["image_id"])].add(int(annotation["category_id"]))
    category_image_counts: Counter[int] = Counter(
        category_id
        for category_ids in image_categories.values()
        for category_id in category_ids
    )
    if not category_image_counts:
        return [1.0] * len(images)
    majority = max(category_image_counts.values())
    category_weights = {
        category_id: min(max_weight, float(np.sqrt(majority / count)))
        for category_id, count in category_image_counts.items()
    }
    return [
        max((category_weights[category_id] for category_id in image_categories[int(image["id"])]), default=1.0)
        for image in images
    ]


def prepare_aircraft_data(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    labels = config["aircraft_labels"]
    dataset_config = config["dataset"]
    sources = {
        "ASDD": Path(paths["asdd"]),
        "aircraftsurface1": Path(paths["aircraftsurface"]),
    }
    all_records: list[AircraftImageRecord] = []
    source_annotation_audits = {
        source: audit_coco_source_annotations(root, labels) for source, root in sources.items()
    }
    source_counts: dict[str, int] = {}
    annotation_record_counts: dict[str, int] = {}
    discovered_paths: list[str] = []
    for source, root in sources.items():
        records = load_aircraft_source(root, source, labels)
        source_images = sorted(
            path for path in root.rglob("*")
            if _is_dataset_image(path)
        )
        source_counts[source] = len(source_images)
        annotation_record_counts[source] = len(records)
        discovered_paths.extend(str(path.resolve()) for path in source_images)
        all_records.extend(records)
    if not all_records:
        raise DatasetConfigurationError("No usable aircraft images with supported annotations were found.")

    duplicate = analyze_duplicates(
        discovered_paths,
        near_hamming=int(dataset_config.get("near_duplicate_hamming", 6)),
        derivative_patterns=dataset_config.get("derivative_stem_patterns", ()),
    )
    records_by_path = {str(Path(record.path).resolve()): record for record in all_records}
    invalid_paths = {item["path"] for item in duplicate.invalid_images}
    exact_removed, exact_label_conflicts = _select_exact_duplicate_removals(
        records_by_path, duplicate.hashes
    )
    records = [
        record
        for path, record in records_by_path.items()
        if path not in invalid_paths and path not in exact_removed
    ]

    class_counts = Counter(
        annotation.normalized_class for record in records for annotation in record.annotations
    )
    corrosion_minimum = int(dataset_config.get("corrosion_min_examples", 30))
    categories = [name for name in labels if name != "corrosion"]
    corrosion_enabled = class_counts.get("corrosion", 0) >= corrosion_minimum
    if corrosion_enabled:
        categories.append("corrosion")
    else:
        for record in records:
            record.annotations = [
                annotation for annotation in record.annotations if annotation.normalized_class != "corrosion"
            ]
    # Images with no compatible labels remain valid negative detector examples.
    record_paths = [str(Path(record.path).resolve()) for record in records]
    splits = _grouped_split(
        record_paths,
        duplicate.groups,
        dataset_config["split"],
        int(config["training"]["seed"]),
    )
    split_lookup = {path: split for split, split_paths in splits.items() for path in split_paths}
    processed_root = Path(paths["processed"]) / "aircraft"
    coco_outputs: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        output_path = processed_root / f"{split}.json"
        records_to_coco(
            [record for record in records if split_lookup.get(str(Path(record.path).resolve())) == split],
            categories,
            output_path,
        )
        coco_outputs[split] = str(output_path)

    agdd_records = load_agdd_source(paths["agdd"], labels)
    agdd_paths = [record.path for split_records in agdd_records.values() for record in split_records]
    agdd_duplicates = analyze_duplicates(
        agdd_paths,
        near_hamming=int(dataset_config.get("near_duplicate_hamming", 6)),
        derivative_patterns=dataset_config.get("derivative_stem_patterns", ()),
    )
    agdd_invalid = {item["path"] for item in agdd_duplicates.invalid_images}
    agdd_exact_removed = {pair[1] for pair in agdd_duplicates.exact_pairs}
    main_exact_hashes = {value["sha256"] for value in duplicate.hashes.values()}
    cross_source_removed = {
        path
        for path, hashes in agdd_duplicates.hashes.items()
        if hashes["sha256"] in main_exact_hashes
    }
    auxiliary_outputs: dict[str, str] = {}
    auxiliary_counts: dict[str, int] = {}
    auxiliary_root = Path(paths["processed"]) / "aircraft_auxiliary"
    for split in ("train", "validation"):
        usable = [
            record for record in agdd_records[split]
            if record.path not in agdd_invalid
            and record.path not in agdd_exact_removed
            and record.path not in cross_source_removed
        ]
        output_path = auxiliary_root / f"agdd_{split}.json"
        records_to_coco(usable, categories, output_path)
        auxiliary_outputs[split] = str(output_path)
        auxiliary_counts[split] = len(usable)

    final_counts = Counter(
        annotation.normalized_class for record in records for annotation in record.annotations
    )
    return {
        "source_counts": source_counts,
        "annotation_record_counts": annotation_record_counts,
        "source_annotation_audits": source_annotation_audits,
        "images_discovered": len(discovered_paths),
        "valid_images": len(records),
        "invalid_images": duplicate.invalid_images,
        "exact_duplicates": len(duplicate.exact_pairs),
        "exact_duplicate_label_conflicts": exact_label_conflicts,
        "exact_duplicate_resolution": (
            "Kept one copy with the most specific/rich annotation set; aircraftsurface1 wins exact ties."
        ),
        "near_duplicates": len(duplicate.near_pairs),
        "derivative_pairs": len(duplicate.derivative_pairs),
        "removed_duplicates": len(exact_removed),
        "class_distribution": dict(sorted(final_counts.items())),
        "corrosion_enabled": corrosion_enabled,
        "splits": {name: len(items) for name, items in splits.items()},
        "coco_annotations": coco_outputs,
        "auxiliary_pretraining": {
            "source": "AGDD",
            "images": auxiliary_counts,
            "coco_annotations": auxiliary_outputs,
            "ignored_unmapped_class": "spot",
            "cross_source_exact_duplicates_removed": len(cross_source_removed),
            "used_for_final_test": False,
        },
        "integrity": {"duplicate_groups_kept_within_one_split": True, "external_iisc_used": False},
    }


class AircraftDetectionDataset:
    """PyTorch dataset backed by AeroInspect's unified COCO JSON."""

    def __init__(self, annotation_file: str | Path, processor: Any, transform: Any = None):
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.images = payload["images"]
        ordered_categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
        self.category_to_model = {
            int(category["id"]): model_id for model_id, category in enumerate(ordered_categories)
        }
        self.model_to_category = {value: key for key, value in self.category_to_model.items()}
        self.labels = [str(category["name"]) for category in ordered_categories]
        self.annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)
        self.processor = processor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.images[index]
        image = load_image(item["file_name"])
        annotations = [
            {**annotation, "category_id": self.category_to_model[int(annotation["category_id"])]}
            for annotation in self.annotations.get(int(item["id"]), [])
        ]
        if self.transform is not None:
            transformed = self.transform(
                image=np.asarray(image),
                bboxes=[annotation["bbox"] for annotation in annotations],
                category_ids=[annotation["category_id"] for annotation in annotations],
            )
            image = Image.fromarray(transformed["image"])
            annotations = [
                {
                    "id": offset + 1,
                    "image_id": int(item["id"]),
                    "bbox": list(bbox),
                    "category_id": int(category_id),
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
                for offset, (bbox, category_id) in enumerate(
                    zip(transformed["bboxes"], transformed["category_ids"])
                )
            ]
        encoded = self.processor(
            images=image,
            annotations={"image_id": int(item["id"]), "annotations": annotations},
            return_tensors="pt",
        )
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": encoded["labels"][0]}


class AeBADDataset:
    """AeBAD-S adapter that preserves official train/test domain boundaries."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 224,
        transform: Any = None,
        paths: Sequence[str | Path] | None = None,
    ):
        supplied_root = Path(root)
        if not supplied_root.is_dir():
            raise DatasetConfigurationError(
                f"AeBAD-S directory is missing: {supplied_root}. Expected the official AeBAD_S/train, test, "
                "and ground_truth structure."
            )
        self.root = _resolve_aebad_s_root(supplied_root)
        if split not in {"train", "validation", "test"}:
            raise ValueError("AeBAD split must be train, validation, or test.")
        self.split = split
        self.image_size = int(image_size)
        self.transform = transform
        base = self.root / ("test" if split == "test" else "train")
        if not base.is_dir():
            raise DatasetConfigurationError(f"AeBAD-S split directory is missing: {base}")
        discovered = sorted(
            path for path in base.rglob("*") if _is_dataset_image(path)
        )
        if paths is not None:
            allowed = {str(Path(path).resolve()) for path in paths}
            discovered = [path for path in discovered if str(path.resolve()) in allowed]
        if split != "test":
            discovered = [path for path in discovered if "good" in {part.lower() for part in path.parts}]
        if not discovered:
            raise DatasetConfigurationError(f"No AeBAD-S {split} images found under {base}.")
        self.paths = discovered

    def __len__(self) -> int:
        return len(self.paths)

    def _mask_path(self, image_path: Path) -> Path | None:
        test_root = self.root / "test"
        relative = image_path.relative_to(test_root)
        candidates = []
        ground_root = self.root / "ground_truth"
        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidates.extend(
                [
                    (ground_root / relative).with_suffix(suffix),
                    (ground_root / relative).with_name(relative.stem + "_mask" + suffix),
                ]
            )
        return next((path for path in candidates if path.is_file()), None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from torchvision.transforms import v2

        path = self.paths[index]
        image = load_image(path)
        resize = round(self.image_size * 256 / 224)
        default_transform = v2.Compose(
            [v2.Resize((resize, resize), antialias=True), v2.CenterCrop(self.image_size), v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )
        tensor = self.transform(image) if self.transform else default_transform(image)
        parts = {part.lower() for part in path.parts}
        is_anomaly = int(self.split == "test" and not ({"good", "normal"} & parts))
        mask_path = self._mask_path(path) if is_anomaly else None
        if is_anomaly and mask_path is None:
            raise DatasetConfigurationError(
                f"AeBAD-S anomaly mask is missing for {path}. Pixel metrics would be invalid."
            )
        if mask_path:
            mask_image = load_image(mask_path).convert("L").resize((resize, resize), Image.Resampling.NEAREST)
            left = (resize - self.image_size) // 2
            mask_image = mask_image.crop((left, left, left + self.image_size, left + self.image_size))
            mask = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        relative_parts = path.relative_to(
            self.root / ("test" if self.split == "test" else "train")
        ).parts
        domain = relative_parts[1] if len(relative_parts) > 2 else relative_parts[0]
        return {
            "image": tensor,
            "is_anomaly": is_anomaly,
            "mask": mask,
            "domain": domain,
            "image_path": str(path),
            "image_name": path.name,
        }


class BladeSynthDataset:
    """Optional synthetic-data adapter; results must remain labeled synthetic."""

    def __init__(self, root: str | Path, transform: Any = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetConfigurationError(f"Optional BladeSynth directory is missing: {self.root}")
        self.paths = sorted(
            path for path in self.root.rglob("*") if _is_dataset_image(path)
        )
        if not self.paths:
            raise DatasetConfigurationError(f"No images found in BladeSynth directory: {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = load_image(self.paths[index])
        return {"image": self.transform(image) if self.transform else image, "source": "BladeSynth", "synthetic": True}


class NormalImageDataset:
    """Normalized image-only dataset for auxiliary normality pretraining."""

    def __init__(self, paths: Sequence[str | Path], image_size: int = 224, transform: Any = None):
        if not paths:
            raise DatasetConfigurationError("Normal-image dataset received no image paths.")
        self.paths = [Path(path) for path in paths]
        self.image_size = int(image_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from torchvision.transforms import v2

        image = load_image(self.paths[index])
        default_transform = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        tensor = self.transform(image) if self.transform else default_transform(image)
        return {"image": tensor, "image_path": str(self.paths[index])}


def sample_aebad_v_training_paths(root: str | Path, stride: int = 10) -> list[str]:
    """Sample only official normal AeBAD-V training frames, independently per video."""
    if stride < 1:
        raise ValueError("AeBAD-V frame stride must be at least one.")
    aebad_s = _resolve_aebad_s_root(Path(root))
    video_root = aebad_s.parent / "AeBAD_V" / "train" / "good"
    if not video_root.is_dir():
        raise DatasetConfigurationError(f"AeBAD-V normal training directory is missing: {video_root}")
    selected: list[str] = []
    video_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    if not video_dirs:
        raise DatasetConfigurationError(f"No AeBAD-V training videos found under {video_root}")
    for video_dir in video_dirs:
        frames = sorted(
            path for path in video_dir.rglob("*")
            if _is_dataset_image(path)
        )
        selected.extend(str(path.resolve()) for path in frames[::stride])
    if not selected:
        raise DatasetConfigurationError("AeBAD-V sampling produced no normal training frames.")
    return selected


def find_bladesynth_normal_paths(root: str | Path) -> list[str]:
    """Find BladeSynth's Normal class without accepting masks as input images."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetConfigurationError(
            f"BladeSynth directory is missing: {root}. The .crdownload file is not a completed dataset."
        )
    normal_names = {"normal", "good", "healthy", "defect free", "defectfree"}
    rejected_names = {"mask", "masks", "label", "labels", "ground truth", "groundtruth"}
    paths = []
    for path in root.rglob("*"):
        if not _is_dataset_image(path):
            continue
        parent_names = {canonical_label(part) for part in path.parts}
        is_annotation_artifact = bool(parent_names & rejected_names) or any(
            "mask" in name or "ground truth" in name for name in parent_names
        )
        if parent_names & normal_names and not is_annotation_artifact:
            paths.append(str(path.resolve()))
    if not paths:
        raise DatasetConfigurationError(
            f"No BladeSynth Normal images were found under {root}. Extract the completed archive first."
        )
    return sorted(paths)


def audit_bladesynth(
    root: str | Path,
    expected_images_per_class: int = 2500,
) -> dict[str, Any]:
    """Validate the official five-class BladeSynth archive before it is used."""
    root = Path(root)
    normal_paths = find_bladesynth_normal_paths(root)
    expected_classes = {"normal", "dent", "nick", "scratch", "corrosion"}
    rejected_names = {"mask", "masks", "label", "labels", "ground truth", "groundtruth"}
    class_paths: dict[str, list[Path]] = defaultdict(list)
    mask_files = 0
    unclassified = []
    for path in root.rglob("*"):
        if not _is_dataset_image(path):
            continue
        parent_names = {canonical_label(part) for part in path.parts}
        if parent_names & rejected_names or any(
            "mask" in name or "ground truth" in name for name in parent_names
        ):
            mask_files += 1
            continue
        matches = sorted(parent_names & expected_classes)
        if len(matches) == 1:
            class_paths[matches[0]].append(path)
        else:
            unclassified.append(str(path.resolve()))
    counts = {name: len(class_paths[name]) for name in sorted(expected_classes)}
    unexpected_counts = {
        name: count for name, count in counts.items() if count != expected_images_per_class
    }
    if unexpected_counts:
        raise DatasetConfigurationError(
            "BladeSynth class counts do not match the official archive "
            f"({expected_images_per_class} images per class): {unexpected_counts}."
        )
    corrupt = []
    dimensions_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for class_name, paths in class_paths.items():
        for path in paths:
            try:
                image = load_image(path)
            except ImageValidationError as exc:
                corrupt.append({"path": str(path), "error": str(exc)})
                continue
            dimensions_by_class[class_name][f"{image.width}x{image.height}"] += 1
    if corrupt:
        raise DatasetConfigurationError(
            f"BladeSynth contains {len(corrupt)} corrupt class images; first: {corrupt[:3]}"
        )
    return {
        "image_class_counts": counts,
        "normal_images": len(normal_paths),
        "mask_files": mask_files,
        "unclassified_image_files": len(unclassified),
        "image_dimensions_by_class": {
            name: dict(sorted(dimensions_by_class[name].items()))
            for name in sorted(expected_classes)
        },
        "synthetic_anomaly_classes_used_as_normal": False,
        "used_for_real_test_metrics": False,
        "role": "synthetic auxiliary experiment only",
    }


def split_aebad_training_paths(root: str | Path, validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    root = _resolve_aebad_s_root(Path(root))
    good_root = root / "train" / "good"
    if not good_root.is_dir():
        raise DatasetConfigurationError(f"AeBAD-S normal training directory is missing: {good_root}")
    discovered = sorted(
        str(path.resolve())
        for path in good_root.rglob("*")
        if _is_dataset_image(path)
    )
    unique_by_hash: dict[str, str] = {}
    for path in discovered:
        unique_by_hash.setdefault(sha256_file(path), path)
    paths = sorted(unique_by_hash.values())
    if len(paths) < 2:
        raise DatasetConfigurationError("AeBAD-S needs at least two normal training images for train/validation separation.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_name[Path(path).name.lower()].append(path)
    splits = _grouped_split(
        paths,
        list(by_name.values()),
        {"train": 1.0 - validation_fraction, "validation": validation_fraction, "test": 0.0},
        seed,
    )
    return splits["train"], splits["validation"]


def _resolve_aebad_s_root(root: Path) -> Path:
    """Accept either the official AeBAD parent directory or its AeBAD_S child."""
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root
    nested = root / "AeBAD_S"
    if (nested / "train").is_dir() and (nested / "test").is_dir():
        return nested
    raise DatasetConfigurationError(
        f"AeBAD-S structure not found under {root}. Expected train/, test/, and ground_truth/ "
        "directly or inside an AeBAD_S/ child."
    )


def prepare_datasets(config: Mapping[str, Any], report_path: str | Path | None = None) -> dict[str, Any]:
    """Prepare configured real datasets and write a truthful, generated report."""
    # Run inexpensive completeness checks before the slow aircraft duplicate
    # analysis so an unfinished archive fails quickly and clearly.
    train_paths, validation_paths = split_aebad_training_paths(
        config["paths"]["aebad"],
        float(config["engine"]["validation_fraction"]),
        int(config["training"]["seed"]),
    )
    aebad_test = AeBADDataset(
        config["paths"]["aebad"], "test", image_size=int(config["engine"]["image_size"])
    )
    missing_masks = []
    for image_path in aebad_test.paths:
        parts = {part.lower() for part in image_path.parts}
        if not ({"good", "normal"} & parts) and aebad_test._mask_path(image_path) is None:
            missing_masks.append(str(image_path))
    if missing_masks:
        raise DatasetConfigurationError(
            f"AeBAD-S has {len(missing_masks)} anomalous test images without masks; first: "
            f"{missing_masks[:3]}"
        )
    video_paths = sample_aebad_v_training_paths(
        config["paths"]["aebad"], int(config["engine"]["aebad_v_frame_stride"])
    )
    bladesynth_report = audit_bladesynth(
        config["paths"]["bladesynth"],
        int(config["engine"].get("bladesynth_expected_images_per_class", 2500)),
    )
    bladesynth_paths = find_bladesynth_normal_paths(config["paths"]["bladesynth"])
    imdd_report = audit_imdd_aircraft_subset(
        config["paths"]["imdd_aircraft_images"], config["paths"]["imdd_aircraft_csv"]
    )
    aircraft_report = prepare_aircraft_data(config)
    report = {
        "aircraft": aircraft_report,
        "aircraft_image_level_auxiliary": {"source": "IMDD aircraft subset", **imdd_report},
        "engine": {
            "source": "AeBAD-S",
            "train_normal": len(train_paths),
            "validation_normal": len(validation_paths),
            "test": len(aebad_test),
            "threshold_selected_on_final_test": False,
            "auxiliary_pretraining": {
                "aebad_v_sampled_normal_frames": len(video_paths),
                "aebad_v_stride": int(config["engine"]["aebad_v_frame_stride"]),
                "bladesynth_normal_images": len(bladesynth_paths),
                "bladesynth_audit": bladesynth_report,
                "synthetic_anomalies_used_as_normal": False,
                "used_for_threshold_calibration": False,
                "used_for_final_test": False,
            },
        },
    }
    destination = Path(report_path or Path(config["paths"]["reports"]) / "dataset_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
