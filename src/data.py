"""Dataset adapters, annotation normalization, split integrity, and COCO export."""

from __future__ import annotations

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
)


class DatasetConfigurationError(RuntimeError):
    """Raised when a required real dataset is absent or structurally unsupported."""


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
    if len(bbox) != 4 or not all(np.isfinite(float(item)) for item in bbox):
        return None
    x, y, w, h = map(float, bbox)
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
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            index[path.name.lower()].append(path)
    return index


def _records_from_coco(
    root: Path,
    annotation_path: Path,
    source: str,
    labels: Mapping[str, Sequence[str]],
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
    index = _image_index(root)
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
    for candidate in json_candidates:
        records = _records_from_coco(root, candidate, source, label_config)
        if records:
            return records
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


def prepare_aircraft_data(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    labels = config["aircraft_labels"]
    dataset_config = config["dataset"]
    sources = {
        "ASDD": Path(paths["asdd"]),
        "aircraftsurface1": Path(paths["aircraftsurface"]),
    }
    all_records: list[AircraftImageRecord] = []
    source_counts: dict[str, int] = {}
    annotation_record_counts: dict[str, int] = {}
    discovered_paths: list[str] = []
    for source, root in sources.items():
        records = load_aircraft_source(root, source, labels)
        source_images = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
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
    invalid_paths = {item["path"] for item in duplicate.invalid_images}
    exact_removed = {pair[1] for pair in duplicate.exact_pairs}
    records_by_path = {str(Path(record.path).resolve()): record for record in all_records}
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

    final_counts = Counter(
        annotation.normalized_class for record in records for annotation in record.annotations
    )
    return {
        "source_counts": source_counts,
        "annotation_record_counts": annotation_record_counts,
        "images_discovered": len(discovered_paths),
        "valid_images": len(records),
        "invalid_images": duplicate.invalid_images,
        "exact_duplicates": len(duplicate.exact_pairs),
        "near_duplicates": len(duplicate.near_pairs),
        "derivative_pairs": len(duplicate.derivative_pairs),
        "removed_duplicates": len(exact_removed),
        "class_distribution": dict(sorted(final_counts.items())),
        "corrosion_enabled": corrosion_enabled,
        "splits": {name: len(items) for name, items in splits.items()},
        "coco_annotations": coco_outputs,
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
            path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
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
            path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise DatasetConfigurationError(f"No images found in BladeSynth directory: {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = load_image(self.paths[index])
        return {"image": self.transform(image) if self.transform else image, "source": "BladeSynth", "synthetic": True}


def split_aebad_training_paths(root: str | Path, validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    root = _resolve_aebad_s_root(Path(root))
    good_root = root / "train" / "good"
    if not good_root.is_dir():
        raise DatasetConfigurationError(f"AeBAD-S normal training directory is missing: {good_root}")
    paths = sorted(
        str(path.resolve())
        for path in good_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if len(paths) < 2:
        raise DatasetConfigurationError("AeBAD-S needs at least two normal training images for train/validation separation.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    random.Random(seed).shuffle(paths)
    validation_count = max(1, min(len(paths) - 1, round(len(paths) * validation_fraction)))
    return sorted(paths[validation_count:]), sorted(paths[:validation_count])


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
    aircraft_report = prepare_aircraft_data(config)
    train_paths, validation_paths = split_aebad_training_paths(
        config["paths"]["aebad"],
        float(config["engine"]["validation_fraction"]),
        int(config["training"]["seed"]),
    )
    aebad_test = AeBADDataset(
        config["paths"]["aebad"], "test", image_size=int(config["engine"]["image_size"])
    )
    report = {
        "aircraft": aircraft_report,
        "engine": {
            "source": "AeBAD-S",
            "train_normal": len(train_paths),
            "validation_normal": len(validation_paths),
            "test": len(aebad_test),
            "threshold_selected_on_final_test": False,
        },
    }
    destination = Path(report_path or Path(config["paths"]["reports"]) / "dataset_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
