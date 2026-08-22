"""Image loading, quality checks, augmentation, and duplicate utilities."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an inspection input cannot be decoded or is unusable."""


def load_image(source: str | Path | bytes | bytearray | BinaryIO | Image.Image | np.ndarray) -> Image.Image:
    """Load an RGB image from a path, bytes, file-like object, PIL image, or ndarray."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, np.ndarray):
            if source.ndim not in (2, 3) or source.size == 0:
                raise ImageValidationError("Image array must be a non-empty 2D or 3D array.")
            array = source
            if array.dtype != np.uint8:
                if not np.isfinite(array).all():
                    raise ImageValidationError("Image array contains NaN or infinite values.")
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.ndim == 3 and array.shape[2] == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            elif array.ndim == 3 and array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
            image = Image.fromarray(array)
        elif isinstance(source, (bytes, bytearray)):
            if not source:
                raise ImageValidationError("Image data is empty.")
            image = Image.open(io.BytesIO(source))
        elif hasattr(source, "read"):
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
            if not raw:
                raise ImageValidationError("Image data is empty.")
            image = Image.open(io.BytesIO(raw))
        else:
            path = Path(source)
            if not path.is_file():
                raise ImageValidationError(f"Image file not found: {path}")
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ImageValidationError(f"Unsupported image format: {path.suffix or 'none'}")
            image = Image.open(path)
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.width <= 0 or image.height <= 0:
            raise ImageValidationError("Image has invalid dimensions.")
        return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Image is corrupted or unsupported: {exc}") from exc


def check_image_quality(
    source: Any,
    *,
    minimum_width: int = 128,
    minimum_height: int = 128,
    blur_variance_warning: float = 35.0,
    dark_mean_warning: float = 25.0,
    bright_mean_warning: float = 235.0,
) -> dict[str, Any]:
    """Run deterministic quality checks; only decoding/dimension failures are rejected."""
    image = load_image(source)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings: list[str] = []
    unusable = False
    if image.width < minimum_width or image.height < minimum_height:
        warnings.append(
            f"Image resolution {image.width}x{image.height} is below the recommended "
            f"minimum {minimum_width}x{minimum_height}."
        )
        unusable = image.width < 16 or image.height < 16
    if laplacian_variance < blur_variance_warning:
        warnings.append("Image appears heavily blurred.")
    if mean < dark_mean_warning:
        warnings.append("Image appears extremely dark.")
    if mean > bright_mean_warning:
        warnings.append("Image appears extremely bright.")
    return {
        "quality": "invalid" if unusable else ("warning" if warnings else "ok"),
        "warnings": warnings,
        "usable": not unusable,
        "width": image.width,
        "height": image.height,
        "brightness_mean": round(mean, 3),
        "laplacian_variance": round(laplacian_variance, 3),
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(path: str | Path) -> tuple[str, str]:
    with Image.open(path) as image:
        image.load()
        return str(imagehash.phash(image)), str(imagehash.dhash(image))


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Image hashes must have equal lengths.")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def derivative_key(path: str | Path, patterns: Sequence[str] = ()) -> str:
    """Return a conservative parent key for common generated/augmented file suffixes."""
    stem = Path(path).stem.lower()
    for pattern in patterns:
        stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class BKTree:
    """Small metric tree used to avoid an O(n^2) perceptual-hash comparison."""

    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        result: list[int] = []
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                result.append(node[0])
            stack.extend(
                child for edge, child in node[1].items() if distance - radius <= edge <= distance + radius
            )
        return result


@dataclass(frozen=True)
class DuplicateAnalysis:
    groups: list[list[str]]
    exact_pairs: list[tuple[str, str]]
    near_pairs: list[tuple[str, str]]
    derivative_pairs: list[tuple[str, str]]
    invalid_images: list[dict[str, str]]
    hashes: dict[str, dict[str, str]]


def analyze_duplicates(
    paths: Sequence[str | Path],
    *,
    near_hamming: int = 6,
    derivative_patterns: Sequence[str] = (),
) -> DuplicateAnalysis:
    """Group exact, perceptually-near, and conservatively named derivative images."""
    canonical = [str(Path(path).resolve()) for path in paths]
    uf = UnionFind(canonical)
    exact_pairs: list[tuple[str, str]] = []
    near_pairs: list[tuple[str, str]] = []
    derivative_pairs: list[tuple[str, str]] = []
    invalid: list[dict[str, str]] = []
    hashes: dict[str, dict[str, str]] = {}
    exact_seen: dict[str, str] = {}
    derivative_seen: dict[str, str] = {}
    phash_seen: dict[int, str] = {}
    tree = BKTree()

    for path in canonical:
        try:
            load_image(path)
            exact = sha256_file(path)
            phash, dhash = perceptual_hashes(path)
        except (ImageValidationError, OSError) as exc:
            invalid.append({"path": path, "error": str(exc)})
            continue
        hashes[path] = {"sha256": exact, "phash": phash, "dhash": dhash}
        if exact in exact_seen:
            pair = (exact_seen[exact], path)
            exact_pairs.append(pair)
            uf.union(*pair)
        else:
            exact_seen[exact] = path

        key = derivative_key(path, derivative_patterns)
        if key and key in derivative_seen:
            pair = (derivative_seen[key], path)
            derivative_pairs.append(pair)
            uf.union(*pair)
        elif key:
            derivative_seen[key] = path

        phash_int = int(phash, 16)
        for match in tree.query(phash_int, near_hamming):
            other = phash_seen[match]
            # Requiring both hash families greatly reduces accidental visual collisions.
            if (
                other != path
                and hashes[other]["sha256"] != exact
                and hamming_distance(hashes[other]["dhash"], dhash) <= near_hamming
            ):
                pair = (other, path)
                near_pairs.append(pair)
                uf.union(*pair)
        if phash_int not in phash_seen:
            phash_seen[phash_int] = path
            tree.add(phash_int)

    grouped: dict[str, list[str]] = {}
    for path in canonical:
        if path in hashes:
            grouped.setdefault(uf.find(path), []).append(path)
    groups = [sorted(group) for group in grouped.values()]
    groups.sort(key=lambda group: group[0])
    return DuplicateAnalysis(groups, exact_pairs, near_pairs, derivative_pairs, invalid, hashes)


def build_aircraft_augmentation(image_size: int = 800):
    """Return bbox-safe, physically mild aircraft training augmentation."""
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError("Albumentations is required for aircraft training augmentation.") from exc
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), translate_percent=(-0.03, 0.03), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.15),
            A.GaussianBlur(blur_limit=(3, 3), p=0.10),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category_ids"], min_visibility=0.25,
            clip=True, filter_invalid_bboxes=True
        ),
    )
