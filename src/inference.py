"""Reusable, fail-closed inference and visualization APIs used by Streamlit."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .preprocessing import ImageValidationError, check_image_quality, load_image


class ModelNotReadyError(RuntimeError):
    """Raised instead of returning fabricated output when a trained artifact is absent."""


def _quality(image: Image.Image, config: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = (config or {}).get("quality", {})
    return check_image_quality(image, **settings)


def annotate_aircraft(image: Any, detections: list[Mapping[str, Any]]) -> Image.Image:
    output = load_image(image)
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for detection in detections:
        box = [float(value) for value in detection["bbox"]]
        label = f"{str(detection['class']).replace('_', ' ')} {float(detection['confidence']):.1%}"
        draw.rectangle(box, outline=(238, 72, 86), width=max(2, round(min(output.size) / 250)))
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        y = max(0, box[1] - (text_box[3] - text_box[1]) - 6)
        draw.rectangle((box[0], y, box[0] + text_box[2] - text_box[0] + 6, box[1]), fill=(238, 72, 86))
        draw.text((box[0] + 3, y + 2), label, fill="white", font=font)
    return output


def _box_iou_xyxy(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    detections: list[dict[str, Any]], iou_threshold: float = 0.45
) -> list[dict[str, Any]]:
    """Merge overlapping full-frame/tile predictions without merging classes."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be between zero and one.")
    kept: list[dict[str, Any]] = []
    for candidate in sorted(detections, key=lambda item: float(item["confidence"]), reverse=True):
        if any(
            candidate["class"] == previous["class"]
            and _box_iou_xyxy(list(candidate["bbox"]), list(previous["bbox"])) >= iou_threshold
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be in [0, tile_size).")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def predict_aircraft_tiled(
    model: Any,
    image: Any,
    *,
    threshold: float | None = None,
    tile_size: int = 512,
    overlap: int = 128,
    nms_iou: float = 0.45,
    include_full_frame: bool = True,
) -> tuple[list[dict[str, Any]], float]:
    """Run overlapping high-resolution crops and map detections to source pixels."""
    pil_image = load_image(image)
    all_detections: list[dict[str, Any]] = []
    elapsed_ms = 0.0
    if include_full_frame:
        all_detections.extend(model.predict(pil_image, threshold=threshold))
        elapsed_ms += float(getattr(model, "last_inference_time_ms", 0.0))
    for top in _tile_starts(pil_image.height, tile_size, overlap):
        for left in _tile_starts(pil_image.width, tile_size, overlap):
            if include_full_frame and left == 0 and top == 0 and pil_image.size == (
                min(tile_size, pil_image.width), min(tile_size, pil_image.height)
            ):
                continue
            right, bottom = min(left + tile_size, pil_image.width), min(top + tile_size, pil_image.height)
            tile = pil_image.crop((left, top, right, bottom))
            for detection in model.predict(tile, threshold=threshold):
                x1, y1, x2, y2 = map(float, detection["bbox"])
                all_detections.append(
                    {
                        **detection,
                        "bbox": [x1 + left, y1 + top, x2 + left, y2 + top],
                        "inference_region": "tile",
                    }
                )
            elapsed_ms += float(getattr(model, "last_inference_time_ms", 0.0))
    return class_aware_nms(all_detections, nms_iou), elapsed_ms


def inspect_aircraft(
    image: Any,
    model: Any,
    *,
    config: Mapping[str, Any] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    pil_image = load_image(image)
    quality = _quality(pil_image, config)
    if not quality["usable"]:
        raise ImageValidationError("Image is too small to analyze reliably.")
    if model is None or not hasattr(model, "predict"):
        raise ModelNotReadyError("A trained transformer-detector checkpoint is required for aircraft inspection.")
    start = time.perf_counter()
    tile_config = (config or {}).get("aircraft", {}).get("tiled_inference", {})
    tiled = bool(tile_config.get("enabled", False)) and max(pil_image.size) >= int(
        tile_config.get("min_image_size", 768)
    )
    if tiled:
        detections, elapsed = predict_aircraft_tiled(
            model,
            pil_image,
            threshold=threshold,
            tile_size=int(tile_config.get("tile_size", 512)),
            overlap=int(tile_config.get("overlap", 128)),
            nms_iou=float(tile_config.get("nms_iou", 0.45)),
        )
    else:
        detections = list(model.predict(pil_image, threshold=threshold))
        elapsed = float(
            getattr(model, "last_inference_time_ms", (time.perf_counter() - start) * 1000)
        )
    result = {
        "inspection_type": "aircraft",
        "status": "defect_detected" if detections else "no_defect_detected",
        "defect_count": len(detections),
        "defects": detections,
        "quality": quality,
        "model_name": str(getattr(model, "model_name", "deformable_detr")),
        "model_version": str(getattr(model, "version", "unknown")),
        "inference_time_ms": round(elapsed, 3),
        "tiled_inference": tiled,
    }
    result["annotated_image"] = annotate_aircraft(pil_image, detections)
    return result

def create_engine_heatmap(
    image: Any, anomaly_map: np.ndarray, pixel_threshold: float
) -> tuple[Image.Image, Image.Image, Image.Image]:
    original = load_image(image)
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    if anomaly_map.ndim != 2 or anomaly_map.size == 0 or not np.isfinite(anomaly_map).all():
        raise ValueError("Anomaly map must be a finite, non-empty 2D array.")
    resized = cv2.resize(anomaly_map, original.size, interpolation=cv2.INTER_LINEAR)
    minimum, maximum = float(resized.min()), float(resized.max())
    normalized = np.zeros_like(resized) if maximum <= minimum else (resized - minimum) / (maximum - minimum)
    colored = cv2.applyColorMap(np.uint8(np.clip(normalized * 255, 0, 255)), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    original_array = np.asarray(original)
    overlay = cv2.addWeighted(original_array, 0.58, colored, 0.42, 0)
    mask = np.uint8(resized >= float(pixel_threshold)) * 255
    return Image.fromarray(colored), Image.fromarray(overlay), Image.fromarray(mask, mode="L")


def inspect_engine(
    image: Any,
    model: Any,
    *,
    config: Mapping[str, Any] | None = None,
    anomaly_threshold: float | None = None,
    pixel_threshold: float | None = None,
) -> dict[str, Any]:
    pil_image = load_image(image)
    quality = _quality(pil_image, config)
    if not quality["usable"]:
        raise ImageValidationError("Image is too small to analyze reliably.")
    if model is None or not (hasattr(model, "predict") or hasattr(model, "predict_image")):
        raise ModelNotReadyError("A trained MMR checkpoint is required for engine inspection.")
    model_threshold = getattr(model, "anomaly_threshold", None)
    model_pixel_threshold = getattr(model, "pixel_threshold", None)
    threshold = anomaly_threshold if anomaly_threshold is not None else model_threshold
    map_threshold = pixel_threshold if pixel_threshold is not None else model_pixel_threshold
    if threshold is None or map_threshold is None:
        raise ModelNotReadyError(
            "MMR checkpoint is missing validation-derived anomaly and pixel thresholds; final test data must not be used."
        )
    prediction = model.predict_image(pil_image) if hasattr(model, "predict_image") else model.predict(pil_image)
    score = float(prediction["anomaly_score"])
    anomaly_map = np.asarray(prediction["anomaly_map"], dtype=np.float32)
    if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
        raise ValueError("MMR returned an invalid anomaly map.")
    affected = float(np.count_nonzero(anomaly_map >= float(map_threshold)) / anomaly_map.size)
    heatmap, overlay, mask = create_engine_heatmap(pil_image, anomaly_map, float(map_threshold))
    result = {
        "inspection_type": "engine",
        "status": "anomalous" if score >= float(threshold) else "normal",
        "anomaly_score": score,
        "affected_visible_area": affected,
        "quality": quality,
        "model_name": str(getattr(model, "model_name", "mmr")),
        "model_version": str(getattr(model, "version", "unknown")),
        "inference_time_ms": round(float(prediction.get("inference_time_ms", 0.0)), 3),
        "heatmap": heatmap,
        "overlay": overlay,
        "anomaly_mask": mask,
    }
    return result
