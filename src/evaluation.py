"""Truthful object-detection and anomaly-detection evaluation utilities."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def bbox_iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0 else 0.0


def detection_prf(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, list[int]]:
    """Greedy, score-ordered class-aware matching at one explicit IoU threshold."""
    category_ids = sorted(
        {int(item["category_id"]) for item in ground_truth}
        | {int(item["category_id"]) for item in predictions}
    )
    index = {category: offset for offset, category in enumerate(category_ids)}
    background = len(category_ids)
    matrix = np.zeros((background + 1, background + 1), dtype=int)
    gt_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_image[int(item["image_id"])].append(item)
    for item in predictions:
        pred_by_image[int(item["image_id"])].append(item)
    tp = fp = fn = 0
    for image_id in sorted(set(gt_by_image) | set(pred_by_image)):
        truths = gt_by_image[image_id]
        used: set[int] = set()
        for prediction in sorted(pred_by_image[image_id], key=lambda item: float(item["score"]), reverse=True):
            best_index, best_iou = None, 0.0
            for truth_index, truth in enumerate(truths):
                if truth_index in used:
                    continue
                overlap = bbox_iou_xywh(prediction["bbox"], truth["bbox"])
                if overlap > best_iou:
                    best_index, best_iou = truth_index, overlap
            predicted_category = int(prediction["category_id"])
            if best_index is not None and best_iou >= iou_threshold:
                truth_category = int(truths[best_index]["category_id"])
                used.add(best_index)
                matrix[index[truth_category], index[predicted_category]] += 1
                if predicted_category == truth_category:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                matrix[background, index[predicted_category]] += 1
                fp += 1
        for truth_index, truth in enumerate(truths):
            if truth_index not in used:
                matrix[index[int(truth["category_id"])], background] += 1
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}, matrix, category_ids


def _plot_confusion(matrix: np.ndarray, names: Sequence[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(6, len(names)), max(5, len(names))))
    rendered = axis.imshow(matrix, cmap="Blues")
    labels = [*names, "background"]
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Detection confusion matrix at IoU 0.50")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(rendered, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_aircraft_predictions(
    coco_ground_truth: str | Path,
    predictions: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    latencies_ms: Sequence[float] = (),
) -> dict[str, Any]:
    """Run COCO AP plus explicit IoU=.50 P/R/F1 and serialize all generated results."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(coco_ground_truth).read_text(encoding="utf-8"))
    ground_truth = payload["annotations"]
    categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    normalized_predictions = [
        {
            "image_id": int(item["image_id"]),
            "category_id": int(item["category_id"]),
            "bbox": [float(value) for value in item["bbox"]],
            "score": float(item["score"]),
        }
        for item in predictions
    ]
    prediction_dir = output_dir / "aircraft_predictions"
    prediction_dir.mkdir(exist_ok=True)
    (prediction_dir / "predictions.json").write_text(
        json.dumps(normalized_predictions, indent=2), encoding="utf-8"
    )

    map_50_95 = map_50 = 0.0
    per_class = {item["name"]: 0.0 for item in categories}
    if normalized_predictions and ground_truth:
        coco_gt = COCO(str(coco_ground_truth))
        coco_dt = coco_gt.loadRes(normalized_predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        map_50_95, map_50 = float(evaluator.stats[0]), float(evaluator.stats[1])
        precision = evaluator.eval["precision"]  # IoU x recall x class x area x maxDet
        for class_offset, category in enumerate(categories):
            values = precision[:, :, class_offset, 0, -1]
            values = values[values > -1]
            per_class[str(category["name"])] = float(values.mean()) if values.size else 0.0

    prf, matrix, matrix_ids = detection_prf(ground_truth, normalized_predictions)
    names = [category_names.get(category, str(category)) for category in matrix_ids]
    _plot_confusion(matrix, names, output_dir / "aircraft_confusion_matrix.png")
    with (output_dir / "aircraft_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "ap_50_95"])
        writer.writeheader()
        writer.writerows({"class": name, "ap_50_95": value} for name, value in per_class.items())
    metrics = {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "counts": {key: prf[key] for key in ("tp", "fp", "fn")},
        "ap_per_class": per_class,
        "inference_latency_ms": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
            "samples": len(latencies_ms),
        },
    }
    (output_dir / "aircraft_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(labels_array) != len(scores_array) or not len(labels_array):
        raise ValueError("Labels and scores must be non-empty and have equal lengths.")
    if not np.isfinite(scores_array).all():
        raise ValueError("Anomaly scores contain NaN or infinite values.")
    if len(np.unique(labels_array)) < 2:
        return None
    return float(roc_auc_score(labels_array, scores_array))


def compute_aupro(
    masks: np.ndarray, anomaly_maps: np.ndarray, *, max_fpr: float = 0.30, thresholds: int = 200
) -> float | None:
    """Area under per-region overlap versus FPR, normalized over [0, max_fpr]."""
    masks = np.asarray(masks).astype(bool)
    anomaly_maps = np.asarray(anomaly_maps, dtype=float)
    if masks.shape != anomaly_maps.shape or masks.ndim != 3:
        raise ValueError("Masks and anomaly maps must have identical [N,H,W] shapes.")
    if not masks.any() or (~masks).sum() == 0:
        return None
    threshold_values = np.linspace(float(anomaly_maps.max()), float(anomaly_maps.min()), thresholds)
    curve: list[tuple[float, float]] = []
    for threshold in threshold_values:
        predicted = anomaly_maps >= threshold
        fpr = float(np.logical_and(predicted, ~masks).sum() / (~masks).sum())
        if fpr > max_fpr:
            continue
        overlaps = []
        for sample_mask, sample_prediction in zip(masks, predicted):
            component_count, components = cv2.connectedComponents(sample_mask.astype(np.uint8))
            for component in range(1, component_count):
                region = components == component
                overlaps.append(float(np.logical_and(region, sample_prediction).sum() / region.sum()))
        if overlaps:
            curve.append((fpr, float(np.mean(overlaps))))
    if len(curve) < 2:
        return None
    curve.sort()
    fpr_values, pro_values = map(np.asarray, zip(*curve))
    unique_fpr, unique_indices = np.unique(fpr_values, return_index=True)
    return float(np.trapezoid(pro_values[unique_indices], unique_fpr) / max_fpr)


def evaluate_engine_predictions(
    labels: Sequence[int],
    scores: Sequence[float],
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    domains: Sequence[str],
    output_dir: str | Path,
    *,
    latencies_ms: Sequence[float] = (),
    pixel_threshold: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_array = np.asarray(labels, dtype=int)
    masks_array = np.asarray(masks).astype(bool)
    maps_array = np.asarray(anomaly_maps, dtype=float)
    image_auroc = safe_auroc(labels, scores)
    pixel_auroc = safe_auroc(masks_array.reshape(-1).astype(int), maps_array.reshape(-1))
    aupro = compute_aupro(masks_array, maps_array)
    dice = iou = None
    if pixel_threshold is not None:
        predicted = maps_array >= pixel_threshold
        intersection = np.logical_and(predicted, masks_array).sum()
        dice_denominator = predicted.sum() + masks_array.sum()
        union = np.logical_or(predicted, masks_array).sum()
        dice = float(2 * intersection / dice_denominator) if dice_denominator else None
        iou = float(intersection / union) if union else None
    metrics = {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "dice": dice,
        "iou": iou,
        "inference_latency_ms": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else None,
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
            "samples": len(latencies_ms),
        },
    }
    (output_dir / "engine_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    examples = output_dir / "engine_examples"
    examples.mkdir(exist_ok=True)
    ranked = np.argsort(np.asarray(scores, dtype=float))[::-1][: min(5, len(scores))]
    for rank, index in enumerate(ranked, start=1):
        anomaly_map = maps_array[index]
        minimum, maximum = float(anomaly_map.min()), float(anomaly_map.max())
        normalized = (
            np.zeros_like(anomaly_map)
            if maximum <= minimum
            else (anomaly_map - minimum) / (maximum - minimum)
        )
        colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
        cv2.imwrite(
            str(examples / f"rank_{rank:02d}_label_{int(labels_array[index])}_heatmap.png"), colored
        )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(domains):
        grouped[str(domain)].append(index)
    rows = []
    for domain, indices in sorted(grouped.items()):
        rows.append(
            {
                "domain": domain,
                "samples": len(indices),
                "image_auroc": safe_auroc(labels_array[indices], np.asarray(scores)[indices]),
            }
        )
    with (output_dir / "engine_domain_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "samples", "image_auroc"])
        writer.writeheader()
        writer.writerows(rows)
    return metrics
