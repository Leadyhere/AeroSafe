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
from sklearn.metrics import average_precision_score, roc_auc_score


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


def select_detection_threshold(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Select a deployment threshold using validation labels only."""
    scores = np.asarray([float(item["score"]) for item in predictions], dtype=float)
    if not len(scores):
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, min(101, len(scores))))
    candidates = sorted(set(np.clip(np.r_[0.05, quantiles, 0.95], 0.0, 1.0).tolist()))
    best: dict[str, float] | None = None
    for threshold in candidates:
        filtered = [item for item in predictions if float(item["score"]) >= threshold]
        metrics, _, _ = detection_prf(
            ground_truth, filtered, iou_threshold=iou_threshold
        )
        candidate = {"threshold": float(threshold), **metrics}
        # Prefer F1, then recall (missed defects are costly), then the higher threshold.
        key = (candidate["f1"], candidate["recall"], candidate["threshold"])
        if best is None or key > (best["f1"], best["recall"], best["threshold"]):
            best = candidate
    assert best is not None
    return {key: float(best[key]) for key in ("threshold", "precision", "recall", "f1")}


def detection_calibration_error(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    bins: int = 10,
) -> float | None:
    """Expected calibration error where correctness means class-aware IoU matching."""
    if bins < 2:
        raise ValueError("Calibration bins must be at least two.")
    if not predictions:
        return None
    gt_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in ground_truth:
        gt_by_image[int(item["image_id"])].append(item)
    for item in predictions:
        pred_by_image[int(item["image_id"])].append(item)
    outcomes: list[tuple[float, int]] = []
    for image_id, image_predictions in pred_by_image.items():
        truths = gt_by_image.get(image_id, [])
        used: set[int] = set()
        for prediction in sorted(image_predictions, key=lambda item: float(item["score"]), reverse=True):
            matches = [
                (bbox_iou_xywh(prediction["bbox"], truth["bbox"]), index)
                for index, truth in enumerate(truths)
                if index not in used
                and int(truth["category_id"]) == int(prediction["category_id"])
            ]
            best_iou, best_index = max(matches, default=(0.0, -1))
            correct = int(best_iou >= iou_threshold)
            if correct:
                used.add(best_index)
            outcomes.append((float(prediction["score"]), correct))
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        members = [
            item for item in outcomes
            if boundaries[index] <= item[0] <= boundaries[index + 1]
            and (index == bins - 1 or item[0] < boundaries[index + 1])
        ]
        if members:
            confidence = float(np.mean([item[0] for item in members]))
            accuracy = float(np.mean([item[1] for item in members]))
            error += len(members) / len(outcomes) * abs(confidence - accuracy)
    return float(error)


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
    confidence_threshold: float | None = None,
    calibrate_threshold: bool = False,
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

    map_50_95 = map_50 = map_small = map_medium = map_large = 0.0
    per_class = {item["name"]: 0.0 for item in categories}
    if normalized_predictions and ground_truth:
        coco_gt = COCO(str(coco_ground_truth))
        coco_dt = coco_gt.loadRes(normalized_predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        map_50_95, map_50 = float(evaluator.stats[0]), float(evaluator.stats[1])
        map_small, map_medium, map_large = map(float, evaluator.stats[3:6])
        precision = evaluator.eval["precision"]  # IoU x recall x class x area x maxDet
        for class_offset, category in enumerate(categories):
            values = precision[:, :, class_offset, 0, -1]
            values = values[values > -1]
            per_class[str(category["name"])] = float(values.mean()) if values.size else 0.0

    calibration = (
        select_detection_threshold(ground_truth, normalized_predictions)
        if calibrate_threshold
        else None
    )
    selected_threshold = float(
        calibration["threshold"] if calibration is not None else (
            0.5 if confidence_threshold is None else confidence_threshold
        )
    )
    thresholded_predictions = [
        item for item in normalized_predictions if float(item["score"]) >= selected_threshold
    ]
    prf, matrix, matrix_ids = detection_prf(ground_truth, thresholded_predictions)
    names = [category_names.get(category, str(category)) for category in matrix_ids]
    _plot_confusion(matrix, names, output_dir / "aircraft_confusion_matrix.png")
    per_class_prf: dict[str, dict[str, float]] = {}
    for category in categories:
        category_id, name = int(category["id"]), str(category["name"])
        class_metrics, _, _ = detection_prf(
            [item for item in ground_truth if int(item["category_id"]) == category_id],
            [item for item in thresholded_predictions if int(item["category_id"]) == category_id],
        )
        positives = class_metrics["tp"] + class_metrics["fn"]
        per_class_prf[name] = {
            "precision": float(class_metrics["precision"]),
            "recall": float(class_metrics["recall"]),
            "missed_defect_rate": float(class_metrics["fn"] / positives) if positives else 0.0,
        }
    with (output_dir / "aircraft_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "ap_50_95", "precision", "recall", "missed_defect_rate"],
        )
        writer.writeheader()
        writer.writerows(
            {"class": name, "ap_50_95": value, **per_class_prf[name]}
            for name, value in per_class.items()
        )
    metrics = {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "map_small": map_small,
        "map_medium": map_medium,
        "map_large": map_large,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "missed_defect_rate": 1.0 - prf["recall"],
        "confidence_threshold": selected_threshold,
        "recommended_confidence_threshold": calibration["threshold"] if calibration else None,
        "expected_calibration_error": detection_calibration_error(
            ground_truth, thresholded_predictions
        ),
        "counts": {key: prf[key] for key in ("tp", "fp", "fn")},
        "ap_per_class": per_class,
        "threshold_metrics_per_class": per_class_prf,
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


def safe_average_precision(
    labels: Sequence[int], scores: Sequence[float]
) -> float | None:
    """Return anomaly average precision without inventing a one-class result."""
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(labels_array) != len(scores_array) or not len(labels_array):
        raise ValueError("Labels and scores must be non-empty and have equal lengths.")
    if not np.isfinite(scores_array).all():
        raise ValueError("Anomaly scores contain NaN or infinite values.")
    if len(np.unique(labels_array)) < 2:
        return None
    return float(average_precision_score(labels_array, scores_array))


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
    anomaly_threshold: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_array = np.asarray(labels, dtype=int)
    masks_array = np.asarray(masks).astype(bool)
    maps_array = np.asarray(anomaly_maps, dtype=float)
    image_auroc = safe_auroc(labels, scores)
    image_average_precision = safe_average_precision(labels, scores)
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
    decision_metrics: dict[str, Any] = {
        "anomaly_threshold": anomaly_threshold,
        "false_positives": None,
        "false_negatives": None,
        "false_alarm_rate": None,
        "missed_anomaly_rate": None,
        "sensitivity": None,
        "specificity": None,
        "balanced_accuracy": None,
    }
    if anomaly_threshold is not None:
        decisions = np.asarray(scores, dtype=float) >= float(anomaly_threshold)
        positives = labels_array == 1
        negatives = ~positives
        tp = int(np.logical_and(decisions, positives).sum())
        fp = int(np.logical_and(decisions, negatives).sum())
        fn = int(np.logical_and(~decisions, positives).sum())
        tn = int(np.logical_and(~decisions, negatives).sum())
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        decision_metrics.update(
            {
                "false_positives": fp,
                "false_negatives": fn,
                "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
                "missed_anomaly_rate": fn / (fn + tp) if fn + tp else None,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": (
                    (sensitivity + specificity) / 2
                    if sensitivity is not None and specificity is not None else None
                ),
            }
        )
    metrics = {
        "image_auroc": image_auroc,
        "image_average_precision": image_average_precision,
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "dice": dice,
        "iou": iou,
        "decision_metrics": decision_metrics,
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
        domain_labels = labels_array[indices]
        domain_scores = np.asarray(scores)[indices]
        domain_decisions = (
            domain_scores >= float(anomaly_threshold) if anomaly_threshold is not None else None
        )
        domain_negatives = domain_labels == 0
        domain_positives = domain_labels == 1
        rows.append(
            {
                "domain": domain,
                "samples": len(indices),
                "image_auroc": safe_auroc(domain_labels, domain_scores),
                "false_alarm_rate": (
                    float(np.logical_and(domain_decisions, domain_negatives).sum() / domain_negatives.sum())
                    if domain_decisions is not None and domain_negatives.any() else None
                ),
                "missed_anomaly_rate": (
                    float(np.logical_and(~domain_decisions, domain_positives).sum() / domain_positives.sum())
                    if domain_decisions is not None and domain_positives.any() else None
                ),
            }
        )
    with (output_dir / "engine_domain_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "samples", "image_auroc", "false_alarm_rate", "missed_anomaly_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return metrics
