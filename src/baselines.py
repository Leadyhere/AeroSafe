"""Reproducible Faster R-CNN and PatchCore comparison baselines."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def select_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FasterRCNNDataset:
    """COCO adapter for torchvision detectors with one-based model labels."""

    def __init__(self, annotation_file: str | Path, *, train: bool = False):
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.images = sorted(payload["images"], key=lambda item: int(item["id"]))
        self.annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload["annotations"]:
            self.annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
        self.category_ids = [int(item["id"]) for item in categories]
        self.category_to_label = {
            category_id: offset + 1 for offset, category_id in enumerate(self.category_ids)
        }
        self.train = train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from torchvision.transforms import functional

        from .preprocessing import load_image

        item = self.images[index]
        image = load_image(item["file_name"])
        boxes, labels, areas, crowds = [], [], [], []
        for annotation in self.annotations.get(int(item["id"]), []):
            x, y, width, height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(image.width), x + width), min(float(image.height), y + height)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.category_to_label[int(annotation["category_id"])])
            areas.append((x2 - x1) * (y2 - y1))
            crowds.append(int(annotation.get("iscrowd", 0)))
        image_tensor = functional.pil_to_tensor(image).float().div(255)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and torch.rand(()) < 0.5:
            image_tensor = functional.hflip(image_tensor)
            if boxes_tensor.numel():
                old_x1 = boxes_tensor[:, 0].clone()
                old_x2 = boxes_tensor[:, 2].clone()
                boxes_tensor[:, 0] = image.width - old_x2
                boxes_tensor[:, 2] = image.width - old_x1
        target = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(item["id"]), dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def collate_faster_rcnn(batch):
    return tuple(zip(*batch))


class FasterRCNNBaseline:
    model_name = "faster_rcnn_resnet50_fpn"

    def __init__(
        self,
        labels: Sequence[str],
        *,
        pretrained: bool = True,
        device: Any = None,
        confidence_threshold: float = 0.5,
    ):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        if not labels or len(labels) != len(set(labels)):
            raise ValueError("Faster R-CNN labels must be non-empty and unique.")
        self.labels = list(labels)
        self.label2id = {label: index for index, label in enumerate(self.labels)}
        self.device = device or select_device()
        self.confidence_threshold = float(confidence_threshold)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model_options = {"weights": weights}
        if not pretrained:
            model_options["weights_backbone"] = None
        self.model = fasterrcnn_resnet50_fpn(**model_options)
        input_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(input_features, len(labels) + 1)
        self.model.to(self.device)
        self.metadata: dict[str, Any] = {}
        self.version = "untrained-transfer-head"
        self.last_inference_time_ms = 0.0

    def predict(self, image: Any, threshold: float | None = None) -> list[dict[str, Any]]:
        import torch
        from torchvision.transforms import functional

        from .preprocessing import load_image

        image = load_image(image)
        threshold = self.confidence_threshold if threshold is None else float(threshold)
        tensor = functional.pil_to_tensor(image).float().div(255).to(self.device)
        self.model.eval()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model([tensor])[0]
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000
        detections = []
        for score, label, box in zip(output["scores"], output["labels"], output["boxes"]):
            score_value, model_label = float(score.item()), int(label.item())
            if score_value < threshold or not 1 <= model_label <= len(self.labels):
                continue
            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append(
                {
                    "class": self.labels[model_label - 1],
                    "confidence": score_value,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections

    def save(self, path: str | Path, metadata: Mapping[str, Any]) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "labels": self.labels,
            "confidence_threshold": self.confidence_threshold,
            **dict(metadata),
        }
        torch.save({"state_dict": self.model.state_dict(), "metadata": payload}, path)
        self.metadata = payload

    @classmethod
    def load(cls, path: str | Path, *, device: Any = None):
        import torch

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Faster R-CNN checkpoint not found: {path}")
        target_device = device or select_device()
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        metadata = checkpoint["metadata"]
        instance = cls(
            metadata["labels"],
            pretrained=False,
            device=target_device,
            confidence_threshold=float(metadata.get("confidence_threshold", 0.5)),
        )
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.metadata = metadata
        instance.version = str(metadata.get("version", path.stem))
        instance.model.eval()
        return instance


class PatchCoreBaseline:
    """PatchCore-style memory-bank baseline with greedy coreset sampling."""

    model_name = "patchcore_wide_resnet50_2"

    def __init__(
        self,
        *,
        image_size: int = 224,
        coreset_fraction: float = 0.01,
        max_patches: int = 50000,
        projection_dim: int = 128,
        pretrained: bool = True,
        device: Any = None,
    ):
        import torch
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        from torchvision.models.feature_extraction import create_feature_extractor

        if not 0 < coreset_fraction <= 1:
            raise ValueError("coreset_fraction must be in (0, 1].")
        if max_patches < 1 or projection_dim < 1:
            raise ValueError("max_patches and projection_dim must be positive.")
        self.image_size = int(image_size)
        self.coreset_fraction = float(coreset_fraction)
        self.max_patches = int(max_patches)
        self.projection_dim = int(projection_dim)
        self.device = device or select_device()
        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        backbone = wide_resnet50_2(weights=weights)
        self.extractor = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(self.device).eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad = False
        self.memory_bank: torch.Tensor | None = None
        self.metadata: dict[str, Any] = {}
        self.anomaly_threshold: float | None = None
        self.pixel_threshold: float | None = None

    def _embeddings(self, images):
        import torch
        from torch.nn import functional

        with torch.inference_mode():
            features = self.extractor(images.to(self.device))
            middle, deep = features["layer2"], features["layer3"]
            deep = functional.interpolate(deep, middle.shape[-2:], mode="bilinear", align_corners=False)
            combined = torch.cat([middle, deep], dim=1)
            # Local average pooling makes the baseline less sensitive to one-pixel noise.
            combined = functional.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)
        return combined

    @staticmethod
    def _greedy_coreset(embeddings, count: int, projection_dim: int):
        import torch

        if count >= len(embeddings):
            return embeddings
        generator = torch.Generator(device="cpu").manual_seed(42)
        projection = torch.randn(
            embeddings.shape[1], min(projection_dim, embeddings.shape[1]), generator=generator
        ) / math.sqrt(min(projection_dim, embeddings.shape[1]))
        reduced = embeddings.float().cpu() @ projection
        selected = [0]
        minimum_distance = torch.full((len(reduced),), float("inf"))
        for _ in range(1, count):
            distance = torch.sum((reduced - reduced[selected[-1]]) ** 2, dim=1)
            minimum_distance = torch.minimum(minimum_distance, distance)
            selected.append(int(torch.argmax(minimum_distance).item()))
        return embeddings[torch.as_tensor(selected)]

    def fit(self, loader, *, smoke: bool = False) -> None:
        import torch

        collected = []
        for batch_index, batch in enumerate(loader):
            features = self._embeddings(batch["image"])
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu()
            collected.append(patches)
            # Bound host RAM before the final concatenation on large AeBAD-S inputs.
            if sum(len(item) for item in collected) > self.max_patches:
                bounded = torch.cat(collected)
                indices = torch.linspace(0, len(bounded) - 1, self.max_patches).long()
                collected = [bounded[indices]]
            if smoke and batch_index == 0:
                break
        if not collected:
            raise RuntimeError("PatchCore received no normal training images.")
        embeddings = torch.cat(collected)
        if len(embeddings) > self.max_patches:
            indices = torch.linspace(0, len(embeddings) - 1, self.max_patches).long()
            embeddings = embeddings[indices]
        target = max(1, round(len(embeddings) * self.coreset_fraction))
        self.memory_bank = self._greedy_coreset(embeddings, target, self.projection_dim).contiguous()

    def _nearest_distances(self, queries, *, chunk_size: int = 2048):
        import torch

        if self.memory_bank is None or not len(self.memory_bank):
            raise RuntimeError("PatchCore memory bank is empty. Fit the baseline first.")
        memory = self.memory_bank.to(self.device)
        results = []
        for start in range(0, len(queries), chunk_size):
            distances = torch.cdist(queries[start : start + chunk_size], memory)
            results.append(distances.min(dim=1).values)
        return torch.cat(results)

    def predict_tensor(self, images) -> list[dict[str, Any]]:
        from scipy.ndimage import gaussian_filter
        from torch.nn import functional

        start = time.perf_counter()
        features = self._embeddings(images)
        batch, channels, height, width = features.shape
        queries = features.permute(0, 2, 3, 1).reshape(-1, channels)
        distances = self._nearest_distances(queries).reshape(batch, 1, height, width)
        maps = functional.interpolate(
            distances, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        ).squeeze(1)
        elapsed = (time.perf_counter() - start) * 1000 / max(1, batch)
        results = []
        for anomaly_map in maps.detach().cpu().numpy():
            smoothed = gaussian_filter(anomaly_map.astype(np.float32), sigma=4)
            results.append(
                {
                    "anomaly_score": float(np.max(smoothed)),
                    "anomaly_map": smoothed,
                    "inference_time_ms": elapsed,
                }
            )
        return results

    def save(self, path: str | Path, metadata: Mapping[str, Any]) -> None:
        import torch

        if self.memory_bank is None:
            raise RuntimeError("Cannot save PatchCore before fitting its memory bank.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "coreset_fraction": self.coreset_fraction,
            "max_patches": self.max_patches,
            "projection_dim": self.projection_dim,
            **dict(metadata),
        }
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "extractor_state_dict": self.extractor.state_dict(),
                "metadata": payload,
            },
            path,
        )
        self.metadata = payload
        self.anomaly_threshold = payload.get("anomaly_threshold")
        self.pixel_threshold = payload.get("pixel_threshold")

    @classmethod
    def load(cls, path: str | Path, *, device: Any = None):
        import torch

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PatchCore checkpoint not found: {path}")
        target_device = device or select_device()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        instance = cls(
            image_size=int(metadata["image_size"]),
            coreset_fraction=float(metadata["coreset_fraction"]),
            max_patches=int(metadata["max_patches"]),
            projection_dim=int(metadata["projection_dim"]),
            pretrained=False,
            device=target_device,
        )
        if "extractor_state_dict" not in checkpoint:
            raise RuntimeError("PatchCore checkpoint is incomplete: frozen extractor weights are missing.")
        instance.extractor.load_state_dict(checkpoint["extractor_state_dict"])
        instance.memory_bank = checkpoint["memory_bank"].cpu()
        instance.metadata = metadata
        instance.anomaly_threshold = metadata.get("anomaly_threshold")
        instance.pixel_threshold = metadata.get("pixel_threshold")
        return instance
