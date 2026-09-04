"""Shared Hugging Face transformer wrapper for aircraft defect detection."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AircraftDetector:
    """Common wrapper for Hugging Face transformer object detectors."""

    model_name = "deformable_detr"

    def __init__(
        self,
        labels: Sequence[str],
        *,
        pretrained_model: str = "SenseTime/deformable-detr",
        architecture: str = "deformable_detr",
        image_size: int | None = None,
        device: Any = None,
        confidence_threshold: float = 0.5,
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Aircraft detector labels must be a non-empty unique sequence.")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one.")
        self.labels = list(labels)
        self.id2label = {index: label for index, label in enumerate(self.labels)}
        self.label2id = {label: index for index, label in self.id2label.items()}
        self.pretrained_model = pretrained_model
        self.architecture = str(architecture)
        self.model_name = self.architecture
        self.device = device or select_device()
        self.confidence_threshold = float(confidence_threshold)
        processor_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if image_size is not None:
            if self.architecture.startswith("rt_detr"):
                processor_kwargs["size"] = {"height": int(image_size), "width": int(image_size)}
            else:
                processor_kwargs["size"] = {
                    "shortest_edge": int(image_size),
                    "longest_edge": int(image_size),
                }
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model, **processor_kwargs)
        self.model = AutoModelForObjectDetection.from_pretrained(
            pretrained_model,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        ).to(self.device)
        self.version = "untrained-transfer-head"
        self.metadata: dict[str, Any] = {}

    def training_forward(
        self, pixel_values: Any, pixel_mask: Any | None, labels: list[dict[str, Any]]
    ):
        arguments = {"pixel_values": pixel_values, "labels": labels}
        if pixel_mask is not None and "pixel_mask" in inspect.signature(self.model.forward).parameters:
            arguments["pixel_mask"] = pixel_mask
        return self.model(**arguments)

    def predict(self, image: Any, threshold: float | None = None) -> list[dict[str, Any]]:
        import torch

        from .preprocessing import load_image

        image = load_image(image)
        threshold = self.confidence_threshold if threshold is None else float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("Detection threshold must be between zero and one.")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.model.eval()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        self.last_inference_time_ms = (time.perf_counter() - start) * 1000
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        decoded = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        detections: list[dict[str, Any]] = []
        for score, label, box in zip(decoded["scores"], decoded["labels"], decoded["boxes"]):
            label_id = int(label.item())
            if label_id not in self.id2label:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            x1, x2 = sorted((max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))))
            y1, y2 = sorted((max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {"class": self.id2label[label_id], "confidence": float(score.item()), "bbox": [x1, y1, x2, y2]}
            )
        return sorted(detections, key=lambda item: item["confidence"], reverse=True)

    def save(self, directory: str | Path, metadata: Mapping[str, Any]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        payload = {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "base_checkpoint": self.pretrained_model,
            "labels": self.labels,
            "confidence_threshold": self.confidence_threshold,
            **dict(metadata),
        }
        (directory / "aeroinspect_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.metadata = payload
        self.version = str(payload.get("version", directory.name))

    @classmethod
    def load(cls, directory: str | Path, *, device: Any = None) -> AircraftDetector:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        directory = Path(directory)
        metadata_path = directory / "aeroinspect_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Aircraft checkpoint metadata not found: {metadata_path}. Train the detector before inference."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        instance = cls.__new__(cls)
        instance.labels = list(metadata["labels"])
        instance.id2label = {index: label for index, label in enumerate(instance.labels)}
        instance.label2id = {label: index for index, label in instance.id2label.items()}
        instance.pretrained_model = str(metadata.get("base_checkpoint", directory))
        instance.architecture = str(metadata.get("architecture", metadata.get("model_name", "deformable_detr")))
        instance.model_name = instance.architecture
        instance.device = device or select_device()
        instance.confidence_threshold = float(metadata.get("confidence_threshold", 0.5))
        instance.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
        instance.model = AutoModelForObjectDetection.from_pretrained(
            directory, local_files_only=True
        ).to(instance.device)
        instance.metadata = metadata
        instance.version = str(metadata.get("version", directory.name))
        return instance


def collate_detection_batch(batch: Sequence[Mapping[str, Any]], processor: Any) -> dict[str, Any]:
    import torch

    images = [item["pixel_values"] for item in batch]
    if images and all(image.shape == images[0].shape for image in images):
        padded: dict[str, Any] = {"pixel_values": torch.stack(images)}
        if "pixel_mask" in batch[0]:
            padded["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    elif hasattr(processor, "pad"):
        padded = processor.pad(images=images, return_tensors="pt")
    else:
        raise ValueError("The selected processor produced variable image sizes but cannot pad them.")
    labels = []
    for item in batch:
        labels.append(
            {key: value for key, value in item["labels"].items()}
        )
    return {
        "pixel_values": padded["pixel_values"],
        "pixel_mask": padded.get("pixel_mask"),
        "labels": labels,
    }
