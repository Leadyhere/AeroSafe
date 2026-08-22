"""Compact MMR implementation adapted from the official AeBAD MMR methodology.

Architecture: a trainable MAE-pretrained ViT receives completely removed random
patches, restores mask tokens, and reconstructs a multi-scale FPN. A frozen
pretrained hierarchical teacher supplies normal-image targets. Per-pixel cosine
feature discrepancy is the anomaly signal.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class SimpleFPN(nn.Module):
    """ViTDet-style simple FPN matching the official MMR reconstruction scales."""

    def __init__(self, embed_dim: int, output_channels: Sequence[int]):
        super().__init__()
        if len(output_channels) != 3:
            raise ValueError("MMR expects three teacher feature scales.")
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
                    ChannelLayerNorm(embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2),
                    nn.Conv2d(embed_dim // 4, output_channels[0], 1, bias=False),
                    ChannelLayerNorm(output_channels[0]),
                    nn.Conv2d(output_channels[0], output_channels[0], 3, padding=1, bias=False),
                    ChannelLayerNorm(output_channels[0]),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),
                    nn.Conv2d(embed_dim // 2, output_channels[1], 1, bias=False),
                    ChannelLayerNorm(output_channels[1]),
                    nn.Conv2d(output_channels[1], output_channels[1], 3, padding=1, bias=False),
                    ChannelLayerNorm(output_channels[1]),
                ),
                nn.Sequential(
                    nn.Conv2d(embed_dim, output_channels[2], 1, bias=False),
                    ChannelLayerNorm(output_channels[2]),
                    nn.Conv2d(output_channels[2], output_channels[2], 3, padding=1, bias=False),
                    ChannelLayerNorm(output_channels[2]),
                ),
            ]
        )

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        return [stage(feature) for stage in self.stages]


class MaskedMultiScaleReconstruction(nn.Module):
    model_name = "mmr"

    def __init__(
        self,
        *,
        image_size: int = 224,
        teacher_backbone: str = "wide_resnet50_2.tv2_in1k",
        mae_backbone: str = "vit_base_patch16_224.mae",
        pretrained: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        import timm

        self.image_size = int(image_size)
        self.teacher_backbone = teacher_backbone
        self.mae_backbone = mae_backbone
        self.teacher = timm.create_model(
            teacher_backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3)
        )
        teacher_channels = list(self.teacher.feature_info.channels())
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

        student = timm.create_model(
            mae_backbone, pretrained=pretrained, num_classes=0, img_size=self.image_size
        )
        self.patch_embed = student.patch_embed
        self.cls_token = student.cls_token
        self.pos_embed = student.pos_embed
        self.pos_drop = getattr(student, "pos_drop", nn.Identity())
        self.blocks = student.blocks
        self.norm = student.norm
        self.embed_dim = int(student.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.fpn = SimpleFPN(self.embed_dim, teacher_channels)
        self.device = device or self._select_device()
        self.to(self.device)
        self.metadata: dict[str, Any] = {}
        self.version = "untrained"
        self.anomaly_threshold: float | None = None
        self.pixel_threshold: float | None = None

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _position_embedding(self, grid_height: int, grid_width: int) -> torch.Tensor:
        position = self.pos_embed[:, 1:, :]
        original = int(math.sqrt(position.shape[1]))
        if original * original != position.shape[1]:
            raise RuntimeError("MAE positional embedding does not describe a square patch grid.")
        if (grid_height, grid_width) != (original, original):
            position = position.reshape(1, original, original, -1).permute(0, 3, 1, 2)
            position = F.interpolate(position, (grid_height, grid_width), mode="bicubic", align_corners=False)
            position = position.permute(0, 2, 3, 1).reshape(1, grid_height * grid_width, -1)
        return position

    @staticmethod
    def random_masking(tokens: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0, 1).")
        batch, length, channels = tokens.shape
        keep = max(1, int(length * (1 - mask_ratio)))
        noise = torch.rand(batch, length, device=tokens.device)
        shuffled = torch.argsort(noise, dim=1)
        restore = torch.argsort(shuffled, dim=1)
        keep_indices = shuffled[:, :keep]
        visible = torch.gather(tokens, 1, keep_indices.unsqueeze(-1).expand(-1, -1, channels))
        mask = torch.ones((batch, length), device=tokens.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, restore)
        return visible, mask, restore

    def reconstruct(self, images: torch.Tensor, mask_ratio: float) -> tuple[list[torch.Tensor], torch.Tensor]:
        tokens = self.patch_embed(images)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        grid_h = images.shape[-2] // self.patch_embed.patch_size[0]
        grid_w = images.shape[-1] // self.patch_embed.patch_size[1]
        tokens = tokens + self._position_embedding(grid_h, grid_w)
        visible, mask, restore = self.random_masking(tokens, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(images.shape[0], -1, -1)
        encoded = torch.cat([cls, visible], dim=1)
        encoded = self.pos_drop(encoded)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        missing = restore.shape[1] - (encoded.shape[1] - 1)
        mask_tokens = self.mask_token.expand(images.shape[0], missing, -1)
        restored = torch.cat([encoded[:, 1:], mask_tokens], dim=1)
        restored = torch.gather(restored, 1, restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        restored = restored + self._position_embedding(grid_h, grid_w)
        feature = restored.transpose(1, 2).reshape(images.shape[0], self.embed_dim, grid_h, grid_w)
        return self.fpn(feature), mask

    def teacher_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            return list(self.teacher(images))

    @staticmethod
    def reconstruction_loss(targets: Sequence[torch.Tensor], reconstructions: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(targets) != len(reconstructions):
            raise ValueError("Teacher and reconstructed feature scale counts differ.")
        loss = torch.zeros((), device=targets[0].device)
        for target, reconstruction in zip(targets, reconstructions):
            if reconstruction.shape[-2:] != target.shape[-2:]:
                reconstruction = F.interpolate(reconstruction, target.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + (1 - F.cosine_similarity(target, reconstruction, dim=1)).mean()
        return loss

    def forward(self, images: torch.Tensor, mask_ratio: float = 0.4) -> dict[str, Any]:
        targets = self.teacher_features(images)
        reconstructions, mask = self.reconstruct(images, mask_ratio)
        return {
            "loss": self.reconstruction_loss(targets, reconstructions),
            "teacher_features": targets,
            "reconstructed_features": reconstructions,
            "mask": mask,
        }

    @staticmethod
    def feature_anomaly_map(
        targets: Sequence[torch.Tensor], reconstructions: Sequence[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        maps = []
        for target, reconstruction in zip(targets, reconstructions):
            if reconstruction.shape[-2:] != target.shape[-2:]:
                reconstruction = F.interpolate(reconstruction, target.shape[-2:], mode="bilinear", align_corners=False)
            discrepancy = 1 - F.cosine_similarity(target, reconstruction, dim=1).unsqueeze(1)
            maps.append(F.interpolate(discrepancy, output_size, mode="bilinear", align_corners=False))
        return torch.stack(maps).sum(dim=0).squeeze(1)

    def anomaly_map(self, images: torch.Tensor, *, mask_ratio: float = 0.0, passes: int = 1) -> torch.Tensor:
        if passes < 1:
            raise ValueError("passes must be at least one.")
        self.eval()
        accumulated = None
        with torch.inference_mode():
            targets = self.teacher_features(images)
            for _ in range(passes):
                reconstructions, _ = self.reconstruct(images, mask_ratio)
                current = self.feature_anomaly_map(targets, reconstructions, images.shape[-2:])
                accumulated = current if accumulated is None else accumulated + current
        return accumulated / passes

    def predict_tensor(self, images: torch.Tensor, *, mask_ratio: float = 0.0, passes: int = 1) -> list[dict[str, Any]]:
        from scipy.ndimage import gaussian_filter

        start = time.perf_counter()
        maps = self.anomaly_map(images.to(self.device), mask_ratio=mask_ratio, passes=passes)
        elapsed = (time.perf_counter() - start) * 1000 / max(1, len(images))
        results = []
        for anomaly_map in maps.detach().cpu().numpy():
            smoothed = gaussian_filter(anomaly_map.astype(np.float32), sigma=4)
            results.append(
                {"anomaly_score": float(np.max(smoothed)), "anomaly_map": smoothed, "inference_time_ms": elapsed}
            )
        return results

    def predict_image(self, image: Any, *, mask_ratio: float = 0.0, passes: int = 1) -> dict[str, Any]:
        from torchvision.transforms import v2

        from .preprocessing import load_image

        transform = v2.Compose(
            [
                v2.Resize((self.image_size, self.image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        tensor = transform(load_image(image)).unsqueeze(0)
        return self.predict_tensor(tensor, mask_ratio=mask_ratio, passes=passes)[0]

    def save_checkpoint(self, path: str | Path, metadata: Mapping[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_metadata = {
            "model_name": self.model_name,
            "teacher_backbone": self.teacher_backbone,
            "mae_backbone": self.mae_backbone,
            "image_size": self.image_size,
            **dict(metadata),
        }
        torch.save({"state_dict": self.state_dict(), "metadata": payload_metadata}, path)
        self.metadata = payload_metadata
        self.version = str(payload_metadata.get("version", path.stem))
        self.anomaly_threshold = payload_metadata.get("anomaly_threshold")
        self.pixel_threshold = payload_metadata.get("pixel_threshold")

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: torch.device | None = None) -> MaskedMultiScaleReconstruction:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MMR checkpoint not found: {path}. Train MMR before inference.")
        target_device = device or cls._select_device()
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        instance = cls(
            image_size=int(metadata.get("image_size", 224)),
            teacher_backbone=metadata.get("teacher_backbone", "wide_resnet50_2.tv2_in1k"),
            mae_backbone=metadata.get("mae_backbone", "vit_base_patch16_224.mae"),
            pretrained=False,
            device=target_device,
        )
        instance.load_state_dict(checkpoint["state_dict"], strict=True)
        instance.metadata = metadata
        instance.version = str(metadata.get("version", path.stem))
        instance.anomaly_threshold = metadata.get("anomaly_threshold")
        instance.pixel_threshold = metadata.get("pixel_threshold")
        instance.eval()
        return instance
