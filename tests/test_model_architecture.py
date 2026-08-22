from __future__ import annotations

import torch


def test_tiny_deformable_detr_forward_loss_and_backward() -> None:
    """Exercise the real Deformable DETR matching/loss stack without a weight download."""
    from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

    config = DeformableDetrConfig(
        num_labels=5,
        use_pretrained_backbone=False,
        backbone="resnet18",
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=10,
        num_feature_levels=3,
    )
    model = DeformableDetrForObjectDetection(config)
    output = model(
        pixel_values=torch.randn(1, 3, 64, 64),
        pixel_mask=torch.ones(1, 64, 64, dtype=torch.long),
        labels=[{"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}],
    )
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (1, 10, 5)
    assert output.pred_boxes.shape == (1, 10, 4)
    output.loss.backward()


def test_tiny_mmr_forward_loss_backward_and_pixel_map() -> None:
    """Exercise masking, FPN reconstruction, teacher alignment, and anomaly mapping."""
    from src.engine_model import MaskedMultiScaleReconstruction

    model = MaskedMultiScaleReconstruction(
        image_size=64,
        teacher_backbone="resnet18",
        mae_backbone="vit_tiny_patch16_224",
        pretrained=False,
        device=torch.device("cpu"),
    )
    images = torch.randn(1, 3, 64, 64)
    output = model(images, mask_ratio=0.5)
    assert [feature.shape[-2:] for feature in output["reconstructed_features"]] == [
        (16, 16), (8, 8), (4, 4)
    ]
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    anomaly_map = model.anomaly_map(images, mask_ratio=0.0, passes=1)
    assert anomaly_map.shape == (1, 64, 64)
    assert torch.isfinite(anomaly_map).all()
