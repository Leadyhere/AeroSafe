from types import SimpleNamespace

import pytest
import torch

from src.deformable_loading import checked_load_state, replace_class_heads


def test_shared_weight_aliases_are_restored_without_missing_parameters():
    layer = torch.nn.Linear(2, 1)
    model = torch.nn.ModuleList([layer, layer])
    checked_load_state(model, {'0.weight': torch.ones(1, 2), '0.bias': torch.ones(1)})
    assert torch.equal(model[1].weight, torch.ones(1, 2))
    with pytest.raises(ValueError, match='Incomplete'):
        checked_load_state(model, {'0.weight': torch.ones(1, 2)})
    with pytest.raises(ValueError, match='Conflicting'):
        checked_load_state(model, {'0.weight': torch.ones(1, 2), '1.weight': torch.zeros(1, 2)})


def test_nonfinite_checkpoint_is_rejected():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='Non-finite'):
        checked_load_state(model, {'weight': torch.full((1, 2), float('nan')), 'bias': torch.ones(1)})


def test_classifier_replacement_preserves_sharing_and_leaves_box_head_unchanged():
    model = torch.nn.Module()
    old = torch.nn.Linear(4, 91)
    model.class_embed = torch.nn.ModuleList([old, old])
    model.bbox_embed = torch.nn.Linear(4, 4)
    boxes = model.bbox_embed.weight.detach().clone()
    model.model = SimpleNamespace(decoder=SimpleNamespace(class_embed=model.class_embed))
    model.config = SimpleNamespace(init_std=.02)
    replace_class_heads(model, ['defect'])
    assert model.class_embed[0] is model.class_embed[1]
    assert model.model.decoder.class_embed is model.class_embed
    assert model.class_embed[0].out_features == 1
    assert torch.equal(boxes, model.bbox_embed.weight)
    assert model.config.id2label == {0: 'defect'}
    assert torch.isfinite(model.class_embed[0].weight).all()


def test_nonfinite_gradient_is_rejected_before_optimizer_update():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full((2,), float('nan'))
    optimizer = torch.optim.AdamW([parameter], lr=.001)
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match='non-finite'):
        torch.nn.utils.clip_grad_norm_([parameter], .1, error_if_nonfinite=True)
        optimizer.step()
    assert torch.equal(before, parameter)
