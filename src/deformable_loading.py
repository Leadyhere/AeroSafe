"""Explicit, checked CPU loading for the supported Deformable DETR checkpoint.

Avoid nested timm pretrained loading and meta-device initialization when replacing
the classifier. The full detector supplies the backbone weights. No NaN repair,
random-backbone fallback, or global Transformers monkey-patch is used.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch


def checked_load_state(model, state):
    """Expand safetensors' omitted shared aliases, then require full coverage."""
    state = dict(state)
    aliases = defaultdict(list)
    for name, tensor in [*model.named_parameters(remove_duplicate=False),
                         *model.named_buffers(remove_duplicate=False)]:
        aliases[id(tensor)].append(name)
    for names in aliases.values():
        present = [name for name in names if name in state]
        if present:
            value = state[present[0]]
            if any(not torch.equal(value, state[name]) for name in present[1:]):
                raise ValueError(f"Conflicting shared checkpoint weights: {present}")
            for name in names:
                state.setdefault(name, value)
    # Frozen BatchNorm does not have these training-only counters.
    expected = model.state_dict()
    state = {name: value for name, value in state.items()
             if name in expected or not name.endswith('.num_batches_tracked')}
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Incomplete detector checkpoint: missing={missing}, unexpected={unexpected}")
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite pretrained tensor: {name}")
    model.load_state_dict(state, strict=True)


def replace_class_heads(model, labels):
    """Replace only classification layers; preserve decoder sharing and boxes."""
    replacements = {}
    heads = []
    for old in model.class_embed:
        if id(old) not in replacements:
            head = torch.nn.Linear(old.in_features, len(labels))
            torch.nn.init.normal_(head.weight, std=model.config.init_std)
            torch.nn.init.constant_(head.bias, -math.log(99.0))
            replacements[id(old)] = head
        heads.append(replacements[id(old)])
    model.class_embed = torch.nn.ModuleList(heads)
    if model.model.decoder.class_embed is not None:
        model.model.decoder.class_embed = model.class_embed
    model.config.num_labels = len(labels)
    model.config.id2label = dict(enumerate(labels))
    model.config.label2id = {label: index for index, label in enumerate(labels)}


def load_deformable_checkpoint(source, *, labels=None, local_files_only=False):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForObjectDetection
    from transformers.utils.hub import cached_file

    config = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
    if config.model_type != 'deformable_detr':
        raise ValueError('This loader is only for Deformable DETR.')
    config.use_pretrained_backbone = False
    config.disable_custom_kernels = True
    # Normal CPU construction initializes real tensors, not meta placeholders.
    with torch.device('cpu'):
        model = AutoModelForObjectDetection.from_config(config).float()
    weights = cached_file(source, 'model.safetensors', local_files_only=local_files_only)
    checked_load_state(model, load_file(weights, device='cpu'))
    if labels is not None:
        replace_class_heads(model, labels)
    print('Deformable DETR: complete CPU weight load verified; native attention; FP32.', flush=True)
    return model
