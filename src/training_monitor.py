"""TensorBoard helpers shared by all AeroInspect training entry points."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any


def create_tensorboard_writer(
    config: Mapping[str, Any], model_name: str, mode: str
) -> tuple[Any | None, Path | None]:
    """Create a TensorBoard writer under the configured reports directory."""
    training = config.get("training", {})
    if not bool(training.get("tensorboard", True)):
        return None, None

    configured_root = training.get("tensorboard_log_dir")
    root = (
        Path(configured_root)
        if configured_root
        else Path(config["paths"]["reports"]) / "tensorboard"
    )
    log_dir = root / mode / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "TensorBoard logging is enabled but tensorboard is not installed. "
            "Run `pip install -r requirements.txt` or set training.tensorboard=false."
        ) from exc

    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=15)
    writer.add_text("run/model", model_name, 0)
    writer.add_text("run/mode", mode, 0)
    writer.add_text(
        "run/config",
        "```json\n" + json.dumps(dict(config), indent=2, default=str) + "\n```",
        0,
    )
    writer.flush()
    print(f"TensorBoard logs: {log_dir}", flush=True)
    return writer, log_dir


def log_epoch_progress(
    writer: Any | None,
    *,
    epoch_index: int,
    total_epochs: int,
    phase: str,
    learning_rate: float,
) -> None:
    """Record and print completed-epoch progress for a training run."""
    completed = epoch_index + 1
    percent = 100.0 * completed / max(1, total_epochs)
    print(
        f"Epoch {completed}/{total_epochs} ({percent:.1f}%) completed; phase={phase}",
        flush=True,
    )
    if writer is None:
        return
    writer.add_scalar("progress/completed_epochs", completed, completed)
    writer.add_scalar("progress/remaining_epochs", total_epochs - completed, completed)
    writer.add_scalar("progress/percent", percent, completed)
    writer.add_scalar("train/learning_rate", learning_rate, completed)
    writer.add_text("progress/phase", phase, completed)


def log_numeric_metrics(
    writer: Any | None,
    prefix: str,
    metrics: Mapping[str, Any] | None,
    step: int,
) -> None:
    """Recursively add finite numeric metric values to TensorBoard."""
    if writer is None or metrics is None:
        return
    for name, value in metrics.items():
        tag = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, Mapping):
            log_numeric_metrics(writer, tag, value, step)
        elif isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                writer.add_scalar(tag, numeric, step)


def finish_tensorboard(writer: Any | None) -> None:
    """Flush and close a writer without burdening disabled runs."""
    if writer is not None:
        writer.flush()
        writer.close()
