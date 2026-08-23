from __future__ import annotations

from src.training_monitor import (
    create_tensorboard_writer,
    finish_tensorboard,
    log_epoch_progress,
    log_numeric_metrics,
)


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.text = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_text(self, tag, value, step):
        self.text.append((tag, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def test_disabled_tensorboard_does_not_require_dependency(tmp_path):
    config = {
        "paths": {"reports": str(tmp_path)},
        "training": {"tensorboard": False},
    }
    writer, log_dir = create_tensorboard_writer(config, "model", "smoke")
    assert writer is None
    assert log_dir is None


def test_progress_and_nested_metrics_are_logged():
    writer = FakeWriter()
    log_epoch_progress(
        writer,
        epoch_index=1,
        total_epochs=4,
        phase="finetuning",
        learning_rate=0.001,
    )
    log_numeric_metrics(
        writer,
        "validation",
        {"map": 0.5, "nested": {"recall": 0.75}, "ignored": None},
        2,
    )

    assert ("progress/percent", 50.0, 2) in writer.scalars
    assert ("train/learning_rate", 0.001, 2) in writer.scalars
    assert ("validation/map", 0.5, 2) in writer.scalars
    assert ("validation/nested/recall", 0.75, 2) in writer.scalars
    assert writer.text == [("progress/phase", "finetuning", 2)]

    finish_tensorboard(writer)
    assert writer.flushed
    assert writer.closed
