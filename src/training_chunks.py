"""Shared quarter planning and Kaggle session time guards."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def training_boundaries(
    main_epochs: int,
    *,
    auxiliary_epochs: int = 0,
    quarter_count: int = 4,
    one_go_max_epochs: int = 20,
) -> list[int]:
    """Return absolute completed-epoch targets for resumable runs.

    Models with at most ``one_go_max_epochs`` main epochs run as one logical
    chunk. Longer runs are split into equal main-training quarters; auxiliary
    pretraining belongs to the first quarter.
    """
    if main_epochs < 1 or auxiliary_epochs < 0:
        raise ValueError("Training requires positive main epochs and non-negative auxiliary epochs.")
    if quarter_count < 1 or one_go_max_epochs < 1:
        raise ValueError("quarter_count and one_go_max_epochs must be positive.")
    if main_epochs <= one_go_max_epochs:
        return [auxiliary_epochs + main_epochs]
    boundaries = [
        auxiliary_epochs + math.ceil(main_epochs * quarter / quarter_count)
        for quarter in range(1, quarter_count + 1)
    ]
    return list(dict.fromkeys(boundaries))


def boundary_for_part(boundaries: list[int], part: int | None) -> int:
    """Resolve a one-based quarter/part number into an absolute epoch target."""
    if not boundaries:
        raise ValueError("At least one training boundary is required.")
    if part is None:
        return boundaries[-1]
    if not 1 <= part <= len(boundaries):
        raise ValueError(f"Training part must be between 1 and {len(boundaries)}, got {part}.")
    return boundaries[part - 1]


@dataclass
class SessionTimeGuard:
    """Stop between epochs before a fixed-duration Kaggle session expires."""

    max_session_hours: float
    packaging_reserve_minutes: float = 45.0
    started_at: float = field(default_factory=time.monotonic)
    epoch_seconds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.max_session_hours <= 12:
            raise ValueError("max_session_hours must be greater than zero and at most 12.")
        if not 0 <= self.packaging_reserve_minutes < self.max_session_hours * 60:
            raise ValueError("Packaging reserve must fit inside the session time budget.")

    def record_epoch(self, seconds: float) -> None:
        if seconds > 0:
            self.epoch_seconds.append(float(seconds))

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def should_stop_before_next_epoch(self, completed_epochs: int, target_epoch: int) -> bool:
        """Reserve enough time for one estimated epoch and artifact packaging."""
        if completed_epochs >= target_epoch or not self.epoch_seconds:
            return False
        average_epoch = sum(self.epoch_seconds) / len(self.epoch_seconds)
        conservative_next_epoch = max(average_epoch, self.epoch_seconds[-1]) * 1.20
        reserve = self.packaging_reserve_minutes * 60
        return self.elapsed_seconds + conservative_next_epoch + reserve >= self.max_session_hours * 3600

