import time

import pytest

from scripts.train_aircraft import resolve_epoch_window, warmup_cosine_multiplier
from src.training_artifacts import artifact_output_path
from src.training_chunks import SessionTimeGuard, boundary_for_part, training_boundaries


def test_chunk_window_starts_at_first_epoch_and_stops_at_absolute_epoch():
    assert list(
        resolve_epoch_window(start_epoch=0, total_epochs=33, stop_after_epoch=8)
    ) == list(range(8))


def test_chunk_window_resumes_without_repeating_completed_epochs():
    assert list(
        resolve_epoch_window(start_epoch=8, total_epochs=33, stop_after_epoch=16)
    ) == list(range(8, 16))


def test_chunk_window_rejects_completed_or_invalid_stop_epoch():
    with pytest.raises(ValueError, match="greater"):
        resolve_epoch_window(start_epoch=8, total_epochs=33, stop_after_epoch=8)
    with pytest.raises(ValueError, match="between"):
        resolve_epoch_window(start_epoch=0, total_epochs=33, stop_after_epoch=34)


def test_warmup_cosine_schedule_warms_then_decays():
    values = [
        warmup_cosine_multiplier(epoch, warmup_epochs=2, total_epochs=6)
        for epoch in range(6)
    ]
    assert values[:2] == [0.5, 1.0]
    assert values[2] == pytest.approx(1.0)
    assert values[-1] > 0.0
    assert warmup_cosine_multiplier(6, warmup_epochs=2, total_epochs=6) == pytest.approx(0.0)


def test_long_training_is_split_into_main_epoch_quarters():
    assert training_boundaries(200) == [50, 100, 150, 200]
    assert training_boundaries(200, auxiliary_epochs=5) == [55, 105, 155, 205]


def test_twenty_epoch_model_defaults_to_four_parts():
    assert training_boundaries(20, auxiliary_epochs=3) == [8, 13, 18, 23]
    assert training_boundaries(20, auxiliary_epochs=3, one_go_max_epochs=20) == [23]
    assert boundary_for_part([23], 1) == 23
    with pytest.raises(ValueError, match="between 1 and 1"):
        boundary_for_part([23], 2)


def test_session_guard_reserves_packaging_and_next_epoch_time():
    guard = SessionTimeGuard(
        max_session_hours=11,
        packaging_reserve_minutes=45,
        started_at=time.monotonic() - 10 * 3600,
    )
    guard.record_epoch(30 * 60)
    assert guard.should_stop_before_next_epoch(5, 20)
    assert not guard.should_stop_before_next_epoch(20, 20)


def test_partial_artifact_name_records_exact_progress():
    path = artifact_output_path(
        "artifacts", "mmr_real", 37, 200, part=1, time_limited=True
    )
    assert path.name == "mmr_real_partial_part_1_epoch_37_of_200.tar.gz"
