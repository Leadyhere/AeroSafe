import pytest

from scripts.train_aircraft import resolve_epoch_window


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
