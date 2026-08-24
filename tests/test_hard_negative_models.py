import pytest

from scripts import mine_hard_negatives


def test_hard_negative_miner_loads_faster_rcnn(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        mine_hard_negatives.FasterRCNNBaseline,
        "load",
        lambda checkpoint: expected,
    )

    assert (
        mine_hard_negatives.load_mining_model("faster_rcnn", "checkpoint.pt")
        is expected
    )


def test_hard_negative_miner_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported"):
        mine_hard_negatives.load_mining_model("unknown", "checkpoint.pt")
