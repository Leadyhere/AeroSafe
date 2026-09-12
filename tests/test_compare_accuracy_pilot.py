import pytest

from scripts.compare_accuracy_pilot import compare_records


def record(map_value, recall, *, small=0.1, missed=0.4, false_positives=0.2):
    return {
        "best_epoch": 1,
        "checkpoint": "fixture",
        "metrics": {
            "map_50_95": map_value,
            "map_50": map_value + 0.1,
            "map_small": small,
            "recall": recall,
            "missed_defect_rate": missed,
            "false_positives_per_image": false_positives,
        },
    }


def test_comparison_prefers_map_before_recall_and_never_uses_test():
    result = compare_records(record(0.20, 0.9), record(0.21, 0.4))
    assert result["recommended"] == "crop_pilot"
    assert result["split"] == "validation"
    assert result["final_test_used"] is False


def test_comparison_uses_recall_to_break_map_tie():
    result = compare_records(record(0.20, 0.5), record(0.20, 0.6))
    assert result["recommended"] == "crop_pilot"


def test_comparison_warns_about_operational_regressions():
    result = compare_records(
        record(0.20, 0.5, small=0.2, missed=0.4, false_positives=0.2),
        record(0.21, 0.6, small=0.1, missed=0.5, false_positives=0.3),
    )
    assert len(result["warnings"]) == 3
    assert result["pilot_minus_baseline"]["map_50_95"] == pytest.approx(0.01)
