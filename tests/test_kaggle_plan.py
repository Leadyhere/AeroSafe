from pathlib import Path

from scripts.kaggle_train_all import MODEL_ORDER, model_boundaries, training_command
from src import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plan_contains_five_transformer_runs_and_two_baselines():
    assert MODEL_ORDER == (
        "rt_detr_v2",
        "rt_detr",
        "deformable_detr",
        "mmr_real",
        "mmr_bladesynth",
        "faster_rcnn",
        "patchcore",
    )


def test_configured_quarter_boundaries_match_training_design():
    config = load_config(PROJECT_ROOT / "config.yaml")
    assert model_boundaries(config, "rt_detr_v2") == [23]
    assert model_boundaries(config, "faster_rcnn") == [23]
    assert model_boundaries(config, "mmr_real") == [50, 100, 150, 200]
    assert model_boundaries(config, "mmr_bladesynth") == [55, 105, 155, 205]


def test_full_command_passes_part_to_resumable_trainer():
    command = training_command("mmr_real", "config.yaml", mode="full", part=3)
    assert command[-2:] == ["--quarter", "3"]
    assert "--variant" in command
    assert "real" in command
