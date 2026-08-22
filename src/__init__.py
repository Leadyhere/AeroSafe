"""AeroInspect core package."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the single project configuration file and resolve no paths implicitly."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must be a YAML mapping: {config_path}")
    return data


DISCLAIMER = (
    "AeroInspect is a research and portfolio project for AI-assisted aircraft visual "
    "inspection. It is not certified aviation-maintenance or airworthiness software. "
    "Model outputs must not be used as the sole basis for aircraft maintenance or "
    "flight-safety decisions."
)
