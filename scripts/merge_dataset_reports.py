"""Merge independently generated aircraft and engine dataset audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_reports(inputs: list[str | Path], output: str | Path) -> dict:
    merged: dict = {}
    for input_path in inputs:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Dataset report must be a JSON object: {input_path}")
        merged.update({key: value for key, value in payload.items() if key != "scope"})
    merged["scope"] = "all" if "aircraft" in merged and "engine" in merged else "merged"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="reports/dataset_report.json")
    args = parser.parse_args()
    report = merge_reports(args.inputs, args.output)
    print(json.dumps({"scope": report["scope"], "sections": sorted(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
