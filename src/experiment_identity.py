"""Portable experiment identities prevent resuming old labels or changed data."""
import hashlib
import json
from pathlib import Path


def aircraft_data_identity(files):
    splits = []
    for file in files:
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        if [c["name"] for c in payload["categories"]] != ["defect"]:
            raise ValueError("This experiment requires binary defect labels. Re-run prepare_data.py.")
        annotations = {}
        for ann in payload["annotations"]:
            annotations.setdefault(ann["image_id"], []).append([ann["category_id"], ann["bbox"]])
        items = [
            [item.get("sha256", Path(item["file_name"]).name), item["width"], item["height"],
             sorted(annotations.get(item["id"], []), key=str)]
            for item in payload["images"]
        ]
        splits.append(sorted(items, key=str))
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def validate_resume_identity(state, identity):
    if state.get("metadata", {}).get("experiment_identity") != identity:
        raise ValueError(
            "Incompatible/legacy checkpoint or changed dataset/recipe. Start a fresh run from "
            "general pretrained weights; do not resume old 7-class or prior-recipe checkpoints."
        )
