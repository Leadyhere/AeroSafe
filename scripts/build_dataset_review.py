"""Export grouping metadata and a training-only annotation review pack; never invent labels."""
import argparse
import csv
import html
import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import load_config
from src.data import LEAKAGE_ID_FIELDS, load_aircraft_source
from src.preprocessing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/dataset_review_v2")
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output folder; existing review decisions must not be overwritten.")
    output.mkdir(parents=True)
    records = []
    for key, source in [("asdd", "ASDD"), ("aircraftsurface", "aircraftsurface1")]:
        records.extend(load_aircraft_source(config["paths"][key], source, config["aircraft_labels"]))
    groups = defaultdict(list)
    for record in records:
        groups[sha256_file(Path(record.path))].append(record)
    splits = {}
    for split in ["train", "validation", "test"]:
        payload = json.loads((Path(config["paths"]["processed"]) / "aircraft" / f"{split}.json").read_text())
        for image in payload["images"]:
            fingerprint = image.get("sha256") or sha256_file(Path(image["file_name"]))
            if fingerprint in splits and splits[fingerprint] != split:
                raise ValueError("Existing split leakage detected; cannot freeze this split.")
            splits[fingerprint] = split
    fields = ["image", "sha256", *LEAKAGE_ID_FIELDS, "split"]
    manifest_path = output / "aerosafe_group_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fingerprint, group in sorted(groups.items()):
            writer.writerow({"image": Path(group[0].path).name, "sha256": fingerprint,
                             "group_id": f"exact:{fingerprint}", "split": splits.get(fingerprint, "")})
    conflicts = []
    for fingerprint, group in groups.items():
        classes = {tuple(sorted({ann.normalized_class for ann in record.annotations})) for record in group}
        boxes = {tuple(sorted(tuple(ann.bbox) for ann in record.annotations)) for record in group}
        if len(group) > 1 and (len(classes) > 1 or len(boxes) > 1):
            conflicts.append({"sha256": fingerprint, "split": splits.get(fingerprint),
                              "class_conflict": len(classes) > 1, "box_conflict": len(boxes) > 1,
                              "status": "unreviewed",
                              "variants": [{"source": r.source, "file": Path(r.path).name,
                                            "boxes": [{"bbox": a.bbox, "label": a.original_class}
                                                      for a in r.annotations]} for r in group]})
    (output / "duplicate_conflicts.json").write_text(json.dumps(conflicts, indent=2))
    priority = [item["sha256"] for item in conflicts if item["split"] == "train"]
    others = [key for key in sorted(groups) if splits.get(key) == "train" and key not in priority]
    random.Random(42).shuffle(others)
    chosen = (priority + others)[:args.count]
    image_root = output / "images"
    image_root.mkdir()
    cards, review_rows = [], []
    for fingerprint in chosen:
        group = groups[fingerprint]
        image = group[0]
        image_name = fingerprint + Path(image.path).suffix.lower()
        shutil.copy2(image.path, image_root / image_name)
        variants = []
        for variant in group:
            rectangles = []
            for ann in variant.annotations:
                x, y, w, h = ann.bbox
                rectangles.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="red" stroke-width="2"><title>{html.escape(ann.original_class)}</title></rect>')
            variants.append(f'<div><p>{html.escape(variant.source)} — {len(variant.annotations)} boxes</p>'
                            f'<svg viewBox="0 0 {image.width} {image.height}"><image href="images/{image_name}" width="{image.width}" height="{image.height}"/>{"".join(rectangles)}</svg></div>')
        cards.append(f'<article><h3>{len(cards)+1}. {fingerprint[:16]}</h3><div class="variants">{"".join(variants)}</div></article>')
        review_rows.append({"sha256": fingerprint, "image": image_name, "status": "unreviewed",
                            "missing_boxes": "", "loose_boxes": "", "incorrect_boxes": "", "notes": ""})
    with (output / "review_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Aircraft annotation review</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:auto;padding:20px}article{border-top:1px solid #ccc;padding:20px 0}.variants{display:flex;gap:12px}.variants>div{flex:1}svg{width:100%;background:#ddd}</style>'
        '<h1>Training-only annotation review</h1><p>All items are UNREVIEWED. Compare source boxes. '
        'Record findings in review_decisions.csv. Red boxes are source annotations, not verified ground truth. '
        'No test images are displayed. Unknown acquisition IDs in the group manifest are blank.</p>'
        + ''.join(cards), encoding="utf-8"
    )
    summary = {"raw_records": len(records), "unique_image_hashes": len(groups),
               "class_conflict_groups": sum(c["class_conflict"] for c in conflicts),
               "geometry_conflict_groups": sum(c["box_conflict"] for c in conflicts),
               "review_training_images": len(chosen), "human_approved_images": 0,
               "new_confirmed_normals": 0, "source_datasets_modified": False,
               "identity_metadata": "Unknown aircraft/session/camera IDs are blank; exact-image grouping only."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(json.dumps(summary, indent=2))
    print(f"Review ZIP: {archive_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
