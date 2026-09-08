"""Optional winner-only ablation: retain full images and append annotated training crops."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.inference import _tile_starts
from src.preprocessing import sha256_file


def clipped_boxes(annotations, left, top, right, bottom):
    result = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        x1, y1 = max(x, left), max(y, top)
        x2, y2 = min(x+w, right), min(y+h, bottom)
        if x2 > x1 and y2 > y1:
            # Retain every visible annotated fragment; do not turn it into background.
            result.append({**ann, "bbox": [x1-left, y1-top, x2-x1, y2-y1],
                           "area": (x2-x1)*(y2-y1)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/aircraft/train.json")
    parser.add_argument("--output", required=True, help="New directory, never the source split directory")
    parser.add_argument("--size", type=int, choices=(512, 640), default=512)
    parser.add_argument("--max-per-image", type=int, default=4)
    args = parser.parse_args()
    source = Path(args.input)
    if source.name != "train.json":
        parser.error("Only the training split may be sliced.")
    root = Path(args.output)
    if root.exists() or args.max_per_image < 1:
        parser.error("Use a new output directory and a positive crop limit.")
    payload = json.loads(source.read_text())
    if [c["name"] for c in payload["categories"]] != ["defect"]:
        parser.error("Expected prepared binary defect data.")
    root.mkdir(parents=True)
    output = copy.deepcopy(payload)
    by_image = {}
    for ann in payload["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    next_image = max(i["id"] for i in payload["images"]) + 1
    next_ann = max((a["id"] for a in payload["annotations"]), default=0) + 1
    randomizer = random.Random(42)
    for item in payload["images"]:
        if max(item["width"], item["height"]) <= args.size:
            continue
        crops = []
        for top in _tile_starts(item["height"], args.size, args.size//4):
            for left in _tile_starts(item["width"], args.size, args.size//4):
                right, bottom = min(left+args.size, item["width"]), min(top+args.size, item["height"])
                annotations = clipped_boxes(by_image.get(item["id"], []), left, top, right, bottom)
                if annotations:
                    crops.append((left, top, right, bottom, annotations))
        randomizer.shuffle(crops)
        if not crops:
            continue
        with Image.open(item["file_name"]) as image:
            for left, top, right, bottom, annotations in crops[:args.max_per_image]:
                destination = root / f"slice_{next_image}.png"
                image.crop((left, top, right, bottom)).save(destination)
                output["images"].append({**item, "id": next_image, "file_name": str(destination.resolve()),
                                         "width": right-left, "height": bottom-top,
                                         "sha256": sha256_file(destination),
                                         "parent_image_sha256": item.get("sha256"), "split": "train"})
                for annotation in annotations:
                    output["annotations"].append({**annotation, "id": next_ann, "image_id": next_image})
                    next_ann += 1
                next_image += 1
    (root / "train.json").write_text(json.dumps(output, indent=2))
    print(f"Retained {len(payload['images'])} full images; added {len(output['images'])-len(payload['images'])} slices.")
    print(f"New training JSON: {root / 'train.json'}; validation/test were not changed.")


if __name__ == "__main__":
    main()
