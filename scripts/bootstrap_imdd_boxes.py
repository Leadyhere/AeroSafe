"""Create model-assisted IMDD COCO proposals for mandatory human review.

The IMDD CSV contains image-level classes, not locations. This script uses a
trained detector to reduce drawing work, but deliberately does not create the
REVIEWED marker required by the training loader. Import the generated COCO into
CVAT/Roboflow, correct/add boxes, export it into the configured
imdd_localization directory, and create REVIEWED only after inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.aircraft_model import AircraftDetector
from src.data import DatasetConfigurationError, canonical_label, normalize_label
from src.inference import predict_aircraft_tiled
from src.preprocessing import SUPPORTED_IMAGE_EXTENSIONS, load_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="data/annotation_tasks/imdd_proposals.json")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    image_root = Path(config["paths"]["imdd_aircraft_images"])
    csv_path = Path(config["paths"]["imdd_aircraft_csv"])
    if not image_root.is_dir() or not csv_path.is_file():
        raise DatasetConfigurationError("IMDD image directory or CSV is missing.")
    model = AircraftDetector.load(args.checkpoint or config["aircraft"]["checkpoint"])
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            by_name[path.name.lower()].append(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    category_to_id = {name: index + 1 for index, name in enumerate(model.labels)}
    output = {
        "info": {
            "description": "UNREVIEWED IMDD model-assisted proposals",
            "aeroinspect_reviewed": False,
            "warning": "Do not train until every selected image is reviewed and corrected.",
        },
        "licenses": [],
        "categories": [
            {"id": category_to_id[name], "name": name, "supercategory": "aircraft_defect"}
            for name in model.labels
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    images_without_proposals = 0
    tile_config = config["aircraft"].get("tiled_inference", {})
    for image_id, row in enumerate(rows, start=1):
        name = str(row["Image Name"]).strip()
        matches = by_name.get(name.lower(), [])
        if len(matches) != 1:
            raise DatasetConfigurationError(f"IMDD image is missing or ambiguous: {name}")
        path = matches[0].resolve()
        image = load_image(path)
        expected = normalize_label(
            canonical_label(str(row["Categories"]).split(",")[-1]),
            config["aircraft_labels"],
        )
        if expected is None or expected not in category_to_id:
            raise DatasetConfigurationError(f"IMDD class has no detector mapping: {row['Categories']}")
        detections, _ = predict_aircraft_tiled(
            model,
            image,
            threshold=args.threshold,
            tile_size=int(tile_config.get("tile_size", 512)),
            overlap=int(tile_config.get("overlap", 128)),
            nms_iou=float(tile_config.get("nms_iou", 0.45)),
        )
        proposals = [item for item in detections if item["class"] == expected]
        output["images"].append(
            {
                "id": image_id,
                "file_name": str(path),
                "width": image.width,
                "height": image.height,
                "source": "IMDD",
                "expected_image_class": expected,
                "description": str(row.get("Description", "")),
                "review_status": "unreviewed",
            }
        )
        images_without_proposals += int(not proposals)
        for proposal in proposals:
            x1, y1, x2, y2 = map(float, proposal["bbox"])
            width, height = x2 - x1, y2 - y1
            output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[expected],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "proposal_score": float(proposal["confidence"]),
                    "review_status": "unreviewed",
                }
            )
            annotation_id += 1
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(output['images'])} IMDD review images and "
        f"{len(output['annotations'])} proposals to {destination}. "
        f"{images_without_proposals} images need boxes drawn from scratch."
    )
    print("This file is UNREVIEWED and is intentionally excluded from training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
