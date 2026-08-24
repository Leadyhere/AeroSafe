"""Mine detector false positives from a human-confirmed normal-image pool."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.aircraft_model import AircraftDetector
from src.baselines import FasterRCNNBaseline
from src.inference import predict_aircraft_tiled
from src.preprocessing import SUPPORTED_IMAGE_EXTENSIONS, load_image, sha256_file


def load_mining_model(model_name: str, checkpoint: str | Path):
    """Load one of the two aircraft detectors behind their shared prediction API."""
    if model_name == "deformable_detr":
        return AircraftDetector.load(checkpoint)
    if model_name == "faster_rcnn":
        return FasterRCNNBaseline.load(checkpoint)
    raise ValueError(f"Unsupported mining model: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--model",
        choices=("deformable_detr", "faster_rcnn"),
        default="deformable_detr",
        help="Detector whose false positives should be mined.",
    )
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument(
        "--confirmed-normal",
        action="store_true",
        help="Required confirmation that every input image was reviewed as defect-free.",
    )
    args = parser.parse_args()
    if not args.confirmed_normal:
        parser.error(
            "--confirmed-normal is required. An unlabeled image is not automatically a true negative."
        )
    config = load_config(args.config)
    source = Path(args.normal_dir)
    if not source.is_dir():
        parser.error(f"Normal-image directory does not exist: {source}")
    candidates = sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not path.name.startswith("._")
    )
    if not candidates:
        parser.error(f"No supported images found under {source}")
    default_checkpoint = (
        config["aircraft"]["checkpoint"]
        if args.model == "deformable_detr"
        else config["baselines"]["faster_rcnn"]["checkpoint"]
    )
    checkpoint = args.checkpoint or default_checkpoint
    model = load_mining_model(args.model, checkpoint)
    tile_config = config["aircraft"].get("tiled_inference", {})
    selected = []
    fingerprints: set[str] = set()
    score_rows = []
    for path in candidates:
        fingerprint = sha256_file(path)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        image = load_image(path)
        detections, _ = predict_aircraft_tiled(
            model,
            image,
            threshold=args.threshold,
            tile_size=int(tile_config.get("tile_size", 512)),
            overlap=int(tile_config.get("overlap", 128)),
            nms_iou=float(tile_config.get("nms_iou", 0.45)),
        )
        if detections:
            selected.append((path.resolve(), image.width, image.height, fingerprint))
            score_rows.append(
                {
                    "file_name": str(path.resolve()),
                    "false_positive_count": len(detections),
                    "maximum_score": max(float(item["confidence"]) for item in detections),
                    "detections": detections,
                }
            )
    output_root = Path(config["paths"]["hard_negatives"]) / f"round_{args.round}"
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir(exist_ok=True)
    portable_images = []
    for path, width, height, fingerprint in selected:
        file_name = f"{fingerprint[:16]}_{path.name}"
        destination = image_root / file_name
        shutil.copy2(path, destination)
        portable_images.append((Path("images") / file_name, width, height, fingerprint))
    categories = [
        {"id": index + 1, "name": name, "supercategory": "aircraft_defect"}
        for index, name in enumerate(model.labels)
    ]
    payload = {
        "info": {
            "description": f"Verified normal hard negatives, mining round {args.round}",
            "aeroinspect_reviewed": True,
        },
        "licenses": [],
        "categories": categories,
        "images": [
            {
                "id": index,
                "file_name": str(path).replace("\\", "/"),
                "width": width,
                "height": height,
                "source": "verified-hard-negatives",
                "group_id": f"hard-negative:{fingerprint}",
                "split": "train",
            }
            for index, (path, width, height, fingerprint) in enumerate(portable_images, start=1)
        ],
        "annotations": [],
    }
    (output_root / "hard_negatives.coco.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_root / "mining_report.json").write_text(
        json.dumps(
            {
                "round": args.round,
                "pool_images": len(candidates),
                "unique_images": len(fingerprints),
                "selected_false_positive_images": len(selected),
                "threshold": args.threshold,
                "model_type": args.model,
                "checkpoint": str(Path(checkpoint)),
                "model_version": model.version,
                "items": score_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # The parent marker enables all accumulated reviewed rounds.
    Path(config["paths"]["hard_negatives"]).mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["hard_negatives"]) / "REVIEWED").write_text(
        "Every listed source image was explicitly confirmed defect-free.\n", encoding="utf-8"
    )
    print(f"Selected {len(selected)} hard-negative images in round {args.round}: {output_root}")
    print("Run prepare_data.py, then retrain both aircraft detectors before the next mining round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
