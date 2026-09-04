"""AeroInspect Streamlit application."""

from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st
from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from src import DISCLAIMER, PROJECT_ROOT, load_config
from src.database import dashboard_statistics, init_database, list_inspections, save_inspection
from src.inference import ModelNotReadyError, inspect_aircraft, inspect_engine
from src.preprocessing import ImageValidationError, check_image_quality, load_image
from src.report import generate_html_report

st.set_page_config(page_title="AeroInspect", page_icon="✈️", layout="wide")
CONFIG = load_config()
SESSIONS = init_database(PROJECT_ROOT / CONFIG["paths"]["database"])


@st.cache_resource(show_spinner="Loading Deformable DETR checkpoint…")
def load_aircraft_model():
    from src.aircraft_model import AircraftDetector

    return AircraftDetector.load(PROJECT_ROOT / CONFIG["aircraft"]["checkpoint"])


@st.cache_resource(show_spinner="Loading MMR checkpoint…")
def load_engine_model():
    from src.engine_model import MaskedMultiScaleReconstruction

    return MaskedMultiScaleReconstruction.load_checkpoint(PROJECT_ROOT / CONFIG["engine"]["checkpoint"])


def max_upload_bytes() -> int:
    return int(CONFIG["app"]["max_upload_mb"]) * 1024 * 1024


def read_upload(upload) -> Image.Image:
    if upload.size > max_upload_bytes():
        raise ImageValidationError(
            f"Upload exceeds the configured {CONFIG['app']['max_upload_mb']} MB limit."
        )
    return load_image(upload.getvalue())


def persist_visual(image: Image.Image, image_name: str, suffix: str) -> str:
    destination = PROJECT_ROOT / CONFIG["paths"]["inspections"]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{uuid.uuid4().hex}_{suffix}.jpg"
    image.convert("RGB").save(path, quality=90)
    return str(path)


def analyze(image: Image.Image, kind: str):
    if kind == "Aircraft Exterior":
        return inspect_aircraft(image, load_aircraft_model(), config=CONFIG)
    return inspect_engine(image, load_engine_model(), config=CONFIG)


def present_result(result, image_name: str, *, persist: bool = True):
    visual = result["annotated_image"] if result["inspection_type"] == "aircraft" else result["overlay"]
    left, right = st.columns(2)
    with left:
        st.image(visual, caption="AI-assisted visual result", use_container_width=True)
    with right:
        status = result["status"].replace("_", " ").upper()
        st.subheader(status)
        if result["inspection_type"] == "aircraft":
            st.metric("Visual defects detected", result["defect_count"])
            if result["defects"]:
                st.dataframe(
                    pd.DataFrame(
                        [{"Class": item["class"], "Confidence": item["confidence"]} for item in result["defects"]]
                    ),
                    hide_index=True,
                )
        else:
            st.metric("Anomaly score", f"{result['anomaly_score']:.4f}")
            st.metric("Approx. visibly flagged area", f"{result['affected_visible_area']:.2%}")
        st.caption(f"{result['model_name']} · {result['model_version']} · {result['inference_time_ms']:.1f} ms")
        if result["quality"]["warnings"]:
            st.warning("\n".join(result["quality"]["warnings"]))
        st.info("AI-assisted visual inspection. Human inspection recommended.")
    if persist and CONFIG["app"]["save_history"]:
        saved_path = persist_visual(visual, image_name, result["inspection_type"])
        inspection_id = save_inspection(
            SESSIONS, image_name=image_name, saved_image_path=saved_path, result=result
        )
        report = generate_html_report(
            inspection_id=inspection_id, image_name=image_name, result=result, visual=visual
        )
        st.download_button(
            "Download inspection report",
            report,
            file_name=f"aeroinspect-{inspection_id}.html",
            mime="text/html",
        )


def dashboard_page():
    st.title("AeroInspect")
    st.caption("Deep-learning aircraft structural and aero-engine visual inspection")
    stats = dashboard_statistics(SESSIONS)
    columns = st.columns(5)
    labels = [
        ("Total inspections", "total_inspections"),
        ("Aircraft", "aircraft_inspections"),
        ("Engine", "engine_inspections"),
        ("Aircraft findings", "aircraft_findings"),
        ("Engine anomalies", "engine_anomaly_findings"),
    ]
    for column, (label, key) in zip(columns, labels):
        column.metric(label, stats[key])
    history = list_inspections(SESSIONS, limit=10)
    st.subheader("Recent inspections")
    if history:
        st.dataframe(pd.DataFrame(history).drop(columns=["detections"]), hide_index=True, use_container_width=True)
    else:
        st.info("No inspections have been stored yet.")


def new_inspection_page():
    st.title("New Inspection")
    kind = st.radio("Inspection type", ["Aircraft Exterior", "Aero-Engine Blade"], horizontal=True)
    upload = st.file_uploader("Upload one inspection image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    if not upload:
        return
    try:
        image = read_upload(upload)
        st.image(image, caption=upload.name, width=520)
        quality = check_image_quality(image, **CONFIG["quality"])
        if quality["warnings"]:
            st.warning("\n".join(quality["warnings"]))
        else:
            st.success("Image passed deterministic quality checks.")
        if st.button("Analyze", type="primary"):
            with st.spinner("Running visual inspection…"):
                present_result(analyze(image, kind), upload.name)
    except (ImageValidationError, ModelNotReadyError, FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))


def batch_page():
    st.title("Batch Inspection")
    kind = st.radio("Batch type", ["Aircraft Exterior", "Aero-Engine Blade"], horizontal=True)
    uploads = st.file_uploader(
        "Upload inspection images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True
    )
    if uploads and st.button("Analyze batch", type="primary"):
        rows = []
        progress = st.progress(0)
        for index, upload in enumerate(uploads):
            try:
                image = read_upload(upload)
                result = analyze(image, kind)
                if result["inspection_type"] == "aircraft":
                    finding = result["defects"][0]["class"] if result["defects"] else "none"
                    score = result["defects"][0]["confidence"] if result["defects"] else None
                else:
                    finding, score = result["status"], result["anomaly_score"]
                visual = result["annotated_image"] if result["inspection_type"] == "aircraft" else result["overlay"]
                saved_path = persist_visual(visual, upload.name, result["inspection_type"])
                save_inspection(SESSIONS, image_name=upload.name, saved_image_path=saved_path, result=result)
                rows.append({"File": upload.name, "Status": result["status"], "Finding": finding,
                             "Score": score, "Inference Time (ms)": result["inference_time_ms"]})
            except (
                ImageValidationError,
                ModelNotReadyError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
                OSError,
                SQLAlchemyError,
            ) as exc:
                rows.append({"File": upload.name, "Status": "error", "Finding": str(exc), "Score": None,
                             "Inference Time (ms)": None})
            progress.progress((index + 1) / len(uploads))
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button("Download batch CSV", frame.to_csv(index=False), "aeroinspect-batch.csv", "text/csv")


def history_page():
    st.title("Inspection History")
    history = list_inspections(SESSIONS, limit=500)
    if not history:
        st.info("No inspections have been stored yet.")
        return
    frame = pd.DataFrame(history).drop(columns=["detections"])
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button("Download history CSV", frame.to_csv(index=False), "aeroinspect-history.csv", "text/csv")


def performance_page():
    st.title("Model Performance")
    report_root = PROJECT_ROOT / CONFIG["paths"]["reports"]
    detector_keys = ["map_50", "map_50_95", "precision", "recall", "f1"]
    anomaly_keys = ["image_auroc", "image_average_precision", "pixel_auroc", "aupro", "dice", "iou"]
    model_reports = [
        ("Aircraft - currently deployed", "aircraft_metrics.json", detector_keys),
        *[
            (
                f"Aircraft transformer - {name.replace('_', ' ').upper()}",
                f"aircraft_transformers/{name}/aircraft_metrics.json",
                detector_keys,
            )
            for name in CONFIG["aircraft"].get("transformer_candidates", {})
        ],
        ("Aircraft baseline - Faster R-CNN", "baselines/faster_rcnn/aircraft_metrics.json", detector_keys),
        ("Engine - MMR", "engine_metrics.json", anomaly_keys),
        (
            "Engine transformer experiment - MMR + BladeSynth",
            "experiments/bladesynth_mmr/engine_metrics.json",
            anomaly_keys,
        ),
        ("Engine baseline - PatchCore", "baselines/patchcore/engine_metrics.json", anomaly_keys),
    ]
    for title, filename, keys in model_reports:
        st.subheader(title)
        path = report_root / filename
        if not path.is_file():
            st.info("This model has not been evaluated yet.")
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        chart = {key: metrics.get(key) for key in keys if metrics.get(key) is not None}
        st.bar_chart(pd.Series(chart))
        st.json(metrics, expanded=False)
    st.subheader("Dataset statistics")
    dataset_report = report_root / "dataset_report.json"
    if dataset_report.is_file():
        st.json(json.loads(dataset_report.read_text(encoding="utf-8")), expanded=False)
    else:
        st.info("Dataset preparation has not been run yet.")


def about_page():
    st.title("About / Methodology")
    st.markdown(
        """
### Aircraft exterior
RT-DETR v2, RT-DETR, and Deformable DETR are fine-tuned independently on the same binary defect-localization
split. The deployed model is changed only after frozen-test mAP and missed-defect comparisons.

### Aero-engine blade
MMR learns normal blade structure. A masked MAE/ViT branch reconstructs frozen hierarchical teacher
features at three scales. Cosine reconstruction discrepancy produces an image score and pixel heatmap.

### Interpretation
`No defect detected` and `normal` mean only that a model threshold was not exceeded. They are not
airworthiness conclusions. Approximate visibly flagged area is not damage severity.
"""
    )
    st.warning(DISCLAIMER)


page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "New Inspection", "Batch Inspection", "Inspection History", "Model Performance", "About / Methodology"],
)
st.sidebar.caption(DISCLAIMER)
{
    "Dashboard": dashboard_page,
    "New Inspection": new_inspection_page,
    "Batch Inspection": batch_page,
    "Inspection History": history_page,
    "Model Performance": performance_page,
    "About / Methodology": about_page,
}[page]()
