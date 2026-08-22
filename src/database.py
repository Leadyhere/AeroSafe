"""Minimal SQLAlchemy persistence for inspections and aircraft detections."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)
    saved_image_path: Mapped[str | None] = mapped_column(String(1024))
    inspection_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    primary_result: Mapped[str | None] = mapped_column(String(128))
    score: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    detections: Mapped[list[Detection]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", lazy="selectin"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    inspection: Mapped[Inspection] = relationship(back_populates="detections")


def database_url(path: str | Path) -> str:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def init_database(path: str | Path):
    engine = create_engine(database_url(path), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _finite_score(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Inspection scores and coordinates must be finite.")
    return result


def save_inspection(
    session_factory,
    *,
    image_name: str,
    saved_image_path: str | None,
    result: Mapping[str, Any],
) -> int:
    inspection_type = str(result.get("inspection_type", ""))
    if inspection_type not in {"aircraft", "engine"}:
        raise ValueError("inspection_type must be aircraft or engine.")
    detections = list(result.get("defects", [])) if inspection_type == "aircraft" else []
    if inspection_type == "aircraft":
        primary = detections[0]["class"] if detections else None
        score = detections[0]["confidence"] if detections else None
    else:
        primary = str(result.get("status", ""))
        score = result.get("anomaly_score")
    row = Inspection(
        image_name=Path(image_name).name[:255] or "unnamed-image",
        saved_image_path=str(saved_image_path) if saved_image_path else None,
        inspection_type=inspection_type,
        status=str(result["status"]),
        primary_result=str(primary) if primary is not None else None,
        score=_finite_score(score),
        model_name=str(result["model_name"]),
        model_version=str(result["model_version"]),
        inference_time_ms=_finite_score(result["inference_time_ms"]) or 0.0,
    )
    for item in detections:
        box = item.get("bbox", [])
        if len(box) != 4:
            raise ValueError("Every aircraft detection must contain four bbox coordinates.")
        row.detections.append(
            Detection(
                class_name=str(item["class"]),
                confidence=_finite_score(item["confidence"]),
                x1=_finite_score(box[0]),
                y1=_finite_score(box[1]),
                x2=_finite_score(box[2]),
                y2=_finite_score(box[3]),
            )
        )
    with session_factory.begin() as session:
        session.add(row)
    return row.id


def list_inspections(session_factory, *, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    with session_factory() as session:
        rows = session.scalars(select(Inspection).order_by(Inspection.created_at.desc()).limit(limit)).all()
        return [serialize_inspection(row) for row in rows]


def get_inspection(session_factory, inspection_id: int) -> dict[str, Any] | None:
    with session_factory() as session:
        row = session.get(Inspection, int(inspection_id))
        return serialize_inspection(row) if row else None


def serialize_inspection(row: Inspection) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "image_name": row.image_name,
        "saved_image_path": row.saved_image_path,
        "inspection_type": row.inspection_type,
        "status": row.status,
        "primary_result": row.primary_result,
        "score": row.score,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "inference_time_ms": row.inference_time_ms,
        "detections": [
            {
                "class": detection.class_name,
                "confidence": detection.confidence,
                "bbox": [detection.x1, detection.y1, detection.x2, detection.y2],
            }
            for detection in row.detections
        ],
    }


def dashboard_statistics(session_factory) -> dict[str, int]:
    with session_factory() as session:
        def count(*criteria) -> int:
            return int(session.scalar(select(func.count(Inspection.id)).where(*criteria)) or 0)

        return {
            "total_inspections": count(),
            "aircraft_inspections": count(Inspection.inspection_type == "aircraft"),
            "engine_inspections": count(Inspection.inspection_type == "engine"),
            "aircraft_findings": count(
                Inspection.inspection_type == "aircraft", Inspection.status == "defect_detected"
            ),
            "engine_anomaly_findings": count(
                Inspection.inspection_type == "engine", Inspection.status == "anomalous"
            ),
        }
