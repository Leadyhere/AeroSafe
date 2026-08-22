"""Self-contained, escaped HTML inspection reports."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from PIL import Image

from . import DISCLAIMER


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _image_data_uri(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_html_report(
    *,
    inspection_id: int,
    image_name: str,
    result: Mapping[str, Any],
    visual: Image.Image | None = None,
    timestamp: datetime | None = None,
) -> str:
    timestamp = timestamp or datetime.now(timezone.utc)
    quality = result.get("quality", {})
    warnings = quality.get("warnings", [])
    if result.get("inspection_type") == "aircraft":
        rows = "".join(
            f"<tr><td>{_escape(item['class'])}</td><td>{float(item['confidence']):.2%}</td>"
            f"<td>{_escape([round(float(v), 1) for v in item['bbox']])}</td></tr>"
            for item in result.get("defects", [])
        )
        findings = (
            "<table><thead><tr><th>Class</th><th>Confidence</th><th>Bounding box</th></tr></thead>"
            f"<tbody>{rows or '<tr><td colspan=\"3\">No visual defect exceeded the configured threshold.</td></tr>'}</tbody></table>"
        )
    else:
        findings = (
            f"<p><strong>Anomaly score:</strong> {float(result['anomaly_score']):.6f}</p>"
            f"<p><strong>Approximate visibly flagged area:</strong> "
            f"{float(result['affected_visible_area']):.2%}</p>"
        )
    warning_html = "".join(f"<li>{_escape(message)}</li>" for message in warnings) or "<li>None</li>"
    image_uri = _image_data_uri(visual)
    image_html = f'<img src="{image_uri}" alt="Inspection result">' if image_uri else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AeroInspect report #{inspection_id}</title>
<style>
body{{font:15px/1.5 Arial,sans-serif;color:#162236;max-width:920px;margin:32px auto;padding:0 24px}}
h1{{color:#0d3b66}} .meta{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;background:#f3f7fb;padding:18px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd7e2;padding:8px;text-align:left}}
img{{max-width:100%;margin-top:20px;border:1px solid #ccd7e2}}.disclaimer{{margin-top:28px;padding:14px;background:#fff4db}}
</style></head><body>
<h1>AeroInspect Inspection Report</h1>
<div class="meta">
<div><strong>Inspection ID:</strong> {_escape(inspection_id)}</div>
<div><strong>Timestamp:</strong> {_escape(timestamp.isoformat())}</div>
<div><strong>Type:</strong> {_escape(result.get('inspection_type'))}</div>
<div><strong>Image:</strong> {_escape(image_name)}</div>
<div><strong>Model:</strong> {_escape(result.get('model_name'))}</div>
<div><strong>Version:</strong> {_escape(result.get('model_version'))}</div>
<div><strong>Status:</strong> {_escape(result.get('status'))}</div>
<div><strong>Inference:</strong> {float(result.get('inference_time_ms', 0)):.2f} ms</div>
</div>
<h2>Image-quality warnings</h2><ul>{warning_html}</ul>
<h2>Visual inspection result</h2>{findings}{image_html}
<p><strong>Manual inspection recommended for all model findings.</strong></p>
<div class="disclaimer">{_escape(DISCLAIMER)}</div>
</body></html>"""
