"""Create verified, downloadable training-state archives."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Keep the previous completed checkpoint intact if saving is interrupted."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_training_archive(
    project: str | Path,
    model: str,
    output: str | Path,
    sources: Sequence[str | Path],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive existing in-project sources and verify every archived byte."""
    project = Path(project).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_sources: list[Path] = []
    for source_value in sources:
        source = Path(source_value)
        if not source.is_absolute():
            source = project / source
        source = source.resolve()
        if source.exists():
            try:
                source.relative_to(project)
            except ValueError as exc:
                raise ValueError(f"Artifact source must be inside the project: {source}") from exc
            resolved_sources.append(source)
    if not resolved_sources:
        raise FileNotFoundError("No artifact sources exist yet.")

    files: list[Path] = []
    for source in resolved_sources:
        if source.is_file():
            files.append(source)
        else:
            files.extend(path for path in source.rglob("*") if path.is_file())
    files = sorted(set(files), key=lambda path: path.relative_to(project).as_posix().lower())
    entries = [
        {
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest: dict[str, Any] = {
        "format_version": 2,
        "model": model,
        "run": run_metadata or {},
        "files": entries,
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = Path(temporary_directory) / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with tarfile.open(output, "w:gz") as archive:
            for source in resolved_sources:
                archive.add(source, arcname=source.relative_to(project), recursive=True)
            archive.add(manifest_path, arcname="artifact_manifest.json")

    expected = {entry["path"]: entry for entry in entries}
    with tarfile.open(output, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for relative_path, entry in expected.items():
            member = members.get(relative_path)
            if member is None:
                raise RuntimeError(f"Archive verification failed; missing {relative_path}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Archive verification failed; unreadable {relative_path}")
            digest = hashlib.sha256()
            size = 0
            while chunk := extracted.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Archive verification failed; corrupt {relative_path}")

    archive_digest = sha256_file(output)
    checksum_path = output.with_name(f"{output.name}.sha256")
    checksum_path.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    manifest["archive"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": archive_digest,
        "checksum_file": str(checksum_path),
    }
    return manifest


def artifact_output_path(
    artifact_root: str | Path,
    model: str,
    completed_epochs: int,
    total_epochs: int,
    *,
    part: int | None = None,
    time_limited: bool = False,
) -> Path:
    label = f"part_{part}" if part is not None else "complete"
    if time_limited:
        label = f"partial_{label}"
    return Path(artifact_root) / (
        f"{model}_{label}_epoch_{completed_epochs}_of_{total_epochs}.tar.gz"
    )
