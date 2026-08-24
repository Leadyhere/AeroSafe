"""Create and verify a portable aircraft-detector training archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sources(project: Path, model: str, include_training_state: bool) -> list[Path]:
    checkpoint_root = project / "checkpoints"
    if model == "faster_rcnn":
        sources = [checkpoint_root / "faster_rcnn_best.pt"]
        if include_training_state:
            sources.append(checkpoint_root / "faster_rcnn_training_state.pt")
    elif model == "deformable_detr":
        sources = [checkpoint_root / "aircraft_best"]
        if include_training_state:
            sources.append(checkpoint_root / "aircraft_training_state.pt")
    else:
        raise ValueError(f"Unsupported model: {model}")
    sources.extend([project / "reports", project / "config.kaggle.yaml"])
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required artifact inputs are missing: {missing}")
    return sources


def _source_files(project: Path, sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        if source.is_file():
            files.append(source)
        else:
            files.extend(path for path in source.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: str(path.relative_to(project)).lower())


def create_verified_archive(
    project: str | Path,
    model: str,
    output: str | Path,
    *,
    include_training_state: bool = True,
) -> dict[str, Any]:
    project = Path(project).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = artifact_sources(project, model, include_training_state)
    files = _source_files(project, sources)
    entries = [
        {
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest: dict[str, Any] = {
        "format_version": 1,
        "model": model,
        "includes_training_state": include_training_state,
        "files": entries,
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        manifest_path = Path(temporary_directory) / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with tarfile.open(output, "w:gz") as archive:
            for source in sources:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--model", choices=("deformable_detr", "faster_rcnn"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Exclude the optimizer/scheduler training state; the result cannot resume training.",
    )
    args = parser.parse_args()
    try:
        manifest = create_verified_archive(
            args.project,
            args.model,
            args.output,
            include_training_state=not args.deployment_only,
        )
    except (FileNotFoundError, RuntimeError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    archive = manifest["archive"]
    print(json.dumps(archive, indent=2))
    print("PASS: archive contents and SHA-256 checksums were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
