"""Download provenance: one JSON line per file fetched from a remote source.

data/manifest.jsonl records source, URL (secrets redacted), local path, size,
SHA-256 and retrieval time (UTC), so every input can be traced and re-checked.
Manually obtained inputs (Largefire, FRAP, counties, LANDFIRE) are tracked by
DVC instead (`dvc add`).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "manifest.jsonl"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def record(data_dir: Path, *, source: str, url: str, path: Path, step: str, **extra) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel = path.resolve().relative_to(data_dir.resolve().parent)
    except ValueError:
        rel = path
    entry = {
        "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "url": url,
        "path": rel.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "step": step,
        **extra,
    }
    with open(data_dir / MANIFEST_NAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read(data_dir: Path) -> list[dict]:
    f = data_dir / MANIFEST_NAME
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
