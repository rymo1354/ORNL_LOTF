"""Manifest I/O: one JSON record per structure (JSON-lines), status tracking.

Record fields follow BUILD_OUT.md §5.4. Unknown fields are preserved so stages
can attach extra metadata without schema changes.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

MANIFEST_NAME = "manifest.jsonl"
STATUSES = ("pending", "running", "done", "unconverged", "failed", "interrupted", "skipped")


@dataclass
class Record:
    id: str
    stage: str
    mp_id: str | None = None
    substrate: str | None = None
    polymorph: str | None = None
    facet: list[int] | None = None
    termination: str | None = None
    film_layers: int | None = None
    stacking: str | None = None
    supercell_film: list[list[int]] | None = None
    supercell_substrate: list[list[int]] | None = None
    matching_strain: Any = None
    applied_strain: Any = None
    twist_angle: float | None = None
    interfacial_distance: float | None = None
    magnetic_config: str = "FM"
    asymmetric: bool | None = None
    defect_type: str | None = None
    defect_count: int = 0
    defect_sites: list[int] = field(default_factory=list)
    seed: int | None = None
    parent: str | None = None
    path: str | None = None
    status: str = "pending"
    attempts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")


def manifest_path(stage_dir: str | os.PathLike) -> Path:
    return Path(stage_dir) / MANIFEST_NAME


def read_manifest(stage_dir: str | os.PathLike) -> list[Record]:
    path = manifest_path(stage_dir)
    if not path.exists():
        return []
    known = set(Record.__dataclass_fields__)
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        extra = {k: raw.pop(k) for k in list(raw) if k not in known}
        rec = Record(**raw)
        rec.extra.update(extra)
        records.append(rec)
    return records


def write_manifest(stage_dir: str | os.PathLike, records: Iterable[Record]) -> Path:
    """Atomically (re)write the manifest; ids must be unique."""
    records = list(records)
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate manifest ids: {dupes}")
    path = manifest_path(stage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
    os.replace(tmp, path)
    return path


def update_status(stage_dir: str | os.PathLike, updates: dict[str, str]) -> list[Record]:
    """Set status for the given ids and rewrite the manifest."""
    records = read_manifest(stage_dir)
    by_id = {r.id: r for r in records}
    missing = set(updates) - set(by_id)
    if missing:
        raise KeyError(f"ids not in manifest: {sorted(missing)}")
    for rid, status in updates.items():
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        by_id[rid].status = status
    write_manifest(stage_dir, records)
    return records
