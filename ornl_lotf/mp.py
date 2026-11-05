"""Materials Project retrieval. The API key is read from $MP_API_KEY only."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from pymatgen.core import Structure

SUMMARY_FIELDS = ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "nsites",
                  "structure", "deprecated", "ordering", "total_magnetization", "is_stable"]


def _mprester():
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY is not set (export it in ~/.bashrc; never commit it)")
    return MPRester(key)


@dataclass
class MPEntry:
    mp_id: str
    formula: str
    spacegroup: str
    spacegroup_number: int
    e_above_hull: float
    nsites: int
    deprecated: bool
    ordering: str | None
    total_magnetization: float | None
    structure: Structure = field(repr=False)

    def as_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "structure"}


def _to_entry(doc) -> MPEntry:
    return MPEntry(
        mp_id=str(doc.material_id), formula=doc.formula_pretty,
        spacegroup=doc.symmetry.symbol, spacegroup_number=int(doc.symmetry.number),
        e_above_hull=float(doc.energy_above_hull), nsites=int(doc.nsites),
        deprecated=bool(doc.deprecated),
        ordering=str(doc.ordering) if doc.ordering is not None else None,
        total_magnetization=doc.total_magnetization, structure=doc.structure,
    )


def fetch_by_ids(mp_ids: list[str]) -> dict[str, MPEntry]:
    with _mprester() as mpr:
        docs = mpr.materials.summary.search(material_ids=mp_ids, fields=SUMMARY_FIELDS)
    found = {str(d.material_id): _to_entry(d) for d in docs}
    missing = sorted(set(mp_ids) - set(found))
    if missing:
        raise KeyError(f"MP IDs not found (deprecated or merged?): {missing}")
    return found


def fetch_by_formula(formula: str) -> list[MPEntry]:
    with _mprester() as mpr:
        docs = mpr.materials.summary.search(formula=formula, fields=SUMMARY_FIELDS)
    return sorted((_to_entry(d) for d in docs), key=lambda e: e.e_above_hull)
