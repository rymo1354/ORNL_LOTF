"""Structure construction. Stage 0 (bulk preparation) lives here; slabs and
heterostructure matching (Stages 2 and 4) are added alongside."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ornl_lotf.manifest import Record
from ornl_lotf.mp import MPEntry

POLYMORPH_BY_SPACEGROUP = {
    "R-3c": "corundum", "Pm-3m": "cubic", "P4mm": "tetragonal", "R3m": "rhombohedral",
    "P-3m1": "1T",
}


def primitive_standard(structure: Structure, symprec: float = 0.01) -> Structure:
    return SpacegroupAnalyzer(structure, symprec=symprec).get_primitive_standard_structure()


def conventional_standard(structure: Structure, symprec: float = 0.01) -> Structure:
    return SpacegroupAnalyzer(structure, symprec=symprec).get_conventional_standard_structure()


def count_layers(structure: Structure) -> int:
    """Number of bonded 2D components per cell (0 if the structure is not layered)."""
    sg = StructureGraph.from_local_env_strategy(structure, CrystalNN())
    comps = get_structure_components(sg)
    return sum(1 for c in comps if c["dimensionality"] == 2)


def describe(entry: MPEntry) -> dict:
    """Space-group, cell and layering checks for one MP entry (Stage 0 table row)."""
    prim = primitive_standard(entry.structure)
    lat = prim.lattice
    return {
        **entry.as_row(),
        "polymorph": POLYMORPH_BY_SPACEGROUP.get(entry.spacegroup, entry.spacegroup),
        "prim_nsites": len(prim),
        "prim_a": round(lat.a, 4), "prim_b": round(lat.b, 4), "prim_c": round(lat.c, 4),
        "prim_alpha": round(lat.alpha, 2), "prim_beta": round(lat.beta, 2), "prim_gamma": round(lat.gamma, 2),
        "prim_spacegroup": SpacegroupAnalyzer(prim, 0.01).get_space_group_symbol(),
        "layers_in_mp_cell": count_layers(entry.structure),
        "layers_in_prim": count_layers(prim),
    }


def select_crte2(entries: list[MPEntry], expected_a: float, tol: float = 0.03) -> tuple[MPEntry, Structure, list[dict]]:
    """Pick the 1T CrTe2 entry: P-3m1, one layer per primitive cell, a within ``tol`` of expected.

    Returns (entry, single-layer primitive cell, description rows for all entries).
    """
    rows = [describe(e) for e in entries]
    candidates = [
        (e, r) for e, r in zip(entries, rows)
        if r["prim_spacegroup"] == "P-3m1" and r["layers_in_prim"] == 1 and not e.deprecated
    ]
    if not candidates:
        raise RuntimeError("no single-layer P-3m1 CrTe2 entry found; inspect the entry table")
    entry, row = min(candidates, key=lambda er: er[0].e_above_hull)
    rel = abs(row["prim_a"] - expected_a) / expected_a
    row["a_vs_expected"] = round(rel, 4)
    if rel > tol:
        raise RuntimeError(f"{entry.mp_id}: a = {row['prim_a']} Å deviates {rel:.1%} from {expected_a} Å")
    return entry, primitive_standard(entry.structure), rows


def bulk_record_id(material: str, polymorph: str, mp_id: str) -> str:
    return f"bulk-{material}-{polymorph}-{mp_id}"


def write_bulk(structure: Structure, out_dir: Path, rec_id: str) -> Path:
    path = out_dir / "structures" / rec_id / "POSCAR"
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt="poscar")
    return path


def prepare_bulks(config: dict, out_dir: Path, fetch_ids, fetch_formula) -> tuple[list[Record], list[dict], list[dict]]:
    """Stage 0. ``fetch_ids``/``fetch_formula`` are ornl_lotf.mp functions (injectable for tests).

    Returns (manifest records, substrate description rows, CrTe2 entry rows).
    """
    records, sub_rows, crte2_rows = [], [], []
    for material, spec in config["materials"].items():
        if spec.get("role") == "film":
            entries = fetch_formula(material) if spec.get("query_all_entries") else []
            entries += [e for e in fetch_ids(spec["mp_ids"]).values()
                        if e.mp_id not in {x.mp_id for x in entries}]
            entry, prim, crte2_rows = select_crte2(entries, float(spec["expected_a"]))
            chosen = [(entry, prim)]
        else:
            chosen = [(e, primitive_standard(e.structure)) for e in fetch_ids(spec["mp_ids"]).values()]

        for entry, prim in chosen:
            row = describe(entry)
            if spec.get("role") != "film":
                sub_rows.append(row)
            rec_id = bulk_record_id(material, row["polymorph"], entry.mp_id)
            path = write_bulk(prim, out_dir, rec_id)
            records.append(Record(
                id=rec_id, stage="00_bulk", mp_id=entry.mp_id, polymorph=row["polymorph"],
                substrate=material if spec.get("role") == "substrate" else None,
                path=str(path), status="done", magnetic_config="FM", asymmetric=False,
                extra={"material": material, "role": spec.get("role"), "spacegroup": entry.spacegroup,
                       "e_above_hull": entry.e_above_hull, "nsites": len(prim), "is_slab": False,
                       "lattice_abc": [round(x, 4) for x in prim.lattice.abc],
                       "lattice_angles": [round(x, 2) for x in prim.lattice.angles]},
            ))
    return records, sub_rows, crte2_rows


def al2o3_c_plane_check(structure: Structure) -> dict:
    """Confirm rhombohedral (111) == hexagonal (0001) for corundum (frame-independent).

    Orients unit cells on rhombohedral-primitive (111) and hexagonal-conventional
    (001) with SlabGenerator and compares the in-plane meshes and the repeat
    distance along the normal; they must coincide if the planes are the same.
    """
    from pymatgen.core.surface import SlabGenerator

    prim, conv = primitive_standard(structure), conventional_standard(structure)
    ou_r = SlabGenerator(prim, (1, 1, 1), 5, 5).oriented_unit_cell
    ou_h = SlabGenerator(conv, (0, 0, 1), 5, 5).oriented_unit_cell

    def mesh(s):
        m = s.lattice.matrix
        area = float(np.linalg.norm(np.cross(m[0], m[1])))
        return area, float(s.volume / area), sorted((round(s.lattice.a, 3), round(s.lattice.b, 3)))

    (area_r, h_r, ab_r), (area_h, h_h, ab_h) = mesh(ou_r), mesh(ou_h)
    # The rhombohedral primitive cell's out-of-plane vector is oblique and spans 1/3 of
    # the hexagonal c (R centring), so only an integer ratio of normal repeats is required.
    ratio = h_h / h_r
    return {
        "prim_alpha": round(prim.lattice.alpha, 2),
        "r111_area": round(area_r, 3), "hex001_area": round(area_h, 3),
        "r111_ab": ab_r, "hex001_ab": ab_h,
        "r111_normal_repeat": round(h_r, 3), "hex001_normal_repeat": round(h_h, 3),
        "normal_repeat_ratio": round(ratio, 3),
        "same_plane": bool(np.isclose(area_r, area_h, rtol=1e-3) and ab_r == ab_h
                           and abs(ratio - round(ratio)) < 1e-3),
        "hex_conventional_nsites": len(conv),
    }
