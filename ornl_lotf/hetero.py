"""Stage 4: CrTe2 / substrate heterostructures (BUILD_OUT §7, §8.2).

For every selected, relaxed Stage 3 slab:

1. Optionally trim bulk-like layers from the bottom (``trim_to``), keeping the
   slab stoichiometric and its bottom termination unchanged.
2. Build the film (monolayer, or the bulk-stacked AA bilayer) from the relaxed bulk CrTe2 with
   HeteroBuilder (``vdW_structures``).
3. Lattice-match with pymatgen's ZSL through HeteroBuilder's
   ``VdWHeterostructureGenerator``. The film takes all the strain and the substrate
   keeps its relaxed in-plane lattice. The in-plane registry is whatever the
   coherent-interface builder produces (no lateral shift scan yet).
4. Describe each match by its principal film strains (polar decomposition of
   the film -> substrate superlattice map) and its twist, the rotation of the
   film lattice relative to the substrate lattice. The twist is folded by the
   lattice symmetries (film 60°, substrate 90°/60°/180°) and mirrors, so that
   equivalent orientations collapse to one value.
5. Keep the ``n_twist_angles`` lowest-strain distinct twists within
   ``max_length_tol`` and ``max_atoms`` (strains within 0.1% are tied, and the smaller cell wins). The lowest-strain one is flagged
   ``primary``: it alone gets the Stage 5 strain series.
6. Write interfacial-distance variants of each kept match: the start gap, the scan
   gaps, and a large-separation reference.

Twist selection is a proof-of-concept placeholder. Lowest strain ignores
surface morphology, local potentials and nucleation, which have to be
discussed with the experimentalists (see CLAUDE.md, "open science questions").
Every distinct candidate is therefore written to ``matches.csv``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Lattice, Structure

from ornl_lotf.manifest import Record
from ornl_lotf.slabs import find_planes, plane_formula


# --------------------------------------------------------------------------- geometry helpers
def principal_strains(film_sl_vectors, substrate_sl_vectors) -> np.ndarray:
    """Principal in-plane strains of the film (fractions; + = film stretched)."""
    a_f = np.array(film_sl_vectors, float)[:, :2].T
    a_s = np.array(substrate_sl_vectors, float)[:, :2].T
    return np.linalg.svd(a_s @ np.linalg.inv(a_f), compute_uv=False) - 1.0


def rotation_angle(film_sl_vectors, substrate_sl_vectors) -> tuple[float, float]:
    """(angle in degrees, det) of the rotation part of the film -> substrate map."""
    a_f = np.array(film_sl_vectors, float)[:, :2].T
    a_s = np.array(substrate_sl_vectors, float)[:, :2].T
    u, _, vt = np.linalg.svd(a_s @ np.linalg.inv(a_f))
    r = u @ vt
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))), float(np.linalg.det(r))


def lattice_period_deg(lattice: Lattice, tol: float = 0.01) -> int:
    """Rotational period of a 2D lattice (a, b, gamma): 60 hexagonal, 90 square, else 180."""
    a, b, gam = lattice.a, lattice.b, lattice.gamma
    if abs(a - b) / a < tol:
        if abs(gam - 90) < 1:
            return 90
        if abs(gam - 60) < 1 or abs(gam - 120) < 1:
            return 60
    return 180


def fold_twist(angle_deg: float, film_period: int, substrate_period: int) -> float:
    """Reduce a film/substrate rotation to [0, P/2], where P = gcd(periods); mirrors fold ±θ."""
    period = math.gcd(int(film_period), int(substrate_period))
    th = angle_deg % period
    return float(min(th, period - th))


def film_mask(structure: Structure, film_elements: set[str]) -> np.ndarray:
    return np.array([sp.symbol in film_elements for sp in structure.species])


def interface_gap(structure: Structure, film_elements: set[str]) -> float:
    z = structure.cart_coords[:, 2]
    m = film_mask(structure, film_elements)
    return float(z[m].min() - z[~m].max())


def set_gap(structure: Structure, film_elements: set[str], gap: float, vacuum: float) -> Structure:
    """Move the film rigidly along z to ``gap`` (top substrate atom to bottom film atom), make c
    the slab thickness plus ``vacuum``, and centre the atoms in the cell (vacuum/2 below and
    above), so that no atom sits near z = 0 and the dipole layer lies inside the vacuum.
    c must lie along z."""
    if not np.allclose(structure.lattice.matrix[2][:2], 0, atol=1e-4):
        raise ValueError("c must be along z")
    m = film_mask(structure, film_elements)
    xyz = structure.cart_coords.copy()
    xyz[m, 2] += gap - interface_gap(structure, film_elements)
    xyz[:, 2] += vacuum / 2 - xyz[:, 2].min()
    c = float(xyz[:, 2].max() + vacuum / 2)
    mat = structure.lattice.matrix.copy()
    mat[2] = [0.0, 0.0, c]
    return Structure(Lattice(mat), structure.species, xyz, coords_are_cartesian=True,
                     site_properties={k: v for k, v in structure.site_properties.items()
                                      if k != "selective_dynamics"})


def trim_bottom(slab: Structure, min_thickness: float, bulk: Composition, plane_tol: float = 0.3
                ) -> tuple[Structure, int]:
    """Remove bottom atomic planes, giving the thinnest slab that is at least ``min_thickness``
    thick (plane centre to plane centre), is stoichiometric with ``bulk``, and keeps the
    original bottom-plane composition. Returns (slab, number of planes removed)."""
    planes = find_planes(slab, plane_tol)
    names = [plane_formula([slab[i].specie.symbol for i in p]) for p in planes]
    z = slab.cart_coords[:, 2]
    zc = [float(np.mean(z[p])) for p in planes]
    best = 0
    for k in range(1, len(planes)):
        if zc[-1] - zc[k] < min_thickness:
            break
        keep = [i for p in planes[k:] for i in p]
        comp = Composition("".join(slab[i].specie.symbol for i in keep))
        ratio = {el: comp[el] / bulk[el] for el in bulk.elements}
        vals = list(ratio.values())
        if (names[k] == names[0] and set(comp.elements) == set(bulk.elements)
                and np.allclose(vals, vals[0]) and float(vals[0]).is_integer()):
            best = k
    if best == 0:
        return slab.copy(), 0
    keep = sorted(i for p in planes[best:] for i in p)
    out = Structure.from_sites([slab[i] for i in keep])
    return out, best


# --------------------------------------------------------------------------- matching
@dataclass
class Match:
    structure: Structure
    natoms: int
    n_film: int
    strains: np.ndarray            # principal strains (fractions)
    von_mises: float
    twist: float                   # folded, degrees
    twist_raw: float               # rotation from the polar decomposition, degrees
    film_transformation: list
    substrate_transformation: list
    film_sl_vectors: list
    substrate_sl_vectors: list
    props: dict = field(repr=False, default_factory=dict)

    @property
    def max_strain(self) -> float:
        return float(np.max(np.abs(self.strains)))


def make_film(bulk_film: Structure, layers: int, min_vdw_gap: float):
    from vdW_structures.vdW_structure import VdWStructure

    vf = VdWStructure(bulk_film, minimum_vdW_gap=min_vdw_gap)
    n = len(vf.vdW_layers)
    order = [i % n for i in range(layers)]          # repeat the bulk stacking (AA for 1T)
    return vf.extend_vdW_layers(order) if order != list(range(n)) else vf


def make_substrate(slab: Structure, min_vdw_gap: float, vacuum: float):
    from vdW_structures.vdW_structure import VdWStructure

    s = slab.copy()
    if "selective_dynamics" in s.site_properties:
        s.remove_site_property("selective_dynamics")
    return VdWStructure(s, minimum_vdW_gap=min_vdw_gap).set_vacuum(vacuum)


def find_matches(bulk_film: Structure, slab: Structure, layers: int, mcfg: dict, icfg: dict,
                 film_elements: set[str]) -> list[Match]:
    """All proper-rotation matches within max_atoms, deduplicated by (folded twist, size)."""
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator
    from vdW_structures.vdW_heterostructure import VdWHeterostructureGenerator

    gap_min = mcfg.get("min_vdw_gap", 2.2)
    vf = make_film(bulk_film, layers, gap_min)
    vs = make_substrate(slab, gap_min, mcfg.get("substrate_vacuum", 2.4))
    max_atoms = int(mcfg.get("max_atoms", 400))
    s_area = slab.lattice.volume / slab.lattice.c
    f_area = bulk_film.lattice.volume / bulk_film.lattice.c
    n_film_cell = len(vf.structure)
    max_area = max_atoms / (len(slab) / s_area + n_film_cell / f_area)
    zsl = ZSLGenerator(max_area_ratio_tol=mcfg.get("max_area_ratio_tol", 0.08), max_area=max_area,
                       max_length_tol=mcfg.get("max_length_tol", 0.025),
                       max_angle_tol=mcfg.get("max_angle_tol", 0.0175), bidirectional=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structs, props = VdWHeterostructureGenerator().generate_vdW_heterostructures(
            film=vf, substrate=vs, zsl_generator=zsl, get_unique=False,
            interface_spacing=icfg.get("start_gap", 3.25), vacuum_over_film=icfg.get("vacuum", 15.0))

    f_per = lattice_period_deg(bulk_film.lattice)
    s_per = lattice_period_deg(slab.lattice)
    best: dict[tuple, Match] = {}
    for vstruct, p in zip(structs, props):
        s = vstruct.structure
        if len(s) > max_atoms:
            continue
        raw, det = rotation_angle(p["film_sl_vectors"], p["substrate_sl_vectors"])
        if det < 0:                     # same superlattice listed with a mirror-related basis
            continue
        st = principal_strains(p["film_sl_vectors"], p["substrate_sl_vectors"])
        if np.max(np.abs(st)) > mcfg.get("max_length_tol", 0.025) + 1e-9:
            continue
        m = Match(structure=s, natoms=len(s), n_film=int(film_mask(s, film_elements).sum()),
                  strains=st, von_mises=float(p.get("von_mises_strain", np.nan)),
                  twist=round(fold_twist(raw, f_per, s_per), 2), twist_raw=round(raw, 2),
                  film_transformation=np.array(p["film_transformation"]).astype(int).tolist(),
                  substrate_transformation=np.array(p["substrate_transformation"]).astype(int).tolist(),
                  film_sl_vectors=np.round(np.array(p["film_sl_vectors"], float), 4).tolist(),
                  substrate_sl_vectors=np.round(np.array(p["substrate_sl_vectors"], float), 4).tolist(),
                  props=p)
        key = (round(m.twist, 1), m.natoms)
        if key not in best or m.max_strain < best[key].max_strain:
            best[key] = m
    return sorted(best.values(), key=lambda m: strain_rank(m))


STRAIN_TIE = 0.001          # strains within 0.1% count as tied; the smaller cell wins


def strain_rank(m: Match) -> tuple:
    return (int(m.max_strain / STRAIN_TIE), m.natoms, m.max_strain)


def select_twists(matches: list[Match], n: int) -> list[Match]:
    """The ``n`` lowest-strain matches with distinct folded twists (ties go to fewer atoms)."""
    out, seen = [], set()
    for m in matches:
        k = round(m.twist, 1)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- stage driver
def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def het_id(slab_id: str, film: str, layers: int, twist: float, gap: float) -> str:
    return f"het-{slab_id.removeprefix('slab-')}-{film}-{layers}L-tw{_fmt(twist)}-g{_fmt(gap)}"


def build_heterostructures(config: dict, slab_records: list[Record], bulk_records: list[Record],
                           out_dir: Path) -> tuple[list[Record], list[dict], list[dict]]:
    """Returns (manifest records, summary rows for the kept structures, rows for all distinct matches)."""
    hcfg = config["heterostructures"]
    mcfg = {**config.get("matching", {}), **hcfg.get("matching", {})}
    icfg = config.get("interface", {})
    out_dir = Path(out_dir)

    film_rec = next(r for r in bulk_records if r.extra.get("material") == hcfg["film"] and r.status == "done")
    bulk_film = Structure.from_file(Path(film_rec.path) / "relaxed.vasp")
    film_elements = {el.symbol for el in bulk_film.composition.elements}
    by_id = {r.id: r for r in slab_records}
    selected = {s["id"]: s for s in hcfg["substrates"]}

    gaps = [("start", icfg.get("start_gap", 3.25))]
    gaps += [("scan", g) for g in icfg.get("scan_gaps", []) if abs(g - gaps[0][1]) > 1e-6]
    if icfg.get("reference_gap"):
        gaps.append(("reference", icfg["reference_gap"]))

    records, rows, all_rows = [], [], []
    for r in slab_records:                          # every Stage 3 slab is listed, used or not
        if r.id not in selected:
            records.append(_skipped(r, "not selected (config heterostructures.substrates)"))
            continue
        # disk, not the manifest: a running Stage 3 only syncs its manifest between rounds
        if not (Path(r.path) / "properties.json").exists() or not (Path(r.path) / "relaxed.vasp").exists():
            records.append(_skipped(r, "parent slab not relaxed (no properties.json in Stage 3)"))
            continue
        sel = selected[r.id]
        smcfg = {**mcfg, **sel.get("matching", {})}     # per-slab overrides, e.g. a looser strain tolerance
        slab = Structure.from_file(Path(r.path) / "relaxed.vasp")
        n_trim = 0
        if sel.get("trim_to"):
            bulk_rec = next(b for b in bulk_records if b.id == r.extra["bulk_id"])
            bulk_comp = Structure.from_file(Path(bulk_rec.path) / "relaxed.vasp").composition.reduced_composition
            slab, n_trim = trim_bottom(slab, float(sel["trim_to"]), bulk_comp,
                                       config.get("slabs", {}).get("plane_tol", 0.3))
        thick = float(np.ptp(slab.cart_coords[:, 2]))
        for layers in hcfg.get("film_layers", [1]):
            matches = find_matches(bulk_film, slab, layers, smcfg, icfg, film_elements)
            for m in matches:
                all_rows.append({"slab": r.id, "film_layers": layers, "twist_deg": m.twist,
                                 "twist_raw_deg": m.twist_raw, "natoms": m.natoms,
                                 "max_strain_pct": round(100 * m.max_strain, 2),
                                 "principal_strains_pct": np.round(100 * m.strains, 2).tolist(),
                                 "von_mises_pct": round(100 * m.von_mises, 2)})
            kept = select_twists(matches, int(smcfg.get("n_twist_angles", 1)))
            if not kept:
                records.append(_skipped(r, f"no match within {smcfg.get('max_length_tol')} strain and "
                                           f"{smcfg.get('max_atoms')} atoms ({layers}L)"))
                continue
            for rank, m in enumerate(kept):
                for role, gap in gaps:
                    s = set_gap(m.structure, film_elements, gap, icfg.get("vacuum", 15.0))
                    rid = het_id(r.id, hcfg["film"], layers, m.twist, gap)
                    path = out_dir / "structures" / rid / "POSCAR"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    s.to(filename=str(path), fmt="poscar")
                    extra = {
                        "material": f"{hcfg['film']}/{r.substrate}", "is_slab": True,
                        "film": hcfg["film"], "film_bulk_id": film_rec.id, "substrate_slab_id": r.id,
                        "substrate_polymorph": r.polymorph, "substrate_facet": r.facet,
                        "substrate_termination": r.termination, "substrate_trimmed_planes": n_trim,
                        "substrate_thickness": round(thick, 3), "natoms": len(s), "n_film_atoms": m.n_film,
                        "formula": s.composition.formula.replace(" ", ""),
                        "twist_raw_deg": m.twist_raw, "twist_rank": rank, "primary": rank == 0,
                        "gap_role": role, "matching_overrides": sel.get("matching", {}),
                        "film_sl_vectors": m.film_sl_vectors,
                        "substrate_sl_vectors": m.substrate_sl_vectors,
                        "epitaxy": f"{hcfg['film']}[100] rotated {m.twist_raw:+.1f} deg from substrate-cell a "
                                   f"(folded {m.twist:.1f} deg)",
                        "parent_path": str(Path(r.path) / "relaxed.vasp"),
                    }
                    records.append(Record(
                        id=rid, stage="04_heterostructures", mp_id=r.mp_id, substrate=r.substrate,
                        polymorph=r.polymorph, facet=r.facet, termination=r.termination,
                        film_layers=layers, stacking="AA" if layers > 1 else None,
                        supercell_film=m.film_transformation, supercell_substrate=m.substrate_transformation,
                        matching_strain={"principal": np.round(m.strains, 5).tolist(),
                                         "max_abs": round(m.max_strain, 5), "von_mises": round(m.von_mises, 5)},
                        applied_strain=None, twist_angle=m.twist, interfacial_distance=gap,
                        magnetic_config="FM", asymmetric=True, parent=r.id, path=str(path),
                        status="done", extra=extra))
                    rows.append({"id": rid, "slab": r.id, "layers": layers, "natoms": len(s),
                                 "n_film": m.n_film, "twist": m.twist, "max_strain_%": round(100 * m.max_strain, 2),
                                 "principal_%": np.round(100 * m.strains, 2).tolist(), "gap": gap,
                                 "role": role, "primary": rank == 0, "trimmed_planes": n_trim,
                                 "strain_tol_%": round(100 * smcfg.get("max_length_tol", 0.025), 2)})
    return records, rows, all_rows


def _skipped(r: Record, reason: str) -> Record:
    return Record(id=f"het-{r.id.removeprefix('slab-')}", stage="04_heterostructures", mp_id=r.mp_id,
                  substrate=r.substrate, polymorph=r.polymorph, facet=r.facet, termination=r.termination,
                  parent=r.id, path=None, status="skipped", extra={"skip_reason": reason})
