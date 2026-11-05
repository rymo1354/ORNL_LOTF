"""Stage 2: substrate slabs and termination enumeration (BUILD_OUT §8.1).

Approach: cut a thick reference slab from the relaxed bulk with pymatgen's
SlabGenerator, rebuild it so that c = [0, 0, c] (HeteroBuilder requirement),
group atoms into atomic planes by z, and enumerate terminations as distinct
cut positions within one bulk repeat of planes. For every termination two
variants are cut from the interior of the reference slab:

* ``symmetric``      - the bottom plane sequence mirrors the top (often
                       non-stoichiometric, e.g. TiO2/.../TiO2);
* ``stoichiometric`` - composition is an integer multiple of the bulk formula
                       (on polar or alternating stacks this is asymmetric).

If one cut satisfies both, it is kept once with both labels. Whether a slab is
geometrically symmetric (an operation maps z -> -z) decides the dipole
correction; this also catches polar bulks (e.g. P4mm (001)) whose plane
compositions look palindromic. The bottom half of each slab is frozen with
selective dynamics for Stage 3, and the in-plane lattice comes from the
relaxed bulk (no lattice relaxation).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# polar axis of each polar polymorph, as a direct-lattice vector of its 5-atom cell
POLAR_AXES = {"tetragonal": (0, 0, 1), "rhombohedral": (1, 1, 1)}


@dataclass
class SlabCandidate:
    structure: Structure
    hkl: tuple[int, int, int]
    top: str
    bottom: str
    variants: list[str]
    stoichiometric: bool
    symmetric: bool
    n_planes: int
    thickness: float
    plane_sequence: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "+".join(self.variants)


# --------------------------------------------------------------------------- helpers
def plane_formula(species: list[str]) -> str:
    c = Counter(species)
    order = sorted(c, key=lambda el: (el == "O", el))   # cations first, O last
    return "".join(f"{el}{c[el] if c[el] > 1 else ''}" for el in order)


def reference_slab(bulk: Structure, hkl, thickness: float) -> Structure:
    """Thick slab with c exactly along z, a along x (atoms not wrapped across vacuum)."""
    gen = SlabGenerator(bulk, hkl, min_slab_size=thickness, min_vacuum_size=15,
                        center_slab=True, primitive=True, reorient_lattice=True)
    slab = gen.get_slab().get_orthogonal_c_slab()
    lat = slab.lattice
    return Structure(Lattice.from_parameters(*lat.abc, *lat.angles), slab.species, slab.frac_coords)


def find_planes(structure: Structure, tol: float) -> list[list[int]]:
    z = structure.cart_coords[:, 2]
    order = np.argsort(z)
    planes = [[int(order[0])]]
    for prev, cur in zip(order[:-1], order[1:]):
        if z[cur] - z[prev] > tol:
            planes.append([int(cur)])
        else:
            planes[-1].append(int(cur))
    return planes


def repeat_length(seq: list[str]) -> int:
    """Smallest period p of the plane-composition sequence."""
    for p in range(1, len(seq) // 2 + 1):
        if all(seq[i] == seq[i + p] for i in range(len(seq) - p)):
            return p
    raise ValueError("no plane periodicity found; increase reference thickness or plane tol")


def is_geometrically_symmetric(structure: Structure, symprec: float = 0.05) -> bool:
    """True if some symmetry operation maps z -> -z (inversion, mirror, 2-fold in-plane, ...)."""
    ops = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations(cartesian=True)
    return any(np.isclose(op.rotation_matrix[2, 2], -1.0, atol=1e-3) for op in ops)


def is_stoichiometric(comp: Composition, bulk: Composition) -> bool:
    ratio = {el: comp[el] / bulk[el] for el in bulk}
    vals = list(ratio.values())
    return set(comp.elements) == set(bulk.elements) and np.allclose(vals, vals[0]) and float(vals[0]).is_integer()


def polarization_relation(bulk: Structure, polymorph: str, hkl) -> dict:
    """Angle between the bulk polar axis and the surface normal (both in the bulk frame)."""
    axis = POLAR_AXES.get(polymorph)
    if axis is None:
        return {"polarization": "none", "polar_axis_angle_deg": None}
    p = bulk.lattice.get_cartesian_coords(axis)
    n = bulk.lattice.reciprocal_lattice_crystallographic.get_cartesian_coords(hkl)
    ang = float(np.degrees(np.arccos(abs(np.dot(p, n)) / (np.linalg.norm(p) * np.linalg.norm(n)))))
    kind = "out-of-plane" if ang < 10 else ("in-plane" if ang > 80 else "oblique")
    return {"polarization": kind, "polar_axis_angle_deg": round(ang, 2)}


def finalize_slab(ref: Structure, plane_idx: list[int], vacuum: float) -> Structure:
    """Atoms of the selected planes in a cell with ``vacuum`` Å total, centred, bottom half frozen."""
    sites = [ref[i] for i in plane_idx]
    z = np.array([s.coords[2] for s in sites])
    thickness = float(z.max() - z.min())
    c = thickness + vacuum
    lat = Lattice.from_parameters(ref.lattice.a, ref.lattice.b, c, 90, 90, ref.lattice.gamma)
    shift = (c - thickness) / 2 - z.min()
    coords = [s.coords + np.array([0, 0, shift]) for s in sites]
    slab = Structure(lat, [s.specie for s in sites], coords, coords_are_cartesian=True)
    zc = slab.cart_coords[:, 2]
    mid = (zc.min() + zc.max()) / 2
    slab.add_site_property("selective_dynamics", [[bool(zz >= mid)] * 3 for zz in zc])
    return slab.get_sorted_structure()


# --------------------------------------------------------------------------- enumeration
def enumerate_slabs(bulk: Structure, hkl, min_thickness: float = 12.0, vacuum: float = 15.0,
                    plane_tol: float = 0.3, symprec: float = 0.05) -> list[SlabCandidate]:
    ref = reference_slab(bulk, hkl, thickness=3 * min_thickness + 10)
    planes = find_planes(ref, plane_tol)
    seq = [plane_formula([ref[i].specie.symbol for i in p]) for p in planes]
    zc = [float(np.mean(ref.cart_coords[p, 2])) for p in planes]
    # drop edge planes (may be split by the cut), then find the plane period
    seq_i, planes_i, zc_i = seq[1:-1], planes[1:-1], zc[1:-1]
    p = repeat_length(seq_i)
    bulk_comp = bulk.composition

    candidates: dict[tuple[int, int], SlabCandidate] = {}
    n = len(seq_i)
    seen_tops: dict[tuple, int] = {}
    for k in range(p):                                    # distinct terminations within one repeat
        t = n - 1 - k                                     # top plane index (read downward)
        sig = tuple(seq_i[t - j] for j in range(p))
        if sig in seen_tops:
            continue
        seen_tops[sig] = t
        top_label = "/".join(sig[:2]) if p > 1 else sig[0]
        found = {"symmetric": None, "stoichiometric": None}
        for b in range(t, -1, -1):                       # grow downward from the top plane
            if zc_i[t] - zc_i[b] < min_thickness:
                continue
            idx = sum((planes_i[j] for j in range(b, t + 1)), [])
            comp = Composition(Counter(ref[i].specie.symbol for i in idx))
            bottom_sig = tuple(seq_i[b + j] for j in range(min(p, t - b + 1)))
            if found["symmetric"] is None and bottom_sig == sig[: len(bottom_sig)]:
                found["symmetric"] = (b, idx, comp)
            if found["stoichiometric"] is None and is_stoichiometric(comp, bulk_comp):
                found["stoichiometric"] = (b, idx, comp)
            if all(found.values()):
                break
        for variant, hit in found.items():
            if hit is None:
                continue
            b, idx, comp = hit
            key = (b, t)
            if key in candidates:
                candidates[key].variants.append(variant)
                continue
            slab = finalize_slab(ref, idx, vacuum)
            bottom_label = "/".join(seq_i[b + j] for j in range(min(2, t - b + 1)))
            candidates[key] = SlabCandidate(
                structure=slab, hkl=tuple(hkl), top=top_label, bottom=bottom_label,
                variants=[variant], stoichiometric=is_stoichiometric(comp, bulk_comp),
                symmetric=is_geometrically_symmetric(slab, symprec),
                n_planes=t - b + 1, thickness=round(zc_i[t] - zc_i[b], 3),
                plane_sequence=seq_i[b : t + 1],
            )
    return list(candidates.values())


# --------------------------------------------------------------------------- Stage 2 driver
VARIANT_TAG = {"symmetric": "sym", "stoichiometric": "stoich"}


def slab_record_id(material: str, polymorph: str, hkl, top: str, variants: list[str]) -> str:
    tag = "-".join(VARIANT_TAG[v] for v in sorted(variants, key=list(VARIANT_TAG).index))
    return f"slab-{material}-{polymorph}-{''.join(map(str, hkl))}-{top.replace('/', '_')}-{tag}"


def build_slabs(config: dict, bulk_records, out_dir):
    """Stage 2: all slabs for config['substrates'] from Stage 1 relaxed bulks.

    Returns (manifest records, summary rows). ``bulk_records`` are the
    01_bulk_relax manifest records (status 'done'; relaxed.vasp in their path).
    """
    from pathlib import Path

    from ornl_lotf.build import conventional_standard
    from ornl_lotf.manifest import Record

    scfg = config.get("slabs", {})
    out_dir = Path(out_dir)
    records, rows = [], []
    for sub in config["substrates"]:
        material = sub["material"]
        bulks = [r for r in bulk_records if r.extra.get("material") == material and r.status == "done"]
        for br in bulks:
            bulk = Structure.from_file(Path(br.path) / "relaxed.vasp")
            if sub.get("setting") == "hexagonal":
                bulk = conventional_standard(bulk)
            facets = [tuple(sub.get("hkl_used", sub["facet"]))]
            facets += [tuple(f) for f in sub.get("extra_facets", {}).get(br.polymorph, [])]
            for hkl in facets:
                pol = polarization_relation(bulk, br.polymorph, hkl)
                for cand in enumerate_slabs(bulk, hkl, scfg.get("min_slab_size", 12.0),
                                            scfg.get("vacuum", 15.0), scfg.get("plane_tol", 0.3),
                                            scfg.get("symprec", 0.05)):
                    rid = slab_record_id(material, br.polymorph, hkl, cand.top, cand.variants)
                    path = out_dir / "structures" / rid / "POSCAR"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cand.structure.to(filename=str(path), fmt="poscar")
                    n_frozen = sum(not sd[0] for sd in cand.structure.site_properties["selective_dynamics"])
                    extra = {
                        "material": material, "is_slab": True, "bulk_id": br.id,
                        "facet_requested": list(sub["facet"]), "setting": sub.get("setting", "primitive"),
                        "top_termination": cand.top, "bottom_termination": cand.bottom,
                        "variants": cand.variants, "stoichiometric": cand.stoichiometric,
                        "geometrically_symmetric": cand.symmetric, "n_planes": cand.n_planes,
                        "thickness": cand.thickness, "natoms": len(cand.structure), "n_frozen": n_frozen,
                        "formula": cand.structure.composition.formula.replace(" ", ""),
                        "plane_sequence": cand.plane_sequence, **pol,
                        "in_plane_abc": [round(cand.structure.lattice.a, 4), round(cand.structure.lattice.b, 4)],
                        "gamma": round(cand.structure.lattice.gamma, 2),
                    }
                    records.append(Record(
                        id=rid, stage="02_slabs", mp_id=br.mp_id, substrate=material,
                        polymorph=br.polymorph, facet=list(hkl), termination=cand.top,
                        asymmetric=not cand.symmetric, parent=br.id, path=str(path), status="done",
                        extra=extra,
                    ))
                    rows.append({"id": rid, **{k: extra[k] for k in (
                        "formula", "natoms", "n_frozen", "thickness", "n_planes", "top_termination",
                        "bottom_termination", "variants", "stoichiometric", "geometrically_symmetric",
                        "polarization", "polar_axis_angle_deg")}, "dipole": not cand.symmetric})
    return records, rows
