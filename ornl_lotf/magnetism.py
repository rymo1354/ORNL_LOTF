"""Structure-dependent DFT settings: Hubbard U, initial moments, dipole, k-points.

These are applied to the INCAR/KPOINTS *after* pymatgen's input set builds them,
so they never depend on MPRelaxSet's rules (which only apply U to transition
metals when O or F is present; BUILD_OUT §5.5). The same U and moment choices
are therefore applied to bulk CrTe2, substrate slabs, and heterostructures.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints

# d-shell elements that take LDAUL = 2 when a U is given for them
_D_BLOCK = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "W"}


def hubbard_incar(site_symbols: Sequence[str], hubbard_u: Mapping[str, float]) -> dict:
    """LDAU block for the POSCAR species order ``site_symbols`` (one entry per species).

    Always returns LDAU=.TRUE. with explicit per-species lists, so a structure
    without any U element still gets an LDAU block of zeros (identical
    treatment across bulk / slab / heterostructure).
    """
    ldaul, ldauu, ldauj = [], [], []
    for sym in site_symbols:
        u = float(hubbard_u.get(sym, 0.0))
        ldaul.append(2 if (u and sym in _D_BLOCK) else -1)
        ldauu.append(u)
        ldauj.append(0.0)
    return {
        "LDAU": True,
        "LDAUTYPE": 2,
        "LDAUL": ldaul,
        "LDAUU": ldauu,
        "LDAUJ": ldauj,
        "LMAXMIX": 4 if any(l == 2 for l in ldaul) else 2,
        "LDAUPRINT": 1,
    }


def initial_magmoms(structure: Structure, magmom: Mapping[str, float], default: float = 0.0) -> list[float]:
    """Ferromagnetic initialization: one moment per site, by element."""
    return [float(magmom.get(site.specie.symbol, default)) for site in structure]


def slab_center_frac_z(structure: Structure) -> float:
    """Fractional z of the atoms' centre, robust to a slab straddling z=0/1.

    Uses the largest gap in sorted fractional z as the vacuum, unwraps the
    slab across it, and returns the mean position wrapped back into [0, 1).
    """
    z = np.sort(np.mod(structure.frac_coords[:, 2], 1.0))
    gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))
    cut = z[(int(np.argmax(gaps)) + 1) % len(z)]       # first atom above the vacuum
    unwrapped = np.mod(structure.frac_coords[:, 2] - cut, 1.0) + cut
    return float(np.mod(unwrapped.mean(), 1.0))


def dipole_incar(structure: Structure) -> dict:
    """Dipole correction along z, centred on the slab (BUILD_OUT §5.5)."""
    return {"LDIPOL": True, "IDIPOL": 3, "DIPOL": [0.5, 0.5, round(slab_center_frac_z(structure), 6)]}


def kpoints_for(structure: Structure, kppa: int, is_slab: bool) -> Kpoints:
    """Gamma-centred grid at ``kppa`` k-points per reciprocal atom; kz=1 for slabs."""
    kp = Kpoints.automatic_density(structure, kppa, force_gamma=True)
    grid = list(kp.kpts[0])
    if is_slab:
        grid[2] = 1
    return Kpoints.gamma_automatic(tuple(int(g) for g in grid))
