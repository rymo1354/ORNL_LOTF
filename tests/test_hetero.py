"""Stage 4 helpers: strain/twist descriptors, gap setting, bottom trimming, and matching on real data."""
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

from ornl_lotf import hetero

STAGES = Path("/pscratch/sd/r/rym/LOTF/stages")


def test_principal_strain_and_rotation():
    f = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    th = np.radians(30)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s = (rot @ np.diag([1.02, 0.99]) @ f[:, :2].T).T
    assert sorted(np.round(hetero.principal_strains(f, s), 6)) == [-0.01, 0.02]
    ang, det = hetero.rotation_angle(f, s)
    assert ang == pytest.approx(30.0, abs=1e-6) and det == pytest.approx(1.0)


def test_twist_folding():
    # hexagonal film on square substrate: period gcd(60, 90) = 30, mirror-folded to [0, 15]
    assert hetero.fold_twist(-7.91, 60, 90) == pytest.approx(7.91)
    assert hetero.fold_twist(22.09, 60, 90) == pytest.approx(7.91)
    assert hetero.fold_twist(135.0, 60, 90) == pytest.approx(15.0)
    # hexagonal on hexagonal: period 60, folded to [0, 30]
    assert hetero.fold_twist(-180.0, 60, 60) == pytest.approx(0.0)
    assert hetero.fold_twist(83.41, 60, 60) == pytest.approx(23.41)
    assert hetero.lattice_period_deg(Lattice.hexagonal(3.8, 20)) == 60
    assert hetero.lattice_period_deg(Lattice.tetragonal(4.0, 20)) == 90
    assert hetero.lattice_period_deg(Lattice.orthorhombic(3.99, 4.11, 20)) == 180


def _toy_het():
    lat = Lattice.tetragonal(4.0, 30.0)
    species = ["Ti", "O", "Ti", "Cr", "Te"]
    z = [5.0, 7.0, 9.0, 12.5, 14.0]
    return Structure(lat, species, [[0, 0, zi / 30.0] for zi in z])


def test_set_gap_moves_film_and_keeps_vacuum():
    s = hetero.set_gap(_toy_het(), {"Cr", "Te"}, 5.0, 15.0)
    assert hetero.interface_gap(s, {"Cr", "Te"}) == pytest.approx(5.0)
    z = s.cart_coords[:, 2]
    assert s.lattice.c - np.ptp(z) == pytest.approx(15.0)
    assert np.ptp(z[:3]) == pytest.approx(4.0)            # substrate untouched
    assert z.min() == pytest.approx(7.5) and s.lattice.c - z.max() == pytest.approx(7.5)   # centred


def test_trim_bottom_keeps_stoichiometry_and_termination():
    # corundum-like stack of (Al | O3 | Al) units, planes 1 Å apart: Al2O3 with Al on both faces
    lat = Lattice.hexagonal(4.8, 40.0)
    species, coords, z = [], [], 5.0
    for _ in range(5):
        for plane in (["Al"], ["O", "O", "O"], ["Al"]):
            for j, el in enumerate(plane):
                species.append(el); coords.append([j / 3, (2 * j % 3) / 3, z / 40])
            z += 1.0
    slab = Structure(lat, species, coords)
    trimmed, n = hetero.trim_bottom(slab, 5.0, Composition("Al2O3"), plane_tol=0.3)
    zt = trimmed.cart_coords[:, 2]
    assert n == 9                                   # 3 whole units removed: 15 -> 6 planes = 5 Å (thinnest >= 5)
    assert np.ptp(zt) == pytest.approx(5.0)
    assert trimmed.composition.reduced_formula == "Al2O3"
    assert [trimmed[i].specie.symbol for i in np.where(np.isclose(zt, zt.min()))[0]] == ["Al"]


needs_stage_data = pytest.mark.skipif(
    not (STAGES / "03_slab_relax/runs/slab-BaTiO3-cubic-100-TiO2_BaO-sym/relaxed.vasp").exists(),
    reason="Stage 1/3 outputs not available")


@needs_stage_data
def test_match_cubic_100_real_data():
    pytest.importorskip("vdW_structures")
    film = Structure.from_file(STAGES / "01_bulk_relax/runs/bulk-CrTe2-1T-mp-685055/relaxed.vasp")
    slab = Structure.from_file(STAGES / "03_slab_relax/runs/slab-BaTiO3-cubic-100-TiO2_BaO-sym/relaxed.vasp")
    mcfg = {"max_atoms": 400, "max_length_tol": 0.025, "max_angle_tol": 0.0175, "max_area_ratio_tol": 0.08}
    icfg = {"start_gap": 3.25, "vacuum": 15.0}
    matches = hetero.find_matches(film, slab, 1, mcfg, icfg, {"Cr", "Te"})
    assert matches and all(m.natoms <= 400 and m.max_strain <= 0.025 + 1e-9 for m in matches)
    best = hetero.select_twists(matches, 1)[0]
    s = best.structure
    assert hetero.interface_gap(s, {"Cr", "Te"}) == pytest.approx(3.25, abs=1e-3)
    assert np.allclose(s.lattice.matrix[2][:2], 0, atol=1e-4)
    assert 0.0 <= best.twist <= 15.0
