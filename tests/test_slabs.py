"""Slab enumeration on an ideal cubic perovskite (exact expectations)."""
import numpy as np
from pymatgen.core import Lattice, Structure

from ornl_lotf.slabs import enumerate_slabs, polarization_relation


def cubic_bto(a=4.0):
    return Structure(Lattice.cubic(a), ["Ba", "Ti", "O", "O", "O"],
                     [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def by_top_and_variant(cands):
    return {(c.top, tuple(c.variants)): c for c in cands}


def test_bto_100_terminations_and_variants():
    cands = by_top_and_variant(enumerate_slabs(cubic_bto(), (1, 0, 0), min_thickness=12.0))
    assert set(cands) == {("TiO2/BaO", ("symmetric",)), ("TiO2/BaO", ("stoichiometric",)),
                          ("BaO/TiO2", ("symmetric",)), ("BaO/TiO2", ("stoichiometric",))}
    sym_ti = cands[("TiO2/BaO", ("symmetric",))]
    assert sym_ti.structure.composition.formula.replace(" ", "") == "Ba3Ti4O11"
    assert sym_ti.symmetric and not sym_ti.stoichiometric
    stoich = cands[("TiO2/BaO", ("stoichiometric",))]
    assert stoich.stoichiometric and not stoich.symmetric        # alternating stack -> dipole


def test_slab_geometry_contract():
    for c in enumerate_slabs(cubic_bto(), (1, 1, 1), min_thickness=12.0):
        s = c.structure
        assert np.allclose(s.lattice.matrix[2][:2], 0) and s.lattice.matrix[2][2] > 0   # c along z
        z = s.cart_coords[:, 2]
        assert c.thickness >= 12.0 - 1e-6
        assert abs((s.lattice.c - (z.max() - z.min())) - 15.0) < 1e-6                     # 15 Å vacuum
        frozen = [not sd[0] for sd in s.site_properties["selective_dynamics"]]
        mid = (z.min() + z.max()) / 2
        assert all(f == (zz < mid) for f, zz in zip(frozen, z))                            # bottom half


def test_polarization_relation():
    tet = Structure(Lattice.tetragonal(4.0, 4.1), ["Ba", "Ti", "O", "O", "O"],
                    [[0, 0, 0], [.5, .5, .52], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    assert polarization_relation(tet, "tetragonal", (0, 0, 1))["polarization"] == "out-of-plane"
    assert polarization_relation(tet, "tetragonal", (1, 0, 0))["polarization"] == "in-plane"
    assert polarization_relation(cubic_bto(), "cubic", (1, 0, 0))["polarization"] == "none"
