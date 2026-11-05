"""Container environment checks: run inside the ORNL_LOTF image.

    podman-hpc run --rm -v $SCRATCH:$SCRATCH -e PYTHONPATH=$PWD -w $PWD localhost/ornl_lotf:v0.1 \
        python -m pytest -q tests/test_container_env.py
"""
import importlib

import pytest

REQUIRED = [
    "vdW_structures", "EnsembleFFFit", "matensemble", "pymatgen", "mp_api",
    "pymatgen.analysis.defects", "papermill", "py3Dmol", "ipykernel", "yaml",
]


@pytest.mark.parametrize("mod", REQUIRED)
def test_import(mod):
    importlib.import_module(mod)


def test_heterobuilder_api_present():
    from vdW_structures.vdW_heterostructure import VdWHeterostructureGenerator
    from vdW_structures.vdW_structure import VdWStructure

    assert hasattr(VdWHeterostructureGenerator, "generate_vdW_heterostructures")
    for attr in ("extend_vdW_layers", "z_solve_shift_vdW_layers", "set_vacuum"):
        assert hasattr(VdWStructure, attr)


def test_vdw_structure_finds_two_layers():
    """Two well-separated Cr-Te-like sheets -> VdWStructure detects 2 layers."""
    from pymatgen.core import Lattice, Structure
    from vdW_structures.vdW_structure import VdWStructure

    a, c = 3.8, 24.0
    lat = Lattice([[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, c]])
    z = [0.10, 0.17, 0.035, 0.60, 0.67, 0.535]  # Cr, Te, Te per sheet (fractional)
    species = ["Cr", "Te", "Te"] * 2
    coords = [[0, 0, z[0]], [1 / 3, 2 / 3, z[1]], [2 / 3, 1 / 3, z[2]],
              [0, 0, z[3]], [1 / 3, 2 / 3, z[4]], [2 / 3, 1 / 3, z[5]]]
    vdw = VdWStructure(Structure(lat, species, coords), minimum_vdW_gap=2.5)
    assert len(vdw.vdW_layers) == 2


def test_eff_dft_builder_importable():
    from EnsembleFFFit.base import DFTMatEnsemble

    assert hasattr(DFTMatEnsemble, "build_dft_dcts")
    assert hasattr(DFTMatEnsemble, "run_individual")
