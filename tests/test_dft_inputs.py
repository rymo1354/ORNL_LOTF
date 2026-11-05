"""INCAR/KPOINTS generation checks (BUILD_OUT §5.5). Needs the POTCAR directory, so run
inside the container on Perlmutter; skipped elsewhere."""
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from ornl_lotf.compute import archive_and_escalate, disk_status
from ornl_lotf.magnetism import hubbard_incar, slab_center_frac_z

PSP = Path("/pscratch/sd/r/rym/MatEnsemble/run_pipeline/DFT/pymatgen_pseudos")
DFT = {"hubbard_u": {"Cr": 3.7}, "initial_magmom": {"Cr": 3.0}, "dipole_correction": "auto"}
RECIPE = {"pseudopotentials_directory": str(PSP), "input_set": "MPRelaxSet",
          "potcar_functional": "PBE_54_W_HASH", "dft": DFT,
          "user_incar_settings": {"ENCUT": 520, "IVDW": 12, "NSW": 0},
          "user_kpoints_settings": {"grid_density": 1000}}
needs_potcars = pytest.mark.skipif(not PSP.exists(), reason="POTCAR directory not available")


def crte2_bulk():
    a, c = 3.86, 6.2
    lat = Lattice.hexagonal(a, c)
    return Structure(lat, ["Cr", "Te", "Te"], [[0, 0, 0], [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])


def crte2_on_oxide_slab():
    """Toy asymmetric 'heterostructure': an Al-O layer under a CrTe2 sheet, with vacuum."""
    a, c = 3.86, 30.0
    lat = Lattice.hexagonal(a, c)
    species = ["Al", "O", "Cr", "Te", "Te"]
    z = np.array([2.0, 3.0, 7.5, 6.0, 9.0]) / c
    coords = [[0, 0, z[0]], [1 / 3, 2 / 3, z[1]], [0, 0, z[2]], [1 / 3, 2 / 3, z[3]], [2 / 3, 1 / 3, z[4]]]
    return Structure(lat, species, coords)


def test_hubbard_block_independent_of_oxygen():
    without_o = hubbard_incar(["Cr", "Te"], DFT["hubbard_u"])
    with_o = hubbard_incar(["Al", "O", "Cr", "Te"], DFT["hubbard_u"])
    assert without_o["LDAU"] and with_o["LDAU"]
    assert without_o["LDAUU"] == [3.7, 0.0] and with_o["LDAUU"] == [0.0, 0.0, 3.7, 0.0]
    assert without_o["LDAUL"] == [2, -1] and with_o["LDAUL"] == [-1, -1, 2, -1]
    assert without_o["LMAXMIX"] == with_o["LMAXMIX"] == 4


def test_slab_center_handles_wrapping():
    s = Structure(Lattice.cubic(10), ["O", "O"], [[0, 0, 0.95], [0, 0, 0.05]])
    assert slab_center_frac_z(s) == pytest.approx(0.0, abs=1e-6) or slab_center_frac_z(s) == pytest.approx(1.0)


@needs_potcars
def test_bulk_and_heterostructure_same_u_and_fm(tmp_path):
    from ornl_lotf.vasp_driver import write_inputs

    results = {}
    for name, struct, task in [
        ("bulk", crte2_bulk(), {"is_slab": False, "asymmetric": False}),
        ("hetero", crte2_on_oxide_slab(), {"is_slab": True, "asymmetric": True}),
    ]:
        wd = tmp_path / name
        wd.mkdir()
        struct.to(filename=str(wd / "POSCAR"), fmt="poscar")
        results[name] = (write_inputs(wd, RECIPE, task), wd)

    (bulk, wd_b), (het, wd_h) = results["bulk"], results["hetero"]
    for inc, wd in ((bulk, wd_b), (het, wd_h)):
        symbols = (wd / "POSCAR").read_text().splitlines()[5].split()
        u = dict(zip(symbols, inc["LDAUU"]))
        assert inc["LDAU"] is True and u["Cr"] == 3.7
        assert all(v == 0.0 for k, v in u.items() if k != "Cr")
        assert inc["ISPIN"] == 2 and inc["IVDW"] == 12
        assert max(inc["MAGMOM"]) == 3.0
    assert "LDIPOL" not in bulk
    assert het["LDIPOL"] is True and het["IDIPOL"] == 3
    kz_b = int((wd_b / "KPOINTS").read_text().splitlines()[3].split()[2])
    kz_h = int((wd_h / "KPOINTS").read_text().splitlines()[3].split()[2])
    assert kz_b > 1 and kz_h == 1


@needs_potcars
def test_escalation_override_reaches_incar(tmp_path):
    from ornl_lotf.vasp_driver import write_inputs

    wd = tmp_path / "run"
    wd.mkdir()
    crte2_bulk().to(filename=str(wd / "POSCAR"), fmt="poscar")
    (wd / "task.json").write_text(json.dumps({"attempt": 1, "incar_overrides": {}}))
    (wd / "status.json").write_text(json.dumps({"converged": False}))
    (wd / "OUTCAR").write_text("x")
    assert disk_status(wd) == "unconverged"
    task = archive_and_escalate(wd, 1, [{}, {"ALGO": "All", "NELM": 200}])
    assert (wd / "attempts" / "1" / "OUTCAR").exists() and not (wd / "OUTCAR").exists()
    assert not list((wd / "attempts").rglob("POSCAR"))      # EFF discovery must not see archives
    inc = write_inputs(wd, RECIPE, task)
    assert inc["ALGO"] == "All" and inc["NELM"] == 200
