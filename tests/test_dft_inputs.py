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


# ---------------------------------------------------------------- crash / restart handling
def _slab(c=30.0, z=(10.0, 12.0, 14.0)):
    return Structure(Lattice.tetragonal(4.0, c), ["Ti", "O", "Ti"], [[0, 0, zi / c] for zi in z])


def test_vasp_error_banner_is_failed_not_interrupted(tmp_path):
    (tmp_path / "INCAR").write_text("x")
    (tmp_path / "OUTCAR").write_text("x")
    (tmp_path / "vasp.out").write_text("running\n")
    assert disk_status(tmp_path) == "interrupted"
    (tmp_path / "vasp.out").write_text("Error EDDDAV: Call to ZHEGV failed\nI REFUSE TO CONTINUE WITH THIS SICK JOB\n")
    assert disk_status(tmp_path) == "failed"


def test_matensemble_failure_record_marks_run_failed(tmp_path):
    import os, time
    from ornl_lotf.compute import crashed_runs

    wd = tmp_path / "runs" / "slab-X-sym"
    wd.mkdir(parents=True)
    (wd / "INCAR").write_text("x")
    (wd / "OUTCAR").write_text("x")               # no banner: abort came from another rank
    wf = tmp_path / "matensemble_workflow-1" / "out" / "chore-run_vasp-0001"
    wf.mkdir(parents=True)
    (wf / "chore.pickle").write_bytes(b"\x80\x04junk" + str(wd.resolve()).encode() + b"\x94more")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() + 5))
    (tmp_path / "matensemble_workflow-1" / "status.json").write_text(
        json.dumps({"failures": [{"chore_id": "chore-run_vasp-0001", "timestamp": stamp}]}))
    crashed = crashed_runs(tmp_path)
    assert str(wd.resolve()) in crashed
    assert disk_status(wd, crashed[str(wd.resolve())]) == "failed"
    # a failure older than the current attempt's INCAR belongs to an earlier attempt
    os.utime(wd / "INCAR", (time.time() + 60, time.time() + 60))
    assert disk_status(wd, crashed[str(wd.resolve())]) == "interrupted"


def test_exploded_contcar_is_discarded(tmp_path):
    from ornl_lotf.compute import geometry_broken

    parent = tmp_path / "parent.vasp"
    _slab().to(filename=str(parent), fmt="poscar")
    assert geometry_broken(_slab(), _slab()) is None
    assert "thicker" in geometry_broken(_slab(z=(10.0, 12.0, 24.0)), _slab())
    assert "apart" in geometry_broken(_slab(z=(10.0, 10.3, 14.0)), _slab())

    wd = tmp_path / "run"
    wd.mkdir()
    (wd / "task.json").write_text(json.dumps({"attempt": 1, "incar_overrides": {}}))
    _slab(z=(10.0, 12.0, 24.0)).to(filename=str(wd / "CONTCAR"), fmt="poscar")
    _slab(z=(10.0, 12.0, 24.0)).to(filename=str(wd / "POSCAR"), fmt="poscar")
    (wd / "WAVECAR").write_bytes(b"w")
    archive_and_escalate(wd, 1, [{}], escalate=False, reference=parent)
    assert not (wd / "CONTCAR").exists() and not (wd / "WAVECAR").exists()
    assert (wd / "attempts" / "1" / "CONTCAR.broken").exists()
    assert Structure.from_file(wd / "POSCAR").cart_coords[:, 2].max() == pytest.approx(14.0)


def test_empty_wavecar_removed(tmp_path):
    (tmp_path / "task.json").write_text(json.dumps({"attempt": 1, "incar_overrides": {}}))
    (tmp_path / "WAVECAR").write_bytes(b"")
    archive_and_escalate(tmp_path, 1, [{}, {"ALGO": "All"}])
    assert not (tmp_path / "WAVECAR").exists()


def test_incar_rules_match_and_merge():
    from ornl_lotf.compute import incar_rules_for
    from ornl_lotf.manifest import Record

    rules = [{"match": {"stage": "03_*", "asymmetric": True}, "incar": {"BMIX": 0.0001, "NELM": 300}},
             {"match": {"id": "slab-BaTiO3-tetragonal-*"}, "incar": {"ISPIN": 1, "NELM": 400}}]
    tet = Record(id="slab-BaTiO3-tetragonal-001-X", stage="03_slab_relax")
    cub = Record(id="slab-BaTiO3-cubic-001-X", stage="03_slab_relax")
    assert incar_rules_for(tet, "03_slab_relax", {"asymmetric": True}, rules) == {"BMIX": 0.0001, "NELM": 400, "ISPIN": 1}
    assert incar_rules_for(cub, "03_slab_relax", {"asymmetric": False}, rules) == {}
    assert incar_rules_for(cub, "06_single_points", {"asymmetric": True}, rules) == {}


@needs_potcars
def test_incar_base_then_overrides(tmp_path):
    from ornl_lotf.vasp_driver import write_inputs

    wd = tmp_path / "run"
    wd.mkdir()
    crte2_on_oxide_slab().to(filename=str(wd / "POSCAR"), fmt="poscar")
    task = {"is_slab": True, "asymmetric": True, "incar_base": {"ISPIN": 1, "NELM": 300, "BMIX": 0.0001},
            "incar_overrides": {"NELM": 400}}
    inc = write_inputs(wd, RECIPE, task)
    assert inc["ISPIN"] == 1 and "MAGMOM" not in inc
    assert inc["NELM"] == 400 and inc["BMIX"] == 0.0001 and inc["LDIPOL"] is True
