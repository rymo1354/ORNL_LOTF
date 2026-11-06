"""VASP driver for ORNL_LOTF compute stages (a ``dft_task`` for EFF's DFTMatEnsemble).

Dispatched by ``EnsembleFFFit.base.DFTMatEnsemble.run_individual`` as
``run_vasp_calculation([working_directory], [recipe_yaml])``. Based on the
Ensemble-FF-Fit example driver (examples/Perlmutter/VASP_MACE_ASE/DFT/vasp_dft.py,
including its confirmed stress sign); additions:

* Per-structure settings come from ``task.json`` in the working directory
  (written by ornl_lotf.compute from the manifest): slab flag, dipole flag,
  attempt number, ``incar_base`` (matching ``dft.incar_rules`` from the config)
  and ``incar_overrides`` (the escalation rung). Precedence: recipe < U/MAGMOM/
  dipole blocks < incar_base < incar_overrides.
* Hubbard U, FM MAGMOM, and the dipole correction are applied explicitly after
  pymatgen builds the INCAR (ornl_lotf.magnetism), independent of MPRelaxSet's
  oxide/fluoride rule.
* Flux runs this function once per MPI rank (mpi=True) and PMI joins the
  ranks' vasp_std processes into one job. Only rank 0 writes inputs and
  outputs; other ranks wait for rank 0's per-attempt ``.inputs_ready.<n>``
  marker before launching vasp_std.
* Convergence: writes ``status.json`` always, and ``properties.json`` (the
  EFF finished_file) only when the run is electronically converged and, for
  relaxations, ionically converged. Relaxations also write ``relaxed.vasp``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
import pymatgen.io.vasp.sets as vasp_input_sets
from pymatgen.core import SETTINGS, Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.sets import VaspInputSet

from ornl_lotf.magnetism import dipole_incar, hubbard_incar, initial_magmoms, kpoints_for

KBAR_TO_EV_PER_A3 = 6.241509074e-4
TASK_FILE = "task.json"
STATUS_FILE = "status.json"
FINISHED_FILE = "properties.json"
READY_TIMEOUT_S = 900


def _rank() -> int:
    for var in ("FLUX_TASK_RANK", "PMI_RANK", "SLURM_PROCID"):
        if var in os.environ:
            return int(os.environ[var])
    return 0


def _resolve_input_set(name: str):
    cls = getattr(vasp_input_sets, name, None)
    if not (isinstance(cls, type) and issubclass(cls, VaspInputSet)):
        raise ValueError(f"input_set={name!r} is not a pymatgen VaspInputSet subclass")
    return cls


def _pick_structure(wd: Path, structure_filename: str) -> tuple[Structure, Path]:
    contcar = wd / "CONTCAR"
    if contcar.is_file() and contcar.stat().st_size > 0:
        try:
            return Structure.from_file(contcar), contcar
        except Exception:
            pass
    path = wd / structure_filename
    return Structure.from_file(path), path


def load_task(wd: Path) -> dict:
    path = wd / TASK_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def write_inputs(wd: str | os.PathLike, recipe: dict, task: dict | None = None) -> dict:
    """Write INCAR/KPOINTS/POTCAR/POSCAR into ``wd``; return the final INCAR as a dict.

    Pure input generation (no VASP), so it is unit-testable on a login node.
    """
    wd = Path(wd)
    task = task if task is not None else load_task(wd)
    dft = recipe.get("dft", {})
    if recipe.get("pseudopotentials_directory"):
        SETTINGS["PMG_VASP_PSP_DIR"] = recipe["pseudopotentials_directory"]

    structure, source = _pick_structure(wd, recipe.get("structure_filename", "POSCAR"))
    InputSet = _resolve_input_set(recipe.get("input_set", "MPRelaxSet"))

    user_incar = deepcopy(recipe.get("user_incar_settings", {}))
    user_incar.update(task.get("incar_overrides", {}))
    iset = InputSet(structure, user_incar_settings=user_incar,
                    user_potcar_functional=recipe.get("potcar_functional", "PBE_54_W_HASH"))

    poscar = iset.poscar                      # input sets may re-sort sites; use this order
    ordered = poscar.structure
    incar = iset.incar
    for key in ("LDAUU", "LDAUL", "LDAUJ", "MAGMOM"):
        incar.pop(key, None)
    incar.update(hubbard_incar(poscar.site_symbols, dft.get("hubbard_u", {})))
    incar["ISPIN"] = 2
    incar["MAGMOM"] = initial_magmoms(ordered, dft.get("initial_magmom", {}))
    if task.get("asymmetric") and dft.get("dipole_correction", "auto") != "off":
        incar.update(dipole_incar(ordered))
    if "band_multiplier" in recipe and "NBANDS" not in user_incar:
        nelect, nions = iset.nelect, len(ordered)
        incar["NBANDS"] = int(max(nelect / 2 + nions / 2, nelect * recipe["band_multiplier"]))
    # per-structure rules (config dft.incar_rules, resolved into task.json by compute.py),
    # then escalation overrides, which win over everything (including the blocks above)
    incar.update(task.get("incar_base", {}))
    incar.update(task.get("incar_overrides", {}))
    if int(incar.get("ISPIN", 2)) == 1:
        incar.pop("MAGMOM", None)

    kppa = int(recipe.get("user_kpoints_settings", {}).get("grid_density", 1000))
    kpoints = kpoints_for(ordered, kppa, is_slab=bool(task.get("is_slab")))
    # KPAR cannot exceed the number of k-points (large heterostructure cells are often Gamma-only).
    # Upper bound on the irreducible count: time reversal pairs k with -k on a Gamma-centred grid.
    grid = [int(g) for g in kpoints.kpts[0]]
    n_irr = (int(np.prod(grid)) + int(np.prod([2 if g % 2 == 0 else 1 for g in grid]))) // 2
    if int(incar.get("KPAR", 1)) > max(1, n_irr):
        incar["KPAR"] = max(1, n_irr)

    incar.write_file(wd / "INCAR")
    kpoints.write_file(wd / "KPOINTS")
    iset.potcar.write_file(wd / "POTCAR")
    poscar.write_file(wd / "POSCAR")
    print(f"{wd}: inputs from {source.name}, {len(ordered)} atoms, kpts {kpoints.kpts[0]}")
    return dict(incar)


def _vasp_env() -> dict:
    env = dict(os.environ)
    if "VASP_LD_LIBRARY_PATH" in env:      # host libs only for vasp_std, never this process
        env["LD_LIBRARY_PATH"] = env["VASP_LD_LIBRARY_PATH"]
    return env


def parse_results(wd: Path, incar: dict, stress_tol_kbar: float = 2.0) -> dict:
    """Parse outputs, write status.json (+ properties.json, relaxed.vasp when converged)."""
    status = {"electronic_converged": False, "ionic_converged": False, "converged": False}
    vr_path = wd / "vasprun.xml"
    try:
        vr = Vasprun(vr_path, parse_dos=False, parse_eigen=False, parse_potcar_file=False,
                     exception_on_bad_xml=False)
    except Exception as exc:                              # crashed / truncated run
        status["error"] = f"vasprun.xml unreadable: {exc}"[:500]
        (wd / STATUS_FILE).write_text(json.dumps(status, indent=2))
        return status

    relaxing = int(incar.get("NSW", 0)) > 0 and int(incar.get("IBRION", -1)) >= 0
    status["electronic_converged"] = bool(vr.converged_electronic)
    status["ionic_converged"] = bool(vr.converged_ionic) if relaxing else True
    # A force-based EDIFFG (< 0) ignores stress, so a high-symmetry cell with zero forces
    # stops after one step without relaxing its lattice. For cell relaxations (ISIF >= 3)
    # also require the final stress to be below stress_tol_kbar (recipe key, default 2 kB).
    final_stress = vr.ionic_steps[-1].get("stress") if vr.ionic_steps else None
    if final_stress is not None:
        status["max_abs_stress_kbar"] = float(max(abs(x) for row in final_stress for x in row))
    if relaxing and int(incar.get("ISIF", 2)) >= 3 and final_stress is not None:
        tol = float(stress_tol_kbar)
        status["stress_converged"] = status["max_abs_stress_kbar"] <= tol
        status["ionic_converged"] = status["ionic_converged"] and status["stress_converged"]
    status["converged"] = status["electronic_converged"] and status["ionic_converged"]
    status["n_ionic_steps"] = len(vr.ionic_steps)
    (wd / STATUS_FILE).write_text(json.dumps(status, indent=2))
    if not status["converged"]:
        return status

    final = vr.ionic_steps[-1]
    forces = final["forces"]
    props = {
        "energy": vr.final_energy,
        "energy_per_atom": vr.final_energy / len(vr.final_structure),
        "fx": [f[0] for f in forces], "fy": [f[1] for f in forces], "fz": [f[2] for f in forces],
    }
    stress = final.get("stress")
    if stress is not None:       # ASE-Voigt sign convention, as in the EFF example driver
        props["stress_voigt"] = [-stress[i][j] * KBAR_TO_EV_PER_A3
                                 for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))]
    try:
        out = Outcar(wd / "OUTCAR")
        props["total_magnetization"] = out.total_mag
        props["site_magnetization"] = [m.get("tot") for m in out.magnetization]
    except Exception:
        pass
    if relaxing:
        vr.final_structure.to(filename=str(wd / "relaxed.vasp"), fmt="poscar")
        lat = vr.final_structure.lattice
        props["final_lattice_abc"] = list(lat.abc)
        props["final_lattice_angles"] = list(lat.angles)
    (wd / FINISHED_FILE).write_text(json.dumps(props, indent=2))
    return status


def run_vasp_calculation(working_directory_list, vasp_yaml_list):
    rank = _rank()
    for working_directory, vasp_yaml in zip(working_directory_list, vasp_yaml_list):
        wd = Path(working_directory)
        recipe = yaml.safe_load(Path(vasp_yaml).read_text())
        task = load_task(wd)
        if task.get("skip"):            # dropped from the stage (task.json "skip": "<reason>"); every rank exits
            print(f"{wd}: skipped ({task['skip']})")
            continue
        ready = wd / f".inputs_ready.{task.get('attempt', 0)}"

        if rank == 0:
            for old in wd.glob(".inputs_ready.*"):
                old.unlink()
            (wd / STATUS_FILE).unlink(missing_ok=True)
            incar = write_inputs(wd, recipe, task)
            (wd / "incar_used.json").write_text(json.dumps(incar, indent=2, default=str))
            ready.touch()
        else:
            t0 = time.time()
            while not ready.exists():
                if time.time() - t0 > READY_TIMEOUT_S:
                    raise TimeoutError(f"rank {rank}: {ready} never appeared")
                time.sleep(1)

        command = recipe.get("command", recipe.get("vasp_executable", "vasp_std"))
        # every rank keeps its own output: when a non-zero rank calls MPI_Abort, its error
        # banner is the only record of why (rank 0's vasp.out then just stops)
        log = open(wd / ("vasp.out" if rank == 0 else f"vasp.out.rank{rank}"), "w")
        try:
            rc = subprocess.run(command, shell=True, cwd=wd, close_fds=False,
                                stdout=log, stderr=subprocess.STDOUT, env=_vasp_env()).returncode
        finally:
            log.close()
        if rank != 0:
            continue
        status = parse_results(wd, incar, recipe.get("stress_tol_kbar", 2.0))
        status["returncode"] = rc
        (wd / STATUS_FILE).write_text(json.dumps(status, indent=2))
        print(f"{wd}: rc={rc} converged={status['converged']}")
    return {"status": "complete"}


if __name__ == "__main__":
    from EnsembleFFFit.utilities.general import parse_list

    run_vasp_calculation(parse_list(sys.argv[1]), parse_list(sys.argv[2]))
