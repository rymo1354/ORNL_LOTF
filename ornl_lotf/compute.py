"""GPU compute stages: manifest -> run tree -> EFF DFTMatEnsemble + MatEnsemble Pipeline.

Layout of a compute stage directory::

    stages/01_bulk_relax/
      manifest.jsonl
      runs/
        bulk_relax.yml            # recipe merged with config['dft'] (exactly one per tree)
        <record id>/POSCAR        # from the parent stage's structure
        <record id>/task.json     # per-structure flags + attempt + escalation overrides
        <record id>/attempts/<n>/ # archived outputs of failed attempts (no POSCAR inside)

Ensemble-FF-Fit's DFTMatEnsemble discovers every ``POSCAR`` under ``runs/`` and
skips directories containing the finished file (``properties.json``), so
re-running a stage is safe. Unconverged runs are retried in the same allocation
up to ``resources.max_attempts`` times, climbing ``resources.escalation``
(entry k-1 is applied on attempt k; entry 0 is normally empty).
"""
from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

import yaml
from pymatgen.core import Structure

from ornl_lotf.manifest import Record, read_manifest, write_manifest
from ornl_lotf.vasp_driver import FINISHED_FILE, STATUS_FILE, TASK_FILE

DRIVER = str(Path(__file__).with_name("vasp_driver.py"))
FLUX_PYTHON_BINDINGS = "/usr/local/lib/python3.12/site-packages"
REPO = str(Path(__file__).resolve().parents[1])

# Host VASP 6.4.2-gpu runtime (copied from EFF VASP_MACE_ASE run_pipeline.py; applied
# only to vasp_std's own subprocess by the driver, never to Python processes).
VASP_LD_LIBRARY_PATH = (
    "/usr/local/lib/flux:"
    "/opt/cray/pe/lib64:/opt/cray/libfabric/1.22.0/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/lib:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/extras/qd/lib:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/math_libs/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/lib64:"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/lib64:"
    "/global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1/lib:"
    "/host_lib64:/host_lib"
)

DEFAULT_ESCALATION = [
    {},
    {"ALGO": "VeryFast", "NELM": 300},
    {"ALGO": "VeryFast", "NELM": 300, "AMIX": 0.05, "BMIX": 0.0001, "AMIX_MAG": 0.2, "BMIX_MAG": 0.0001},
]

# Text VASP prints before MPI_Abort. MPI_Abort kills rank 0's Python too, so no status.json
# is written; these markers (or a MatEnsemble failure record, see crashed_runs) tell a
# crash apart from a walltime kill.
CRASH_MARKERS = ("I REFUSE TO CONTINUE", "MPI_Abort", "Error EDDDAV", "ZBRENT: fatal")
CRASH_LOGS = ("vasp.out", "OUTCAR")          # plus vasp.out.rank<N> written by non-zero ranks

# A restart geometry is discarded (the run restarts from the parent structure) when it has
# atoms closer than MIN_BOND_A, or its slab got thicker than the parent by more than
# MAX_THICKENING_A (atoms leaving the slab / crossing the vacuum).
MIN_BOND_A = 0.7
MAX_THICKENING_A = 3.0

# Which parent stage feeds each compute stage, and the per-record flags it implies.
COMPUTE_STAGES = {
    "01_bulk_relax": {"parent": "00_bulk", "is_slab": False},
    "03_slab_relax": {"parent": "02_slabs", "is_slab": True},
    "06_single_points": {"parent": "05_variants", "is_slab": True},
    # tree plan (2026-11-06): gap scan on the pristine, unstrained Stage 4 structures first
    "06a_gap_scan": {"parent": "04_heterostructures", "is_slab": True},
}


def select_parents(parents: list[Record], sel: dict | None) -> list[Record]:
    """Filter parent records by ``config.compute_select[stage]``: ``ids`` (globs) and/or ``extra``
    equality filters, e.g. ``{extra: {gap_role: [start]}}``. With no selection, every record is used."""
    if not sel:
        return parents
    out = []
    for p in parents:
        if "ids" in sel and not any(fnmatch.fnmatch(p.id, g) for g in sel["ids"]):
            continue
        if any(p.extra.get(k) not in (v if isinstance(v, list) else [v]) for k, v in sel.get("extra", {}).items()):
            continue
        out.append(p)
    return out

ARCHIVE_FILES = ("INCAR", "KPOINTS", "OUTCAR", "OSZICAR", "vasprun.xml", "vasp.out",
                 STATUS_FILE, "incar_used.json", "EIGENVAL", "DOSCAR", "PROCAR", "XDATCAR", "REPORT")


def vasp_container_env(cores_per_task: int) -> dict:
    return {
        "PYTHONPATH": f"{FLUX_PYTHON_BINDINGS}:{REPO}",
        "OMP_NUM_THREADS": str(cores_per_task),
        "VASP_LD_LIBRARY_PATH": VASP_LD_LIBRARY_PATH,
        "PSEUDOPOTENTIAL_DIR": "/global/common/software/nersc9/vasp/dependencies/pseudopotentials",
        "VDW_KERNEL_DIR": "/global/common/software/nersc9/vasp/dependencies/vdw_kernel",
        "NO_STOP_MESSAGE": "1",
        "MPICH_NO_BUFFER_ALIAS_CHECK": "1",
    }


def vasp_resource_estimator(structure, input_args, options):
    """Same atoms-per-node estimate as the EFF example (recipe 'nodes' > 0 pins it)."""
    nodes = input_args.get("nodes", 0)
    if nodes and nodes > 0:
        return nodes
    apn = input_args.get("atoms_per_node", options.get("atoms_per_node", 100))
    return max(1, -(-len(structure) // apn))


# --------------------------------------------------------------------------- run tree
def materialize(stage: str, recipe_path: Path, config: dict, stage_dir: Path) -> list[Record]:
    """Create/merge this stage's manifest and run tree from the parent stage."""
    spec = COMPUTE_STAGES[stage]
    parent_dir = stage_dir.parent / spec["parent"]
    parents = [r for r in read_manifest(parent_dir) if r.status == "done"]
    parents = select_parents(parents, config.get("compute_select", {}).get(stage))
    if not parents:
        raise RuntimeError(f"{stage}: no 'done' records in {parent_dir}/manifest.jsonl")

    runs = stage_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["dft"] = config.get("dft", {})
    (runs / recipe_path.name).write_text(yaml.safe_dump(recipe, sort_keys=False))

    existing = {r.id: r for r in read_manifest(stage_dir)}
    records = []
    for p in parents:
        if p.id in existing:
            records.append(existing[p.id])
            continue
        wd = runs / p.id
        wd.mkdir(parents=True, exist_ok=True)
        src = Path(p.path)
        Structure.from_file(src).to(filename=str(wd / "POSCAR"), fmt="poscar")
        is_slab = bool(p.extra.get("is_slab", spec["is_slab"]))
        escalation = config.get("resources", {}).get("escalation", DEFAULT_ESCALATION)
        task = {"id": p.id, "is_slab": is_slab, "asymmetric": bool(p.asymmetric), "attempt": 0,
                "incar_overrides": dict(escalation[0]) if escalation else {}}
        (wd / TASK_FILE).write_text(json.dumps(task, indent=2))
        rec = replace(p, stage=stage, parent=p.id, path=str(wd), status="pending", attempts=0,
                      extra={**p.extra, "parent_path": str(src)})
        records.append(rec)
    write_manifest(stage_dir, records)
    return records


def crashed_runs(stage_dir: Path) -> dict[str, float]:
    """Run directories whose MatEnsemble chore failed (e.g. PMI_Abort), with the failure time.

    Read from every ``matensemble_workflow-*/status.json`` failure record. The chore's
    pickle names its working directory as a plain string.
    """
    runs = str((Path(stage_dir) / "runs").resolve())
    pat = re.compile(re.escape(runs).encode() + rb"/[A-Za-z0-9_.+=-]+")
    out: dict[str, float] = {}
    for st in Path(stage_dir).glob("matensemble_workflow-*/status.json"):
        try:
            failures = json.loads(st.read_text()).get("failures", [])
        except (OSError, ValueError):
            continue
        for f in failures:
            pkl = st.parent / "out" / str(f.get("chore_id")) / "chore.pickle"
            try:
                t = datetime.fromisoformat(f["timestamp"].replace("Z", "+00:00")).timestamp()
                paths = set(pat.findall(pkl.read_bytes()))
            except (OSError, KeyError, ValueError):
                continue
            for raw in paths:
                wd = raw.decode()
                out[wd] = max(out.get(wd, 0.0), t)
    return out


def _vasp_crashed(wd: Path) -> bool:
    for name in [*CRASH_LOGS, *(p.name for p in wd.glob("vasp.out.rank*"))]:
        try:
            with open(wd / name, "rb") as fh:
                fh.seek(max(0, (wd / name).stat().st_size - 200_000))
                tail = fh.read().decode(errors="replace")
        except OSError:
            continue
        if any(m in tail for m in CRASH_MARKERS):
            return True
    return False


def disk_status(wd: Path, crashed_at: float | None = None) -> str:
    """done | unconverged | failed | interrupted | pending, from the files in ``wd``.

    ``crashed_at``: time of a MatEnsemble failure record for this directory. It counts
    only if it is newer than the current attempt's INCAR (older records belong to
    earlier attempts).
    """
    if (wd / FINISHED_FILE).exists():
        return "done"
    if (wd / STATUS_FILE).exists():
        st = json.loads((wd / STATUS_FILE).read_text())
        return "unconverged" if "error" not in st else "failed"
    if (wd / "vasprun.xml").exists() or (wd / "OUTCAR").exists():
        incar_t = (wd / "INCAR").stat().st_mtime if (wd / "INCAR").exists() else 0.0
        if _vasp_crashed(wd) or (crashed_at is not None and crashed_at >= incar_t):
            return "failed"         # VASP aborted: escalate, don't just resubmit
        return "interrupted"        # walltime: outputs but the driver never wrote status.json
    return "pending"


def _read_structure(path: Path) -> Structure | None:
    try:
        if path.is_file() and path.stat().st_size > 0:
            return Structure.from_file(path)
    except Exception:
        pass
    return None


def slab_thickness(structure: Structure) -> float:
    """Extent of the atoms along c (Å), measured across the largest vacuum gap."""
    z = np.sort(np.mod(structure.frac_coords[:, 2], 1.0))
    gaps = np.diff(np.concatenate([z, [z[0] + 1.0]]))
    return float((1.0 - gaps.max()) * structure.lattice.c)


def geometry_broken(structure: Structure, reference: Structure | None = None) -> str | None:
    """Reason a restart geometry must not be used, or None if it looks sane."""
    if len(structure) > 1:
        dmat = structure.distance_matrix
        np.fill_diagonal(dmat, np.inf)
        if dmat.min() < MIN_BOND_A:
            return f"atoms {dmat.min():.2f} A apart"
    if reference is not None:
        grow = slab_thickness(structure) - slab_thickness(reference)
        if grow > MAX_THICKENING_A:
            return f"slab {grow:.1f} A thicker than its parent"
    return None


def archive_and_escalate(wd: Path, attempt: int, escalation: list[dict], escalate: bool = True,
                         reference: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """Move a failed attempt's outputs aside (keeps CONTCAR/WAVECAR to restart) and escalate.

    With ``escalate=False`` (interrupted runs) the current INCAR overrides are kept.
    Empty restart files are removed. If CONTCAR is broken (see geometry_broken, compared
    with ``reference``, normally the parent-stage structure), CONTCAR and WAVECAR are moved
    aside and POSCAR is reset to ``reference``, so the next attempt starts over.
    """
    dest = wd / "attempts" / str(attempt)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        if (wd / name).exists():
            shutil.move(str(wd / name), str(dest / name))
    for extra in wd.glob("vasp.out.rank*"):
        shutil.move(str(extra), str(dest / extra.name))
    for name in ("CONTCAR", "WAVECAR", "CHGCAR"):
        if (wd / name).exists() and (wd / name).stat().st_size == 0:
            (wd / name).unlink()
    if (wd / "CONTCAR").exists():
        shutil.copy(wd / "CONTCAR", dest / "CONTCAR")
        ref = _read_structure(reference) if reference else None
        cont = _read_structure(wd / "CONTCAR")
        why = "unreadable" if cont is None else geometry_broken(cont, ref)
        if why:
            log(f"{wd.name}: discarding CONTCAR ({why}); restarting from the parent structure")
            shutil.move(str(wd / "CONTCAR"), str(dest / "CONTCAR.broken"))
            (wd / "WAVECAR").unlink(missing_ok=True)
            if ref is not None:
                ref.to(filename=str(wd / "POSCAR"), fmt="poscar")
    task = json.loads((wd / TASK_FILE).read_text())
    if escalate:
        task["incar_overrides"] = dict(escalation[min(attempt, len(escalation) - 1)])
    (wd / TASK_FILE).write_text(json.dumps(task, indent=2))
    return task


def incar_rules_for(record: Record, stage: str, task: dict, rules: list[dict]) -> dict:
    """Merge the ``dft.incar_rules`` entries that match this record (later rules win).

    Each rule is ``{match: {...}, incar: {...}}``. Match keys, all optional and ANDed:
    ``id`` (glob), ``stage`` (glob), ``asymmetric`` / ``is_slab`` (bool, from task.json).
    """
    merged: dict = {}
    for rule in rules or []:
        m = rule.get("match", {})
        if "id" in m and not fnmatch.fnmatch(record.id, m["id"]):
            continue
        if "stage" in m and not fnmatch.fnmatch(stage, m["stage"]):
            continue
        if any(k in m and bool(task.get(k)) != bool(m[k]) for k in ("asymmetric", "is_slab")):
            continue
        merged.update(rule.get("incar", {}))
    return merged


def sync_manifest(stage_dir: Path) -> list[Record]:
    records = read_manifest(stage_dir)
    crashed = crashed_runs(stage_dir)
    for r in records:
        if r.status == "skipped":
            continue
        wd = Path(r.path)
        if json.loads((wd / TASK_FILE).read_text()).get("skip"):
            r.status = "skipped"
            continue
        r.status = disk_status(wd, crashed.get(str(wd.resolve())))
    write_manifest(stage_dir, records)
    return records


# --------------------------------------------------------------------------- submission
def submit_round(runs: Path, recipe_name: str, wds: set[str], res: dict,
                 log: Callable[[str], None] = print) -> None:
    from EnsembleFFFit.base import DFTMatEnsemble
    from matensemble.model import Resources
    from matensemble.pipeline import Pipeline

    options = {"structure_filename": "POSCAR", "dft_recipe": recipe_name,
               "atoms_per_node": res.get("atoms_per_node", 100)}
    dft = DFTMatEnsemble(str(runs), str(runs), **options)
    dcts = dft.build_dft_dcts(DRIVER, ["dft_recipe"], "run_vasp_calculation",
                              finished_file=FINISHED_FILE, resource_estimator=vasp_resource_estimator)
    dcts = [d for d in dcts if str(Path(d["working_directory"]).resolve()) in wds]
    prio = res.get("priority", [])               # globs on run-directory names, submitted first, in order
    rank = lambda d: next((i for i, g in enumerate(prio) if fnmatch.fnmatch(Path(d["working_directory"]).name, g)),
                          len(prio))
    dcts.sort(key=rank)
    if not dcts:
        log("nothing to submit")
        return

    cores, gpus = res.get("cores_per_task", 15), res.get("gpus_per_task", 1)
    rpn, max_nodes = res.get("ranks_per_node", 4), res.get("max_nodes_per_job", 4)
    env = vasp_container_env(cores)
    pipe = Pipeline(reserve_broker_node=False)

    @pipe.chore(name="run_vasp", num_tasks=1, cores_per_task=cores, gpus_per_task=gpus,
                env=env, inherit_env=True, mpi=True)
    def run_vasp_chore(task_dict):
        return DFTMatEnsemble.run_individual(task_dict)

    for d in dcts:
        nodes = min(d["allocated_nodes"], max_nodes)
        d = {**d, "allocated_nodes": nodes}
        log(f"submit {d['working_directory']}: {nodes} node(s) x {rpn} ranks x {cores} cores x {gpus} GPU")
        pipe.call("run_vasp", d, resources=Resources(num_tasks=nodes * rpn, cores_per_task=cores,
                                                      gpus_per_task=gpus, env=env, inherit_env=True,
                                                      mpi=True))
    pipe.submit(log_delay=25, set_gpu_affinity=True, set_cpu_affinity=False).result()


def run_compute_stage(stage: str, recipe: Path, config: dict, stage_dir: Path) -> list[Record]:
    res = config.get("resources", {})
    max_attempts = int(res.get("max_attempts", 3))
    escalation = res.get("escalation", DEFAULT_ESCALATION)
    rules = config.get("dft", {}).get("incar_rules", [])
    runs = stage_dir / "runs"

    materialize(stage, Path(recipe), config, stage_dir)
    for round_ in range(max_attempts):
        records = sync_manifest(stage_dir)
        todo = []
        for r in records:
            if r.status in ("done", "skipped"):
                continue
            if r.status == "pending" and r.attempts > 0:
                r.attempts -= 1          # submitted but never started (e.g. walltime): not an attempt
            if r.attempts >= max_attempts:
                continue
            wd = Path(r.path)
            print(f"[{stage}] {r.id}: {r.status} after {r.attempts} attempt(s)")
            if r.status in ("unconverged", "failed", "interrupted"):
                parent = r.extra.get("parent_path")
                archive_and_escalate(wd, r.attempts, escalation, escalate=r.status != "interrupted",
                                     reference=Path(parent) if parent else None)
            task = json.loads((wd / TASK_FILE).read_text())
            base = incar_rules_for(r, stage, task, rules)
            if base != task.get("incar_base", {}):
                # spin/mixing setup changed: an old WAVECAR may not match (e.g. ISPIN 2 -> 1)
                (wd / "WAVECAR").unlink(missing_ok=True)
            task["incar_base"] = base
            task["attempt"] = r.attempts + 1
            (wd / TASK_FILE).write_text(json.dumps(task, indent=2))
            r.attempts += 1
            r.status = "running"
            todo.append(r)
        write_manifest(stage_dir, records)
        if not todo:
            break
        print(f"[{stage}] round {round_ + 1}: {len(todo)} job(s)")
        with contextlib.chdir(stage_dir):        # MatEnsemble writes matensemble_workflow-*/ to cwd
            submit_round(runs, Path(recipe).name, {str(Path(r.path).resolve()) for r in todo}, res)

    records = sync_manifest(stage_dir)
    summary = {s: sum(r.status == s for r in records) for s in {r.status for r in records}}
    print(f"[{stage}] final status: {summary}")
    if any(r.status != "done" for r in records):
        print(f"[{stage}] some runs are not done. Re-running the stage retries records with "
              f"attempts < resources.max_attempts ({max_attempts}); raise it (attempt counters "
              "persist in the manifest) or inspect runs/<id>/attempts/ first.", file=sys.stderr)
    return records
