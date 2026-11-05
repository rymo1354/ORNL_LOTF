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
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

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
    {"ALGO": "All", "NELM": 200},
    {"ALGO": "All", "NELM": 300, "AMIX": 0.1, "BMIX": 0.01, "AMIX_MAG": 0.4, "BMIX_MAG": 0.01},
]

# Which parent stage feeds each compute stage, and the per-record flags it implies.
COMPUTE_STAGES = {
    "01_bulk_relax": {"parent": "00_bulk", "is_slab": False},
    "03_slab_relax": {"parent": "02_slabs", "is_slab": True},
    "06_single_points": {"parent": "05_variants", "is_slab": True},
}

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


def disk_status(wd: Path) -> str:
    if (wd / FINISHED_FILE).exists():
        return "done"
    if (wd / STATUS_FILE).exists():
        st = json.loads((wd / STATUS_FILE).read_text())
        return "unconverged" if "error" not in st else "failed"
    if (wd / "vasprun.xml").exists() or (wd / "OUTCAR").exists():
        return "failed"
    return "pending"


def archive_and_escalate(wd: Path, attempt: int, escalation: list[dict]) -> dict:
    """Move a failed attempt's outputs aside (keeps CONTCAR/WAVECAR to restart) and escalate."""
    dest = wd / "attempts" / str(attempt)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_FILES:
        if (wd / name).exists():
            shutil.move(str(wd / name), str(dest / name))
    if (wd / "CONTCAR").exists():
        shutil.copy(wd / "CONTCAR", dest / "CONTCAR")
    task = json.loads((wd / TASK_FILE).read_text())
    task["incar_overrides"] = dict(escalation[min(attempt, len(escalation) - 1)])
    (wd / TASK_FILE).write_text(json.dumps(task, indent=2))
    return task


def sync_manifest(stage_dir: Path) -> list[Record]:
    records = read_manifest(stage_dir)
    for r in records:
        r.status = disk_status(Path(r.path)) if r.status != "skipped" else "skipped"
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
    runs = stage_dir / "runs"

    materialize(stage, Path(recipe), config, stage_dir)
    for round_ in range(max_attempts):
        records = sync_manifest(stage_dir)
        todo = []
        for r in records:
            if r.status in ("done", "skipped"):
                continue
            if r.attempts >= max_attempts:
                continue
            wd = Path(r.path)
            if r.status in ("unconverged", "failed"):
                archive_and_escalate(wd, r.attempts, escalation)
            task = json.loads((wd / TASK_FILE).read_text())
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
