#!/usr/bin/env python
"""ORNL_LOTF stage launcher.

    python run_pipeline.py --stage 00_bulk [--config config/pipeline.yaml]

Construction stages (CPU) execute their notebook headlessly with papermill and
save an executed copy into the stage directory. Compute stages (GPU) build a
run tree from the previous stage's manifest and submit VASP through
Ensemble-FF-Fit's DFTMatEnsemble + MatEnsemble Pipeline (must be run inside
``flux start``; see container/launch_gpu.slurm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent

# name -> (kind, notebook or compute recipe)
STAGES: dict[str, tuple[str, str]] = {
    "00_bulk": ("construct", "notebooks/00_bulk.ipynb"),
    "01_bulk_relax": ("compute", "config/vasp/bulk_relax.yml"),
    "02_slabs": ("construct", "notebooks/02_slabs.ipynb"),
    "03_slab_relax": ("compute", "config/vasp/slab_relax.yml"),
    "04_heterostructures": ("construct", "notebooks/04_heterostructures.ipynb"),
    "05_variants": ("construct", "notebooks/05_variants.ipynb"),
    "06_single_points": ("compute", "config/vasp/single_point.yml"),
    "07_analysis": ("construct", "notebooks/07_analysis.ipynb"),
}


def stage_dir(config: dict, stage: str) -> Path:
    root = Path(config["stages_root"])
    if not root.is_absolute():
        root = (REPO / root).resolve()
    d = root / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_construct(stage: str, notebook: str, config_path: Path, config: dict) -> None:
    import papermill as pm

    out_dir = stage_dir(config, stage)
    pm.execute_notebook(
        str(REPO / notebook),
        str(out_dir / f"{stage}.executed.ipynb"),
        parameters={"config_path": str(config_path), "stage": stage, "stage_dir": str(out_dir)},
        cwd=str(out_dir),
        kernel_name=config.get("papermill_kernel", "python3"),
    )


def run_compute(stage: str, recipe: str, config_path: Path, config: dict) -> None:
    from ornl_lotf.compute import run_compute_stage

    run_compute_stage(stage=stage, recipe=REPO / recipe, config=config, stage_dir=stage_dir(config, stage))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    kind, target = STAGES[args.stage]
    if kind == "construct":
        run_construct(args.stage, target, config_path, config)
    else:
        run_compute(args.stage, target, config_path, config)


if __name__ == "__main__":
    main()
