# ORNL_LOTF

ORNL_LOTF is a MatEnsemble pipeline. It builds CrTe2 heterostructures on Al2O3 (111), BaTiO3 (100) and BaTiO3 (111), generates variants of them (point defects, strain, CrTe2 thickness, interfacial distance), and runs DFT single points on those variants on NERSC Perlmutter.

The project plan is in `../BUILD_OUT.md`, and the investigation of the provided materials is in `docs/FINDINGS.md`.

## Layout

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Single entry point: `python run_pipeline.py --stage <name>` |
| `config/pipeline.yaml` | MP IDs, DFT settings, matching tolerances, scans, defect specs, resources |
| `config/vasp/*.yml` | VASP recipes, one per compute stage |
| `ornl_lotf/` | Importable construction, manifest and DFT logic |
| `notebooks/` | Thin construction-stage notebooks, run headlessly with papermill |
| `container/` | podman-hpc Dockerfile, build and launch scripts, JupyterHub kernel |
| `../stages/` | Stage directories and manifests (git-ignored) |

## Stages

| Stage | Type | Name |
|---|---|---|
| 0 | CPU | `00_bulk` |
| 1 | GPU | `01_bulk_relax` |
| 2 | CPU | `02_slabs` |
| 3 | GPU | `03_slab_relax` |
| 4 | CPU | `04_heterostructures` |
| 5 | CPU | `05_variants` |
| 6 | GPU | `06_single_points` |
| 7 | CPU | `07_analysis` |

Each stage reads only the previous stage's directory and `manifest.jsonl`. It writes its own outputs and, for construction stages, an executed notebook.

## Container and kernel

These commands all run on a Perlmutter login node from the repo root.

**Build the image.** `container/build_image.sh v0.1` builds `localhost/ornl_lotf:v0.1` and migrates it. It also writes `container/pip_freeze.v0.1.txt`, which lists every package version the build resolved.

The image is `nersc/flux:26.05` plus the following pinned components. Each pin is a `--build-arg` in `container/Dockerfile` and can be overridden at build time:

| Component | Pin |
|---|---|
| MatEnsemble | `5d8be305f1` (v0.5.8) |
| Ensemble-FF-Fit | `frontier-rmg-update` branch |
| HeteroBuilder (`vdW_structures`) | `9c7293c`, the rank-drift fix |
| Notebook stack | pymatgen, mp-api, papermill, jupyter, py3Dmol |

No MACE layers are added. The base image is already about 40 GB and ships torch 2.11+cu128, so the final image is about the base plus 3 GB. VASP is bind-mounted from the host at run time.

**Check the image.**

```bash
podman-hpc run --rm -v $SCRATCH:$SCRATCH -e PYTHONPATH=$PWD -w $PWD localhost/ornl_lotf:v0.1 \
    python -m pytest -q tests
```

**Install the JupyterHub kernel.** Run `container/jupyter/install_kernel.sh`, then open `notebooks/container_check.ipynb` with the **ORNL_LOTF (container)** kernel. The kernel runs `podman-hpc run --network=host`, bind-mounts `/tmp`, `$HOME` and `$SCRATCH`, and puts this checkout on `PYTHONPATH`.

**Run a stage.**

| Stage type | Command |
|---|---|
| Construction (CPU) | `STAGE=00_bulk sbatch container/launch_cpu.slurm` (account `m2113`) |
| Compute (GPU) | `STAGE=01_bulk_relax sbatch container/launch_gpu.slurm` (account `m2113_g`) |
| Flux check (GPU) | `STAGE=flux_smoke sbatch -N 2 container/launch_gpu.slurm` |

The Flux check confirms that the brokers on separate nodes merge into a single Flux instance.

Both launchers run `container/flux_run.sh` inside the container. That script writes a static Flux `R.json` and calls `run_pipeline.py` from inside `flux start`.

## Secrets

The Materials Project key is read from the `MP_API_KEY` environment variable and is never stored in this repository.
