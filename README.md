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

## Secrets

The Materials Project key is read from the `MP_API_KEY` environment variable and is never stored in this repository.
