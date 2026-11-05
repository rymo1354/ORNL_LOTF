# Section 4 Findings (BUILD_OUT.md)

Investigated 2026-11-05. All paths are absolute on Perlmutter. `EFF` = `/pscratch/sd/r/rym/LOTF/Ensemble-FF-Fit`
(clone of `Q-CAD/Ensemble-FF-Fit`, branch `frontier-rmg-update`, HEAD `ac5702c`). `EX` =
`EFF/examples/Perlmutter/VASP_MACE_ASE`.

## (a) Structures, MatEnsemble and HeteroBuilder in the construction templates

- `provided/Construct_LAMMPs.ipynb` (kernel `pyrmg`, Python 3.11):
  - It works only with pymatgen `Structure` objects read from and written to POSCAR.
  - From HeteroBuilder (the import name is **`vdW_structures`**) it uses:
    - `VdWStructure(structure, minimum_vdW_gap=..., shift_gap=True)`
    - `.vdW_layers`, `.vdW_spacings` and `.structure`
    - `.extend_vdW_layers([0, 1, 2, ...])` to select, reorder and repeat layers
    - `vdW_structures.viewer.view_structure` for py3Dmol views
  - Supercells come from `CubicSupercellTransformation` and `make_supercell`. The hand-written `stack_substrate_and_film(gap, vacuum)` assumes the in-plane cells already match and that c lies along z.
  - **MatEnsemble is not used**, and neither is lattice matching.
- The matching example is `/pscratch/sd/r/rym/Bi-Se_Heteros_w_P_tar/Bi_Se_Heteros.ipynb`. It includes an Al2O3 mp-1143 substrate.
  - The slab is built with `SlabGenerator(structure, facet, min_slab_size, min_vacuum_size)`, then a slab index is chosen, it is optionally inverted, a z-cut is applied, and `set_vacuum(slab, 2.4)` is set. The vacuum is kept just above `minimum_vdW_gap=2.2`, so the whole slab counts as one vdW layer.
  - `VdWStructure(...).extend_vdW_layers(...)` is applied to both the substrate and the film.
  - Matching uses `ZSLGenerator(max_area_ratio_tol, max_area, max_length_tol, max_angle_tol)` followed by `VdWHeterostructureGenerator().generate_vdW_heterostructures(film=, substrate=, zsl_generator=, vacuum_over_film=, get_unique=False)`.
  - Results are filtered by atom count and the "most cubic" cell is selected.
  - Interfacial and exfoliation gaps are scanned with EFF's `generate_vdW_spacing`, which calls `z_solve_shift_vdW_layers`.
- MatEnsemble is a Flux task orchestrator, not a structure library. The current API is `matensemble.pipeline.Pipeline`, `@pipe.chore`, `pipe.call(..., resources=Resources(...))` and `pipe.submit()`.

## (b) How EFF builds, submits and checks VASP jobs

- `EX/run_pipeline.py --config workflow_config.yaml --stage converge_dft_data` runs `run_converge_dft_data`, which:
  - calls `EnsembleFFFit.base.DFTMatEnsemble(dir, dir, **opts).build_dft_dcts(dft_task, check_files, entry_point, finished_file, resource_estimator)`;
  - registers one `run_vasp` chore and calls `pipe.call` for each structure, with `Resources(num_tasks=nodes*4, cores_per_task, gpus_per_task=1, mpi=True)`;
  - submits with `pipe.submit(set_gpu_affinity=True)`.
- The layout is one MPI rank per GPU with `cores_per_task` OpenMP threads, and pmi2 joins the ranks. Nodes per job = ceil(natoms/atoms_per_node), capped at `max_nodes_per_job`.
- `EX/DFT/vasp_dft.py::run_vasp_calculation([wd], [recipe])` builds the inputs:
  - It uses the pymatgen input set named by `input_set` (default `MPRelaxSet`) plus `user_incar_settings` and `user_kpoints_settings` (`grid_density`). POTCARs come from `pseudopotentials_directory`, and NBANDS is set automatically.
  - It prefers CONTCAR over POSCAR on a restart.
  - It runs `/host_lib/ld-linux-x86-64.so.2 <vasp_std>` with `VASP_LD_LIBRARY_PATH`.
  - It checks `Vasprun.converged` and writes `properties.json` (energy, forces, stress).
- Recipe INCAR (`EX/DFT/vdW_single_point.yml`): ALGO Normal, EDIFF 1e-5, EDIFFG -0.01, ISIF 2, NPAR 16, KPAR 4, ISMEAR 0, SIGMA 0.03, IVDW 12, NSW 0, SYMPREC 1e-6, NELM 100.
  - MAGMOM and U come implicitly from MPRelaxSet, which gives no U to non-oxides.
  - The recipe has no LDIPOL.
- Convergence: a rerun skips any directory that has `properties.json`. There is no resubmission or escalation, and no relaxed-structure export.
- `EFF/EnsembleFFFit/density_functional_theory/vasp/vasp_dft.py` (the package copy) differs from the example. It does not negate stress and is marked UNVERIFIED. **Use the example driver as the reference.**

## (c) Can it be pointed at an arbitrary stage directory and manifest?

- **Directory: yes, without modification.** The pipeline discovers work from a root containing `**/POSCAR` and **exactly one** recipe yaml, with `dft_task` set to any driver path.
- **Manifest: no.** There is no manifest support. ORNL_LOTF therefore writes the `runs/<id>/POSCAR` tree from its manifest, calls `DFTMatEnsemble` and `Pipeline`, and syncs status back to the manifest.
- **Hard-coded values:** VASP 6.4.2-gpu, the NVHPC 25.9/26.5 LD paths, POTCAR, the vdW kernel paths, and 4 GPUs / 64 cores per node.

## (d) The podman-hpc container

- `EX/Dockerfile.mace` is based on `FROM nersc/flux:26.05`. It installs MatEnsemble (cloned **unpinned**; `PIPELINE_CHECKS_FROM_FRONTIER.md` recommends `5d8be305f1`, v0.5.8), then torch, MACE and the pymatgen stack.
  - VASP is **not** in the image; it is bind-mounted from `/global/common/software/nersc9/vasp`.
  - EFF is put on PYTHONPATH from a bind mount.
- **Build:** `podman-hpc build -f Dockerfile.mace -t matensemble:mace .`, then `podman-hpc migrate matensemble:mace`.
- **Launch** (`EX/launch_multi_node.slurm`):
  - `srun -N N -n N --mpi=pmi2 podman-hpc run --gpu --privileged ... -v /usr/lib64:/host_lib64 -v /lib64:/host_lib -v <vasp,nvhpc,cray> -v $SCRATCH:$SCRATCH -w $PWD <image> bash <script>`
  - The script writes a static Flux `R.json` and runs `flux start python run_pipeline.py`.
  - PYTHONPATH must be set on the `flux start` command only; setting it container-wide breaks rc1.
- **Still to add:** `vdW_structures` (HeteroBuilder), papermill, jupyter/ipykernel and py3Dmol, plus EFF as a pinned dependency. Live `pip install` in `--rm` containers does not persist, so all of these go into the image.
- **JupyterHub:** nothing exists in EFF for this. ORNL_LOTF adds a kernelspec whose argv launches the image with podman-hpc. The fallback is a Shifter kernel.

## (e) HeteroBuilder version and matching API

- **Installed:** `vdW_structures` 0.1.0 @ `9a7e80e` (Jul 2025), in `~/bin/Ensemble-FF-Fit/unified_build/vdw_structures`.
- **Remote:**
  - `main` = `8b49594` (old)
  - `develop` = `9fecff7`
  - `fix/extend-vdw-layers-rank-drift` = `9c7293c`. **ORNL_LOTF pins this one** until it is merged.
- **API:**
  - `VdWHeterostructureGenerator().generate_vdW_heterostructures(film: VdWStructure, substrate: VdWStructure, zsl_generator: ZSLGenerator, get_unique=True, n_chunks=4, vacuum_over_film=..., interface_spacing=..., **matcher_kwargs)`
    - Returns `(VdWStructure[], props[])`, where `props` are pymatgen `interface_properties` plus `signed_angle` and `decomposition_angle`.
  - `write_heterostructures_data(dir, structs, props)`
  - `VdWStructure.z_solve_shift_vdW_layers(layer_inds, solve_shift, adjust_z_cell=True)`
  - `set_vacuum(v)`
- The matching parameters are pymatgen's `ZSLGenerator(max_area_ratio_tol, max_area, max_length_tol, max_angle_tol, bidirectional)`.
- **Limitations:**
  - Miller indices are fixed at (0,0,1)/(0,0,1). Slabs must be pre-oriented with the normal along z, and c must be `[0, 0, z]`.
  - There is no termination search and no twist-angle input; twists are reported after the fact.
  - The default interface gap and top vacuum are inherited from the inputs' vdW spacings, so pass both explicitly.
  - `get_unique_structures` is missing an import (`tqdm`).
  - `UniqueStructureGetter` opens a `Pool()` on every core.

## Security note

Real-looking Materials Project API keys are committed in `EFF/EnsembleFFFit/structures/materials_project/api_key.yml` (tracked in git) and hard-coded in cell 31 of `Bi_Se_Heteros.ipynb`. Rotation is recommended. ORNL_LOTF reads the key only from `$MP_API_KEY`.
