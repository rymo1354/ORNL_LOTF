# ORNL_LOTF image: NERSC Flux base + pinned MatEnsemble + Ensemble-FF-Fit +
# HeteroBuilder (vdW_structures) + notebook stack. No MACE/cuequivariance layers
# are added; note the nersc/flux:26.05 base itself (~40 GB, NGC-derived) already
# ships torch 2.11+cu128, so the image is ~base + 3 GB.
#
# Adapted from Ensemble-FF-Fit examples/Perlmutter/VASP_MACE_ASE/Dockerfile.mace.
# VASP is NOT in the image: the host's NERSC vasp module is bind-mounted in by
# launch_gpu.slurm. ORNL_LOTF itself is NOT installed either: the launchers and
# the Jupyter kernel put the bind-mounted checkout on PYTHONPATH, so code edits
# need no rebuild. Everything else is installed at build time, because a
# `pip install` inside a `podman-hpc run --rm` container does not persist.
#
# Build (from this directory):  ./build_image.sh  [tag]

FROM docker.io/nersc/flux:26.05

# uv :latest as in the confirmed EFF build (only used for `uv export` of the lock).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH=/opt/basic/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-dev libglu1-mesa-dev mesa-utils && \
    rm -rf /var/lib/apt/lists/*

# ---- Pins (override with --build-arg) ---------------------------------------
# MatEnsemble main @ 2026-08-31, v0.5.8 (recommended in EFF PIPELINE_CHECKS).
ARG MATENSEMBLE_REF=5d8be305f1fbdb478d9a18878d826229b05f190f
# Ensemble-FF-Fit: tracks this branch for now (was ac5702c at setup).
ARG EFF_REF=frontier-rmg-update
# HeteroBuilder: rank-drift fix branch head; move to develop once merged.
ARG HETEROBUILDER_REF=9c7293cc70f87dfc43f65bcace425d565a960cbe

# ---- MatEnsemble (locked runtime deps only; --no-dev keeps torch & docs out) --
WORKDIR /opt
RUN git clone https://github.com/Q-CAD/MatEnsemble.git matensemble && \
    cd matensemble && git checkout ${MATENSEMBLE_REF} && \
    uv export --format requirements.txt --no-hashes --no-dev --no-emit-project \
        --output-file requirements.txt && \
    python -m pip install -r requirements.txt && \
    python -m pip install --no-deps .

# ---- Science stack + notebook tooling ---------------------------------------
RUN python -m pip install \
        pymatgen pymatgen-analysis-defects mp-api ase numpy scipy \
        matminer scikit-learn pyarrow tqdm pyyaml matplotlib \
        papermill jupyter ipykernel py3Dmol pytest

# ---- HeteroBuilder + Ensemble-FF-Fit (fixed, non-editable) ------------------
# --no-deps on EFF: its only extra requirement is matensemble>=0.5.1, which is
# satisfied above; this stops pip from trying to resolve matensemble from PyPI.
RUN python -m pip install \
        "vdW_structures @ git+https://github.com/Q-CAD/HeteroBuilder.git@${HETEROBUILDER_REF}" && \
    python -m pip install --no-deps \
        "EnsembleFFFit @ git+https://github.com/Q-CAD/Ensemble-FF-Fit.git@${EFF_REF}"

# Record exactly what was resolved, then fail the build (not the first chore)
# if anything critical cannot import. Flux bindings are checked separately at
# runtime (the complete copy lives in /usr/local/lib/python3.12/site-packages).
RUN python -m pip freeze > /opt/ornl_lotf_pip_freeze.txt && \
    python -c "import vdW_structures, EnsembleFFFit, matensemble, pymatgen, mp_api, papermill, py3Dmol, ipykernel; \
from vdW_structures.vdW_heterostructure import VdWHeterostructureGenerator; \
from EnsembleFFFit.base import DFTMatEnsemble; print('imports ok')"

WORKDIR /
