#!/usr/bin/bash
# CPU construction stages (papermill-executed notebooks; MatEnsemble chores on CPU cores).
#   STAGE=00_bulk sbatch container/launch_cpu.slurm
#   STAGE=00_bulk bash   container/launch_cpu.slurm     # inside `salloc -C cpu -q interactive ...`
# MP_API_KEY is forwarded from your environment if set (never stored in the repo).

#SBATCH -A m2113
#SBATCH -C cpu
#SBATCH --qos regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -J lotf_cpu
#SBATCH --ntasks-per-node=1   # one Flux broker per node

set -euo pipefail
ORNL_LOTF="${ORNL_LOTF:-/pscratch/sd/r/rym/LOTF/ORNL_LOTF}"
IMAGE="${IMAGE:-localhost/ornl_lotf:v0.1}"
export ORNL_LOTF STAGE="${STAGE:?set STAGE}" CONFIG="${CONFIG:-${ORNL_LOTF}/config/pipeline.yaml}"
export NCORES=128 NGPUS=0     # physical cores per Perlmutter CPU node (2 x 64-core Milan)
export OMP_NUM_THREADS=1

PODMAN_ARGS=(--rm --group-add keep-groups
  -e 'SLURM_*' -e 'PALS_*' -e 'PMI_*'
  -e NCORES -e NGPUS -e ORNL_LOTF -e STAGE -e CONFIG -e OMP_NUM_THREADS -e MP_API_KEY
  --ipc=host --network=host --pid=host
  -v /dev/shm:/dev/shm
  -v /var/spool/slurmd:/var/spool/slurmd -v /run/munge:/run/munge -v /run/nscd:/run/nscd
  -v "${SCRATCH}:${SCRATCH}"
  -w "${ORNL_LOTF}")

srun -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --cpu-bind=none --mpi=pmi2 \
  podman-hpc run "${PODMAN_ARGS[@]}" "${IMAGE}" bash "${ORNL_LOTF}/container/flux_run.sh"
