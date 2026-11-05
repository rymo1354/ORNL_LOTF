#!/usr/bin/bash
# GPU compute stages (VASP via Ensemble-FF-Fit DFTMatEnsemble + MatEnsemble).
#   STAGE=01_bulk_relax sbatch container/launch_gpu.slurm
#   STAGE=flux_smoke    sbatch -N 2 container/launch_gpu.slurm   # check Flux merges nodes
# Also works as `bash container/launch_gpu.slurm` inside an interactive GPU allocation.
#
# Adapted from Ensemble-FF-Fit examples/Perlmutter/VASP_MACE_ASE/launch_multi_node.slurm;
# see that file for why each podman-hpc flag is needed.

#SBATCH -A m2113_g
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -J lotf_gpu
#SBATCH --ntasks-per-node=1   # one Flux broker per node
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

set -euo pipefail
ORNL_LOTF="${ORNL_LOTF:-/pscratch/sd/r/rym/LOTF/ORNL_LOTF}"
IMAGE="${IMAGE:-localhost/ornl_lotf:v0.1}"
export ORNL_LOTF STAGE="${STAGE:?set STAGE}" CONFIG="${CONFIG:-${ORNL_LOTF}/config/pipeline.yaml}"
export NCORES=64 NGPUS=4      # physical cores / GPUs per Perlmutter GPU node

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

PODMAN_ARGS=(--rm --gpu --group-add keep-groups --privileged
  -e 'SLURM_*' -e 'PALS_*' -e 'PMI_*'
  -e NCORES -e NGPUS -e ORNL_LOTF -e STAGE -e CONFIG
  -e MPICH_GPU_SUPPORT_ENABLED -e CRAY_ACCEL_TARGET -e OMP_NUM_THREADS -e SLURM_CPU_BIND
  --ipc=host --network=host --pid=host
  -v /dev/shm:/dev/shm
  -v /dev/cxi0:/dev/cxi0 -v /dev/cxi1:/dev/cxi1 -v /dev/cxi2:/dev/cxi2 -v /dev/cxi3:/dev/cxi3
  -v /dev/xpmem:/dev/xpmem
  -v /var/spool/slurmd:/var/spool/slurmd -v /run/munge:/run/munge -v /run/nscd:/run/nscd
  -v /usr/lib64:/host_lib64 -v /lib64:/host_lib
  -v /global/common/software/nersc9/vasp:/global/common/software/nersc9/vasp
  -v /opt/nvidia/hpc_sdk/Linux_x86_64/25.9:/opt/nvidia/hpc_sdk/Linux_x86_64/25.9
  -v /opt/nvidia/hpc_sdk/Linux_x86_64/26.5:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
  -v /opt/cray:/opt/cray
  -v /global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1:/global/common/software/nersc9/darshan/3.4.6-gcc-13.2.1
  -v "${SCRATCH}:${SCRATCH}"
  -w "${ORNL_LOTF}")

srun -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --cpu-bind=none --mpi=pmi2 \
  podman-hpc run "${PODMAN_ARGS[@]}" "${IMAGE}" bash "${ORNL_LOTF}/container/flux_run.sh"
