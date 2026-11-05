#!/usr/bin/bash
# NERSC JupyterHub kernel helper: runs the ipykernel INSIDE the ORNL_LOTF image.
# Installed (with kernel.json) by container/jupyter/install_kernel.sh; Jupyter
# calls it as:  kernel-helper.sh python -m ipykernel_launcher -f {connection_file}
#
#  -v /tmp, $HOME  what podman-hpc's --jupyter module binds (the connection file
#                  lives under $HOME/.local/share/jupyter/runtime or /tmp)
#  -v $SCRATCH     what --scratch binds (repo, stages/, POTCARs)
#                  Explicit -v is used instead of the --jupyter/--scratch module
#                  flags because podman validates -w before module binds exist
#                  ("workdir ... does not exist", seen 2026-11-05).
#  --network=host  the kernel's ZMQ ports must be reachable from the Jupyter
#                  server on the host; default rootless networking isolates them
#  PYTHONPATH      the bind-mounted ORNL_LOTF checkout (code edits need no rebuild)
set -euo pipefail

IMAGE="__IMAGE__"
ORNL_LOTF="__ORNL_LOTF__"

# Start in the notebook's directory when it is visible inside the container.
WORKDIR_ARGS=()
case "${PWD}" in
    "${HOME}"*|"${SCRATCH}"*|/tmp*) WORKDIR_ARGS=(-w "${PWD}") ;;
esac

exec podman-hpc run --rm --network=host \
    -v /tmp:/tmp -v "${HOME}:${HOME}" -v "${SCRATCH}:${SCRATCH}" \
    -e MP_API_KEY \
    -e PYTHONPATH="${ORNL_LOTF}" \
    "${WORKDIR_ARGS[@]}" \
    "${IMAGE}" "$@"
