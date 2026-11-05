#!/usr/bin/bash
# Build and migrate the ORNL_LOTF podman-hpc image (run on a Perlmutter login node).
#   ./build_image.sh            -> ornl_lotf:latest
#   ./build_image.sh v0.1       -> ornl_lotf:v0.1
# Extra podman build args pass through, e.g.
#   ./build_image.sh v0.1 --build-arg EFF_REF=<sha>
set -euo pipefail
TAG="${1:-latest}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman-hpc build -f "${HERE}/Dockerfile" -t "ornl_lotf:${TAG}" "$@" "${HERE}"
# migrate = squash into $SCRATCH-backed storage so compute + Jupyter nodes see it
podman-hpc migrate "ornl_lotf:${TAG}"
podman-hpc run --rm "localhost/ornl_lotf:${TAG}" cat /opt/ornl_lotf_pip_freeze.txt \
    > "${HERE}/pip_freeze.${TAG}.txt"
echo "Built localhost/ornl_lotf:${TAG}; resolved packages in ${HERE}/pip_freeze.${TAG}.txt"
