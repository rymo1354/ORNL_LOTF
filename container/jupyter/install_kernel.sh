#!/usr/bin/bash
# Register the ORNL_LOTF container as a Jupyter kernel for NERSC JupyterHub.
#   container/jupyter/install_kernel.sh [image] [kernel-name]
# Defaults: localhost/ornl_lotf:v0.1, ornl_lotf
set -euo pipefail
IMAGE="${1:-localhost/ornl_lotf:v0.1}"
NAME="${2:-ornl_lotf}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORNL_LOTF="$(cd "${HERE}/../.." && pwd)"
DEST="${HOME}/.local/share/jupyter/kernels/${NAME}"

mkdir -p "${DEST}"
cp "${HERE}/kernel.json" "${DEST}/kernel.json"
sed -e "s|__IMAGE__|${IMAGE}|" -e "s|__ORNL_LOTF__|${ORNL_LOTF}|" \
    "${HERE}/kernel-helper.sh" > "${DEST}/kernel-helper.sh"
chmod u+x "${DEST}/kernel-helper.sh"
if [[ "${NAME}" != "ornl_lotf" ]]; then
    sed -i "s|\"ORNL_LOTF (container)\"|\"ORNL_LOTF (${IMAGE##*:})\"|" "${DEST}/kernel.json"
fi
echo "Installed kernel '${NAME}' -> ${DEST} (image ${IMAGE}, repo ${ORNL_LOTF})"
