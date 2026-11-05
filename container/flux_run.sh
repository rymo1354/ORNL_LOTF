#!/usr/bin/bash
# Runs INSIDE the container, one copy per node (launched by srun from
# launch_gpu.slurm / launch_cpu.slurm). Writes a static Flux R.json for the
# allocation and starts one Flux instance spanning all nodes, then runs the
# ORNL_LOTF stage inside it.
#
# Env (set by the launcher): NCORES (physical cores/node), NGPUS (0 on CPU nodes),
# ORNL_LOTF (repo path), CONFIG, STAGE.
#
# Why a static R.json instead of live hwloc/NVML discovery, why NCORES must be
# PHYSICAL cores (64 on GPU nodes), and why PYTHONPATH is scoped to the final
# command only (setting it container-wide breaks Flux's rc1): see the header
# and inline notes of Ensemble-FF-Fit examples/Perlmutter/VASP_MACE_ASE/
# launch_multi_node.slurm, from which this is adapted.
set -euo pipefail

: "${NCORES:?}" "${NGPUS:?}" "${ORNL_LOTF:?}" "${CONFIG:?}" "${STAGE:?}"

# Complete Flux python bindings (the /opt/basic copy lacks _flux._rlist).
FLUX_PYTHON_BINDINGS="/usr/local/lib/python3.12/site-packages"

python3 - > /tmp/R.json <<'PYEOF'
import json, os, re, socket

def expand_nodelist(s):
    nodes = []
    for part in re.split(r",(?![^\[]*\])", s):
        m = re.match(r"^(.*?)\[(.+)\]$", part)
        if not m:
            nodes.append(part)
            continue
        prefix, ranges = m.groups()
        for r in ranges.split(","):
            if "-" in r:
                a, b = r.split("-")
                nodes += [f"{prefix}{str(i).zfill(len(a))}" for i in range(int(a), int(b) + 1)]
            else:
                nodes.append(f"{prefix}{r}")
    return nodes

ncores, ngpus = int(os.environ["NCORES"]), int(os.environ["NGPUS"])
nodes = expand_nodelist(os.environ.get("SLURM_JOB_NODELIST", socket.gethostname()))
children = {"core": f"0-{ncores - 1}"}
if ngpus:
    children["gpu"] = f"0-{ngpus - 1}"
print(json.dumps({
    "version": 1,
    "execution": {
        "R_lite": [{"rank": str(i), "children": children} for i in range(len(nodes))],
        "starttime": 0.0, "expiration": 0.0, "nodelist": nodes,
    },
}))
PYEOF

mkdir -p /tmp/fluxcfg
cat > /tmp/fluxcfg/resource.toml <<'TOML'
[resource]
path = "/tmp/R.json"
noverify = true
TOML

cd "${ORNL_LOTF}"
if [[ "${STAGE}" == "flux_smoke" ]]; then
    FLUX_CONF_DIR=/tmp/fluxcfg flux start flux resource list
else
    FLUX_CONF_DIR=/tmp/fluxcfg flux start \
        env PYTHONPATH="${FLUX_PYTHON_BINDINGS}:${ORNL_LOTF}" \
        python "${ORNL_LOTF}/run_pipeline.py" --config "${CONFIG}" --stage "${STAGE}"
fi
