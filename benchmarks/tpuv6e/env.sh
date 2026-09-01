#!/usr/bin/env bash
# Shared environment for the TPU v6e measurement harness (spec 4).
# Source this before any other harness script:  source benchmarks/tpuv6e/env.sh
#
# Quota map discovered empirically 2026-08-31 (project rta-tpu-research):
#   us-east5-a/b   : on-demand v6e ALLOWED  <- home zone
#   us-east1-d     : spot-only (on-demand denied); spot pool preempted 2x/40min
#   europe-west4-* : blocked by org resourceLocations policy
export TPU_PROJECT="${TPU_PROJECT:-rta-tpu-research}"
export TPU_ZONE="${TPU_ZONE:-us-east5-a}"
export TPU_NAME="${TPU_NAME:-cocossim-bench}"
export TPU_TYPE="${TPU_TYPE:-v6e-1}"
export TPU_RUNTIME="${TPU_RUNTIME:-v2-alpha-tpuv6e}"

# gcloud config and SSH key live on /data2: the home directory is quota-full
# and gcloud/ssh-keygen writes there fail (learned the hard way in session 1).
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-/data2/s2chitni/.gcloud}"
export TPU_SSH_KEY="${TPU_SSH_KEY:-/data2/s2chitni/.gcloud/gce_key}"

# Where pulled traces/CSVs land locally (never the home directory).
export RESULTS_DIR="${RESULTS_DIR:-/data2/s2chitni/.tmp/tpuv6e_results}"

tpu_ssh() {  # tpu_ssh "<remote command>"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" \
    --ssh-key-file="$TPU_SSH_KEY" --command="$1" 2>&1 | grep -vE "known_hosts|Warning: Permanently|Attempting to connect"
}
tpu_scp() {  # tpu_scp <src> <dst>  (either side may be $TPU_NAME:path)
  gcloud compute tpus tpu-vm scp --recurse "$1" "$2" --zone="$TPU_ZONE" \
    --ssh-key-file="$TPU_SSH_KEY" 2>&1 | grep -vE "known_hosts|Warning: Permanently|Attempting to connect"
}
