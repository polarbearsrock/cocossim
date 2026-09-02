#!/usr/bin/env bash
# Provision the benchmark TPU VM (ON-DEMAND per user ruling 2026-08-31; spot
# in us-east1-d preempted twice in 40 minutes) and bootstrap it.
# Usage: provision.sh            -> create + bootstrap + verify
#        provision.sh --if-needed -> reuse a READY VM, else create
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

state=$(gcloud compute tpus tpu-vm list --zone="$TPU_ZONE" \
        --filter="name~$TPU_NAME" --format="value(state)" 2>/dev/null || true)
if [ "${1:-}" = "--if-needed" ] && [ "$state" = "READY" ]; then
  echo "reusing READY $TPU_NAME in $TPU_ZONE"
else
  if [ -n "$state" ]; then
    echo "existing $TPU_NAME in state $state - deleting first"
    gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$TPU_ZONE" --quiet
  fi
  date +"provision start: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log" || true
  gcloud compute tpus tpu-vm create "$TPU_NAME" --zone="$TPU_ZONE" \
    --accelerator-type="$TPU_TYPE" --version="$TPU_RUNTIME"
fi

# Bootstrap: uv + python 3.12 venv + vllm-tpu (system pip on the VM's 3.10
# cannot build vllm-tpu - session-1 finding). Idempotent.
tpu_scp ./setup_vm.sh "$TPU_NAME":~/
tpu_ssh "SKIP_MODEL=${SKIP_MODEL:-0} HF_MODEL=${HF_MODEL:-Qwen/Qwen3-8B} bash setup_vm.sh"
echo "provisioned. Remember: ./teardown.sh when the session ends."
