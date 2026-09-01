#!/usr/bin/env bash
# Delete the benchmark TPU and verify NO orphaned TPUs remain in any zone we
# use. Run at the end of EVERY session; a forgotten on-demand v6e bills ~$3/hr.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$TPU_ZONE" --quiet 2>/dev/null \
  && echo "deleted $TPU_NAME ($TPU_ZONE)" || echo "nothing named $TPU_NAME in $TPU_ZONE"
date +"teardown: %F %H:%M (%s)" >> "$RESULTS_DIR/cost.log" 2>/dev/null || true
echo "--- orphan check (must all be empty) ---"
for z in us-east5-a us-east5-b us-east1-d; do
  echo "$z: $(gcloud compute tpus tpu-vm list --zone=$z --format='value(name,state)' 2>/dev/null | tr '\n' ' ')"
done
