#!/usr/bin/env bash
# Tier-1 session driver (spec 3.1): runs the probes listed in PROBES (space
# separated "script.py args...") on a fresh remote probe dir, pulls CSVs and
# traces to $RESULTS_DIR/fidelity/$SESSION. --keep skips teardown so more
# probes can be pushed to the same VM later. Same deploy discipline as
# run_h1.sh (wipe, redeploy, verify, resumable probes).
# Usage: SESSION=tier1 PROBES="g_sweep.py --cells G1,G2,G3 --out g_sweep_slope.csv|e1_chained.py --out e1_slope.csv" run_tier1.sh [--keep]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
SESSION="${SESSION:-tier1}"
OUT="$RESULTS_DIR/fidelity/$SESSION"; mkdir -p "$OUT"
date +"$SESSION start: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"
SKIP_MODEL=1 ./provision.sh --if-needed
tpu_ssh "rm -rf ~/probes"
tpu_scp ./probes "$TPU_NAME":~/probes
tpu_ssh "grep -q 'time_chain_slope' ~/probes/common.py && echo PROBES_DEPLOYED"
IFS='|' read -ra LIST <<< "${PROBES:?PROBES}"
for p in "${LIST[@]}"; do
  echo "== running: $p"
  tpu_ssh "source ~/venv/bin/activate && cd ~/probes && mkdir -p traces && python $p --trace traces && echo PROBE_OK"
done
tpu_scp "$TPU_NAME":~/probes/'*.csv' "$OUT"/
tpu_scp "$TPU_NAME":~/probes/traces "$OUT"/
date +"$SESSION end: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"
[ "${1:-}" = "--keep" ] || ./teardown.sh
