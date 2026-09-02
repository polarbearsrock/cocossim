#!/usr/bin/env bash
# Tier-2 session driver (spec 3.2): one whole-model grid on its own VM.
# Usage: TPU_NAME=cocossim-t2q SESSION=tier2_qwen HF_MODEL=Qwen/Qwen3-8B GRID=qwen run_tier2.sh [--keep]
#        TPU_NAME=cocossim-t2m SESSION=tier2_mistral HF_MODEL=mistralai/Mistral-7B-v0.3 GRID=mistral run_tier2.sh
# Provisions (model prefetched), runs holdout/dh_offline.py with windowed
# traces, pulls CSV + traces to $RESULTS_DIR/fidelity/$SESSION, tears down.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
SESSION="${SESSION:?SESSION}"; GRID="${GRID:?GRID}"; export HF_MODEL="${HF_MODEL:?HF_MODEL}"
OUT="$RESULTS_DIR/fidelity/$SESSION"; mkdir -p "$OUT"
date +"$SESSION start: %F %H:%M (%s) $TPU_NAME" | tee -a "$RESULTS_DIR/cost.log"
./provision.sh --if-needed
tpu_ssh "rm -rf ~/holdout"
tpu_scp ./holdout "$TPU_NAME":~/holdout
tpu_ssh "grep -q 'TRACE_STEPS' ~/holdout/dh_offline.py && echo HOLDOUT_DEPLOYED"
tpu_ssh "source ~/venv/bin/activate && cd ~/holdout && python dh_offline.py --model $HF_MODEL --grid $GRID --dry-run"
tpu_ssh "source ~/venv/bin/activate && cd ~/holdout && mkdir -p traces && python dh_offline.py --model $HF_MODEL --grid $GRID --out dh_$GRID.csv --trace-dir traces < /dev/null && echo TIER2_OK"
tpu_scp "$TPU_NAME":~/holdout/'*.csv' "$OUT"/
tpu_scp "$TPU_NAME":~/holdout/traces "$OUT"/
date +"$SESSION end: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"
[ "${1:-}" = "--keep" ] || ./teardown.sh
