#!/usr/bin/env bash
# Fidelity-benchmark session H2 (spec 2026-09-01 section 8): tier-1 cells A1
# (attention kernel in isolation) and K1 (paged-KV gather derate), chained,
# device-side, one XProf trace per point. Same discipline as run_h1.sh: fresh
# remote probe dir, deployed-version check, pull to $RESULTS_DIR/fidelity/h2,
# teardown unless --keep.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="$RESULTS_DIR/fidelity/h2"
mkdir -p "$OUT"
date +"H2 start: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"

SKIP_MODEL=1 ./provision.sh --if-needed
tpu_ssh "rm -rf ~/probes"
tpu_scp ./probes "$TPU_NAME":~/probes
tpu_ssh "test -f ~/probes/a1_attention.py && test -f ~/probes/k1_kv_gather.py && echo PROBES_DEPLOYED"
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && python a1_attention.py --probe-api && python k1_kv_gather.py --probe-api"
# H1b (spec 6.2): the H1 cells re-measured by the slope method (chain at C
# and 2C; the per-call host floor cancels), plus E1's read_only stream.
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && mkdir -p traces && \
  python g_sweep.py --cells G1,G2,G3 --out g_sweep_slope.csv --trace traces && \
  python e1_chained.py --out e1_slope.csv --trace traces && echo H1B_PROBES_OK"
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && mkdir -p traces && \
  python a1_attention.py --mode both --out a1_attention.csv --trace traces && \
  python k1_kv_gather.py --out k1_kv_gather.csv --trace traces && echo H2_PROBES_OK"
tpu_scp "$TPU_NAME":~/probes/'*.csv' "$OUT"/
tpu_scp "$TPU_NAME":~/probes/traces "$OUT"/
ls -l "$OUT"
date +"H2 end: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"
if [ "${1:-}" != "--keep" ]; then
  ./teardown.sh
fi
