#!/usr/bin/env bash
# Fidelity-benchmark session H1 (spec 2026-09-01 section 8): tier-1 GEMM cells
# G1/G2/G3 and elementwise E1, chained (device-side) with one XProf trace per
# point. Provisions an ON-DEMAND v6e-1 if needed (no model download: the tier-1
# probes need only JAX), runs the probes, pulls CSVs + traces to
# $RESULTS_DIR/fidelity/h1, and tears the VM down. Idempotent: probes resume
# from their CSVs, so a re-run after an interruption continues where it stopped.
# Usage: run_h1.sh [--keep]   (--keep skips the teardown, e.g. to chain H2)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
OUT="$RESULTS_DIR/fidelity/h1"
mkdir -p "$OUT"
date +"H1 start: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"

SKIP_MODEL=1 ./provision.sh --if-needed
tpu_scp ./probes "$TPU_NAME":~/probes
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && mkdir -p traces && \
  python g_sweep.py --cells G1,G2,G3 --out g_sweep.csv --trace traces && \
  python e1_chained.py --out e1_chained.csv --trace traces && echo H1_PROBES_OK"
tpu_scp "$TPU_NAME":~/probes/'*.csv' "$OUT"/
tpu_scp "$TPU_NAME":~/probes/traces "$OUT"/
ls -l "$OUT"
date +"H1 end: %F %H:%M (%s)" | tee -a "$RESULTS_DIR/cost.log"
if [ "${1:-}" != "--keep" ]; then
  ./teardown.sh
fi
