#!/usr/bin/env bash
# Tier-1 fit, coordinate scan (spec 2026-09-01 section 6): run sim_matrix at
# each parameter combination and score it against H1; the objective is the
# mean |ln(sim/silicon)| over the GEMM cells and the HBM (non-VMEM) E1 rows.
# Usage: fit_tier1.sh OUT_ROOT H1_DIR  (H1_DIR holds g_sweep.csv, e1_chained.csv)
# Combos are sequential (each is 8-way parallel inside); resumable per combo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:?OUT_ROOT}"; H1="${2:?H1_DIR}"
mkdir -p "$ROOT"
COMBOS=(
  "dram_enq=32 op_overhead=0"
  "dram_enq=15 op_overhead=0"
  "dram_enq=32 op_overhead=12250"
  "dram_enq=15 op_overhead=12250"
  "dram_enq=32 op_overhead=15750"
  "dram_enq=15 op_overhead=15750"
)
for c in "${COMBOS[@]}"; do
  tag=$(echo "$c" | tr ' =' '_-')
  flags=""; for kv in $c; do flags="$flags -${kv%%=*} ${kv##*=}"; done
  d="$ROOT/$tag"
  [ -s "$d/sim_matrix.csv" ] || SIM_JOBS="${SIM_JOBS:-8}" bash "$HERE/sim_matrix.sh" "$d" $flags > "$d.log" 2>&1
  python3 "$HERE/score_matrix.py" "$d/sim_matrix.csv" "$H1/g_sweep.csv" "$H1/e1_chained.csv" --csv "$d/scorecard.csv" > "$d/scorecard.txt"
  python3 - "$d/scorecard.csv" "$c" <<'EOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
use = [r for r in rows if r["cell"] != "E1v"]
obj = sum(abs(math.log(float(r["sim_us"]) / float(r["si_us"]))) for r in use) / len(use)
by = {}
for r in use:
    by.setdefault(r["cell"], []).append(abs(math.log(float(r["sim_us"]) / float(r["si_us"]))))
cells = "  ".join(f"{k}:{sum(v)/len(v):.3f}" for k, v in sorted(by.items()))
npass = sum(1 for r in use if r["verdict"] == "PASS")
print(f"{sys.argv[2]:40s} objective={obj:.4f}  PASS {npass}/{len(use)}  {cells}")
EOF
done | tee "$ROOT/fit_summary.txt"
