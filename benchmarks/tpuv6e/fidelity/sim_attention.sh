#!/usr/bin/env bash
# Simulator side of the A1 / K1 cells (spec 2026-09-01 sections 3.1, 4): the
# attention stage's wall-clock span from the OPSPAN ATTN line of a 1-layer
# Transformer run at the kernel's dims (attention starts only after q/k/v
# finish, so the span isolates the stage regardless of the GEMM sizes).
#   A1 prefill: MHA nh = nkv = 32 (the Pallas flash kernel has no GQA), causal
#   A1 decode:  GQA 32/8, KV context S, batch B
#   K1: decode cells at -kv_bw_pct 100 and at the sweep values, so the
#       silicon derate (paged / dense) can be matched to a knob value
# Output: OUT/sim_attention.csv with mode,S,B,nkv,kv_bw_pct,attn_first,attn_last,attn_us,cycles
# Usage: sim_attention.sh OUT_DIR [extra perf_model flags...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="${1:?OUT_DIR}"; shift || true
EXTRA=("$@")
mkdir -p "$OUT"
JOBS="${SIM_JOBS:-8}"
HD=128; NH=32; DM=$((NH * HD)); DFF=12288

: > "$OUT/points.txt"
mk() { # name line kvpct
  printf '%s\n' "$2" > "$OUT/$1.txt"; echo "$1 $3" >> "$OUT/points.txt"
}
for S in 512 2048 8192; do
  for B in 1 8 32; do
    # prefill cells the probe can afford (see a1_attention.py --dry-run)
    if { [ $S -eq 512 ]; } || { [ $S -eq 2048 ] && [ $B -le 8 ]; } || { [ $S -eq 8192 ] && [ $B -eq 1 ]; }; then
      mk "a1_prefill_${S}_${B}" "Transformer 1 $DM $NH $NH $DFF $S 0 $B" 100
    fi
    mk "a1_decode_${S}_${B}" "Transformer 1 $DM $NH 8 $DFF $S 1 $B" 100
  done
done
for S in 512 2048 8192; do for B in 8 32; do for P in 75 50 35 25; do
  mk "k1_decode_${S}_${B}_p${P}" "Transformer 1 $DM $NH 8 $DFF $S 1 $B" $P
done; done; done

run_one() {
  name="$1"; pct="$2"; shift 2
  [ -s "$OUT/${name}_s.txt" ] && return 0
  bash "$REPO/configs/tpuv6e.sh" "$OUT/$name.txt" "$OUT/${name}_s.txt" -kv_bw_pct "$pct" "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
}
export -f run_one; export OUT REPO
while read -r name pct; do echo "$name $pct"; done < "$OUT/points.txt" | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ "${EXTRA[@]}" 2>/dev/null || true
# xargs -L1 passes "name pct" as two args; EXTRA flags follow

python3 - "$OUT" <<'EOF'
import sys, os, re, csv
out = sys.argv[1]
rows = []
for l in open(os.path.join(out, "points.txt")):
    name, pct = l.split()
    sp = os.path.join(out, name + "_s.txt")
    if not os.path.exists(sp): continue
    cyc = None; first = last = None
    for ln in open(sp):
        m = re.match(r"Cycles\s+(\d+)", ln)
        if m: cyc = int(m.group(1))
        m = re.match(r"OPSPAN ATTN first (\d+) last (\d+)", ln)
        if m: first, last = int(m.group(1)), int(m.group(2))
    parts = name.split("_")  # a1_prefill_S_B  |  k1_decode_S_B_pP
    mode = parts[1]; S = int(parts[2]); B = int(parts[3])
    nkv = 32 if mode == "prefill" else 8
    rows.append(dict(cell=parts[0].upper(), mode=mode, S=S, B=B, nkv=nkv, kv_bw_pct=int(pct),
                     attn_first=first, attn_last=last,
                     attn_us=(last - first) / 1.75e3 if first is not None else None, cycles=cyc))
with open(os.path.join(out, "sim_attention.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows")
EOF
