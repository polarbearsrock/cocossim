#!/usr/bin/env bash
# Simulator side of tier 2 (spec 2026-09-01 section 3.2): every whole-model
# grid point of holdout/dh_offline.py (Qwen3-8B and Mistral-7B-v0.3) through
# configs/tpuv6e.sh. Per point four runs:
#   full  all layers + LM head              (the number scored against silicon)
#   l1    1 layer, no head   l2  2 layers   lh  1 layer + head
# The triplet reproduces the session-3 extrapolation
#   t_extrap = t_l1 + (L-1)(t_l2 - t_l1) + (t_lh - t_l1)
# so the cheap method can be validated against the full run once and reused.
# One CSV row per run:
#   model,mode,seq,batch,variant,layers,head,cycles,us,cmds,
#   sa_<class>_busy/underfilled/memstall (share of core-cycles, SA cores),
#   vpu_<class>_busy/memstall (VPU cores), span_<class>_us (OPSPAN first..last)
# Usage: sim_tier2.sh OUT_DIR [extra perf_model flags...]
# SIM_JOBS parallel runs (default 32); resumable (skips runs with a stats file).
# MODELS="qwen mistral" and VARIANTS="full l1 l2 lh" (defaults) restrict the
# grid, e.g. VARIANTS="l1 l2 lh" MODELS=qwen for a cheap calibration sweep.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="${1:?OUT_DIR}"; shift || true
EXTRA=("$@")
mkdir -p "$OUT"
JOBS="${SIM_JOBS:-32}"
export MODELS="${MODELS:-qwen mistral}" VARIANTS="${VARIANTS:-full l1 l2 lh}"

# COLLECT_ONLY must not regenerate points.txt: a driver's xargs reads it
# lazily, so rewriting it with the default MODELS/VARIANTS mid-run would
# feed that driver the whole grid (happened 2026-09-02).
if [ "${COLLECT_ONLY:-0}" != "1" ]; then
python3 - "$OUT" "$HERE/../holdout" <<'EOF'
import sys, os
out, holdout = sys.argv[1], sys.argv[2]
sys.path.insert(0, holdout)
from dh_offline import GRIDS
# model -> (n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab)
MODELS = {"qwen": (36, 4096, 32, 8, 12288, 151936),
          "mistral": (32, 4096, 32, 8, 14336, 32768)}
want_models = os.environ["MODELS"].split(); want_variants = os.environ["VARIANTS"].split()
runs = []
for model, (L, d, nh, nkv, dff, vocab) in MODELS.items():
    if model not in want_models:
        continue
    for (mode, seq, batch) in GRIDS[model]["points"]:
        m = 0 if mode == "prefill" else 1
        base = f"{d} {nh} {nkv} {dff} {seq} {m} {batch}"
        rows = seq * batch if mode == "prefill" else batch
        for variant, layers, head in (("full", L, 1), ("l1", 1, 0), ("l2", 2, 0), ("lh", 1, 1)):
            if variant not in want_variants:
                continue
            name = f"{model}_{mode}_{seq}x{batch}_{variant}"
            line = f"Transformer {layers} {base}" + (f" {vocab}" if head else "")
            runs.append((rows * layers, name, line))
# longest runs first so the parallel pool tails off instead of starting late
runs.sort(key=lambda r: -r[0])
with open(os.path.join(out, "points.txt"), "w") as f:
    for _, name, line in runs:
        with open(os.path.join(out, name + ".txt"), "w") as g:
            g.write(line + "\n")
        f.write(name + "\n")
print(len(runs), "runs")
EOF
fi

run_one() {
  name="$1"; shift
  [ -s "$OUT/${name}_s.txt" ] && return 0
  local t0=$SECONDS
  bash "$REPO/configs/tpuv6e.sh" "$OUT/$name.txt" "$OUT/${name}_s.txt" "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
  echo "done $name in $((SECONDS - t0))s"
}
export -f run_one; export OUT REPO
# COLLECT_ONLY=1 skips the runs and just (re)builds the CSV from whatever
# stats files exist, so partial grids can be scored while the rest runs.
if [ "${COLLECT_ONLY:-0}" != "1" ]; then
  xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "${EXTRA[@]}" < "$OUT/points.txt"
fi

python3 - "$OUT" <<'EOF'
import sys, os, re, csv
out = sys.argv[1]
CLASSES = ["OTHER", "QKV", "O", "GATE_UP", "DOWN", "HEAD", "ATTN", "VPU_NORM", "VPU_EW"]
LAYERS = {"qwen": 36, "mistral": 32}
rows = []
for name in open(os.path.join(out, "points.txt")).read().split():
    sp = os.path.join(out, name + "_s.txt"); lg = os.path.join(out, name + ".log")
    if not os.path.exists(sp) or os.path.getsize(sp) == 0:
        continue
    cyc = None; acctc = {}; ncore = {}; span = {}
    for l in open(sp):
        m = re.match(r"Cycles\s+(\d+)", l)
        if m: cyc = int(m.group(1))
        p = l.split()
        if p and p[0] == "ACCTC":
            unit, cls = p[1], p[3]
            d = acctc.setdefault((unit, cls), {"busy": 0, "underfilled": 0, "memstall": 0})
            for i in range(4, len(p) - 1, 2):
                d[p[i]] += int(p[i + 1])
            ncore.setdefault(unit, set()).add(p[2])
        elif p and p[0] == "OPSPAN":
            span[p[1]] = (int(p[3]), int(p[5]))
    cmds = re.findall(r"DRAM CMDs: (\d+)", open(lg, errors="ignore").read())
    cmds = int(cmds[-1]) if cmds else None
    model, mode, shape, variant = name.split("_")
    seq, batch = map(int, shape.split("x"))
    layers = {"full": LAYERS[model], "l1": 1, "l2": 2, "lh": 1}[variant]
    r = dict(model=model, mode=mode, seq=seq, batch=batch, variant=variant, layers=layers,
             head=int(variant in ("full", "lh")), cycles=cyc, us=cyc / 1.75e3 if cyc else None, cmds=cmds)
    def share(unit, cls, key):
        d = acctc.get((unit, cls)); n = len(ncore.get(unit, ()))
        return d[key] / (n * cyc) if d and n and cyc else 0.0
    for c in CLASSES:
        for k in ("busy", "underfilled", "memstall"):
            r[f"sa_{c}_{k}"] = share("SYSTOLIC_ARRAY", c, k)
        for k in ("busy", "memstall"):
            r[f"vpu_{c}_{k}"] = share("VECTOR_UNIT", c, k)
        r[f"span_{c}_us"] = (span[c][1] - span[c][0]) / 1.75e3 if c in span else 0.0
    rows.append(r)
rows.sort(key=lambda r: (r["model"], r["mode"], r["seq"], r["batch"], r["variant"]))
with open(os.path.join(out, "sim_tier2.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows to", os.path.join(out, "sim_tier2.csv"))
EOF
