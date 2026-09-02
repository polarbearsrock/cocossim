#!/usr/bin/env bash
# Simulator side of the tier-1 fidelity matrix (spec 2026-09-01 section 9):
# run configs/tpuv6e.sh at every H1 shape (G1/G2/G3 from probes/g_sweep.py,
# E1 from probes/e1_chained.py) and collect one CSV row per point:
#   cell,M,K,N,op,n,cycles,us,cmds,sa_busy,sa_underfilled,sa_memstall,vpu_busy,vpu_memstall
# Usage: sim_matrix.sh OUT_DIR [extra perf_model flags...]
# Runs in parallel (SIM_JOBS, default 8); resumable (skips points with a stats file).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="${1:?OUT_DIR}"; shift || true
EXTRA=("$@")
mkdir -p "$OUT"
JOBS="${SIM_JOBS:-8}"

python3 - "$OUT" <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])) if False else "", ""))
out = sys.argv[1]
# Shapes must match the probes exactly.
G1 = [(n, n, n) for n in (1024, 2048, 4096, 8192)]
G2 = [(m, k, n) for (k, n) in ((8192, 8192), (4096, 4096)) for m in (128, 256, 512, 1024, 2048)]
G3_KN = [(4096, 4096), (4096, 1024), (4096, 12288), (12288, 4096), (4096, 151936),
         (4096, 14336), (14336, 4096), (4096, 32768)]
G3 = [(m, k, n) for (k, n) in G3_KN for m in (1, 4, 8, 16, 32, 64, 128, 256)]
E1 = [1 << p for p in range(15, 29)]
lines = []
for cell, pts in (("G1", G1), ("G2", G2), ("G3", G3)):
    for (m, k, n) in pts:
        lines.append((f"{cell}_{m}x{k}x{n}", f"Matmul {m} {k} {n}"))
for n in E1:
    lines.append((f"E1_add_{n}", f"Add {n}"))
    lines.append((f"E1_exp_{n}", f"Activation {n}"))
with open(os.path.join(out, "points.txt"), "w") as f:
    for name, line in lines:
        with open(os.path.join(out, name + ".txt"), "w") as g:
            g.write(line + "\n")
        f.write(name + "\n")
print(len(lines), "points")
EOF

run_one() {
  name="$1"; shift
  [ -s "$OUT/${name}_s.txt" ] && return 0
  bash "$REPO/configs/tpuv6e.sh" "$OUT/$name.txt" "$OUT/${name}_s.txt" "$@" > "$OUT/$name.log" 2>&1 || echo "FAILED $name"
}
export -f run_one; export OUT REPO
xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "${EXTRA[@]}" < "$OUT/points.txt"

python3 - "$OUT" <<'EOF'
import sys, os, re, csv
out = sys.argv[1]
rows = []
for name in open(os.path.join(out, "points.txt")).read().split():
    sp = os.path.join(out, name + "_s.txt"); lg = os.path.join(out, name + ".log")
    if not os.path.exists(sp): continue
    cyc = None; acct = {}
    for l in open(sp):
        m = re.match(r"Cycles\s+(\d+)", l)
        if m: cyc = int(m.group(1))
        if l.startswith("ACCT "):
            p = l.split(); u = p[1]
            d = {p[i]: float(p[i + 1]) for i in range(3, len(p) - 1, 2)}
            acct.setdefault(u, []).append(d)
    cmds = re.findall(r"DRAM CMDs: (\d+)", open(lg, errors="ignore").read())
    cmds = int(cmds[-1]) if cmds else None
    def share(u, key):
        lst = acct.get(u, [])
        return sum(d[key] for d in lst) / (len(lst) * cyc) if lst and cyc else 0.0
    parts = name.split("_")
    cell = parts[0]
    if cell.startswith("G"):
        m, k, n = map(int, parts[1].split("x")); op = "matmul"; nn = 0
    else:
        op = parts[1]; nn = int(parts[2]); m = k = n = 0
    rows.append(dict(cell=cell, M=m, K=k, N=n, op=op, n=nn, cycles=cyc, us=cyc / 1.75e3 if cyc else None, cmds=cmds,
                     sa_busy=share("SYSTOLIC_ARRAY", "busy"), sa_underfilled=share("SYSTOLIC_ARRAY", "underfilled"),
                     sa_memstall=share("SYSTOLIC_ARRAY", "memstall"),
                     vpu_busy=share("VECTOR_UNIT", "busy"), vpu_memstall=share("VECTOR_UNIT", "memstall")))
with open(os.path.join(out, "sim_matrix.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows to", os.path.join(out, "sim_matrix.csv"))
EOF
