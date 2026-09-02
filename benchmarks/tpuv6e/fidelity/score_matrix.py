#!/usr/bin/env python3
"""Tier-1 fidelity matrix: join the simulator rows (sim_matrix.sh ->
sim_matrix.csv) with the silicon rows (probes/g_sweep.py -> g_sweep.csv,
probes/e1_chained.py -> e1_chained.csv) at identical shapes and emit one
verdict per cell (spec 2026-09-01 section 7):
  PASS        |error| <= 10%
  CONDITIONAL 10% < |error| <= 25%
  FAIL        otherwise
error = (sim - silicon) / silicon on per-step device time. Silicon rows are
chained (device-side) medians; the p10/p90 band is printed beside them.

Usage: score_matrix.py SIM_CSV G_CSV E1_CSV [--csv out.csv]
"""
import argparse
import csv
import sys
from collections import defaultdict


def verdict(err):
    a = abs(err)
    return "PASS" if a <= 0.10 else ("CONDITIONAL" if a <= 0.25 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_csv")
    ap.add_argument("g_csv")
    ap.add_argument("e1_csv")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    sim = {}
    for r in csv.DictReader(open(a.sim_csv)):
        if r["op"] == "matmul":
            sim[("G", int(r["M"]), int(r["K"]), int(r["N"]))] = r
        else:
            sim[("E", r["op"], int(r["n"]))] = r
    rows = []
    for r in csv.DictReader(open(a.g_csv)):
        key = ("G", int(r["M"]), int(r["K"]), int(r["N"]))
        s = sim.get(key)
        if not s:
            continue
        si = float(r["per_step_us"]); sm = float(s["us"])
        p10 = float(r["p10_s"]) / int(r["chain"]) * 1e6; p90 = float(r["p90_s"]) / int(r["chain"]) * 1e6
        err = (sm - si) / si
        rows.append(dict(cell=r["cell"], label=r["label"], shape=f"{r['M']}x{r['K']}x{r['N']}",
                         si_us=si, si_p10=p10, si_p90=p90, sim_us=sm, err=err,
                         si_tflops=float(r["tflops"]), sim_tflops=2.0 * int(r["M"]) * int(r["K"]) * int(r["N"]) / sm / 1e6,
                         sim_sa_busy=float(s["sa_busy"]), sim_sa_memstall=float(s["sa_memstall"]),
                         verdict=verdict(err)))
    # E1: a scan carry under VMEM (~128 MiB) never touches HBM on silicon, so
    # rows moving < 150 MB measure t0 + VMEM streaming, not HBM. They are
    # reported (cell E1v) but get no HBM verdict; only the large rows do.
    VMEM_BYTES = 150e6
    for r in csv.DictReader(open(a.e1_csv)):
        key = ("E", r["op"], int(r["n"]))
        s = sim.get(key)
        if not s:
            continue
        si = float(r["per_step_us"]); sm = float(s["us"])
        p10 = float(r["p10_s"]) / int(r["chain"]) * 1e6; p90 = float(r["p90_s"]) / int(r["chain"]) * 1e6
        err = (sm - si) / si
        vmem = int(r["bytes_moved"]) < VMEM_BYTES
        rows.append(dict(cell="E1v" if vmem else "E1", label=r["op"], shape=f"n={r['n']}", si_us=si, si_p10=p10, si_p90=p90,
                         sim_us=sm, err=err, si_tflops=0.0, sim_tflops=0.0,
                         sim_sa_busy=0.0, sim_sa_memstall=0.0,
                         verdict="VMEM-RESIDENT (no HBM verdict)" if vmem else verdict(err)))

    print(f"{'cell':4s} {'label':16s} {'shape':18s} {'si_us':>9s} {'[p10,p90]':>17s} {'sim_us':>9s} {'err':>7s}  verdict")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["cell"], r["verdict"])] += 1
        print(f"{r['cell']:4s} {r['label']:16s} {r['shape']:18s} {r['si_us']:9.2f} [{r['si_p10']:7.2f},{r['si_p90']:7.2f}] "
              f"{r['sim_us']:9.2f} {100 * r['err']:+6.1f}%  {r['verdict']}")
    print()
    for cell in ("G1", "G2", "G3", "E1"):
        if cell == "E1":
            print(f"E1v: {sum(v for (c, _), v in counts.items() if c == 'E1v')} VMEM-resident rows (silicon carry fits VMEM; not HBM cells)")
        tot = sum(v for (c, _), v in counts.items() if c == cell)
        if tot:
            print(f"{cell}: PASS {counts[(cell, 'PASS')]}  CONDITIONAL {counts[(cell, 'CONDITIONAL')]}  "
                  f"FAIL {counts[(cell, 'FAIL')]}  (of {tot})")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
