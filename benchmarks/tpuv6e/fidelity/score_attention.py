#!/usr/bin/env python3
"""Score the A1 / K1 cells (spec 2026-09-01 sections 3.1, 4, 7).

A1: the simulator's attention-stage span (OPSPAN ATTN, sim_attention.sh) vs
the silicon kernel's per-step time at identical dims -> error + verdict.
K1: the silicon paged-gather derate d = t_paged_shuffled / t_dense per cell,
and the -kv_bw_pct value P* at which the simulator's decode attention span
ratio attn_us(P) / attn_us(100) matches d (linear interpolation over the
sweep run by sim_attention.sh) -> the knob value the fit should adopt per
cell, plus how consistent P* is across cells (one number if the model is
right, a spread if it is not).

Usage: score_attention.py SIM_CSV A1_CSV [K1_CSV] [--csv out.csv]
"""
import argparse
import csv
from collections import defaultdict


def verdict(err):
    a = abs(err)
    return "PASS" if a <= 0.10 else ("CONDITIONAL" if a <= 0.25 else "FAIL")


def fnum(r, *keys, default=None):
    for k in keys:
        if k in r and r[k] not in ("", None):
            try:
                return float(r[k])
            except ValueError:
                pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_csv")
    ap.add_argument("a1_csv")
    ap.add_argument("k1_csv", nargs="?")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    sim = {}
    for r in csv.DictReader(open(a.sim_csv)):
        sim[(r["mode"], int(r["S"]), int(r["B"]), int(r["kv_bw_pct"]))] = r

    out = []
    print("=== A1: attention kernel vs simulator attention span ===")
    print(f"{'mode':8s} {'S':>5s} {'B':>3s} {'si_us':>9s} {'[p10,p90]':>19s} {'sim_us':>9s} {'err':>7s}  verdict  note")
    for r in csv.DictReader(open(a.a1_csv)):
        mode, S, B = r["mode"], int(r["S"]), int(r["B"])
        s = sim.get((mode, S, B, 100))
        if not s or not s.get("attn_us"):
            continue
        si = fnum(r, "per_step_us"); sm = float(s["attn_us"])
        chain = fnum(r, "chain", default=1)
        # slope-method rows carry a per-step band; older rows a per-call one
        p10 = fnum(r, "per_step_p10_us", default=fnum(r, "p10_s", default=0) / chain * 1e6)
        p90 = fnum(r, "per_step_p90_us", default=fnum(r, "p90_s", default=0) / chain * 1e6)
        err = (sm - si) / si
        vm = int(fnum(r, "vmem_resident", default=0))
        note = "silicon KV fits VMEM" if vm else ""
        v = verdict(err)
        print(f"{mode:8s} {S:5d} {B:3d} {si:9.2f} [{p10:8.2f},{p90:8.2f}] {sm:9.2f} {100*err:+6.1f}%  {v:11s} {note}")
        out.append(dict(cell="A1", mode=mode, S=S, B=B, si_us=si, sim_us=sm, err=err, verdict=v, note=note))

    if a.k1_csv:
        print("\n=== K1: paged-gather derate and the matching -kv_bw_pct ===")
        k1 = defaultdict(dict)
        for r in csv.DictReader(open(a.k1_csv)):
            k1[(int(r["S"]), int(r["B"]))][r["variant"]] = r
        print(f"{'S':>5s} {'B':>3s} {'dense_us':>9s} {'seq_us':>9s} {'shuf_us':>9s} {'d_seq':>6s} {'d_shuf':>6s} {'sim100':>8s} {'P*_shuf':>7s} {'P*_seq':>6s}  note")
        pstars = []
        for (S, B), v in sorted(k1.items()):
            dense = fnum(v.get("dense", {}), "per_step_us"); seq = fnum(v.get("paged_seq", {}), "per_step_us")
            shuf = fnum(v.get("paged_shuffled", {}), "per_step_us")
            if dense is None:
                continue
            d_seq = seq / dense if seq else None; d_shuf = shuf / dense if shuf else None
            base = sim.get(("decode", S, B, 100))
            sweep = sorted([(int(p), float(sim[("decode", S, B, p)]["attn_us"])) for p in (25, 35, 50, 75, 100)
                            if ("decode", S, B, p) in sim and sim[("decode", S, B, p)].get("attn_us")])
            def pstar(d):
                if d is None or not base or len(sweep) < 2:
                    return None
                b0 = float(base["attn_us"])
                pts = [(p, t / b0) for p, t in sweep]  # ratio vs P, decreasing in P
                if d <= pts[-1][1]:
                    return 100.0
                for (p1, r1), (p2, r2) in zip(pts, pts[1:]):
                    if r2 <= d <= r1:
                        return p1 + (p2 - p1) * (d - r1) / (r2 - r1)
                return pts[0][0]
            ps, pq = pstar(d_shuf), pstar(d_seq)
            vm = int(fnum(v.get("dense", {}), "vmem_resident", default=0))
            note = "KV fits VMEM on silicon" if vm else ""
            print(f"{S:5d} {B:3d} {dense:9.2f} {seq or 0:9.2f} {shuf or 0:9.2f} {d_seq or 0:6.2f} {d_shuf or 0:6.2f} "
                  f"{float(base['attn_us']) if base else 0:8.2f} {ps or 0:7.1f} {pq or 0:6.1f}  {note}")
            if ps and not vm:
                pstars.append(ps)
            out.append(dict(cell="K1", mode="decode", S=S, B=B, si_us=dense, sim_us=float(base["attn_us"]) if base else None,
                            err=(float(base["attn_us"]) - dense) / dense if base else None,
                            verdict=f"d_shuf={d_shuf:.2f} P*={ps:.0f}" if ps else "n/a", note=note))
        if pstars:
            print(f"\nP* (shuffled, HBM cells): {', '.join(f'{p:.0f}' for p in pstars)}  -> mean {sum(pstars)/len(pstars):.0f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
