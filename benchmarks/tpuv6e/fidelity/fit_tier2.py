#!/usr/bin/env python3
"""Tier-2 calibration table: score several simulator parameter combos against
the SAME silicon device times and rank them (spec 2026-09-01 section 6).

Each combo is a sim_tier2.sh output CSV (full runs or the validated l1/l2/lh
extrapolation, which matched full runs within 2% on 2026-09-01). The fit set
is one model's grid (Qwen3-8B); the holdout (Mistral-7B-v0.3) is scored only
for combos that also carry its rows, and never drives the choice.

Usage:
  fit_tier2.py --dh dh_qwen.csv --census census_qwen.csv --model qwen \
               --combo priors=sim_tier2/sim_tier2.csv --combo ovh8000=fit2_ovh8000/sim_tier2.csv ...
               [--holdout-dh dh_mistral.csv --holdout-census census_mistral.csv]
Prints per combo: MAPE, mean bias, PASS/CONDITIONAL/FAIL, split by mode, and
the per-point errors; ranks by MAPE on the fit set.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_tier2 import load_sim, extrap, load_census, verdict, point_key  # noqa: E402

LAYERS = {"qwen": 36, "mistral": 32}


def device_times(dh, census_path):
    """point -> per-step device ms (bucket rows summed) for points with a trace."""
    census, _ = load_census(census_path)
    pts = set()
    for r in csv.DictReader(open(dh)):
        pts.add(point_key(r["mode"], r["seq"], r["batch"]))
    out = {}
    for k in pts:
        c = census.get(k)
        if c:
            out[k] = sum(t for (kind, cls), t in c.items() if kind == "bucket")
    return out


def sim_times(path, model):
    sim = load_sim(path, model)
    out = {}
    for k, v in sim.items():
        if "full" in v:
            out[k] = float(v["full"]["us"]) / 1e3
        else:
            e = extrap(v, LAYERS[model])
            if e is not None:
                out[k] = e / 1e3
    return out


def score(dev, sim):
    errs = {k: (sim[k] - dev[k]) / dev[k] for k in dev if k in sim}
    if not errs:
        return None
    by_mode = defaultdict(list)
    for k, e in errs.items():
        by_mode[k[0]].append(e)
    cnt = defaultdict(int)
    for e in errs.values():
        cnt[verdict(e)] += 1
    return dict(n=len(errs), mape=sum(abs(e) for e in errs.values()) / len(errs),
                bias=sum(errs.values()) / len(errs),
                mape_prefill=(sum(abs(e) for e in by_mode["prefill"]) / len(by_mode["prefill"])) if by_mode["prefill"] else None,
                mape_decode=(sum(abs(e) for e in by_mode["decode"]) / len(by_mode["decode"])) if by_mode["decode"] else None,
                cnt=cnt, errs=errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--model", default="qwen", choices=("qwen", "mistral"))
    ap.add_argument("--combo", action="append", required=True, help="label=path/to/sim_tier2.csv")
    ap.add_argument("--holdout-dh")
    ap.add_argument("--holdout-census")
    ap.add_argument("--holdout-model", default="mistral")
    a = ap.parse_args()

    dev = device_times(a.dh, a.census)
    hold = device_times(a.holdout_dh, a.holdout_census) if a.holdout_dh and a.holdout_census else {}
    rows = []
    for c in a.combo:
        label, path = c.split("=", 1)
        s = score(dev, sim_times(path, a.model))
        h = score(hold, sim_times(path, a.holdout_model)) if hold else None
        rows.append((label, s, h))
    rows.sort(key=lambda r: r[1]["mape"] if r[1] else 9)

    pf = lambda x: f"{100 * x:6.1f}%" if x is not None else "     -"
    print(f"{'combo':16s} {'n':>3s} {'MAPE':>7s} {'bias':>7s} {'prefill':>8s} {'decode':>7s}  P/C/F      holdout MAPE  bias  (n)")
    for label, s, h in rows:
        if not s:
            print(f"{label:16s}   - (no sim rows)"); continue
        hs = f"{pf(h['mape'])} {100 * h['bias']:+6.1f}% ({h['n']})" if h else "-"
        print(f"{label:16s} {s['n']:3d} {pf(s['mape'])} {100 * s['bias']:+6.1f}% {pf(s['mape_prefill'])} {pf(s['mape_decode'])}  "
              f"{s['cnt']['PASS']:2d}/{s['cnt']['CONDITIONAL']:2d}/{s['cnt']['FAIL']:2d}   {hs}")
    print()
    keys = sorted(dev, key=lambda k: (k[0] != "prefill", k[1], k[2]))
    print(f"{'point':18s} " + " ".join(f"{label[:9]:>9s}" for label, _, _ in rows))
    for k in keys:
        cells = []
        for _, s, _ in rows:
            e = s["errs"].get(k) if s else None
            cells.append(f"{100 * e:+8.1f}%" if e is not None else f"{'-':>9s}")
        print(f"{k[0]} {k[1]}x{k[2]}".ljust(18), " ".join(cells))


if __name__ == "__main__":
    main()
