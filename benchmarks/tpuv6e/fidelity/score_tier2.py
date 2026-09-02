#!/usr/bin/env python3
"""Tier-2 fidelity: whole-model grid points, simulator vs silicon, with the
per-class attribution that says WHERE the residual sits (spec 2026-09-01
sections 3.2, 7, 8).

Inputs
  --sim     fidelity/sim_tier2.sh output (sim_tier2.csv): one row per
            (model, mode, seq, batch, variant); 'full' is scored, the
            l1/l2/lh triplet gives the session-3 extrapolation cross-check.
  --dh      holdout/dh_offline.py output (dh_<grid>.csv): timed generate
            wall time per point (+ anchor repeats).
  --census  analysis/kernel_census.py --steady --per-class --csv over the
            traces the same run captured: per-step device time per class.
Silicon truth for a point is the per-step DEVICE time from its trace
(bucket rows summed); wall time is the host-inclusive upper bound. Verdicts
(PASS <= 10%, CONDITIONAL <= 25%, FAIL) use the device time; the
error is (sim - silicon) / silicon.

Class mapping (silicon census -> simulator ACCTC/OPSPAN):
  qkv->QKV  o->O  gate_up|mlp_fused->GATE_UP  down->DOWN  head->HEAD
  attention->ATTN  norm->VPU_NORM  elementwise->VPU_EW
  data/other/idle -> no simulator counterpart (unmodeled: reported as such)
Simulator class time = (busy + underfilled + memstall) core-cycle share of
that class on its unit x total cycles, i.e. the time the unit spent on the
class; spans (OPSPAN first..last) are printed beside it.

Usage: score_tier2.py --sim sim_tier2.csv --dh dh_qwen.csv --census census_qwen.csv
                      --model qwen [--csv out.csv]
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

DECODE_TOKENS = 64
CLASS_MAP = [  # (label, census classes, sim class, sim unit)
    ("qkv", ("qkv",), "QKV", "sa"),
    ("o", ("o",), "O", "sa"),
    ("gate_up", ("gate_up", "mlp_fused"), "GATE_UP", "sa"),
    ("down", ("down",), "DOWN", "sa"),
    ("head", ("head",), "HEAD", "sa"),
    ("attention", ("attention",), "ATTN", "both"),
    ("norm", ("norm",), "VPU_NORM", "vpu"),
    ("elementwise", ("elementwise",), "VPU_EW", "vpu"),
]
UNMODELED = ("data", "other", "gemm_other")


def verdict(err):
    a = abs(err)
    return "PASS" if a <= 0.10 else ("CONDITIONAL" if a <= 0.25 else "FAIL")


def point_key(mode, seq, batch):
    return (mode, int(seq), int(batch))


def load_sim(path, model):
    sim = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["model"] != model or not r["cycles"]:
            continue
        sim[point_key(r["mode"], r["seq"], r["batch"])][r["variant"]] = r
    return sim


def sim_class_us(r, sim_cls, unit):
    us = float(r["us"])
    sa = sum(float(r[f"sa_{sim_cls}_{k}"]) for k in ("busy", "underfilled", "memstall")) * us
    vpu = sum(float(r[f"vpu_{sim_cls}_{k}"]) for k in ("busy", "memstall")) * us
    return {"sa": sa, "vpu": vpu, "both": max(sa, vpu)}[unit], float(r[f"span_{sim_cls}_us"])


def extrap(v, layers):
    """Session-3 method from the l1/l2/lh triplet; None if incomplete."""
    if not all(k in v for k in ("l1", "l2", "lh")):
        return None
    t1, t2, th = (float(v[k]["us"]) for k in ("l1", "l2", "lh"))
    return t1 + (layers - 1) * (t2 - t1) + (th - t1)


def load_census(path):
    """point -> {class: time_ms per step}, plus steps."""
    pts = defaultdict(dict); steps = {}
    if not path:
        return pts, steps
    for r in csv.DictReader(open(path)):
        m = re.match(r"(prefill|decode)_(\d+)_(\d+)(?:_r\d+)?$", r["point"])
        if not m:
            continue
        k = point_key(m.group(1), m.group(2), m.group(3))
        pts[k][(r["kind"], r["class"])] = float(r["time_ms"])
        steps[k] = int(r["steps"])
    return pts, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--dh", required=True)
    ap.add_argument("--census", default=None)
    ap.add_argument("--model", required=True, choices=("qwen", "mistral"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    sim = load_sim(a.sim, a.model)
    census, steps = load_census(a.census)
    wall = defaultdict(list)
    layers = {"qwen": 36, "mistral": 32}[a.model]
    for r in csv.DictReader(open(a.dh)):
        wall[point_key(r["mode"], r["seq"], r["batch"])].append(float(r["wall_s"]))

    out = []
    print(f"{'point':22s} {'sim_full_ms':>11s} {'sim_extrap':>10s} {'dev_ms':>9s} {'dev_busy':>9s} {'wall_ms':>9s} {'err_dev':>8s} {'err_busy':>8s}  verdict  n_wall")
    for k in sorted(wall, key=lambda k: (k[0] != "prefill", k[1], k[2])):
        mode, seq, batch = k
        v = sim.get(k, {})
        full = float(v["full"]["us"]) / 1e3 if "full" in v else None
        ext = extrap(v, layers)
        ext = ext / 1e3 if ext is not None else None
        ws = sorted(wall[k])
        w_ms = ws[len(ws) // 2] * 1e3
        if mode == "decode":
            w_ms = w_ms / DECODE_TOKENS  # host-inclusive, and smeared with the context prefill
        c = census.get(k)
        dev = busy = None
        if c:
            dev = sum(t for (kind, cls), t in c.items() if kind == "bucket")
            busy = dev - c.get(("bucket", "idle"), 0.0)
        err_dev = (full - dev) / dev if (full is not None and dev) else None
        err_busy = (full - busy) / busy if (full is not None and busy) else None
        vd = verdict(err_dev) if err_dev is not None else ("no-sim" if full is None else "no-trace")
        f = lambda x, w=9: f"{x:{w}.3f}" if x is not None else f"{'-':>{w}s}"
        p = lambda x: f"{100 * x:+7.1f}%" if x is not None else f"{'-':>8s}"
        name = f"{mode} {seq}x{batch}"
        print(f"{name:22s} {f(full, 11)} {f(ext, 10)} {f(dev)} {f(busy)} {f(w_ms)} {p(err_dev)} {p(err_busy)}  {vd:11s} {len(ws)}"
              + (f"  wall spread {100 * (ws[-1] - ws[0]) / ws[0]:.1f}%" if len(ws) > 1 else ""))
        row = dict(model=a.model, mode=mode, seq=seq, batch=batch, sim_full_ms=full, sim_extrap_ms=ext,
                   dev_ms=dev, dev_busy_ms=busy, wall_ms=w_ms, n_wall=len(ws), err_dev=err_dev, err_busy=err_busy,
                   verdict=vd, steps=steps.get(k, 0))
        # per-class attribution
        if c and "full" in v:
            print(f"    {'class':12s} {'silicon_ms':>10s} {'sim_ms':>8s} {'sim_span':>8s} {'delta_ms':>9s}")
            unm = 0.0
            mlp = {"si": 0.0, "sm": 0.0}
            for label, ccls, scls, unit in CLASS_MAP:
                si = sum(c.get(("gemm_class", x), 0.0) + (c.get(("bucket", x), 0.0) if x in ("attention", "norm", "elementwise") else 0.0)
                         for x in ccls)
                sm_us, span_us = sim_class_us(v["full"], scls, unit)
                sm, sp = sm_us / 1e3, span_us / 1e3
                print(f"    {label:12s} {si:10.3f} {sm:8.3f} {sp:8.3f} {sm - si:+9.3f}")
                row[f"si_{label}_ms"] = si; row[f"sim_{label}_ms"] = sm; row[f"sim_{label}_span_ms"] = sp
                if label in ("gate_up", "down"):
                    mlp["si"] += si; mlp["sm"] += sm
            # XLA fuses one D->F projection with F->D (mlp_fused), so gate_up vs
            # down are not separable on silicon; the MLP as a whole is.
            print(f"    {'mlp (sum)':12s} {mlp['si']:10.3f} {mlp['sm']:8.3f} {'-':>8s} {mlp['sm'] - mlp['si']:+9.3f}")
            row["si_mlp_ms"] = mlp["si"]; row["sim_mlp_ms"] = mlp["sm"]
            for x in UNMODELED:
                t = c.get(("bucket", x), 0.0) + c.get(("gemm_class", x), 0.0)
                unm += t
            idle = c.get(("bucket", "idle"), 0.0)
            print(f"    {'unmodeled':12s} {unm:10.3f} {'-':>8s} {'-':>8s}   (data/other/gemm_other: no sim counterpart)")
            print(f"    {'idle':12s} {idle:10.3f} {'-':>8s} {'-':>8s}   (device idle inside the step)")
            row["si_unmodeled_ms"] = unm; row["si_idle_ms"] = idle
        out.append(row)

    scored = [r for r in out if r["err_dev"] is not None]
    if scored:
        mape = sum(abs(r["err_dev"]) for r in scored) / len(scored)
        bias = sum(r["err_dev"] for r in scored) / len(scored)
        cnt = defaultdict(int)
        for r in scored:
            cnt[r["verdict"]] += 1
        print(f"\n{a.model}: {len(scored)} points scored vs device time: MAPE {100 * mape:.1f}%  mean bias {100 * bias:+.1f}%  "
              f"PASS {cnt['PASS']} CONDITIONAL {cnt['CONDITIONAL']} FAIL {cnt['FAIL']}")
        ex = [r for r in scored if r["sim_extrap_ms"] is not None and r["sim_full_ms"]]
        if ex:
            d = max(abs(r["sim_extrap_ms"] - r["sim_full_ms"]) / r["sim_full_ms"] for r in ex)
            print(f"extrapolation (l1/l2/lh) vs full run: max |delta| {100 * d:.1f}% over {len(ex)} points")
    if a.csv and out:
        keys = []
        for r in out:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
