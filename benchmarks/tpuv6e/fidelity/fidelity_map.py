#!/usr/bin/env python3
"""Fidelity map figure (spec 2026-09-01 section 9): simulator error vs silicon
per cell, one panel per tier, with the PASS (+-10%) and CONDITIONAL (+-25%)
bands. Error = (sim - silicon) / silicon; positive = simulator too slow.

Usage: fidelity_map.py --tier1 scorecard_tier1.csv --a1 score_a1.csv
                       --tier2 label=score_qwen.csv [--tier2 label=score_mistral.csv ...]
                       --out fidelity_map.png [--title ...]
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def bands(ax):
    ax.axhspan(-10, 10, color="green", alpha=0.08, lw=0)
    ax.axhspan(-25, -10, color="orange", alpha=0.07, lw=0)
    ax.axhspan(10, 25, color="orange", alpha=0.07, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("(sim - silicon) / silicon  [%]")


def panel_gemm(ax, rows):
    by = defaultdict(list)
    for r in rows:
        if r["cell"] not in ("G1", "G2", "G3"):
            continue
        m, k, n = (int(x) for x in r["shape"].split("x"))
        by[(r["cell"], r["label"] if r["cell"] == "G3" else f"{k}x{n}")].append((m, 100 * float(r["err"])))
    for (cell, lab), pts in sorted(by.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=1,
                label=f"{cell} {lab}")
    bands(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (rows)")
    ax.set_title("Tier 1: GEMM cells (device per-step, slope/chain method)")
    ax.legend(fontsize=6, ncol=2)


def panel_e1(ax, rows):
    for op, color in (("add", "C0"), ("exp", "C1")):
        hb = [(int(r["shape"].split("=")[1]), 100 * float(r["err"])) for r in rows if r["cell"] == "E1" and r["label"] == op]
        vm = [(int(r["shape"].split("=")[1]), 100 * float(r["err"])) for r in rows if r["cell"] == "E1v" and r["label"] == op]
        if hb:
            hb.sort(); ax.plot([p[0] for p in hb], [p[1] for p in hb], marker="o", color=color, lw=1, label=f"{op} (HBM stream)")
        if vm:
            vm.sort(); ax.plot([p[0] for p in vm], [p[1] for p in vm], marker="o", mfc="none", color=color, lw=0.5, ls=":",
                               label=f"{op} (VMEM-resident on silicon: no HBM verdict)")
    bands(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements")
    ax.set_title("Tier 1: elementwise E1 (sim models an HBM stream)")
    ax.legend(fontsize=6)
    ax.set_ylim(-60, 120)


def panel_a1(ax, rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["mode"], int(r["B"]))].append((int(r["S"]), 100 * float(r["err"])))
    for (mode, b), pts in sorted(by.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s" if mode == "decode" else "o", ms=4, lw=1,
                ls="-" if mode == "prefill" else "--", label=f"{mode} B={b}")
    bands(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("S (context / sequence length)")
    ax.set_title("Tier 1: attention kernels A1 (prefill: Pallas flash_attention;\ndecode: legacy Pallas paged_attention, not vLLM's ragged_paged_attention)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)


def panel_tier2(ax, labelled):
    names = None
    width = 0.8 / max(1, len(labelled))
    for i, (label, rows) in enumerate(labelled):
        rows = [r for r in rows if r.get("err_dev") not in (None, "")]
        rows.sort(key=lambda r: (r["mode"] != "prefill", int(r["seq"]), int(r["batch"])))
        if names is None:
            names = [f"{r['mode'][0]} {r['seq']}x{r['batch']}" for r in rows]
        xs = [j + i * width for j in range(len(rows))]
        ax.bar(xs, [100 * float(r["err_dev"]) for r in rows], width=width, label=label)
    bands(ax)
    if names:
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(names))])
        ax.set_xticklabels(names, rotation=60, fontsize=6)
    ax.set_title("Tier 2: whole model vs per-step device time\n(p = prefill seq x batch, d = decode context x batch)", fontsize=10)
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--a1", required=True)
    ap.add_argument("--tier2", action="append", default=[], help="label=score_tier2.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="COCOSSim TPU v6e-1 fidelity map (2026-09-01)")
    a = ap.parse_args()
    t1 = list(csv.DictReader(open(a.tier1)))
    a1 = list(csv.DictReader(open(a.a1)))
    t2 = [(c.split("=", 1)[0], list(csv.DictReader(open(c.split("=", 1)[1])))) for c in a.tier2]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    panel_gemm(axs[0][0], t1)
    panel_e1(axs[0][1], t1)
    panel_a1(axs[1][0], a1)
    panel_tier2(axs[1][1], t2)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
