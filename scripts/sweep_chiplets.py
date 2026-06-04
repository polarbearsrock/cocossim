#!/usr/bin/env python3
"""
sweep_chiplets.py — COCOSSim multi-chiplet scaling sweep

Runs perf_model_chiplet across chiplet counts and topologies, collects
per-workload metrics, and prints a summary table.

Usage:
    python3 scripts/sweep_chiplets.py [--binary BUILD/perf_model_chiplet]
                                       [--workloads examples/llama7b_decode.txt ...]
                                       [--chiplets 1 2 4 8]
                                       [--sa_sz 128]
                                       [--freq 1.0]
"""

import subprocess
import re
import sys
import os
import argparse
from itertools import product


# ── helpers ──────────────────────────────────────────────────────────────────

TOPO_NAMES = {0: "ring", 1: "mesh", 2: "torus"}
TOPO_IDS   = {"ring": 0, "mesh": 1, "torus": 2}


def parse_output(text: str) -> dict:
    """Extract key metrics from perf_model_chiplet stdout."""
    metrics = {}

    def _find(pattern, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    metrics["total_cycles"]    = _find(r"Total Cycles:\s+(\d+)")
    metrics["compute_cycles"]  = _find(r"Compute Cycles:\s+(\d+)")
    metrics["comm_cycles"]     = _find(r"Communication Cycles:\s+(\d+)")
    metrics["jobs_completed"]  = _find(r"Completed:\s+(\d+)")
    metrics["bytes_xfr_mb"]    = _find(r"Bytes Transferred:\s+([\d.]+)", float)
    metrics["packets_sent"]    = _find(r"Packets Sent:\s+(\d+)")
    metrics["throughput_tops"] = _find(r"Throughput:\s+([\d.]+)", float)
    metrics["latency_ms"]      = _find(r"Latency:\s+([\d.]+)", float)
    metrics["total_energy_mj"] = _find(r"TOTAL:\s+([\d.]+)", float)
    metrics["efficiency"]      = _find(r"Energy Efficiency:\s+([\d.]+)", float)
    metrics["timed_out"]       = "WARNING: Simulation timeout" in text

    # Comm overhead: comm_cycles / total_cycles
    if metrics["total_cycles"] and metrics["comm_cycles"]:
        metrics["comm_overhead_pct"] = (
            100.0 * metrics["comm_cycles"] / metrics["total_cycles"]
        )
    else:
        metrics["comm_overhead_pct"] = None

    return metrics


def run_sim(binary, workload, chiplets, topo_id, sa_sz, freq, max_cycles):
    cmd = [
        binary,
        "-i", workload,
        "-c", str(chiplets),
        "-topo", str(topo_id),
        "-sa_sz", str(sa_sz),
        "-f", str(freq),
        "-max_cycles", str(max_cycles),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary",
        default=os.path.join(os.path.dirname(__file__), "../build/perf_model_chiplet"))
    parser.add_argument("--workloads", nargs="+",
        default=[
            "examples/llama7b_decode.txt",
            "examples/llama7b_prefill.txt",
        ])
    parser.add_argument("--chiplets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--topologies", nargs="+", default=["ring", "mesh"])
    parser.add_argument("--sa_sz", type=int, default=128,
        help="Systolic array size per chiplet (default: 128)")
    parser.add_argument("--freq", type=float, default=1.0)
    parser.add_argument("--max_cycles", type=int, default=100_000_000)
    args = parser.parse_args()

    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        print(f"ERROR: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    # The binary expects to be run from its own directory because it resolves
    # DRAMSim3 configs relative to that (../dramsim3/configs/...).
    build_dir = os.path.dirname(binary)
    repo_root = os.path.dirname(build_dir)
    os.chdir(build_dir)

    # Make workload paths absolute before chdir
    args.workloads = [
        w if os.path.isabs(w) else os.path.join(repo_root, w)
        for w in args.workloads
    ]

    topo_ids = [TOPO_IDS[t] for t in args.topologies]

    rows = []
    total = len(args.workloads) * len(args.chiplets) * len(topo_ids)
    done  = 0

    for workload in args.workloads:
        wl_name = os.path.splitext(os.path.basename(workload))[0]
        for chiplets, topo_id in product(args.chiplets, topo_ids):
            topo_name = TOPO_NAMES[topo_id]
            # Mesh requires >= 4 chiplets (2×2 minimum)
            if topo_id == 1 and chiplets < 4:
                done += 1
                continue

            done += 1
            print(f"[{done}/{total}] {wl_name}  c={chiplets}  topo={topo_name} ...",
                  end="", flush=True)

            out = run_sim(binary, workload, chiplets, topo_id,
                          args.sa_sz, args.freq, args.max_cycles)
            m   = parse_output(out)

            status = "TIMEOUT" if m["timed_out"] else "OK"
            tops = m['throughput_tops'] or 0.0
            comm = m['comm_overhead_pct'] or 0.0
            print(f"  {status}  {m['total_cycles']} cycles  "
                  f"{tops:.2f} TOPS  comm={comm:.1f}%")

            rows.append({
                "workload": wl_name,
                "chiplets": chiplets,
                "topology": topo_name,
                **m,
            })

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'Workload':<25} {'Chips':>5} {'Topo':>6}  "
          f"{'Cycles':>12}  {'Throughput':>12}  {'Comm%':>7}  "
          f"{'Latency_ms':>11}  {'Energy_mJ':>10}  {'Eff_TOPS/W':>11}  Status")
    print("-" * 110)

    for r in rows:
        print(
            f"{r['workload']:<25} {r['chiplets']:>5} {r['topology']:>6}  "
            f"{r['total_cycles'] or 'N/A':>12}  "
            f"{r['throughput_tops'] or 0:>12.3f}  "
            f"{r['comm_overhead_pct'] or 0:>7.1f}  "
            f"{r['latency_ms'] or 0:>11.4f}  "
            f"{r['total_energy_mj'] or 0:>10.3f}  "
            f"{r['efficiency'] or 0:>11.4f}  "
            f"{'TIMEOUT' if r['timed_out'] else 'OK'}"
        )

    print("=" * 110)

    # ── speedup table (relative to 1 chiplet, same workload+topo) ────────────
    print("\nSpeedup vs 1-chiplet (same workload, ring topology):")
    baseline = {
        r["workload"]: r["total_cycles"]
        for r in rows
        if r["chiplets"] == 1 and r["topology"] == "ring"
    }
    if baseline:
        print(f"  {'Workload':<25} {'Chips':>5}  {'Speedup':>8}  {'Comm%':>7}")
        for r in rows:
            if r["topology"] != "ring" or r["chiplets"] == 1:
                continue
            base = baseline.get(r["workload"])
            if base and r["total_cycles"]:
                speedup = base / r["total_cycles"]
                print(f"  {r['workload']:<25} {r['chiplets']:>5}  {speedup:>8.2f}x  "
                      f"{r['comm_overhead_pct'] or 0:>7.1f}%")

    # Write CSV
    csv_path = "results_sweep.csv"
    with open(csv_path, "w") as f:
        headers = ["workload", "chiplets", "topology", "total_cycles",
                   "compute_cycles", "comm_cycles", "comm_overhead_pct",
                   "throughput_tops", "latency_ms", "total_energy_mj",
                   "efficiency", "timed_out"]
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
