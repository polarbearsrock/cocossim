#!/usr/bin/env python3
"""
decoupling_tax.py — Quantify the "decoupling tax" for RQ2.

Compares two prediction approaches for the same LLM workload:

  (A) Naive analytical (decoupled):
        compute = MACs / (chips × sa_ops_per_cycle)   [pure arithmetic, no DRAM]
        comm    = allreduce_bytes / ucie_bw + fixed_latency
        total   = sum(compute + comm) across all layers

  (B) COCOSSim unified:
        Actual simulated cycles with DRAMSim3 + UCIe in one tick loop.

The gap between (A) and (B) is the decoupling tax.  For compute-bound
(prefill) layers the gap is small; for memory-bound (decode) layers it
is large because the analytical model has no DRAM stall term.

Usage:
    python3 scripts/decoupling_tax.py [--chiplets 1 2 4 8] [--sa_sz 128]
"""

import subprocess, re, os, sys, math, argparse

# ── UCIe parameters (standard_16GT_x16, matching ChipletArch default) ─────────
UCIe_BW_GB_S      = 28.0   # effective bandwidth in GB/s (= bytes/ns at 1 GHz)
UCIe_FIXED_LATENCY = 60    # cycles: phy_tx + phy_rx + link + adapter_tx + adapter_rx

# ── Analytical model ──────────────────────────────────────────────────────────

def parse_layers(filepath):
    """Return list of (M, K, N) for every Matmul line in the workload file."""
    layers = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "Matmul" and len(parts) >= 4:
                layers.append((int(parts[1]), int(parts[2]), int(parts[3])))
    return layers


def analytical_total(layers, chiplets, sa_sz, freq_ghz=1.0):
    """
    Analytical (decoupled) prediction.

    Compute  = pure MAC throughput — no DRAM stall term.
    Comm     = ring AllReduce volume / UCIe bandwidth + fixed latency.
    Total    = sum over layers (compute + comm), phases independent.

    Returns (total_cycles, compute_cycles, comm_cycles) as floats.
    """
    tp = chiplets

    total_compute = 0.0
    total_comm    = 0.0

    for (M, K, N) in layers:
        # ── compute (arithmetic roofline) ─────────────────────────────────────
        # For M=1 (decode): vector unit — sa_sz MACs/cycle per chiplet.
        # For M≥sa_sz (prefill): full systolic array — sa_sz² MACs/cycle.
        # General: min(M, sa_sz) × sa_sz (matches COCOSSim hardware model).
        effective_rows = min(M, sa_sz)
        sa_ops         = effective_rows * sa_sz   # MACs per cycle per chiplet
        macs_per_chip  = M * K * (N // tp)
        arith_cycles   = math.ceil(macs_per_chip / sa_ops)

        # ── allreduce traffic (ring, 2-phase) ─────────────────────────────────
        if tp > 1:
            output_bytes      = M * N * 2                  # FP16
            allreduce_bytes   = 2 * (tp - 1) * output_bytes // tp   # ring formula
            bytes_per_link    = allreduce_bytes // tp      # one hop per step
            # UCIe_BW_GB_S is GB/s = bytes/ns; at freq_ghz GHz → bytes/cycle
            bw_bytes_per_cycle = UCIe_BW_GB_S / freq_ghz   # bytes/cycle
            ser_cycles        = math.ceil(bytes_per_link / bw_bytes_per_cycle)
            comm_cycles_layer = ser_cycles + UCIe_FIXED_LATENCY
        else:
            comm_cycles_layer = 0

        total_compute += arith_cycles
        total_comm    += comm_cycles_layer

    return total_compute + total_comm, total_compute, total_comm


# ── COCOSSim runner (re-uses sweep_chiplets logic) ────────────────────────────

def run_cocossim(binary, workload, chiplets, sa_sz, freq, max_cycles=100_000_000):
    cmd = [binary, "-i", workload,
           "-c", str(chiplets),
           "-sa_sz", str(sa_sz),
           "-f", str(freq),
           "-topo", "0",                # ring
           "-max_cycles", str(max_cycles)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stdout + result.stderr

    def _find(pat, cast=int):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    return {
        "total":   _find(r"Total Cycles:\s+(\d+)"),
        "compute": _find(r"Compute Cycles:\s+(\d+)"),
        "comm":    _find(r"Communication Cycles:\s+(\d+)"),
        "timed_out": "WARNING: Simulation timeout" in text,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary",
        default=os.path.join(os.path.dirname(__file__), "../build/perf_model_chiplet"))
    parser.add_argument("--workloads", nargs="+",
        default=["examples/llama7b_decode.txt",
                 "examples/llama7b_prefill.txt"])
    parser.add_argument("--chiplets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--sa_sz",  type=int,   default=128)
    parser.add_argument("--freq",   type=float, default=1.0)
    args = parser.parse_args()

    binary    = os.path.abspath(args.binary)
    build_dir = os.path.dirname(binary)
    repo_root = os.path.dirname(build_dir)
    os.chdir(build_dir)

    args.workloads = [
        w if os.path.isabs(w) else os.path.join(repo_root, w)
        for w in args.workloads
    ]

    rows = []
    total_runs = len(args.workloads) * len(args.chiplets)
    done = 0

    for wl_path in args.workloads:
        wl_name = os.path.splitext(os.path.basename(wl_path))[0]
        layers  = parse_layers(wl_path)

        for chips in args.chiplets:
            done += 1
            print(f"[{done}/{total_runs}] {wl_name}  c={chips} ...", end="", flush=True)

            # Analytical
            a_total, a_compute, a_comm = analytical_total(
                layers, chips, args.sa_sz, args.freq)

            # COCOSSim
            coco = run_cocossim(binary, wl_path, chips,
                                args.sa_sz, args.freq)

            if coco["timed_out"] or coco["total"] is None:
                print("  TIMEOUT/ERROR")
                continue

            c_total   = coco["total"]
            c_compute = coco["compute"] or 0
            c_comm    = coco["comm"] or 0

            # Error = (analytical - cocossim) / cocossim
            err_total   = (a_total   - c_total)   / c_total   * 100
            err_compute = (a_compute - c_compute)  / c_compute * 100 if c_compute else float("nan")
            err_comm    = (a_comm    - c_comm)     / c_comm    * 100 if c_comm    else float("nan")

            print(f"  analytical={int(a_total):>10,}  cocossim={c_total:>10,}  "
                  f"err={err_total:+.1f}%")

            rows.append({
                "workload":    wl_name,
                "chiplets":    chips,
                "a_total":     int(a_total),
                "a_compute":   int(a_compute),
                "a_comm":      int(a_comm),
                "c_total":     c_total,
                "c_compute":   c_compute,
                "c_comm":      c_comm,
                "err_total":   err_total,
                "err_compute": err_compute,
                "err_comm":    err_comm,
            })

    # ── print summary ──────────────────────────────────────────────────────────
    W = 95
    print("\n" + "=" * W)
    print("Decoupling Tax — Analytical vs COCOSSim  (ring topology, sa_sz="
          f"{args.sa_sz}, freq={args.freq}GHz)")
    print("-" * W)
    print(f"{'Workload':<25} {'Chips':>5}  "
          f"{'Analytical':>12}  {'COCOSSim':>12}  {'Err%':>7}  "
          f"{'Δ compute%':>11}  {'Δ comm%':>9}  Regime")
    print("-" * W)

    for r in rows:
        regime = "memory-bound" if abs(r["err_compute"]) > 20 else "compute-bound"
        print(f"{r['workload']:<25} {r['chiplets']:>5}  "
              f"{r['a_total']:>12,}  {r['c_total']:>12,}  "
              f"{r['err_total']:>+7.1f}%  "
              f"{r['err_compute']:>+10.1f}%  "
              f"{r['err_comm']:>+8.1f}%  {regime}")

    print("=" * W)
    print()
    print("Interpretation:")
    print("  Err% > 0  → analytical OVERESTIMATES  (decoupled tool over-predicts latency)")
    print("  Err% < 0  → analytical UNDERESTIMATES (decoupled tool under-predicts latency)")
    print("  Δ compute% measures DRAM stall error: analytical ignores HBM timing,")
    print("             COCOSSim captures it via DRAMSim3. Large gap = memory-bound regime.")
    print("  Δ comm%   measures UCIe model error: both use bandwidth formula,")
    print("             so gap here reflects credit/queueing effects.")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(repo_root, "results_decoupling_tax.csv")
    with open(csv_path, "w") as f:
        headers = ["workload", "chiplets",
                   "analytical_total", "analytical_compute", "analytical_comm",
                   "cocossim_total",   "cocossim_compute",   "cocossim_comm",
                   "err_total_pct", "err_compute_pct", "err_comm_pct"]
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join([
                r["workload"], str(r["chiplets"]),
                str(r["a_total"]), str(r["a_compute"]), str(r["a_comm"]),
                str(r["c_total"]), str(r["c_compute"]), str(r["c_comm"]),
                f"{r['err_total']:.2f}", f"{r['err_compute']:.2f}", f"{r['err_comm']:.2f}"
            ]) + "\n")
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
