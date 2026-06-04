#!/usr/bin/env python3
"""
sweep_seqlen.py — Sweep sequence length across the memory-bound / compute-bound transition.

For LLaMA-7B, varies M (batch×seq tokens processed in parallel) from 1 to 2048.
At M=1 the workload is pure decode (GEMV, memory-bound).
As M increases it transitions to prefill (GEMM), becoming compute-bound around M~sa_sz.

Reports:
  - COCOSSim cycles (SA FSM + DRAMSim3) per sequence length
  - Analytical (roofline) cycles
  - Gap % — how much the analytical tool under/over-estimates
  - Compute vs. memory bound regime

Usage:
    python3 scripts/sweep_seqlen.py [--chiplets 1 4] [--sa_sz 128]
"""

import os, sys, math, re, subprocess, argparse, tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
BUILD_DIR  = os.path.join(REPO_ROOT, "build")

# LLaMA-7B architecture constants
H    = 4096
FFN  = 11008
QKV  = 3 * H   # 12288

UCIe_BW_GB_S    = 28.0
UCIe_FIXED_LAT  = 60
HBM2_BW_GB_S    = 256.0
WORD_SIZE        = 2      # FP16


# ── workload generation ───────────────────────────────────────────────────────

def llama7b_layers(seq_len):
    """Return [(M, K, N), ...] for one LLaMA-7B transformer block at given seq_len."""
    M = seq_len
    return [
        (M, H,   QKV),   # QKV projection
        (M, H,   H),     # output projection
        (M, H,   FFN),   # FFN gate  (SwiGLU)
        (M, H,   FFN),   # FFN up    (SwiGLU)
        (M, FFN, H),     # FFN down
    ]


def write_workload(seq_len, path):
    layers = llama7b_layers(seq_len)
    with open(path, "w") as f:
        f.write(f"# LLaMA-7B seq_len={seq_len} (one transformer block)\n")
        f.write("# Format: Matmul M K N\n")
        for (M, K, N) in layers:
            f.write(f"Matmul {M} {K} {N}\n")


# ── analytical model (matches decoupling_tax.py) ─────────────────────────────

def analytical_cycles(layers, tp, sa_sz, freq_ghz=1.0):
    """Roofline analytical: compute = min(M,sa_sz)*sa_sz MACs/cycle; no DRAM stall."""
    total_compute = 0
    total_comm    = 0
    bw = UCIe_BW_GB_S / freq_ghz

    for (M, K, N) in layers:
        eff_rows  = min(M, sa_sz)
        sa_ops    = eff_rows * sa_sz
        macs_chip = M * K * (N // tp)
        total_compute += math.ceil(macs_chip / sa_ops)

        if tp > 1:
            out_bytes       = M * N * 2
            allreduce_bytes = 2 * (tp - 1) * out_bytes // tp
            bytes_per_link  = allreduce_bytes // tp
            ser_cycles      = math.ceil(bytes_per_link / bw)
            total_comm     += ser_cycles + UCIe_FIXED_LAT

    return total_compute + total_comm, total_compute, total_comm


def scalesim_roofline(layers, tp, sa_sz, hbm_bw=HBM2_BW_GB_S, freq_ghz=1.0):
    """SCALE-Sim roofline: total = max(compute, memory_load) per layer, summed."""
    hbm_bpc = hbm_bw / freq_ghz   # bytes per cycle
    total = 0
    for (M, K, N) in layers:
        N_chip = max(1, N // tp)
        row_fold = math.ceil(M / sa_sz)
        col_fold = math.ceil(N_chip / sa_sz)
        compute  = row_fold * col_fold * (K + sa_sz - 1)
        memory   = math.ceil(K * N_chip * WORD_SIZE / hbm_bpc)
        total   += max(compute, memory)
    return total


def allreduce_analytical(layers, tp, ucie_bw_gb_s=UCIe_BW_GB_S, freq_ghz=1.0):
    """Ring AllReduce latency via bandwidth formula (same as decoupling_tax.py)."""
    if tp == 1:
        return 0
    bw = ucie_bw_gb_s / freq_ghz
    total = 0
    for (M, K, N) in layers:
        out_bytes       = M * N * 2
        allreduce_bytes = 2 * (tp - 1) * out_bytes // tp
        bytes_per_link  = allreduce_bytes // tp
        ser_cycles      = math.ceil(bytes_per_link / bw)
        total          += ser_cycles + UCIe_FIXED_LAT
    return total


# ── COCOSSim runner ───────────────────────────────────────────────────────────

def run_cocossim(wl_path, chiplets, sa_sz, freq=1.0, max_cycles=200_000_000):
    binary = os.path.join(BUILD_DIR, "perf_model_chiplet")
    cmd = [binary, "-i", wl_path, "-c", str(chiplets), "-sa_sz", str(sa_sz),
           "-f", str(freq), "-topo", "0", "-max_cycles", str(max_cycles)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=BUILD_DIR)
    text = r.stdout + r.stderr

    def find(pat, cast=int):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    return {
        "total":   find(r"Total Cycles:\s+(\d+)"),
        "compute": find(r"Compute Cycles:\s+(\d+)"),
        "comm":    find(r"Communication Cycles:\s+(\d+)"),
        "timeout": "WARNING: Simulation timeout" in text,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chiplets",  nargs="+", type=int, default=[1, 4])
    ap.add_argument("--sa_sz",     type=int,  default=128)
    ap.add_argument("--freq",      type=float, default=1.0)
    ap.add_argument("--seq_lens",  nargs="+", type=int,
                    default=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--max_cycles", type=int, default=200_000_000)
    args = ap.parse_args()

    rows = []
    total_runs = len(args.seq_lens) * len(args.chiplets)
    done = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for seq_len in args.seq_lens:
            layers = llama7b_layers(seq_len)
            wl_path = os.path.join(tmpdir, f"llama7b_s{seq_len}.txt")
            write_workload(seq_len, wl_path)

            for chips in args.chiplets:
                done += 1
                print(f"[{done}/{total_runs}] seq={seq_len:>5}  c={chips}",
                      end="", flush=True)

                a_total, a_compute, a_comm = analytical_cycles(
                    layers, chips, args.sa_sz, args.freq)
                ss_compute = scalesim_roofline(
                    layers, chips, args.sa_sz, freq_ghz=args.freq)
                ss_comm  = allreduce_analytical(layers, chips, freq_ghz=args.freq)
                ss_total = ss_compute + ss_comm

                coco = run_cocossim(wl_path, chips, args.sa_sz,
                                    args.freq, args.max_cycles)
                if coco["timeout"] or coco["total"] is None:
                    print("  TIMEOUT/ERROR")
                    continue

                c_total   = coco["total"]
                c_compute = coco["compute"] or 0

                err_analytic  = (a_total  - c_total) / c_total * 100
                err_scalesim  = (ss_total - c_total) / c_total * 100
                err_compute   = (a_compute - c_compute) / c_compute * 100 \
                                if c_compute else float("nan")

                # Classify regime: memory-bound when DRAM load time > compute time
                # Roofline crossing: weight_bytes/hbm_bw > compute_cycles
                # For a single representative layer (QKV, largest):
                M, K, N = layers[0]
                N_chip = N // chips
                compute_bound_M = args.sa_sz  # M where compute = memory at sa_sz utilisation
                regime = "memory" if seq_len < compute_bound_M else "compute"

                print(f"  analytic={int(a_total):>10,}  scalesim={int(ss_total):>10,}"
                      f"  cocossim={c_total:>10,}"
                      f"  Δanalytic={err_analytic:+.1f}%  Δss={err_scalesim:+.1f}%"
                      f"  [{regime}]")

                rows.append({
                    "seq_len": seq_len, "chiplets": chips,
                    "analytical": int(a_total), "scalesim": int(ss_total),
                    "cocossim": c_total,
                    "err_analytic": err_analytic,
                    "err_scalesim": err_scalesim,
                    "err_compute": err_compute,
                    "regime": regime,
                })

    # ── summary table ─────────────────────────────────────────────────────────
    for chips in args.chiplets:
        subset = [r for r in rows if r["chiplets"] == chips]
        if not subset:
            continue
        W = 100
        print(f"\n{'='*W}")
        print(f"LLaMA-7B  Sequence-Length Sweep  (chiplets={chips}, sa_sz={args.sa_sz})")
        print(f"{'-'*W}")
        print(f"{'seq_len':>8}  {'Analytical':>12}  {'SCALE-Sim':>12}  {'COCOSSim':>12}"
              f"  {'Δanalytic':>10}  {'Δscalesim':>10}  Regime")
        print(f"{'-'*W}")
        for r in subset:
            print(f"{r['seq_len']:>8}  {r['analytical']:>12,}  {r['scalesim']:>12,}"
                  f"  {r['cocossim']:>12,}"
                  f"  {r['err_analytic']:>+9.1f}%  {r['err_scalesim']:>+9.1f}%"
                  f"  {r['regime']}")
        print(f"{'='*W}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(REPO_ROOT, "results_seqlen_sweep.csv")
    with open(csv_path, "w") as f:
        hdrs = ["seq_len", "chiplets", "analytical", "scalesim_roofline",
                "cocossim", "err_analytic_pct", "err_scalesim_pct", "regime"]
        f.write(",".join(hdrs) + "\n")
        for r in rows:
            f.write(",".join([
                str(r["seq_len"]), str(r["chiplets"]),
                str(r["analytical"]), str(r["scalesim"]), str(r["cocossim"]),
                f"{r['err_analytic']:.2f}", f"{r['err_scalesim']:.2f}",
                r["regime"]
            ]) + "\n")
    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
