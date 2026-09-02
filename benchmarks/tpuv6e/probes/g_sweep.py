#!/usr/bin/env python3
"""G-sweep: tier-1 GEMM cells G1/G2/G3 of the fidelity-benchmark spec
(docs/superpowers/specs/2026-09-01-tpuv6e-fidelity-benchmark-design.md 3.1),
device-side only: every timed call runs CHAIN GEMMs inside one jit
(lax.scan) so the ~113 us host-dispatch floor is amortized to <1 us/step.

Shapes are NOT square in general (G3 uses the real Qwen3-8B / Mistral-7B
projection shapes), so the chain cannot feed y back as x. The carry is x
itself: each step adds the row-sums of y = x @ w (over ALL N columns) back
into every row of x, and the jit returns the full-array sum of the final x.
Every element of every y is therefore live and every row of the next x
depends on the previous step -- nothing can be sliced or hoisted. (A first
version returned y[0,0]; XLA reduced the whole GEMM to one K-length dot
product and reported 100 PFLOP/s. Never reduce to a slice.)

Per point: per-step us, TF/s, weight+activation bytes per step, effective
GB/s. With --trace DIR one XProf trace per point (a single chained call) is
captured for per-op MXU / HBM utilization (census v2).

Usage: g_sweep.py [--cells G1,G2,G3] [--dry-run] [--out g_sweep.csv] [--trace DIR]
"""
import argparse
import os

CHAIN = 16
G1 = [(n, n, n) for n in (1024, 2048, 4096, 8192)]
G2 = [(m, k, n) for (k, n) in ((8192, 8192), (4096, 4096))
      for m in (128, 256, 512, 1024, 2048)]
G3_SHAPES = {  # (K, N): label
    (4096, 4096): "qwen_q", (4096, 1024): "qwen_kv", (4096, 12288): "qwen_gate_up",
    (12288, 4096): "qwen_down", (4096, 151936): "qwen_head",
    (4096, 14336): "mistral_gate_up", (14336, 4096): "mistral_down", (4096, 32768): "mistral_head",
}
G3 = [(m, k, n) for (k, n) in G3_SHAPES for m in (1, 4, 8, 16, 32, 64, 128, 256)]
CELLS = {"G1": G1, "G2": G2, "G3": G3}
PEAK_TFLOPS = 918.0


def label_of(cell, k, n):
    return G3_SHAPES.get((k, n), f"{cell}_{k}x{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="G1,G2,G3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="g_sweep.csv")
    ap.add_argument("--trace", default=None, help="capture one XProf trace per point under DIR/<cell>_<M>x<K>x<N>")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",")]
    points = [(c, m, k, n) for c in cells for (m, k, n) in CELLS[c]]
    if args.dry_run:
        for (c, m, k, n) in points:
            print(f"{c}  M={m:5d} K={k:6d} N={n:6d}  {label_of(c, k, n)}")
        print(f"{len(points)} points, chain {CHAIN}")
        return

    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done

    @jax.jit
    def jchain(x0, w):
        def step(x, _):
            y = x @ w                                            # M x N, bf16
            r = jnp.sum(y, axis=1, dtype=jnp.float32)            # every column of every row
            x_next = x + (r * 1e-3).astype(x.dtype)[:, None]     # every row of next x depends on y
            return x_next, None
        xf, _ = jax.lax.scan(step, x0, None, length=CHAIN)
        return jnp.sum(xf.astype(jnp.float32))

    for (c, m, k, n) in points:
        if already_done(args.out, {"cell": c, "M": m, "K": k, "N": n}):
            continue
        w = jnp.full((k, n), 1e-3, dtype=jnp.bfloat16)
        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        r = time_op(lambda: jchain(x, w))
        per_step = r["median_s"] / CHAIN
        flops = 2.0 * m * k * n
        wbytes = k * n * 2
        abytes = (m * k + m * n) * 2
        tflops = flops / per_step / 1e12
        row = {"cell": c, "label": label_of(c, k, n), "M": m, "K": k, "N": n, "chain": CHAIN, **r,
               "per_step_us": per_step * 1e6, "tflops": tflops, "mfu": tflops / PEAK_TFLOPS,
               "weight_mb": wbytes / 1e6, "act_mb": abytes / 1e6,
               "eff_gbs": (wbytes + abytes) / per_step / 1e9}
        if tflops > PEAK_TFLOPS * 1.05:
            print(f"SANITY FAIL: {tflops:.0f} TF/s exceeds peak -- XLA elided work; row NOT written", flush=True)
            continue
        csv_append(args.out, row)
        print(f"{c} M={m:5d} K={k:6d} N={n:6d}  {row['per_step_us']:9.2f} us/step  "
              f"{tflops:7.1f} TF/s  {row['eff_gbs']:7.0f} GB/s", flush=True)
        if args.trace:
            d = os.path.join(args.trace, f"{c}_{m}x{k}x{n}")
            os.makedirs(d, exist_ok=True)
            jax.profiler.start_trace(d)
            jax.block_until_ready(jchain(x, w))
            jax.profiler.stop_trace()
        del w, x


if __name__ == "__main__":
    main()
