#!/usr/bin/env python3
"""G-sweep: tier-1 GEMM cells G1/G2/G3 of the fidelity-benchmark spec
(docs/superpowers/specs/2026-09-01-tpuv6e-fidelity-benchmark-design.md 3.1),
device-side only, by the SLOPE method (common.time_chain_slope): every timed
call runs C GEMMs inside one jit (lax.scan), the chain is timed at C and 2C,
and per-step time is (t_2C - t_C) / C, so the per-call launch + completion
cost (~113 us, spec 2) cancels instead of leaking floor/C into every point
(session H1's chain-16 numbers carried ~7 us/step of it; spec 6.2).

Shapes are NOT square in general (G3 uses the real Qwen3-8B / Mistral-7B
projection shapes), so the chain cannot feed y back as x. The carry is x
itself: each step adds the row-sums of y = x @ w (over ALL N columns) back
into every row of x, and the jit returns the full-array sum of the final x.
Every element of every y is live and every row of the next x depends on the
previous step -- nothing can be sliced or hoisted. (A first version returned
y[0,0]; XLA reduced the whole GEMM to one K-length dot product and reported
100 PFLOP/s. Never reduce to a slice.)

Per point: per-step us (slope), the measured per-call intercept, TF/s,
weight+activation bytes per step, effective GB/s. With --trace DIR one
XProf trace per point (the 2C call) for per-op MXU / HBM utilization.

Usage: g_sweep.py [--cells G1,G2,G3] [--dry-run] [--out g_sweep.csv] [--trace DIR]
"""
import argparse
import functools
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
        print(f"{len(points)} points, base chain {CHAIN} (grown until a call >= 1 ms), slope method")
        return

    import jax
    import jax.numpy as jnp
    from common import time_chain_slope, csv_append, already_done

    def chain_fn(C, x0, w):
        def step(x, _):
            y = x @ w                                            # M x N, bf16
            r = jnp.sum(y, axis=1, dtype=jnp.float32)            # every column of every row
            x_next = x + (r * 1e-3).astype(x.dtype)[:, None]     # every row of next x depends on y
            return x_next, None
        xf, _ = jax.lax.scan(step, x0, None, length=C)
        return jnp.sum(xf.astype(jnp.float32))

    for (c, m, k, n) in points:
        if already_done(args.out, {"cell": c, "M": m, "K": k, "N": n}):
            continue
        w = jnp.full((k, n), 1e-3, dtype=jnp.bfloat16)
        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        jitted = {}

        def make_fn(C):
            if C not in jitted:
                jitted[C] = jax.jit(functools.partial(chain_fn, C))
            f = jitted[C]
            return lambda: f(x, w)

        r = time_chain_slope(make_fn, CHAIN)
        per_step = r["per_step_s"]
        flops = 2.0 * m * k * n
        wbytes = k * n * 2
        abytes = (m * k + m * n) * 2
        tflops = flops / per_step / 1e12 if per_step > 0 else float("inf")
        row = {"cell": c, "label": label_of(c, k, n), "M": m, "K": k, "N": n, "chain": r["chain"],
               "per_step_us": per_step * 1e6, "per_step_p10_us": r["per_step_p10_s"] * 1e6,
               "per_step_p90_us": r["per_step_p90_s"] * 1e6, "intercept_us": r["intercept_s"] * 1e6,
               "t_c_us": r["t_c_s"] * 1e6, "t_2c_us": r["t_2c_s"] * 1e6, "reps": r["reps"],
               "tflops": tflops, "mfu": tflops / PEAK_TFLOPS,
               "weight_mb": wbytes / 1e6, "act_mb": abytes / 1e6,
               "eff_gbs": (wbytes + abytes) / per_step / 1e9 if per_step > 0 else float("inf")}
        if tflops > PEAK_TFLOPS * 1.05 or per_step <= 0:
            print(f"SANITY FAIL: {tflops:.0f} TF/s (slope {per_step*1e6:.2f} us) -- XLA elided work; row NOT written", flush=True)
            continue
        csv_append(args.out, row)
        print(f"{c} M={m:5d} K={k:6d} N={n:6d}  {row['per_step_us']:9.2f} us/step  (chain {r['chain']}, "
              f"intercept {row['intercept_us']:6.1f} us)  {tflops:7.1f} TF/s  {row['eff_gbs']:7.0f} GB/s", flush=True)
        if args.trace:
            d = os.path.join(args.trace, f"{c}_{m}x{k}x{n}")
            os.makedirs(d, exist_ok=True)
            jax.profiler.start_trace(d)
            jax.block_until_ready(make_fn(2 * r["chain"])())
            jax.profiler.stop_trace()
        del w, x, jitted


if __name__ == "__main__":
    main()
