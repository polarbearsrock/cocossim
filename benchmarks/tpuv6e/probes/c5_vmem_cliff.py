#!/usr/bin/env python3
"""C5: working-set sweep -> the VMEM capacity cliff. A chain of GEMMs reuses
the SAME square weight matrix W sequentially (lax.scan carries the activation,
so XLA cannot batch or CSE the reuse away); per-step time vs W's footprint
shows a cliff where W stops fitting on-chip and must re-stream from HBM.
Consumes: -buf_mb (cliff location) and -vmem_headroom (cliff sharpness) - the
simulator's V18b crossover, measured on silicon.

Usage: c5_vmem_cliff.py [--dry-run] [--out c5.csv]
"""
import argparse

M = 128
CHAIN = 8
# W is NxN bf16: footprint = 2*N^2. Dense sampling around the 128 MB hypothesis.
N_POINTS = [1024, 2048, 2896, 4096, 5120, 5792, 6144, 6656, 7168, 7680, 8192,
            9216, 10240, 11585]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="c5_vmem_cliff.csv")
    args = ap.parse_args()
    if args.dry_run:
        for n in N_POINTS:
            print(f"N={n:6d}  W={2*n*n/1e6:7.1f} MB")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done
    for n in N_POINTS:
        if already_done(args.out, {"N": n}):
            continue
        w = jnp.eye(n, dtype=jnp.bfloat16)  # identity keeps activations bounded
        x0 = jnp.ones((M, n), dtype=jnp.bfloat16)

        @jax.jit
        def chain(x0=x0, w=w):
            def step(x, _):
                return x @ w, None
            x, _ = jax.lax.scan(step, x0, None, length=CHAIN)
            return x

        r = time_op(chain)
        per_step = r["median_s"] / CHAIN
        mb = 2 * n * n / 1e6
        csv_append(args.out, {"N": n, "w_mb": mb, "chain": CHAIN, **r,
                              "per_step_us": per_step * 1e6,
                              "tflops": 2 * M * n * n / per_step / 1e12})
        print(f"W={mb:7.1f} MB  {per_step*1e6:9.1f} us/step", flush=True)


if __name__ == "__main__":
    main()
