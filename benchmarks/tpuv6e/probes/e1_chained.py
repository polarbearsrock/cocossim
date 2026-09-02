#!/usr/bin/env python3
"""E1: elementwise streaming, chained (device-side) edition of B1. Each timed
call runs CHAIN elementwise ops inside one jit (lax.scan) with a data
dependence through the carry, so the host-dispatch floor is amortized and
per-step time reflects HBM streaming plus any device-side per-kernel cost.

Ops: add (two inputs, one output: 3 arrays moved) and exp (one input, one
output: 2 arrays moved), bf16, n in 2^15 .. 2^28 elements. Per point: per-step
us and GB/s moved. With --trace DIR one XProf trace per point.

Usage: e1_chained.py [--dry-run] [--out e1_chained.csv] [--trace DIR]
"""
import argparse
import os

CHAIN = 16
SIZES = [1 << p for p in range(15, 29)]
OPS = ("add", "exp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="e1_chained.csv")
    ap.add_argument("--trace", default=None)
    args = ap.parse_args()
    if args.dry_run:
        for op in OPS:
            for n in SIZES:
                print(f"{op:4s} n={n:10d} ({n * 2 / 1e6:8.1f} MB per array)")
        print(f"{len(OPS) * len(SIZES)} points, chain {CHAIN}")
        return

    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done

    @jax.jit
    def add_chain(a, b):
        def step(carry, _):
            y = (a + carry.astype(a.dtype)) + b
            return y[0].astype(jnp.float32), None
        acc, _ = jax.lax.scan(step, jnp.float32(0), None, length=CHAIN)
        return acc

    @jax.jit
    def exp_chain(a):
        def step(carry, _):
            y = jnp.exp(a + carry.astype(a.dtype))
            return y[0].astype(jnp.float32), None
        acc, _ = jax.lax.scan(step, jnp.float32(0), None, length=CHAIN)
        return acc

    for op in OPS:
        for n in SIZES:
            if already_done(args.out, {"op": op, "n": n}):
                continue
            a = jnp.full((n,), 0.5, dtype=jnp.bfloat16)
            if op == "add":
                b = jnp.full((n,), 0.25, dtype=jnp.bfloat16)
                fn = lambda: add_chain(a, b)
                moved = 3 * n * 2
            else:
                fn = lambda: exp_chain(a)
                moved = 2 * n * 2
            r = time_op(fn)
            per_step = r["median_s"] / CHAIN
            row = {"op": op, "n": n, "bytes_moved": moved, "chain": CHAIN, **r,
                   "per_step_us": per_step * 1e6, "gbs": moved / per_step / 1e9}
            csv_append(args.out, row)
            print(f"{op:4s} n={n:10d}  {row['per_step_us']:9.2f} us/step  {row['gbs']:7.0f} GB/s", flush=True)
            if args.trace:
                d = os.path.join(args.trace, f"E1_{op}_{n}")
                os.makedirs(d, exist_ok=True)
                jax.profiler.start_trace(d)
                jax.block_until_ready(fn())
                jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
