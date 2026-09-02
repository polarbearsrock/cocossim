#!/usr/bin/env python3
"""E1: elementwise streaming, chained (device-side) by the SLOPE method
(common.time_chain_slope: per-step = (t_2C - t_C) / C, so the ~113 us
per-call launch cost cancels; session H1's chain-16 numbers carried 7 us of
it per step -- spec 6.2). The scan carry IS the full output array, so every
step must read its input(s) and write its whole result.

Ops (bf16, n in 2^15 .. 2^28 elements):
  add        carry + b            2 reads, 1 write  (3 arrays moved)
  exp        exp(-|carry|)        1 read,  1 write  (2 arrays; values stay in (0,1])
  read_only  acc' = sum(exp(a + acc)), acc scalar: reads a, writes nothing --
             the nonlinearity keeps XLA from hoisting the loop-invariant
             array out of the scan (E1b, spec 5.2 item 4: isolates the read
             path so the write cost is add/exp minus read_only)
Rows whose working set fits VMEM (~128 MiB) are VMEM-resident carries on
silicon and are flagged vmem_resident=1 (a VMEM-bandwidth reading, not HBM).

Usage: e1_chained.py [--dry-run] [--out e1_chained.csv] [--trace DIR]
"""
import argparse
import functools
import os

CHAIN = 16
SIZES = [1 << p for p in range(15, 29)]
OPS = ("add", "exp", "read_only")
PLATE_GBS = 1638.0
VMEM_BYTES = 150e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="e1_chained.csv")
    ap.add_argument("--trace", default=None)
    args = ap.parse_args()
    if args.dry_run:
        for op in OPS:
            for n in SIZES:
                print(f"{op:9s} n={n:10d} ({n * 2 / 1e6:8.1f} MB per array)")
        print(f"{len(OPS) * len(SIZES)} points, base chain {CHAIN} (grown until a call >= 1 ms), slope method")
        return

    import jax
    import jax.numpy as jnp
    from common import time_chain_slope, csv_append, already_done

    def add_chain(C, a0, b):
        def step(a, _):
            return a + b, None
        af, _ = jax.lax.scan(step, a0, None, length=C)
        return jnp.sum(af.astype(jnp.float32))

    def exp_chain(C, a0):
        def step(a, _):
            return jnp.exp(-jnp.abs(a)), None
        af, _ = jax.lax.scan(step, a0, None, length=C)
        return jnp.sum(af.astype(jnp.float32))

    def read_chain(C, a):
        def step(acc, _):
            return jnp.sum(jnp.exp(-(a + acc.astype(a.dtype))), dtype=jnp.float32) * 1e-9, None
        acc, _ = jax.lax.scan(step, jnp.float32(0), None, length=C)
        return acc

    for op in OPS:
        for n in SIZES:
            if already_done(args.out, {"op": op, "n": n}):
                continue
            a = jnp.full((n,), 0.5, dtype=jnp.bfloat16)
            b = jnp.full((n,), 0.001, dtype=jnp.bfloat16) if op == "add" else None
            jitted = {}

            def make_fn(C):
                if C not in jitted:
                    if op == "add":
                        jitted[C] = jax.jit(functools.partial(add_chain, C))
                    elif op == "exp":
                        jitted[C] = jax.jit(functools.partial(exp_chain, C))
                    else:
                        jitted[C] = jax.jit(functools.partial(read_chain, C))
                f = jitted[C]
                if op == "add":
                    return lambda: f(a, b)
                return lambda: f(a)

            moved = {"add": 3, "exp": 2, "read_only": 1}[op] * n * 2
            r = time_chain_slope(make_fn, CHAIN)
            per_step = r["per_step_s"]
            gbs = moved / per_step / 1e9 if per_step > 0 else float("inf")
            vmem = moved < VMEM_BYTES or gbs > PLATE_GBS * 1.05
            row = {"op": op, "n": n, "bytes_moved": moved, "chain": r["chain"],
                   "per_step_us": per_step * 1e6, "per_step_p10_us": r["per_step_p10_s"] * 1e6,
                   "per_step_p90_us": r["per_step_p90_s"] * 1e6, "intercept_us": r["intercept_s"] * 1e6,
                   "t_c_us": r["t_c_s"] * 1e6, "t_2c_us": r["t_2c_s"] * 1e6, "reps": r["reps"],
                   "gbs": gbs, "vmem_resident": int(vmem)}
            csv_append(args.out, row)
            print(f"{op:9s} n={n:10d}  {row['per_step_us']:9.2f} us/step (chain {r['chain']}, intercept "
                  f"{row['intercept_us']:6.1f} us)  {gbs:7.0f} GB/s{'  [VMEM-resident]' if vmem else ''}", flush=True)
            if args.trace:
                d = os.path.join(args.trace, f"E1_{op}_{n}")
                os.makedirs(d, exist_ok=True)
                jax.profiler.start_trace(d)
                jax.block_until_ready(make_fn(2 * r["chain"])())
                jax.profiler.stop_trace()
            del jitted


if __name__ == "__main__":
    main()
