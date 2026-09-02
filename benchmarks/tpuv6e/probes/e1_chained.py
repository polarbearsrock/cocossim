#!/usr/bin/env python3
"""E1: elementwise streaming, chained (device-side) edition of B1. Each timed
call runs CHAIN elementwise ops inside one jit (lax.scan) whose CARRY IS THE
FULL OUTPUT ARRAY, so every step must read its input(s) and write its whole
result (nothing can be sliced away: the next step consumes the entire
array). The host-dispatch floor is amortized; per-step time reflects HBM
streaming plus any device-side per-kernel cost.

Ops: add (carry + b: two inputs, one output = 3 arrays moved) and exp
(exp(-|carry|): one input, one output = 2 arrays moved; values stay in
(0, 1] so nothing overflows), bf16, n in 2^15 .. 2^28 elements. Per point:
per-step us and GB/s moved. With --trace DIR one XProf trace per point.

Usage: e1_chained.py [--dry-run] [--out e1_chained.csv] [--trace DIR]
"""
import argparse
import os

CHAIN = 16
SIZES = [1 << p for p in range(15, 29)]
OPS = ("add", "exp")
PLATE_GBS = 1638.0


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
    def add_chain(a0, b):
        def step(a, _):
            return a + b, None
        af, _ = jax.lax.scan(step, a0, None, length=CHAIN)
        return jnp.sum(af.astype(jnp.float32))

    @jax.jit
    def exp_chain(a0):
        def step(a, _):
            return jnp.exp(-jnp.abs(a)), None
        af, _ = jax.lax.scan(step, a0, None, length=CHAIN)
        return jnp.sum(af.astype(jnp.float32))

    for op in OPS:
        for n in SIZES:
            if already_done(args.out, {"op": op, "n": n}):
                continue
            a = jnp.full((n,), 0.5, dtype=jnp.bfloat16)
            if op == "add":
                b = jnp.full((n,), 0.001, dtype=jnp.bfloat16)
                fn = lambda: add_chain(a, b)
                moved = 3 * n * 2
            else:
                fn = lambda: exp_chain(a)
                moved = 2 * n * 2
            r = time_op(fn)
            per_step = r["median_s"] / CHAIN
            gbs = moved / per_step / 1e9
            # Above the HBM plate the scan carry stayed VMEM-resident across
            # iterations (arrays under ~128 MiB never touch HBM): that is a
            # VMEM-bandwidth reading, not an HBM one -- keep it, flagged, so
            # score_matrix can use the HBM rows and report VMEM separately.
            # (H1 session: 2.2-4.5 TB/s for 25-200 MB moved; 1.1 TB/s above.)
            vmem = gbs > PLATE_GBS * 1.05
            row = {"op": op, "n": n, "bytes_moved": moved, "chain": CHAIN, **r,
                   "per_step_us": per_step * 1e6, "gbs": gbs, "vmem_resident": int(vmem)}
            csv_append(args.out, row)
            print(f"{op:4s} n={n:10d}  {row['per_step_us']:9.2f} us/step  {gbs:7.0f} GB/s"
                  f"{'  [VMEM-resident carry]' if vmem else ''}", flush=True)
            if args.trace:
                d = os.path.join(args.trace, f"E1_{op}_{n}")
                os.makedirs(d, exist_ok=True)
                jax.profiler.start_trace(d)
                jax.block_until_ready(fn())
                jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
