#!/usr/bin/env python3
"""B2: DEVICE-side per-kernel overhead via async dispatch. Session 2's 113 us
small-size floor was host dispatch latency (timing loop blocked per call);
the simulator's -job_overhead models the device/queue cost per kernel in a
STREAM of kernels. So: dispatch NCALLS tiny ops back-to-back without blocking
(JAX queues them asynchronously), block once at the end - total/NCALLS is the
per-kernel cost as the device experiences it. Sweep sizes to separate the
fixed overhead (intercept) from the bandwidth term (slope).
Consumes: -job_overhead prior (in seconds; convert at the fitted clock).

Usage: b2_devoverhead.py [--dry-run] [--out b2.csv]
"""
import argparse
import time

SIZES = [2**13, 2**15, 2**17, 2**19, 2**21]
NCALLS = 256
REPS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="b2_devoverhead.csv")
    args = ap.parse_args()
    if args.dry_run:
        print(f"B2: async chains of {NCALLS} adds at sizes {SIZES}")
        return
    import jax
    import jax.numpy as jnp
    from common import csv_append, already_done
    jadd = jax.jit(lambda x, y: x + y)
    for n in SIZES:
        if already_done(args.out, {"n": n}):
            continue
        x = jnp.ones((n,), dtype=jnp.bfloat16)
        y = jnp.ones((n,), dtype=jnp.bfloat16)
        jax.block_until_ready(jadd(x, y))  # compile, discarded
        meds = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = x
            for _ in range(NCALLS):
                out = jadd(out, y)  # async: no block inside the loop
            jax.block_until_ready(out)
            meds.append((time.perf_counter() - t0) / NCALLS)
        meds.sort()
        per_call_us = meds[len(meds) // 2] * 1e6
        csv_append(args.out, {"n": n, "bytes": 3 * n * 2, "ncalls": NCALLS,
                              "per_call_us": per_call_us})
        print(f"n=2^{n.bit_length()-1:2d}  {per_call_us:8.2f} us/kernel (async)",
              flush=True)


if __name__ == "__main__":
    main()
