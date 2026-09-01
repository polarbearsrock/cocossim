#!/usr/bin/env python3
"""C1v2: array-row quantization, chained edition. Session 2 showed a ~115 us
host-dispatch floor drowns single-call latencies, so each timed call now runs
CHAIN GEMMs inside one jit (lax.scan; C5 established weights re-stream per
step, which is fine - the M-dependent compute term is the signal). K=N=2048
keeps the per-step weight stream small (~6 us) so a 256-row quantization step
would appear as a several-us jump in per-step time at M=257 and M=513,
versus a ~0.03 us smooth slope.
Consumes: -sa_sz row confirmation (256 hypothesis), row-fill behavior.

Usage: c1v2_chained.py [--dry-run] [--out c1v2.csv]
"""
import argparse

K = N = 2048
CHAIN = 32
M_POINTS = list(range(192, 321, 2)) + list(range(448, 577, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="c1v2_chained.csv")
    args = ap.parse_args()
    if args.dry_run:
        print(f"C1v2: {len(M_POINTS)} points, K=N={K}, chain={CHAIN}, "
              f"M in [{M_POINTS[0]}..{M_POINTS[-1]}]")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done

    @jax.jit
    def jchain(x0, w):
        def step(x, _):
            return x @ w, None
        x, _ = jax.lax.scan(step, x0, None, length=CHAIN)
        return x

    w = jnp.eye(K, dtype=jnp.bfloat16)
    for m in M_POINTS:
        if already_done(args.out, {"M": m}):
            continue
        x0 = jnp.ones((m, K), dtype=jnp.bfloat16)
        r = time_op(lambda: jchain(x0, w))
        per_step_us = r["median_s"] / CHAIN * 1e6
        csv_append(args.out, {"M": m, "K": K, "N": N, "chain": CHAIN, **r,
                              "per_step_us": per_step_us})
        print(f"M={m:4d}  {per_step_us:7.2f} us/step", flush=True)


if __name__ == "__main__":
    main()
