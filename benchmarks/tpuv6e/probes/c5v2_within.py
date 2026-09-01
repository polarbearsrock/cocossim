#!/usr/bin/env python3
"""C5v2: within-kernel VMEM capacity. Session 2's scan probe showed NO
cross-op residency (weights re-stream every step), so capacity must be probed
INSIDE one GEMM: for fixed weights W (K x N), sweep M and take the slope of
time vs M - the marginal cost per row block. If W stays VMEM-resident across
the kernel's internal M-tiles, the slope excludes W's bytes; once W exceeds
capacity the compiler must re-stream it per M-tile and the slope jumps.
Compare slopes across W footprints from 32 MB to ~254 MB: the knee locates
effective VMEM (-buf_mb) and its sharpness bounds -vmem_headroom.

Usage: c5v2_within.py [--dry-run] [--out c5v2.csv]
"""
import argparse

M_POINTS = [2048, 4096, 8192]
KN = [(4096, 4096), (8192, 4096), (8192, 8192), (11264, 8192), (11264, 11264)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="c5v2_within.csv")
    args = ap.parse_args()
    if args.dry_run:
        for (k, n) in KN:
            print(f"K={k:6d} N={n:6d}  W={2*k*n/1e6:7.1f} MB  x M={M_POINTS}")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done
    jf = jax.jit(lambda a, b: a @ b)
    for (k, n) in KN:
        for m in M_POINTS:
            if already_done(args.out, {"M": m, "K": k, "N": n}):
                continue
            a = jnp.ones((m, k), dtype=jnp.bfloat16)
            b = jnp.ones((k, n), dtype=jnp.bfloat16)
            r = time_op(lambda: jf(a, b))
            tf = 2 * m * k * n / r["median_s"] / 1e12
            csv_append(args.out, {"M": m, "K": k, "N": n, "w_mb": 2 * k * n / 1e6,
                                  **r, "tflops": tf})
            print(f"W={2*k*n/1e6:7.1f} MB M={m:5d}: {r['median_s']*1e6:9.1f} us "
                  f"{tf:6.1f} TF/s", flush=True)


if __name__ == "__main__":
    main()
