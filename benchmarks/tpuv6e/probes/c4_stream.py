#!/usr/bin/env python3
"""C4: streaming bandwidth -> achievable HBM GB/s (plate: 1638). Copy-scale
(read x, write y: 2 bytes moved per element-byte) and triad (read x,y, write
z: 3x). Consumes: freezes HBM2e_v6e.ini's achieved-bandwidth target; the
simulator currently achieves 80-86% of plate on GEMM patterns.

Usage: c4_stream.py [--dry-run] [--out c4.csv]
"""
import argparse

SIZES_MB = [64, 256, 1024]  # working set per array, bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="c4_stream.csv")
    args = ap.parse_args()
    if args.dry_run:
        print(f"C4: copy-scale + triad at {SIZES_MB} MB working sets")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append
    for mb in SIZES_MB:
        n = mb * 1024 * 1024 // 2  # bf16 elements
        x = jnp.ones((n,), dtype=jnp.bfloat16)
        y = jnp.ones((n,), dtype=jnp.bfloat16)
        jcopy = jax.jit(lambda x: x * jnp.bfloat16(1.5))
        jtriad = jax.jit(lambda x, y: x * jnp.bfloat16(1.5) + y)
        for kind, f, factor in [
            ("copy_scale", lambda: jcopy(x), 2),
            ("triad", lambda: jtriad(x, y), 3),
        ]:
            r = time_op(f)
            gbs = factor * n * 2 / r["median_s"] / 1e9
            csv_append(args.out, {"kind": kind, "mb": mb, **r, "gbs": gbs,
                                  "pct_of_1638": 100 * gbs / 1638})
            print(f"{kind} {mb}MB: {gbs:6.0f} GB/s ({100*gbs/1638:4.1f}%)", flush=True)


if __name__ == "__main__":
    main()
