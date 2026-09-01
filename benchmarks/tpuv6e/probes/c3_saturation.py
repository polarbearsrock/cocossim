#!/usr/bin/env python3
"""C3: saturation GEMMs -> sustained TFLOPS pins the clock x MXUs x MACs/PE
product empirically (published peak: 918 bf16 TFLOPS). The M in {256,512,1024}
slice at huge K,N shows how row blocks spread across the TWO MXUs.
Consumes: -mxu_macs_per_pe 2 and -f 1.75 jointly; falsifies either if
sustained TFLOPS is inconsistent with 918 x plausible efficiency.

Usage: c3_saturation.py [--dry-run] [--out c3.csv]
"""
import argparse

SQUARE = [2048, 4096, 8192]
M_SLICE = [(m, 8192, 8192) for m in (256, 512, 1024)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="c3_saturation.csv")
    args = ap.parse_args()
    shapes = [(s, s, s) for s in SQUARE] + M_SLICE
    if args.dry_run:
        print(f"C3: {len(shapes)} shapes: {shapes}")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done
    for (m, k, n) in shapes:
        if already_done(args.out, {"M": m, "K": k, "N": n}):
            continue
        a = jnp.ones((m, k), dtype=jnp.bfloat16)
        b = jnp.ones((k, n), dtype=jnp.bfloat16)
        f = jax.jit(lambda a=a, b=b: a @ b)
        r = time_op(f)
        tf = 2 * m * k * n / r["median_s"] / 1e12
        csv_append(args.out, {"M": m, "K": k, "N": n, **r, "tflops": tf,
                              "pct_of_918": 100 * tf / 918})
        print(f"{m}x{k}x{n}: {tf:6.1f} TF/s ({100*tf/918:4.1f}% of 918)", flush=True)


if __name__ == "__main__":
    main()
