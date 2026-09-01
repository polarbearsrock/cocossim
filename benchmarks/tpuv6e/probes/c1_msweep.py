#!/usr/bin/env python3
"""C1: fine M-sweep at fixed large K, N -> latency stair-steps reveal the
effective systolic-array ROW count (hypothesis: 256) and padding behavior.
Consumes: -sa_sz confirmation; the model's true-M under-fill honesty.

Usage: c1_msweep.py [--dry-run] [--quick] [--out c1.csv]
"""
import argparse

K = N = 4096
# Dense where quantization steps could hide (1..64 and around 128/256/512
# multiples), coarser elsewhere: ~140 points.
M_POINTS = (
    list(range(1, 65))
    + list(range(66, 257, 2))
    + list(range(260, 516, 4))
    + list(range(520, 769, 8))
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="every 8th point")
    ap.add_argument("--out", default="c1_msweep.csv")
    args = ap.parse_args()
    points = M_POINTS[:: 8 if args.quick else 1]
    if args.dry_run:
        print(f"C1: {len(points)} points, K=N={K}, M in [{points[0]}..{points[-1]}]")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done
    for m in points:
        if already_done(args.out, {"M": m, "K": K, "N": N}):
            continue
        a = jnp.ones((m, K), dtype=jnp.bfloat16)
        b = jnp.ones((K, N), dtype=jnp.bfloat16)
        jf = jax.jit(lambda a, b: a @ b)
        f = lambda: jf(a, b)
        r = time_op(f)
        flops = 2 * m * K * N
        csv_append(args.out, {"M": m, "K": K, "N": N, **r,
                              "tflops": flops / r["median_s"] / 1e12})
        print(f"M={m:4d}  {r['median_s']*1e6:9.1f} us  "
              f"{flops / r['median_s'] / 1e12:6.1f} TF/s", flush=True)


if __name__ == "__main__":
    main()
