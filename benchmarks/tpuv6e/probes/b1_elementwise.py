#!/usr/bin/env python3
"""B1: elementwise/VPU sweep -> two fitted parameters at once. Large sizes
give the vector path's HBM roofline (feeds -vu_sz and achievable bandwidth);
small sizes expose the per-kernel launch overhead as the latency intercept
(the first real number behind -job_overhead, currently 0).
Consumes: -job_overhead intercept, -vu_sz roofline, VPU phase-cost sanity.

Usage: b1_elementwise.py [--dry-run] [--out b1.csv]
"""
import argparse

# Elements per array, bf16 (64 KB .. 512 MB working sets).
SIZES = [2**15, 2**17, 2**19, 2**21, 2**23, 2**25, 2**26, 2**27, 2**28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="b1_elementwise.csv")
    args = ap.parse_args()
    if args.dry_run:
        print(f"B1: 5 ops x {len(SIZES)} sizes "
              f"({SIZES[0]*2//1024} KB .. {SIZES[-1]*2//2**20} MB)")
        return
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done
    for n in SIZES:
        x = jnp.ones((n,), dtype=jnp.bfloat16)
        y = jnp.ones((n,), dtype=jnp.bfloat16)
        j2 = {"add": jax.jit(lambda x, y: x + y), "mul": jax.jit(lambda x, y: x * y)}
        j1 = {"exp": jax.jit(jnp.exp), "rsqrt": jax.jit(jax.lax.rsqrt),
              "silu": jax.jit(jax.nn.silu)}
        ops = {name: ((lambda f=f: f(x, y)), 3) for name, f in j2.items()}
        ops.update({name: ((lambda f=f: f(x)), 2) for name, f in j1.items()})
        for name, (f, streams) in ops.items():
            if already_done(args.out, {"op": name, "n": n}):
                continue
            r = time_op(f)
            gbs = streams * n * 2 / r["median_s"] / 1e9
            csv_append(args.out, {"op": name, "n": n, "bytes": streams * n * 2,
                                  **r, "gbs": gbs})
            print(f"{name:6s} n=2^{n.bit_length()-1:2d}  "
                  f"{r['median_s']*1e6:9.1f} us  {gbs:6.0f} GB/s", flush=True)


if __name__ == "__main__":
    main()
