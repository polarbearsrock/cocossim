#!/usr/bin/env python3
"""Minimal TPU v6e LLO feasibility experiment.

Captures three representative compiled programs, one XProf trace each:

* ``gemm``: dependent bf16 M=256, K=N=4096 matmuls (MXU underfill regime)
* ``softmax``: dependent f32-accumulating bf16 softmaxes (vector/fusion)
* ``attention``: dependent Pallas causal flash-attention custom calls

The caller must set the LLO flags before Python imports JAX, for example:

  LIBTPU_INIT_ARGS="--xla_xprof_enable_custom_call_tracing=true \
    --xla_xprof_register_llo_debug_info=true" \
    python llo_three_kernel.py --out-dir /path/to/results

Each workload is compiled and warmed before tracing.  The optimized HLO text,
environment metadata, host-observed timing distribution, and raw XProf profile
are retained.  Timings are only a capture sanity check; XProf kernel/device
times are the experiment's primary measurements.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


CASES = ("gemm", "softmax", "attention")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def build_gemm(jax, jnp):
    """A live dependent chain of small-row GEMMs with no source epilogue."""
    m, k, n, chain = 256, 4096, 4096, 64
    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    # The value 1/K keeps the dependent chain finite and approximately stable.
    w = jnp.full((k, n), 1.0 / k, dtype=jnp.bfloat16)

    @jax.jit
    def run(a, b):
        def step(carry, _):
            return carry @ b, None

        out, _ = jax.lax.scan(step, a, None, length=chain)
        return out

    return run, (x, w), {
        "shape": {"M": m, "K": k, "N": n},
        "dtype": "bfloat16",
        "chain": chain,
        "useful_flops_per_step": 2 * m * k * n,
    }


def build_softmax(jax, jnp):
    """A live dependent chain expected to compile as a vector fusion."""
    rows, width, chain = 256, 2048, 256
    x = jnp.linspace(-2.0, 2.0, rows * width, dtype=jnp.float32)
    x = x.reshape(rows, width).astype(jnp.bfloat16)

    @jax.jit
    def run(a):
        def step(carry, _):
            out = jax.nn.softmax(carry.astype(jnp.float32), axis=-1)
            return out.astype(jnp.bfloat16), None

        out, _ = jax.lax.scan(step, a, None, length=chain)
        return out

    return run, (x,), {
        "shape": {"rows": rows, "width": width},
        "dtype": "bfloat16_in_out_f32_accumulate",
        "chain": chain,
        "elements_per_step": rows * width,
    }


def build_attention(jax, jnp):
    """Four unrolled dependent Pallas flash-attention custom calls."""
    from jax.experimental.pallas.ops.tpu import flash_attention as fa

    batch, heads, seq, head_dim, chain = 1, 32, 512, 128, 4
    shape = (batch, heads, seq, head_dim)
    q = jnp.full(shape, 0.01, dtype=jnp.bfloat16)
    k = jnp.full(shape, 0.02, dtype=jnp.bfloat16)
    v = jnp.linspace(-0.1, 0.1, batch * heads * seq * head_dim,
                     dtype=jnp.float32).reshape(shape).astype(jnp.bfloat16)
    blocks = fa.BlockSizes(
        block_q=seq,
        block_k_major=seq,
        block_k=seq,
        block_b=1,
    )

    @jax.jit
    def run(q0, kk, vv):
        # Unroll the short outer chain so its loop is not confused with any
        # rolled-loop limitations in LLO reporting.  Each output is the next
        # call's full query, which keeps all four custom calls live.
        out = q0
        for _ in range(chain):
            out = fa.flash_attention(
                out,
                kk,
                vv,
                causal=True,
                sm_scale=head_dim ** -0.5,
                block_sizes=blocks,
            )
        return out

    return run, (q, k, v), {
        "shape": {
            "batch": batch,
            "heads": heads,
            "sequence": seq,
            "head_dim": head_dim,
        },
        "dtype": "bfloat16",
        "chain": chain,
        "implementation": "jax.experimental.pallas.ops.tpu.flash_attention",
        "block_q": seq,
        "block_k_major": seq,
        "block_k": seq,
    }


BUILDERS = {
    "gemm": build_gemm,
    "softmax": build_softmax,
    "attention": build_attention,
}


def capture_case(name, builder, root: Path, reps: int, jax, jnp):
    case_dir = root / name
    trace_dir = case_dir / "trace"
    case_dir.mkdir(parents=True, exist_ok=True)

    fn, inputs, description = builder(jax, jnp)
    lowered = fn.lower(*inputs)
    compiled = lowered.compile()
    (case_dir / "optimized_hlo.txt").write_text(compiled.as_text())

    # Compile/warm outside the profile, then obtain a small timing distribution
    # to catch failures, asynchronous timing mistakes, or gross perturbation.
    jax.block_until_ready(compiled(*inputs))
    elapsed = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*inputs))
        elapsed.append(time.perf_counter() - start)

    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    jax.block_until_ready(compiled(*inputs))
    jax.profiler.stop_trace()

    xplanes = list(trace_dir.glob("plugins/profile/*/*.xplane.pb"))
    record = {
        "case": name,
        **description,
        "host_timing_us": {
            "median": statistics.median(elapsed) * 1e6,
            "p10": percentile(elapsed, 0.10) * 1e6,
            "p90": percentile(elapsed, 0.90) * 1e6,
            "repetitions": reps,
        },
        "trace_dir": str(trace_dir),
        "xplane_files": [str(p) for p in xplanes],
        "trace_ok": bool(xplanes),
    }
    (case_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)
    if not xplanes:
        raise RuntimeError(f"{name}: no xplane profile produced under {trace_dir}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases", default=",".join(CASES),
                        help="comma-separated subset of gemm,softmax,attention")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [x.strip() for x in args.cases.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.dry_run:
        print("cases:", ", ".join(selected))
        print("gemm: M=256 K=N=4096 bf16, dependent chain=64")
        print("softmax: rows=256 width=2048 bf16/f32, dependent chain=256")
        print("attention: B=1 H=32 S=512 D=128 bf16 Pallas, unrolled chain=4")
        return

    # Import only after argument validation so --dry-run works off-TPU.  The
    # launching shell must set LIBTPU_INIT_ARGS before this point.
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    if not devices or devices[0].platform != "tpu":
        raise RuntimeError(f"TPU device required, found {devices}")

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "libtpu": package_version("libtpu"),
        "xprof": package_version("xprof-nightly"),
        "python": os.sys.version,
        "devices": [str(d) for d in devices],
        "device_kind": devices[0].device_kind,
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "selected_cases": selected,
    }
    (root / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)

    records = [capture_case(name, BUILDERS[name], root, args.reps, jax, jnp)
               for name in selected]
    (root / "summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
