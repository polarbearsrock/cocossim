#!/usr/bin/env python3
"""Kernel census: XProf trace -> phase-class time breakdown comparable to the
simulator's ACCT output. Wraps xprof's own converters (pip install xprof) so
we never parse xplane.pb ourselves; validated against the S0 Qwen3-8B trace.

Phase classes mirror the simulator's units/ops:
  gemm       -> SA GEMM jobs           (dot/matmul/einsum/conv)
  attention  -> SA score/AV + KV reads (ragged_paged_attention et al.)
  norm       -> VPU reduce phases      (rms/layer norm, rsqrt-reduce fusions)
  elementwise-> VPU broadcast phases   (add/mul/silu/gelu/exp residuals...)
  data       -> layout/copy/transpose  (unmodeled in the simulator - report it)
  other      -> everything else (report loudly; big 'other' = census gap)

Usage: kernel_census.py TRACE_DIR [--csv out.csv] [--top 15]
TRACE_DIR is the directory passed to jax.profiler.start_trace (it contains
plugins/profile/<ts>/*.xplane.pb).
"""
import argparse
import glob
import json
import re
import sys

BUCKETS = [  # first match wins - attention before gemm (its fusions contain dots)
    ("attention", re.compile(r"ragged|paged|attention|flash", re.I)),
    ("gemm", re.compile(r"dot|matmul|einsum|conv|gemm", re.I)),
    ("norm", re.compile(r"rsqrt|norm|reduce|mean|variance|rms", re.I)),
    ("elementwise", re.compile(
        r"add|mul|sub|div|silu|gelu|exp|tanh|sigmoid|max|min|select|compare", re.I)),
    ("data", re.compile(r"copy|transpose|reshape|broadcast|concat|slice|pad|"
                        r"gather|scatter|convert|bitcast|tuple", re.I)),
]


def bucket_of(op_type: str, op_name: str) -> str:
    for name, rx in BUCKETS:
        if rx.search(op_name) or rx.search(op_type):
            return name
    return "other"


def rows_from_table(table_json):
    """Google-visualization table -> list of dicts."""
    cols = [c["id"] for c in table_json["cols"]]
    for r in table_json.get("rows", []):
        yield dict(zip(cols, [c.get("v") for c in r["c"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    xp = sorted(glob.glob(f"{args.trace_dir}/**/*.xplane.pb", recursive=True))
    if not xp:
        sys.exit(f"no *.xplane.pb under {args.trace_dir}")
    from xprof.convert import raw_to_tool_data as r
    data, _ = r.xspace_to_tool_data(list(xp), "framework_op_stats", {})
    payload = json.loads(data if isinstance(data, str) else data.decode())
    table = payload[0] if isinstance(payload, list) else payload

    per_bucket = {}
    per_op = {}
    for row in rows_from_table(table):
        if str(row.get("host_or_device", "")).lower().startswith("host"):
            continue
        t = float(row.get("total_self_time") or 0)  # microseconds
        op_type = str(row.get("type") or "")
        op_name = str(row.get("operation") or "")
        b = "idle" if op_name.strip() == "IDLE" else bucket_of(op_type, op_name)
        per_bucket[b] = per_bucket.get(b, 0.0) + t
        per_op[(b, op_type, op_name)] = per_op.get((b, op_type, op_name), 0.0) + t

    total = sum(per_bucket.values()) or 1.0
    print(f"device self-time total: {total/1e3:.3f} ms   ({len(per_op)} distinct ops)")
    for b, t in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {b:12s} {t/1e3:9.3f} ms  {100*t/total:5.1f}%")
    print(f"\ntop {args.top} ops:")
    for (b, ot, on), t in sorted(per_op.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {t/1e3:9.3f} ms  {100*t/total:5.1f}%  [{b:11s}] {on[:70]}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["bucket", "op_type", "operation", "self_time_us"])
            for (b, ot, on), t in sorted(per_op.items(), key=lambda kv: -kv[1]):
                w.writerow([b, ot, on, f"{t:.1f}"])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
