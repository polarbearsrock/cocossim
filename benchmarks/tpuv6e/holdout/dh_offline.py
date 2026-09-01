#!/usr/bin/env python3
"""D-holdout: fixed-shape Qwen3-8B points via vLLM OFFLINE mode - the
controllable tier that maps one-to-one onto simulator Transformer runs.
Serving-mode (continuous batching) characterization is a separate script;
these points exist to be exactly comparable, so shapes are pinned:
exact prompt token counts (TokensPrompt), ignore_eos, fixed output lengths.

Session-1 gotchas encoded here:
- VLLM_ENABLE_V1_MULTIPROCESSING=0: the v1 engine otherwise owns the TPU in a
  subprocess and jax.profiler cannot attach ("TPU is already in use").
- warmup generate first: the first call pays compilation; never time it.
- a shutdown AttributeError after stop_trace is cosmetic (tpu-inference
  teardown path); results are already on disk when it prints.

Usage: dh_offline.py [--dry-run] [--points prefill:512:1,decode:2048:32,...]
                     [--trace-dir DIR] [--out dh.csv]
Point syntax: mode:seqlen:batch  (prefill times seq tokens in, 1 out;
decode times steady-state tokens out at fixed context).
"""
import argparse
import json
import os
import time

DH6 = [
    ("prefill", 512, 1), ("prefill", 512, 8),
    ("prefill", 2048, 1), ("prefill", 2048, 8),
    ("decode", 512, 8), ("decode", 2048, 32),
]
DECODE_TOKENS = 64


def parse_points(s):
    out = []
    for p in s.split(","):
        mode, seq, batch = p.split(":")
        out.append((mode, int(seq), int(batch)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--points", default=None)
    ap.add_argument("--trace-dir", default=None,
                    help="capture one xplane trace per point under DIR/<point>")
    ap.add_argument("--out", default="dh_offline.csv")
    args = ap.parse_args()
    points = parse_points(args.points) if args.points else DH6
    if args.dry_run:
        for (m, s, b) in points:
            print(f"{m:8s} seq/ctx={s:5d} batch={b:3d}")
        return

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf"))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from common_holdout import csv_append

    max_len = max(s for (_, s, _) in points) + DECODE_TOKENS + 8
    llm = LLM(model="Qwen/Qwen3-8B", max_model_len=max_len)
    tok_id = 872  # any mid-vocab id; timing is value-independent

    for (mode, seq, batch) in points:
        ctx = seq if mode == "decode" else seq
        n_out = 1 if mode == "prefill" else DECODE_TOKENS
        prompts = [TokensPrompt(prompt_token_ids=[tok_id] * ctx)] * batch
        sp = SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)
        llm.generate(prompts, sp)  # warmup/compile for this shape
        trace = None
        if args.trace_dir:
            import jax
            trace = os.path.join(args.trace_dir, f"{mode}_{seq}_{batch}")
            jax.profiler.start_trace(trace)
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        if trace:
            import jax
            jax.profiler.stop_trace()
        row = {"mode": mode, "seq": seq, "batch": batch, "n_out": n_out,
               "wall_s": dt, "trace": trace or ""}
        if mode == "decode":
            # first generated token rides the prefill; steady tokens = n_out-1
            row["note"] = "wall includes ctx prefill + n_out steps"
        csv_append(args.out, row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
