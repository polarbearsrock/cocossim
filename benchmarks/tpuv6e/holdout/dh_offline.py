#!/usr/bin/env python3
"""Tier-2 holdout: fixed-shape whole-model points via vLLM OFFLINE mode - the
controllable tier that maps one-to-one onto simulator Transformer runs
(fidelity spec 3.2). Shapes are pinned: exact prompt token counts
(TokensPrompt), ignore_eos, fixed output lengths, prefix caching OFF.

Per point: a warmup generate (compile), the TIMED generate (prefill: 1 output
token; decode: DECODE_TOKENS steps), and -- with --trace-dir -- a SHORT
traced generate (prefill: the same single forward; decode: TRACE_STEPS
steps) so the XProf trace stays complete and per-step device time comes
from the model program's occurrences (census v2). Anchor points repeat
--repeat-anchors times with a rep index for noise bounds.

Session-1/3 gotchas encoded here:
- VLLM_ENABLE_V1_MULTIPROCESSING=0: the v1 engine otherwise owns the TPU in a
  subprocess and jax.profiler cannot attach ("TPU is already in use").
- enable_prefix_caching=False: identical warmup/timed prompts otherwise HIT
  the prefix cache and the timed prefill is nearly free (session 3 finding:
  the stored session-3 prefill traces were cache hits, 256-token programs).
- a discard point absorbs one-time init (~30 ms observed).
- a shutdown AttributeError after stop_trace is cosmetic.

Usage: dh_offline.py [--model Qwen/Qwen3-8B] [--grid qwen|mistral] [--points ...]
                     [--trace-dir DIR] [--out dh.csv] [--repeat-anchors 3] [--dry-run]
Point syntax: mode:seqlen:batch  (prefill times seq tokens in, 1 out;
decode times steady-state tokens out at fixed context).
"""
import argparse
import json
import os
import time

DECODE_TOKENS = 64
TRACE_STEPS = 8

# KV budget on a 32 GB v6e-1 with ~16 GB of bf16 weights: ~147 KB/token for
# Qwen3-8B (36 layers x 8 kv heads x 128 x 2 x 2 B), so batch x context must
# stay under ~80k tokens; 8192x32 and 2048x64 do not fit and are replaced.
GRIDS = {
    "qwen": {
        "points": [("prefill", 256, 1), ("prefill", 512, 1), ("prefill", 1024, 1), ("prefill", 2048, 1),
                   ("prefill", 4096, 1), ("prefill", 512, 4), ("prefill", 512, 8), ("prefill", 2048, 4),
                   ("prefill", 2048, 8),
                   ("decode", 512, 1), ("decode", 512, 8), ("decode", 512, 32), ("decode", 2048, 8),
                   ("decode", 2048, 32), ("decode", 4096, 16), ("decode", 8192, 8)],
        "anchors": [("prefill", 512, 1), ("prefill", 2048, 1), ("decode", 512, 8), ("decode", 2048, 32)],
    },
    "mistral": {
        "points": [("prefill", 512, 1), ("prefill", 2048, 1), ("prefill", 512, 8), ("prefill", 2048, 8),
                   ("decode", 512, 8), ("decode", 2048, 32), ("decode", 8192, 8)],
        "anchors": [("decode", 2048, 32)],
    },
}


def parse_points(s):
    out = []
    for p in s.split(","):
        mode, seq, batch = p.split(":")
        out.append((mode, int(seq), int(batch)))
    return out


def done_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    import csv
    with open(path) as f:
        for r in csv.DictReader(f):
            keys.add((r["mode"], int(r["seq"]), int(r["batch"]), int(r.get("rep", 0))))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--grid", default="qwen", choices=sorted(GRIDS))
    ap.add_argument("--points", default=None)
    ap.add_argument("--repeat-anchors", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trace-dir", default=None,
                    help="capture one xplane trace per point under DIR/<mode>_<seq>_<batch>[_rN]")
    ap.add_argument("--out", default="dh_offline.csv")
    args = ap.parse_args()
    grid = GRIDS[args.grid]
    base = parse_points(args.points) if args.points else grid["points"]
    plan = [(m, s, b, 0) for (m, s, b) in base]
    if not args.points:
        for rep in range(1, args.repeat_anchors):
            plan += [(m, s, b, rep) for (m, s, b) in grid["anchors"]]
    if args.dry_run:
        for (m, s, b, rep) in plan:
            print(f"{m:8s} seq/ctx={s:5d} batch={b:3d} rep={rep}")
        print(f"{len(plan)} runs, model {args.model}")
        return

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf"))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from common_holdout import csv_append

    max_len = max(s for (_, s, _, _) in plan) + DECODE_TOKENS + 8
    llm = LLM(model=args.model, max_model_len=max_len, enable_prefix_caching=False)
    tok_id = 872  # any mid-vocab id; timing is value-independent
    done = done_keys(args.out)

    def gen(prompts, n_out):
        sp = SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        return time.perf_counter() - t0

    # discard point: absorbs one-time init
    gen([TokensPrompt(prompt_token_ids=[tok_id] * 128)], 1)
    gen([TokensPrompt(prompt_token_ids=[tok_id] * 128)], 1)

    for (mode, seq, batch, rep) in plan:
        if (mode, seq, batch, rep) in done:
            continue
        n_out = 1 if mode == "prefill" else DECODE_TOKENS
        prompts = [TokensPrompt(prompt_token_ids=[tok_id] * seq)] * batch
        gen(prompts, n_out)                      # warmup / compile for this shape
        dt = gen(prompts, n_out)                 # timed
        trace = ""
        if args.trace_dir and rep == 0:
            import jax
            trace = os.path.join(args.trace_dir, f"{mode}_{seq}_{batch}")
            steps = 1 if mode == "prefill" else TRACE_STEPS
            gen(prompts, steps)                  # warm the short shape too
            jax.profiler.start_trace(trace)
            gen(prompts, steps)
            jax.profiler.stop_trace()
        row = {"mode": mode, "seq": seq, "batch": batch, "rep": rep, "n_out": n_out,
               "wall_s": dt, "model": args.model, "trace": trace,
               "trace_steps": (1 if mode == "prefill" else TRACE_STEPS) if trace else 0,
               "note": "wall includes ctx prefill + n_out steps" if mode == "decode" else ""}
        csv_append(args.out, row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
