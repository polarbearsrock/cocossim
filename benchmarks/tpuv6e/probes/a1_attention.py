#!/usr/bin/env python3
"""A1: the attention kernel in isolation (spec 3.1 cell A1), bf16, head_dim 128,
device-side only: every timed call runs CHAIN kernel invocations inside one
jit (lax.scan) so the ~113 us host-dispatch floor is amortized.

Two modes, two Pallas TPU kernels from the installed JAX:

PREFILL (query length = context, causal):
  kernel = jax.experimental.pallas.ops.tpu.flash_attention.flash_attention
  That kernel is multi-head WITHOUT GQA (num_heads must match on q/k/v), so
  prefill runs MHA with nh = nkv = 32 (recorded in the CSV as nkv=32). The
  simulator counterpart is the Transformer attention sub-DAG run with
  n_kv_heads = 32 at the same dims.
  Chain (default --carry out): carry = q [B,nh,S,hd]; step: q_next = flash(q, k, v).
  k, v are fixed (like weights). The kernel output IS the next step's input,
  so every element of every out is live, and the jit returns the full-array
  sum of the final q -- nothing can be sliced or hoisted (spec 5.2: a step
  that returned one element let XLA reduce a GEMM to a single dot product
  and report 100 PFLOP/s). This carry adds NO XLA fusion between kernel
  calls. The first version (--carry sum, kept as an explicit option) used
  q_next = q + (out.sum(-1, f32) * 1e-3)[..., None], whose XLA fusion
  streams three full [B,nh,S,hd] arrays through HBM per step (read out,
  read q, write q_next) -- 20-90 % of the kernel's own time at the prefill
  cells, all charged to the kernel. Do not use it for attribution; it is
  retained only so the two can be compared on silicon.
  Chain FORM (--chain-form scan|unroll, comma list sweeps; default scan):
    scan   = lax.scan over CHAIN steps (rule 1). The scan body is exactly
             one pallas_call, but XLA's while loop keeps its state in one
             buffer that the body parameter and the body root must share,
             and a custom call cannot write its output over its own operand
             (flash_attention's pallas_call declares no input/output
             aliasing), so copy insertion adds a root copy of out
             [B,nh,S,hd] into the loop-carry buffer: two extra full-array
             passes per step OUTSIDE the kernel (~200 us at (512,32) against
             a >= 328 us kernel; +37 % at (2048,8); +13 % at (8192,1)) that
             the wall-clock per_step_us still contains.
    unroll = a Python for-loop of CHAIN kernel calls inside the SAME single
             jit: out_i is the direct operand of call_{i+1}, there is no
             loop state, hence no carry copy; the host-dispatch
             amortization rule (1) intends is identical (one dispatch per
             timed call). Chain honesty is unchanged: every element of
             every out feeds the next call and the jit returns the
             full-array sum of the last one.
    Run both (--chain-form scan,unroll) and compare per_step_us and the
    trace's glue_us: their difference IS the carry copy. Neither form can
    be verified on CPU JAX (CPU XLA rewrites bf16 dots), so the session
    decides which one the scorer uses; chain_form is a CSV column and part
    of the resumability key and the trace directory name (_unroll suffix).
  FLOPs: the kernel SKIPS kv blocks strictly above the diagonal
  (below_or_on_diag on the (block_q, block_k_major) grid), so the FLOPs it
  actually executes are flops_done = 4*B*nh*hd*block_q*block_k_major *
  n_blocks_run (masked entries inside diagonal blocks are still computed).
  With block_q == block_k_major == b and n = S/b that is the causal
  (n+1)/(2n) fraction of the dense 4*B*S*S*nh*hd. tflops/mfu use
  flops_done; flops_full is also written so the dense figure is recoverable.
  Bytes: two figures are written.
    bytes_mb  = q + k + v + out (each B*nh*S*hd*2): the ALGORITHMIC MINIMUM,
                what the simulator's attention sub-DAG moves. gbs uses it.
    hbm_bytes_mb = q + out + K/V re-fetch traffic: what the KERNEL actually
                moves. Its grid is (B, nh, q_blocks, kv_blocks) with kv
                innermost, and kv_index_map restarts at kv block 0 for every
                q block (skipped causal blocks are remapped to block 0);
                Pallas re-DMAs a block whenever its block index changes
                between consecutive grid steps, so K and V are each fetched
                kv_fetches = 1 + (#index changes) times per (b, h) instead of
                once: 1 / 9 / 135 block fetches for S = 512 / 2048 / 8192 at
                block 512 (kv_fetch_factor = kv_fetches / n_kv_blocks =
                1.0 / 2.25 / 8.44). hbm_gbs, t_plate, exp_us and the plate
                sanity gate use hbm_bytes; a block-size sweep changes it.
                (The single-step path, block == S, fetches K/V once.)

DECODE (query length 1):
  kernel = jax.experimental.pallas.ops.tpu.paged_attention.paged_attention
    (q [B,nh,hd], k_pages / v_pages [nkv, num_pages, page_size, hd],
     lengths i32[B], page_indices i32[B, pages_per_seq],
     pages_per_compute_block=PPCB, megacore_mode=None (v6e: 1 TensorCore),
     inline_seq_dim=True)
  GQA nh = 32, nkv = 8, page_size 16, pages laid out SEQUENTIALLY per
  sequence (page_indices[b] = b*pps + arange(pps); K1 covers shuffled
  layouts), lengths[b] = S.
  Chain (--carry out): carry = q [B,nh,hd]; step: q_next = paged_attention(q, ...);
  return sum(final q). (--carry sum: q_next = q + (out * 1e-3).astype(bf16);
  q is <= 256 KB here so the glue is negligible either way.)
  bytes per step = KV bytes read = B*S*nkv*hd*2*2 (+ q, out); the kernel's
  grid is (cores, B, nkv) with the nh/nkv-head query group as the q block,
  so every page is DMA'd exactly once per step: hbm_bytes == bytes,
  kv_fetch_factor = 1. FLOPs = 4*B*S*nh*hd.
  q dtype (--decode-q-dtype f32|bf16, default f32): with nh/nkv = 4 groups
  (not a multiple of 8) the kernel's Python wrapper reshapes q to
  [B,nh,1,hd] and launches the pallas_call with q in FLOAT32
  (q.astype(q_dtype_for_kernel_launch)), then casts the output back to
  q.dtype. With a bf16 carry that puts two XLA convert_element_type
  kernels in the scan body per step (bf16->f32 [B,nh,1,hd] before the
  pallas_call, f32->bf16 after; verified in the jaxpr), each carrying the
  ~7-9 us fixed launch cost H1 measured -- more than the kernel's whole
  expected time at the six decode cells whose KV is < 128 MiB (1.3-41 us
  at plate). With the carry generated in float32 [B,nh,hd] both converts
  are no-ops and disappear from the scan body (verified in the jaxpr; the
  kernel casts its q block to f32 internally anyway, so the numerics are
  identical). The two reshapes [B,nh,hd] <-> [B,nh,1,hd] remain and may be
  relayout copies on TPU (the kernel forces a <1x128> layout); they show
  up in the trace's glue_us. q/out bytes are counted at the carry dtype
  (4 B with f32); they are < 0.2 % of the KV bytes at every cell. q_dtype
  is a CSV column and part of the resumability key; --decode-q-dtype bf16
  reproduces the converted variant (trace dir suffix _qbf16). Prefill q
  is always bf16 (the flash kernel runs its MXU dots in the q dtype).

Chain honesty (both modes): the scan carry is the live data the next step
depends on, every kernel output element influences it, the jit returns a
full-array reduction of the final carry, and per point the bytes the kernel
must move and the FLOPs it must do are computed independently of the
measurement. A row above 1.05x peak (918 TF/s) or 1.05x plate (1638 GB/s,
on hbm_bytes) is refused (SANITY FAIL, not written to --out; the refused
row goes to <out>.rejected.csv with a reason so the session is not lost)
UNLESS its working set fits VMEM (< 128 MiB), in which case it is written
with vmem_resident=1 -- a legitimate VMEM-bandwidth reading (E1 found
2.2-4.5 TB/s for such carries).

Kernel device time (spec 2: utilization ground truth is XProf): with
--trace DIR one XProf trace per point (a single chained call) is captured
under DIR/<point> (the point name carries the kernel config, e.g.
A1_prefill_S2048_B1_bq256_bk256, A1_decode_S8192_B32_ps16_ppcb8). If the
`xprof` package is importable, the trace is parsed right away
(framework_op_stats: rows of type `pallas_call`, device side) and the row
gets kernel_us = sum(pallas total_self_time) / chain, trace_step_us = all
device self time / chain, glue_us = trace_step_us - kernel_us, and
kernel_gbs / kernel_tflops / kernel_mfu computed from kernel_us. Without
xprof on the VM those columns stay blank and `--annotate` fills them
offline from the same trace directory (writes <out>.kernel.csv; run it
with --out <out>.rejected.csv as well to annotate refused rows).
The trace is captured BEFORE the sanity gate, so a refused row reaches
<out>.rejected.csv with its trace directory and kernel_* columns: the
trace is the arbiter for exactly those rows.

WHICH COLUMNS TO READ. per_step_us and the gbs / tflops / mfu / hbm_gbs
derived from it are WALL-CLOCK figures: median chained-call time / chain,
which contains whatever XLA runs around the kernel each step (the scan
carry copy, the decode reshapes, the final reduce / chain). kernel_us and
kernel_gbs / kernel_tflops / kernel_mfu are the KERNEL's own device time
from the trace and are the primary attribution whenever present; the
`attribution` column says which the row has ("xprof" when kernel_us is
filled, "wall" otherwise) so a blank kernel_us is never silently read as
the kernel's number. The brief's column names are kept for the scorer.

Every kernel call goes through a wrapper that logs the exact call
signature used. --probe-api prints the resolved import paths, signatures
and a shape smoke test (jax.eval_shape through the kernels' own validation)
without executing anything; it runs on CPU JAX. The run mode fails loudly
with the exception text; there is NO silent fallback. --fallback-xla
(documented, off by default) substitutes a plain XLA attention
(jax.nn.dot_product_attention for prefill, an einsum GQA for decode over
the same paged KV gathered contiguously) ONLY when the Pallas kernel raises,
and records kernel=xla_* in the row.

Resumable: points whose key fields (cell, kernel config, carry, chain,
chain_form, q_dtype) are already in --out are skipped.

Usage: a1_attention.py [--mode prefill|decode|both] [--out a1_attention.csv]
         [--trace DIR] [--dry-run] [--probe-api] [--annotate] [--chain 8]
         [--carry out|sum] [--chain-form scan[,unroll]] [--decode-q-dtype f32|bf16]
         [--cells 512x1,2048x8] [--block 512[,256,...]]
         [--ppcb 16[,8,...]] [--page-size 16] [--fallback-xla]
--block / --ppcb / --chain-form accept comma lists (the config is part of
the CSV key), so the block-size choice and the chain form can be swept in
the same (cheap) session: the whole default 15-point session is < 1 s of
device time at peak/plate.
"""
import argparse
import csv
import glob
import inspect
import math
import os
import sys
import traceback

HD = 128
NH = 32
NKV_DECODE = 8
PEAK_TFLOPS = 918.0
PLATE_GBS = 1638.0
VMEM_BYTES = 128 * 2**20

# (S, B)
PREFILL_CELLS = [(512, 1), (512, 8), (512, 32), (2048, 1), (2048, 8), (8192, 1)]
DECODE_CELLS = [(s, b) for s in (512, 2048, 8192) for b in (1, 8, 32)]

KERNEL_FLASH = "pallas_flash_attention"
KERNEL_PAGED = "pallas_paged_attention"
KERNEL_XLA_PREFILL = "xla_dot_product_attention"
KERNEL_XLA_DECODE = "xla_gqa_einsum"
CARRIES = ("out", "sum")
CHAIN_FORMS = ("scan", "unroll")
Q_DTYPES = {"f32": "float32", "bf16": "bfloat16"}   # --decode-q-dtype -> numpy/jnp dtype name
ITEMSIZE = {"float32": 4, "bfloat16": 2}
PREFILL_Q_DTYPE = "bfloat16"

# Columns derived from the XProf trace (blank when no trace / no xprof).
TRACE_COLS = ("kernel_us", "kernel_gbs", "kernel_tflops", "kernel_mfu", "glue_us", "trace_step_us",
              "pallas_rows", "pallas_occurrences")


# ----------------------------------------------------------------------------
# Budgets (pure Python; used by --dry-run, the CSV and the sanity gate)
# ----------------------------------------------------------------------------

def below_or_on_diag(r, r_blk, c, c_blk):
    """Same predicate as flash_attention.below_or_on_diag (kv block c runs for
    q block r iff the block's bottom-left corner is on/below the diagonal)."""
    return ((r + 1) * r_blk - 1) > (c * c_blk)


def prefill_kv_index_sequence(S, block_q, block_k_major):
    """The kv block index flash_attention's kv_index_map yields for one (b, h)
    as the grid walks q blocks (outer) x kv blocks (inner), causal=True:
    the kv index when the block runs, else 0 (the kernel prefetches block 0
    for the next q row). Pallas re-DMAs K and V whenever this index changes."""
    nq = math.ceil(S / block_q)
    nk = S // block_k_major
    return [c if below_or_on_diag(r, block_q, c, block_k_major) else 0
            for r in range(nq) for c in range(nk)]


def prefill_budget(S, B, block_q, block_k_major):
    nq = math.ceil(S / block_q)
    nk = S // block_k_major
    seq = prefill_kv_index_sequence(S, block_q, block_k_major)
    n_run = sum(1 for r in range(nq) for c in range(nk)
                if below_or_on_diag(r, block_q, c, block_k_major))
    kv_fetches = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)   # per (b, h), K and V each
    flops_full = 4.0 * B * S * S * NH * HD
    flops_done = 4.0 * B * NH * HD * block_q * block_k_major * n_run
    arr = B * NH * S * HD * 2
    kv_hbm = 2 * B * NH * kv_fetches * block_k_major * HD * 2
    return {
        "flops_full": flops_full, "flops_done": flops_done,
        "causal_frac": flops_done / flops_full, "blocks_run": n_run, "blocks_total": nq * nk,
        "bytes": 4 * arr, "kv_bytes": 2 * arr, "q_bytes": arr,
        "kv_fetches": kv_fetches, "n_kv_blocks": nk, "kv_fetch_factor": kv_fetches / nk,
        "hbm_bytes": 2 * arr + kv_hbm,
    }


def decode_budget(S, B, page_size, q_dtype="float32"):
    kv = B * S * NKV_DECODE * HD * 2 * 2
    q = B * NH * HD * ITEMSIZE[q_dtype]          # q and out at the carry dtype
    pps = S // page_size
    return {
        "flops_full": 4.0 * B * S * NH * HD, "flops_done": 4.0 * B * S * NH * HD,
        "causal_frac": 1.0, "blocks_run": B * NKV_DECODE * pps, "blocks_total": B * NKV_DECODE * pps,
        "bytes": kv + 2 * q, "kv_bytes": kv, "q_bytes": q,
        "kv_fetches": pps, "n_kv_blocks": pps, "kv_fetch_factor": 1.0,
        "hbm_bytes": kv + 2 * q,
        "pages_per_seq": pps, "num_pages": B * pps,
    }


def expected_step_s(bud):
    """(t_peak, t_plate_min, t_plate_kernel, expected): MXU time at peak on
    flops_done; HBM time at plate on the algorithmic minimum bytes and on the
    kernel's actual hbm_bytes; expected = max(t_peak, t_plate_kernel)."""
    t_peak = bud["flops_done"] / (PEAK_TFLOPS * 1e12)
    t_plate_min = bud["bytes"] / (PLATE_GBS * 1e9)
    t_plate = bud["hbm_bytes"] / (PLATE_GBS * 1e9)
    return t_peak, t_plate_min, t_plate, max(t_peak, t_plate)


def prefill_blocks(S, block):
    """block_q = block_k_major = block_k = min(block, S). All must be multiples
    of 128 and block_k_major / block_k must divide S (kernel _verify_block).
    When block_k == S the kernel takes its single-step path (no online
    softmax rescaling), which is the case for S=512 at the default block."""
    b = min(block, S)
    if b % 128 or S % b:
        raise ValueError(f"block {b} must be a multiple of 128 and divide S={S}")
    return b


def _int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def _cell_filter(spec):
    """--cells 512x1,2048x8 -> {(512, 1), (2048, 8)}; None = all."""
    if not spec:
        return None
    out = set()
    for tok in str(spec).split(","):
        s, b = tok.lower().split("x")
        out.add((int(s), int(b)))
    return out


def _chain_forms(args):
    forms = [f.strip() for f in str(getattr(args, "chain_form", "scan")).split(",") if f.strip()]
    for f in forms:
        if f not in CHAIN_FORMS:
            raise ValueError(f"--chain-form {f!r}: choose from {CHAIN_FORMS}")
    return forms


def decode_q_dtype(args):
    return Q_DTYPES[getattr(args, "decode_q_dtype", "f32")]


def points_for(mode, args):
    """One point per (cell x kernel config x chain form). --block / --ppcb /
    --chain-form accept comma lists so the block-size choice and the chain
    form can be swept in the same (cheap) session; each is part of the
    resumability key. cfg carries chain_form and q_dtype for the run."""
    want = _cell_filter(getattr(args, "cells", None))
    forms = _chain_forms(args)
    qdt = decode_q_dtype(args)
    pts = []
    if mode in ("prefill", "both"):
        for (S, B) in PREFILL_CELLS:
            if want is not None and (S, B) not in want:
                continue
            seen = set()
            for blk in _int_list(args.block):
                b = prefill_blocks(S, blk)
                if b in seen:      # min(block, S) collapses e.g. 512 and 1024 at S=512
                    continue
                seen.add(b)
                for form in forms:
                    pts.append(("prefill", S, B,
                                dict(block_q=b, block_k_major=b, block_k=b, block_b=1,
                                     fa_path="single_step" if b == S else "online",
                                     chain_form=form, q_dtype=PREFILL_Q_DTYPE),
                                prefill_budget(S, B, b, b)))
    if mode in ("decode", "both"):
        for (S, B) in DECODE_CELLS:
            if want is not None and (S, B) not in want:
                continue
            pps = S // args.page_size
            for ppcb in _int_list(args.ppcb):
                if S % args.page_size or pps % ppcb:
                    raise ValueError(f"S={S}: page_size {args.page_size} must divide S and "
                                     f"pages_per_compute_block {ppcb} must divide pages_per_seq {pps}")
                for form in forms:
                    pts.append(("decode", S, B,
                                dict(page_size=args.page_size, ppcb=ppcb, pages_per_seq=pps,
                                     chain_form=form, q_dtype=qdt),
                                decode_budget(S, B, args.page_size, qdt)))
    return pts


def cfg_suffix(mode, cfg):
    """Kernel-config part of a point's name (prefill: bq/bk, decode: ps/ppcb)."""
    if mode == "prefill":
        return f"bq{cfg['block_q']}_bk{cfg['block_k']}"
    return f"ps{cfg['page_size']}_ppcb{cfg['ppcb']}"


def point_name(mode, S, B, cfg, carry):
    """Trace directory name and [kernel-call] log tag: cell + kernel config
    (+ the carry / chain form / decode q dtype when not the default), so a
    sweep never writes two configs' xplanes into one directory or
    suppresses the second config's call-signature line."""
    n = f"A1_{mode}_S{S}_B{B}_{cfg_suffix(mode, cfg)}"
    if carry != "out":
        n += f"_carry{carry}"
    if cfg.get("chain_form", "scan") != "scan":
        n += f"_{cfg['chain_form']}"
    if mode == "decode" and cfg.get("q_dtype", "float32") != "float32":
        n += "_q" + ("bf16" if cfg["q_dtype"] == "bfloat16" else cfg["q_dtype"])
    return n


def key_fields(mode, S, B, cfg, chain, carry):
    """Resumability key: the cell plus the kernel config, carry, chain form
    and q dtype that produced it."""
    k = {"mode": mode, "S": S, "B": B, "nh": NH, "nkv": nkv_of(mode), "hd": HD, "chain": chain, "carry": carry,
         "chain_form": cfg.get("chain_form", "scan"), "q_dtype": cfg.get("q_dtype", PREFILL_Q_DTYPE)}
    if mode == "prefill":
        k.update(block_q=cfg["block_q"], block_k=cfg["block_k"])
    else:
        k.update(page_size=cfg["page_size"], ppcb=cfg["ppcb"])
    return k


def nkv_of(mode):
    return NH if mode == "prefill" else NKV_DECODE


# ----------------------------------------------------------------------------
# Kernel wrappers: every kernel call is logged with the exact signature used
# ----------------------------------------------------------------------------

_LOGGED = set()


def _log_call(tag, msg):
    if tag not in _LOGGED:
        _LOGGED.add(tag)
        print(f"[kernel-call] {tag}: {msg}", flush=True)


def _fmt(x):
    return f"{x.dtype}{list(x.shape)}"


def call_flash(q, k, v, *, block_sizes, sm_scale, tag):
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    _log_call(tag, f"flash_attention(q={_fmt(q)}, k={_fmt(k)}, v={_fmt(v)}, ab=None, segment_ids=None, "
                   f"causal=True, sm_scale={sm_scale:.6g}, block_sizes={block_sizes}, debug=False)  "
                   f"[{fa.__file__}]")
    return fa.flash_attention(q, k, v, causal=True, sm_scale=sm_scale, block_sizes=block_sizes)


def call_paged(q, k_pages, v_pages, lengths, page_indices, *, ppcb, megacore_mode, tag):
    from jax.experimental.pallas.ops.tpu import paged_attention as pa
    _log_call(tag, f"paged_attention(q={_fmt(q)}, k_pages={_fmt(k_pages)}, v_pages={_fmt(v_pages)}, "
                   f"lengths={_fmt(lengths)}, page_indices={_fmt(page_indices)}, "
                   f"pages_per_compute_block={ppcb}, megacore_mode={megacore_mode!r}, "
                   f"inline_seq_dim=True, attn_logits_soft_cap=None)  [{pa.__file__}]")
    return pa.paged_attention(q, k_pages, v_pages, lengths, page_indices,
                              pages_per_compute_block=ppcb, megacore_mode=megacore_mode)


def call_xla_prefill(q, k, v, *, sm_scale, tag):
    import jax
    _log_call(tag, f"jax.nn.dot_product_attention(q={_fmt(q)} [BNSH<-BHSN transposed], k, v, "
                   f"is_causal=True, scale={sm_scale:.6g}, implementation='xla')")
    qt, kt, vt = (x.swapaxes(1, 2) for x in (q, k, v))  # kernel layout is [B,N,S,H]; jax.nn wants [B,S,N,H]
    o = jax.nn.dot_product_attention(qt, kt, vt, is_causal=True, scale=sm_scale, implementation="xla")
    return o.swapaxes(1, 2)


def call_xla_decode(q, k_pages, v_pages, lengths, page_indices, *, tag):
    """GQA over the same paged KV, gathered contiguously (all lengths == S)."""
    import jax.numpy as jnp
    _log_call(tag, f"xla_gqa_einsum(q={_fmt(q)}, k_pages={_fmt(k_pages)}, v_pages={_fmt(v_pages)}, "
                   f"gather k_pages[:, page_indices] -> [B,nkv,S,hd], softmax(q k^T / sqrt(hd)) v)")
    B, nh, hd = q.shape
    nkv = k_pages.shape[0]
    g = nh // nkv
    k = k_pages[:, page_indices].reshape(nkv, B, -1, hd).swapaxes(0, 1)  # [B,nkv,S,hd]
    v = v_pages[:, page_indices].reshape(nkv, B, -1, hd).swapaxes(0, 1)
    qg = q.reshape(B, nkv, g, hd)
    s = jnp.einsum("bkgd,bktd->bkgt", qg, k, preferred_element_type=jnp.float32) * (hd ** -0.5)
    p = jnp.exp(s - s.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    o = jnp.einsum("bkgt,bktd->bkgd", p.astype(q.dtype), v, preferred_element_type=jnp.float32)
    return o.reshape(B, nh, hd).astype(q.dtype)


# ----------------------------------------------------------------------------
# Chained jits
# ----------------------------------------------------------------------------

def _next_carry(q, out, carry):
    """The scan carry for the next step. 'out': the kernel output itself is
    the next q (no XLA glue; every element of out is live because the next
    kernel call reads all of it and the jit sums the final carry). 'sum':
    the first version's reduction+broadcast, which streams 3 full arrays
    through HBM per step outside the kernel (see module docstring)."""
    import jax.numpy as jnp
    if carry == "out":
        return out.astype(q.dtype)
    if q.ndim == 4:                                              # prefill [B,nh,S,hd]
        r = jnp.sum(out, axis=-1, dtype=jnp.float32)             # every element of out
        return q + (r * 1e-3).astype(q.dtype)[..., None]         # feeds every element of next q
    return q + (out * 1e-3).astype(q.dtype)                      # decode [B,nh,hd]


def _run_chain(step, q0, chain, chain_form):
    """CHAIN dependent kernel calls inside the enclosing jit.
    scan:   lax.scan carry (rule 1; the while loop's state buffer forces a
            root copy of each step's output, see module docstring).
    unroll: Python for-loop -- out_i is the direct operand of call_{i+1},
            no loop state, no copy. Same single dispatch per timed call."""
    import jax
    assert chain_form in CHAIN_FORMS, chain_form
    if chain_form == "scan":
        qf, _ = jax.lax.scan(lambda q, _: (step(q), None), q0, None, length=chain)
        return qf
    q = q0
    for _ in range(chain):
        q = step(q)
    return q


def make_prefill_chain(chain, block_sizes, sm_scale, tag, carry="out", use_xla=False, chain_form="scan"):
    import jax
    import jax.numpy as jnp
    assert carry in CARRIES, carry

    @jax.jit
    def jchain(q0, k, v):
        def step(q):
            if use_xla:
                out = call_xla_prefill(q, k, v, sm_scale=sm_scale, tag=tag)
            else:
                out = call_flash(q, k, v, block_sizes=block_sizes, sm_scale=sm_scale, tag=tag)
            return _next_carry(q, out, carry)
        qf = _run_chain(step, q0, chain, chain_form)
        return jnp.sum(qf.astype(jnp.float32))
    return jchain


def make_decode_chain(chain, ppcb, megacore_mode, tag, carry="out", use_xla=False, chain_form="scan"):
    import jax
    import jax.numpy as jnp
    assert carry in CARRIES, carry

    @jax.jit
    def jchain(q0, k_pages, v_pages, lengths, page_indices):
        def step(q):
            if use_xla:
                out = call_xla_decode(q, k_pages, v_pages, lengths, page_indices, tag=tag)
            else:
                out = call_paged(q, k_pages, v_pages, lengths, page_indices,
                                 ppcb=ppcb, megacore_mode=megacore_mode, tag=tag)
            return _next_carry(q, out, carry)
        qf = _run_chain(step, q0, chain, chain_form)
        return jnp.sum(qf.astype(jnp.float32))
    return jchain


def prefill_inputs(S, B, key):
    import jax
    import jax.numpy as jnp
    kq, kk, kv = jax.random.split(key, 3)
    shape = (B, NH, S, HD)
    q = (jax.random.normal(kq, shape, jnp.float32) * 0.1).astype(jnp.bfloat16)
    k = (jax.random.normal(kk, shape, jnp.float32) * 0.1).astype(jnp.bfloat16)
    v = (jax.random.normal(kv, shape, jnp.float32) * 0.1).astype(jnp.bfloat16)
    return q, k, v


def decode_inputs(S, B, page_size, key, q_dtype="float32"):
    """q at the carry dtype (default float32: paged_attention launches its
    kernel with q in f32 for nh/nkv = 4 groups, so an f32 carry makes the
    wrapper's two converts no-ops); K/V pages always bf16."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    kq, kk, kv = jax.random.split(key, 3)
    pps = S // page_size
    num_pages = B * pps
    q = (jax.random.normal(kq, (B, NH, HD), jnp.float32) * 0.1).astype(jnp.dtype(q_dtype))
    k_pages = (jax.random.normal(kk, (NKV_DECODE, num_pages, page_size, HD), jnp.float32) * 0.1).astype(jnp.bfloat16)
    v_pages = (jax.random.normal(kv, (NKV_DECODE, num_pages, page_size, HD), jnp.float32) * 0.1).astype(jnp.bfloat16)
    lengths = jnp.full((B,), S, dtype=jnp.int32)
    # SEQUENTIAL layout: sequence b owns pages [b*pps, (b+1)*pps) in order.
    page_indices = jnp.asarray(np.arange(num_pages, dtype=np.int32).reshape(B, pps))
    return q, k_pages, v_pages, lengths, page_indices


# ----------------------------------------------------------------------------
# Kernel device time from the XProf trace (spec 2)
# ----------------------------------------------------------------------------

def trace_kernel_time(trace_dir, chain):
    """Parse the newest xplane under trace_dir/plugins/profile/*/ with xprof's
    framework_op_stats converter (the same one analysis/kernel_census.py
    uses). Returns (stats, None) or (None, reason). stats:
      kernel_us      sum of total_self_time over device rows of type
                     'pallas_call' (the Mosaic custom-calls), / chain
      trace_step_us  sum of total_self_time over ALL device rows, / chain
                     (the trace is exactly one chained call)
      glue_us        trace_step_us - kernel_us (scan/carry copies, the final
                     reduce, any XLA fusion between kernel calls)
      pallas_rows    the pallas rows found (name, type, occurrences, total us)
      other_top      the 3 largest non-pallas device rows
    Never raises on a parse problem: the caller records the reason and
    leaves the columns blank."""
    xps = sorted(glob.glob(os.path.join(trace_dir, "plugins", "profile", "*", "*.xplane.pb")))
    if not xps:
        return None, f"no *.xplane.pb under {trace_dir}/plugins/profile/*/"
    try:
        from xprof.convert import raw_to_tool_data as r
    except ImportError as e:  # xprof is a separate pip package; may be absent on the VM
        return None, f"xprof not importable ({e}); fill kernel_us offline with --annotate"
    try:
        import json
        out = r.xspace_to_tool_data([xps[-1]], "framework_op_stats", {})
        data = out[0] if isinstance(out, tuple) else out
        if isinstance(data, bytes):
            data = data.decode()
        payload = json.loads(data)
        table = payload[0] if isinstance(payload, list) else payload
        cols = [c["id"] for c in table["cols"]]
        rows = [dict(zip(cols, [c.get("v") for c in row["c"]])) for row in table.get("rows", [])]
    except Exception as e:  # noqa: BLE001
        return None, f"xprof converter failed on {xps[-1]}: {type(e).__name__}: {e}"
    dev = [d for d in rows if str(d.get("host_or_device", "")).lower().startswith("device")]
    if not dev:
        return None, f"no device rows in {xps[-1]} (CPU trace?)"

    def is_pallas(d):
        t = str(d.get("type") or "").lower()
        n = str(d.get("operation") or "").lower()
        return t in ("pallas_call", "tpu_custom_call") or n.endswith("/pallas_call") or "tpu_custom_call" in n

    pallas = [d for d in dev if is_pallas(d)]
    if not pallas:
        return None, f"no pallas_call device rows in {xps[-1]} (XLA fallback ran, or naming changed)"
    total = sum(float(d.get("total_self_time") or 0) for d in dev)
    kern = sum(float(d.get("total_self_time") or 0) for d in pallas)
    other = sorted((d for d in dev if not is_pallas(d)), key=lambda d: -float(d.get("total_self_time") or 0))
    return {
        "kernel_us": kern / chain, "trace_step_us": total / chain, "glue_us": (total - kern) / chain,
        "pallas_rows": [(d["operation"], d.get("type"), d.get("occurrences"), float(d.get("total_self_time") or 0))
                        for d in pallas],
        "pallas_occurrences": max(float(d.get("occurrences") or 0) for d in pallas),
        "other_top": [(d["operation"], d.get("type"), float(d.get("total_self_time") or 0)) for d in other[:3]],
        "xplane": xps[-1],
    }, None


def trace_columns(stats, bud):
    """CSV columns derived from trace_kernel_time's stats (all blank if None).
    `attribution` names which figures the row has: 'xprof' when kernel_us is
    filled (kernel_* are the kernel's own device time), 'wall' when only the
    wall-clock per_step_us / gbs / tflops / mfu are available."""
    if not stats:
        return {**{c: "" for c in TRACE_COLS}, "attribution": "wall"}
    ks = stats["kernel_us"] * 1e-6
    ktf = bud["flops_done"] / ks / 1e12 if ks > 0 else ""
    return {"kernel_us": stats["kernel_us"],
            "kernel_gbs": bud["hbm_bytes"] / ks / 1e9 if ks > 0 else "",
            "kernel_tflops": ktf, "kernel_mfu": (ktf / PEAK_TFLOPS) if ktf != "" else "",
            "glue_us": stats["glue_us"], "trace_step_us": stats["trace_step_us"],
            "pallas_rows": len(stats["pallas_rows"]), "pallas_occurrences": stats["pallas_occurrences"],
            "attribution": "xprof"}


def annotate_from_traces(stats_or_none, name, err):
    if stats_or_none is None:
        print(f"  trace {name}: kernel_us unavailable -- {err}", flush=True)
        return
    s = stats_or_none
    print(f"  trace {name}: kernel_us {s['kernel_us']:.2f}  glue_us {s['glue_us']:.2f}  "
          f"trace_step_us {s['trace_step_us']:.2f}  ({s['xplane']})", flush=True)
    for (n, t, occ, us) in s["pallas_rows"]:
        print(f"    pallas  {us:10.2f} us total  occ {occ}  {t}  {n}", flush=True)
    for (n, t, us) in s["other_top"]:
        print(f"    other   {us:10.2f} us total  {t}  {n}", flush=True)


def budget_of_row(row):
    """Rebuild the budget for a CSV row (used by --annotate)."""
    S, B = int(row["S"]), int(row["B"])
    if row["mode"] == "prefill":
        return prefill_budget(S, B, int(row["block_q"]), int(row["block_k_major"]))
    return decode_budget(S, B, int(row["page_size"]), row.get("q_dtype") or "bfloat16")


def cfg_of_row(row):
    """Config dict for point_name() from a CSV row (rows from the round-1
    layout without chain_form / q_dtype get the values that layout used)."""
    common = dict(chain_form=row.get("chain_form") or "scan",
                  q_dtype=row.get("q_dtype") or (PREFILL_Q_DTYPE if row["mode"] == "prefill" else "bfloat16"))
    if row["mode"] == "prefill":
        return dict(block_q=int(row["block_q"]), block_k=int(row["block_k"]), **common)
    return dict(page_size=int(row["page_size"]), ppcb=int(row["ppcb"]), **common)


def annotate(args):
    """Offline: for every row of --out, find its trace under --trace by the
    point name, compute the kernel-time columns and write <out>.kernel.csv
    (the original CSV is never rewritten)."""
    if not args.trace:
        sys.exit("--annotate needs --trace DIR (the directory the run traced into)")
    if not os.path.exists(args.out):
        sys.exit(f"--annotate: {args.out} does not exist")
    dst = args.annotate_out or (os.path.splitext(args.out)[0] + ".kernel.csv")
    with open(args.out) as f:
        rows = list(csv.DictReader(f))
    n_ok = 0
    with open(dst, "w", newline="") as f:
        w = None
        for row in rows:
            name = point_name(row["mode"], int(row["S"]), int(row["B"]), cfg_of_row(row), row.get("carry", "out"))
            stats, err = trace_kernel_time(os.path.join(args.trace, name), int(row["chain"]))
            annotate_from_traces(stats, name, err)
            n_ok += stats is not None
            out = dict(row)
            out.update(trace_columns(stats, budget_of_row(row)))
            out["trace_dir"] = os.path.join(args.trace, name)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(out.keys()))
                w.writeheader()
            w.writerow(out)
    print(f"--annotate: {n_ok}/{len(rows)} rows got kernel_us; wrote {dst}", flush=True)


# ----------------------------------------------------------------------------
# --dry-run / --probe-api
# ----------------------------------------------------------------------------

def dry_run(points, args):
    print(f"A1 attention probe: chain {args.chain}, carry {args.carry}, chain_form {args.chain_form}, "
          f"hd {HD}, nh {NH}, prefill nkv {NH} (MHA, q bf16), decode nkv {NKV_DECODE} (GQA, q {decode_q_dtype(args)}), "
          f"page_size {args.page_size}, pages_per_compute_block {args.ppcb}, block {args.block}")
    print(f"peak {PEAK_TFLOPS:.0f} TF/s, plate {PLATE_GBS:.0f} GB/s, VMEM {VMEM_BYTES / 2**20:.0f} MiB; "
          f"bytes_MB = algorithmic minimum (q+k+v+out | KV+q+out), hbm_MB = what the kernel moves "
          f"(prefill re-fetches K/V kvx times per head), t_plate/exp_us use hbm_MB; "
          f"carry_MB = the per-step scan-carry copy (2 passes over out) the scan form adds OUTSIDE the "
          f"kernel, copy_us its cost at plate (0 for unroll)")
    total = 0.0
    hdr = (f"{'mode':7s} {'S':>5s} {'B':>3s} {'nkv':>3s}  {'kernel':24s} {'cfg':28s} {'form':6s} "
           f"{'flops_full':>11s} {'flops_done':>11s} {'frac':>5s} {'bytes_MB':>9s} {'kv_MB':>8s} "
           f"{'kvx':>5s} {'hbm_MB':>8s} {'t_peak_us':>10s} {'t_plate_us':>10s} {'exp_us':>10s} "
           f"{'carry_MB':>8s} {'copy_us':>8s} {'vmem?':>5s}")
    print(hdr)
    for (mode, S, B, cfg, bud) in points:
        t_peak, _, t_plate, t_exp = expected_step_s(bud)
        kernel = KERNEL_FLASH if mode == "prefill" else KERNEL_PAGED
        cfgs = (f"bq={cfg['block_q']} bk={cfg['block_k']} {cfg['fa_path']}" if mode == "prefill"
                else f"ps={cfg['page_size']} ppcb={cfg['ppcb']} pps={cfg['pages_per_seq']}")
        fits = bud["bytes"] < VMEM_BYTES
        carry_b = 2 * bud["q_bytes"] if cfg["chain_form"] == "scan" else 0
        copy_us = carry_b / (PLATE_GBS * 1e9) * 1e6
        print(f"{mode:7s} {S:5d} {B:3d} {nkv_of(mode):3d}  {kernel:24s} {cfgs:28s} {cfg['chain_form']:6s} "
              f"{bud['flops_full']:11.3e} {bud['flops_done']:11.3e} {bud['causal_frac']:5.3f} "
              f"{bud['bytes'] / 1e6:9.1f} {bud['kv_bytes'] / 1e6:8.1f} "
              f"{bud['kv_fetch_factor']:5.2f} {bud['hbm_bytes'] / 1e6:8.1f} "
              f"{t_peak * 1e6:10.1f} {t_plate * 1e6:10.1f} {t_exp * 1e6:10.1f} "
              f"{carry_b / 1e6:8.1f} {copy_us:8.1f} {'yes' if fits else 'no':>5s}")
        total += (t_exp + copy_us * 1e-6) * args.chain * (args.reps + 1)
    print(f"{len(points)} points; expected device time at peak/plate (+ scan carry copies) for (reps+1)*chain "
          f"calls: {total:.1f} s (reps {args.reps}); traces and compile add to that")


def probe_api(points, args):
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    from jax.experimental.pallas.ops.tpu import paged_attention as pa
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}")
    print(f"flash_attention   module {fa.__file__}")
    print(f"  flash_attention{inspect.signature(fa.flash_attention)}")
    print(f"  BlockSizes{inspect.signature(fa.BlockSizes)}")
    print(f"  MIN_BLOCK_SIZE={fa.MIN_BLOCK_SIZE} NUM_LANES={fa.NUM_LANES} NUM_SUBLANES={fa.NUM_SUBLANES}")
    print(f"paged_attention   module {pa.__file__}")
    print(f"  paged_attention{inspect.signature(pa.paged_attention)}")
    print(f"  kernel module {pa.paged_attention_kernel.__file__}")
    try:
        import xprof  # noqa: F401
        print(f"xprof importable ({xprof.__file__}): kernel_us will be filled in-run from each trace")
    except ImportError:
        print("xprof NOT importable here: kernel_us stays blank in-run; fill it offline with --annotate")
    print(f"chain carry: {args.carry} (see docstring; 'out' = kernel output is the next q, no XLA glue); "
          f"chain form(s): {args.chain_form}; decode q dtype: {decode_q_dtype(args)}")
    print("shape smoke test (jax.eval_shape through each kernel's own validation; nothing executes); "
          "'body' lists the primitives the chain runs per step besides the pallas_call (jaxpr, nested jits "
          "flattened) -- the XLA glue the wall-clock per_step_us contains:")
    ok = True
    for (mode, S, B, cfg, bud) in points:
        tag = point_name(mode, S, B, cfg, args.carry)
        try:
            if mode == "prefill":
                bs = fa.BlockSizes(block_q=cfg["block_q"], block_k_major=cfg["block_k_major"],
                                   block_k=cfg["block_k"], block_b=cfg["block_b"])
                sm = HD ** -0.5
                spec = jax.ShapeDtypeStruct((B, NH, S, HD), jnp.bfloat16)
                o = jax.eval_shape(lambda q, k, v: call_flash(q, k, v, block_sizes=bs, sm_scale=sm, tag=tag),
                                   spec, spec, spec)
                ch = make_prefill_chain(args.chain, bs, sm, tag, carry=args.carry, chain_form=cfg["chain_form"])
                ins = (spec, spec, spec)
                assert o.shape == (B, NH, S, HD) and o.dtype == jnp.bfloat16, o
            else:
                pps = cfg["pages_per_seq"]
                qdt = jnp.dtype(cfg["q_dtype"])
                qs = jax.ShapeDtypeStruct((B, NH, HD), qdt)
                ks = jax.ShapeDtypeStruct((NKV_DECODE, B * pps, cfg["page_size"], HD), jnp.bfloat16)
                ls = jax.ShapeDtypeStruct((B,), jnp.int32)
                ps = jax.ShapeDtypeStruct((B, pps), jnp.int32)
                o = jax.eval_shape(lambda q, k, v, l, p: call_paged(q, k, v, l, p, ppcb=cfg["ppcb"],
                                                                     megacore_mode=args.megacore, tag=tag),
                                   qs, ks, ks, ls, ps)
                ch = make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag, carry=args.carry,
                                       chain_form=cfg["chain_form"])
                ins = (qs, ks, ks, ls, ps)
                assert o.shape == (B, NH, HD) and o.dtype == qdt, o
            r = jax.eval_shape(ch, *ins)
            assert r.shape == () and r.dtype == jnp.float32, r
            body = chain_body_primitives(jax.make_jaxpr(ch)(*ins).jaxpr, cfg["chain_form"])
            print(f"  OK   {mode:7s} S={S:5d} B={B:3d} {cfg_suffix(mode, cfg):16s} {cfg['chain_form']:6s} "
                  f"kernel out {o.dtype}{list(o.shape)}  chain -> {r.dtype}[]  kvx {bud['kv_fetch_factor']:.2f}"
                  f"  body: {body}")
        except Exception as e:  # noqa: BLE001 - report every failing cell, then exit non-zero
            ok = False
            print(f"  FAIL {mode:7s} S={S:5d} B={B:3d} {cfg_suffix(mode, cfg):16s} {type(e).__name__}: {e}")
    if not ok:
        sys.exit(1)


def chain_body_primitives(jaxpr, chain_form):
    """Names of the primitives one chain step runs (the scan body for the
    scan form; the whole jit for the unroll form, so they appear CHAIN
    times), with nested jit / custom_vjp wrappers flattened. Diagnostic
    only: shows the XLA converts / reshapes around the pallas_call."""
    def flat(j, out):
        for e in j.eqns:
            if e.primitive.name == "scan" and chain_form == "scan":
                flat(e.params["jaxpr"].jaxpr, out)
                out.append("<scan>")
            elif e.primitive.name in ("jit", "pjit", "closed_call", "custom_vjp_call", "custom_jvp_call"):
                inner = e.params.get("jaxpr") or e.params.get("call_jaxpr") or e.params.get("fun_jaxpr")
                inner = getattr(inner, "jaxpr", inner)
                if inner is None:
                    out.append(e.primitive.name)
                else:
                    flat(inner, out)
            else:
                out.append(e.primitive.name)
        return out
    names = flat(jaxpr, [])
    if chain_form == "scan" and "<scan>" in names:
        names = names[:names.index("<scan>")]          # the body only
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return " ".join(f"{n}x{c}" if c > 1 else n for n, c in counts.items())


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both", choices=("prefill", "decode", "both"))
    ap.add_argument("--out", default="a1_attention.csv")
    ap.add_argument("--trace", default=None,
                    help="capture one XProf trace per point under DIR/A1_<mode>_S<S>_B<B>_<cfg>; with xprof "
                         "importable the kernel_us columns are filled from it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-api", action="store_true",
                    help="print resolved import paths, signatures and a shape smoke test; no kernel executes")
    ap.add_argument("--annotate", action="store_true",
                    help="offline: fill kernel_us etc. for the rows of --out from the traces under --trace; "
                         "writes <out>.kernel.csv (or --annotate-out)")
    ap.add_argument("--annotate-out", default=None)
    ap.add_argument("--chain", type=int, default=8)
    ap.add_argument("--carry", default="out", choices=CARRIES,
                    help="scan carry: 'out' (kernel output is the next q; no XLA glue) or 'sum' "
                         "(q + sum(out)*1e-3 broadcast; 3 extra array passes per step, kept for comparison)")
    ap.add_argument("--chain-form", default="scan",
                    help="scan (lax.scan; rule 1) or unroll (Python loop of CHAIN calls in the same jit, no "
                         "loop-carry copy); comma list runs both (part of the CSV key)")
    ap.add_argument("--decode-q-dtype", default="f32", choices=tuple(Q_DTYPES),
                    help="decode q / carry dtype; f32 (default) makes paged_attention's wrapper converts "
                         "no-ops, bf16 reproduces the converted variant")
    ap.add_argument("--cells", default=None, help="restrict to cells, e.g. 512x1,2048x8 (SxB)")
    ap.add_argument("--block", default="512",
                    help="prefill block_q = block_k_major = block_k = min(block, S); multiple of 128; "
                         "comma list sweeps (e.g. 256,512)")
    ap.add_argument("--ppcb", default="16", help="decode pages_per_compute_block; comma list sweeps")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--megacore", default=None, choices=(None, "kv_head", "batch"),
                    help="paged_attention megacore_mode; leave None on v6e (single TensorCore)")
    ap.add_argument("--fallback-xla", action="store_true",
                    help="documented fallback: if the Pallas kernel raises, run the XLA attention instead "
                         "and record kernel=xla_* in the row")
    ap.add_argument("--reps", type=int, default=int(os.environ.get("PROBE_REPS", "20")))
    args = ap.parse_args()

    if args.annotate:
        annotate(args)
        return
    points = points_for(args.mode, args)
    if args.dry_run:
        dry_run(points, args)
        return
    if args.probe_api:
        probe_api(points, args)
        return

    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    from common import time_op, csv_append, already_done

    print(f"jax {jax.__version__} backend {jax.default_backend()} devices {jax.devices()}", flush=True)
    print(f"carry {args.carry}; chain {args.chain}; chain_form {args.chain_form}; "
          f"decode q dtype {decode_q_dtype(args)}; reps {args.reps}", flush=True)
    key = jax.random.PRNGKey(0)
    rejected = os.path.splitext(args.out)[0] + ".rejected.csv"

    for (mode, S, B, cfg, bud) in points:
        nkv = nkv_of(mode)
        form = cfg["chain_form"]
        name = point_name(mode, S, B, cfg, args.carry)
        tag = name
        kernel = KERNEL_FLASH if mode == "prefill" else KERNEL_PAGED
        if already_done(args.out, key_fields(mode, S, B, cfg, args.chain, args.carry)):
            print(f"skip (done) {name}", flush=True)
            continue

        if mode == "prefill":
            bs = fa.BlockSizes(block_q=cfg["block_q"], block_k_major=cfg["block_k_major"],
                               block_k=cfg["block_k"], block_b=cfg["block_b"])
            sm = HD ** -0.5
            q, k, v = prefill_inputs(S, B, key)
            jchain = make_prefill_chain(args.chain, bs, sm, tag, carry=args.carry, chain_form=form)
            fn = lambda: jchain(q, k, v)  # noqa: E731
            fb = (lambda: make_prefill_chain(args.chain, bs, sm, tag + "_xla", carry=args.carry, use_xla=True,
                                             chain_form=form),
                  KERNEL_XLA_PREFILL)
            cfg_cols = {"block_q": cfg["block_q"], "block_k_major": cfg["block_k_major"], "block_k": cfg["block_k"],
                        "fa_path": cfg["fa_path"], "page_size": "", "ppcb": ""}
        else:
            q, k_pages, v_pages, lengths, page_indices = decode_inputs(S, B, cfg["page_size"], key, cfg["q_dtype"])
            jchain = make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag, carry=args.carry,
                                       chain_form=form)
            fn = lambda: jchain(q, k_pages, v_pages, lengths, page_indices)  # noqa: E731
            fb = (lambda: make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag + "_xla",
                                            carry=args.carry, use_xla=True, chain_form=form),
                  KERNEL_XLA_DECODE)
            cfg_cols = {"block_q": "", "block_k_major": "", "block_k": "", "fa_path": "",
                        "page_size": cfg["page_size"], "ppcb": cfg["ppcb"]}
        cfg_cols.update(chain_form=form, q_dtype=cfg["q_dtype"])

        def timed(fn):
            # One extra (compiling) call whose value is checked: the chain
            # result must be finite, otherwise the carry drifted and the
            # kernel may have been fed NaN/inf (still timed, but meaningless).
            val = float(jax.block_until_ready(fn()))
            if not math.isfinite(val):
                raise FloatingPointError(f"chain result is {val}: carry '{args.carry}' drifted to non-finite values")
            return time_op(fn, reps=args.reps)

        try:
            r = timed(fn)
        except Exception as e:  # noqa: BLE001
            print(f"KERNEL FAILURE at {name} ({kernel}):\n{traceback.format_exc()}", flush=True)
            if not args.fallback_xla:
                print("no fallback enabled (--fallback-xla); aborting", flush=True)
                raise
            print(f"--fallback-xla: retrying {name} with the XLA attention", flush=True)
            mk, kernel = fb
            jchain = mk()
            if mode == "prefill":
                fn = lambda: jchain(q, k, v)  # noqa: E731
            else:
                fn = lambda: jchain(q, k_pages, v_pages, lengths, page_indices)  # noqa: E731
            r = timed(fn)

        per_step = r["median_s"] / args.chain
        gbs = bud["bytes"] / per_step / 1e9              # on the algorithmic minimum (sim counterpart)
        hbm_gbs = bud["hbm_bytes"] / per_step / 1e9      # on what the kernel actually moves
        tflops = bud["flops_done"] / per_step / 1e12
        fits_vmem = bud["bytes"] < VMEM_BYTES            # working set (q, k, v, out | KV, q, out)
        over = tflops > PEAK_TFLOPS * 1.05 or hbm_gbs > PLATE_GBS * 1.05
        vmem = int(over and fits_vmem)
        row = {"mode": mode, "S": S, "B": B, "nh": NH, "nkv": nkv, "hd": HD, "chain": args.chain,
               "kernel": kernel, **r, "per_step_us": per_step * 1e6,
               "kv_mb": bud["kv_bytes"] / 1e6, "bytes_mb": bud["bytes"] / 1e6, "gbs": gbs,
               "tflops": tflops, "mfu": tflops / PEAK_TFLOPS, "vmem_resident": vmem,
               "hbm_bytes_mb": bud["hbm_bytes"] / 1e6, "hbm_gbs": hbm_gbs,
               "kv_fetch_factor": bud["kv_fetch_factor"], "carry": args.carry,
               "flops_full": bud["flops_full"], "flops_done": bud["flops_done"],
               "causal_frac": bud["causal_frac"], **cfg_cols}

        # The trace is captured BEFORE the sanity gate: a refused row is
        # exactly the one whose trace (kernel_us / kernel_gbs) is the arbiter,
        # so it must exist and its columns go into the rejected row too.
        stats = None
        trace_dir = ""
        if args.trace:
            trace_dir = os.path.join(args.trace, name)
            os.makedirs(trace_dir, exist_ok=True)
            jax.profiler.start_trace(trace_dir)
            jax.block_until_ready(fn())
            jax.profiler.stop_trace()
            try:
                stats, err = trace_kernel_time(trace_dir, args.chain)
            except Exception as e:  # noqa: BLE001 - annotation must never abort a session
                stats, err = None, f"trace_kernel_time raised {type(e).__name__}: {e}"
            annotate_from_traces(stats, name, err)
        row.update(trace_columns(stats, bud))
        row["trace_dir"] = trace_dir
        ks = f"  kernel {stats['kernel_us']:.2f} us (glue {stats['glue_us']:.2f})" if stats else ""

        if over and not fits_vmem:
            reason = (f"{tflops:.0f} TF/s / {hbm_gbs:.0f} GB/s (on hbm_bytes {bud['hbm_bytes'] / 1e6:.0f} MB; "
                      f"{gbs:.0f} GB/s on the {bud['bytes'] / 1e6:.0f} MB minimum) exceeds 1.05x peak/plate "
                      f"with a {bud['bytes'] / 1e6:.0f} MB working set (> VMEM)")
            where = f"trace {trace_dir}{ks}" if trace_dir else "no --trace given: re-run this cell with --trace DIR"
            print(f"SANITY FAIL {name}: {reason} -- work was elided or the fetch model is wrong "
                  f"({where}); row NOT written to {args.out} (kept in {rejected})", flush=True)
            csv_append(rejected, {**row, "reason": reason})
            continue

        csv_append(args.out, row)
        print(f"{mode:7s} S={S:5d} B={B:3d} nkv={nkv:2d} {kernel:24s} {form:6s} {row['per_step_us']:10.2f} us/step  "
              f"{tflops:7.1f} TF/s  {gbs:7.0f} GB/s min  {hbm_gbs:7.0f} GB/s hbm{ks}"
              f"{'  [VMEM-resident]' if vmem else ''}", flush=True)

        if mode == "prefill":
            del q, k, v
        else:
            del q, k_pages, v_pages, lengths, page_indices


if __name__ == "__main__":
    main()
