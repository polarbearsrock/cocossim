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
  Chain: carry = q [B,nh,S,hd]; step: out = flash(q, k, v);
    q_next = q + (out.sum(-1, f32) * 1e-3).astype(bf16)[..., None]
  k, v are fixed (like weights). Every element of every out feeds every
  row of the next q (a full reduction over hd, then broadcast), and the jit
  returns the full-array sum of the final q -- nothing can be sliced or
  hoisted (spec 5.2: a step that returned one element let XLA reduce a GEMM
  to a single dot product and report 100 PFLOP/s).
  FLOPs: the kernel SKIPS kv blocks strictly above the diagonal
  (below_or_on_diag on the (block_q, block_k_major) grid), so the FLOPs it
  actually executes are flops_done = 4*B*nh*hd*block_q*block_k_major *
  n_blocks_run (masked entries inside diagonal blocks are still computed).
  With block_q == block_k_major == b and n = S/b that is the causal
  (n+1)/(2n) fraction of the dense 4*B*S*S*nh*hd. tflops/mfu use
  flops_done; flops_full is also written so the dense figure is recoverable.
  bytes per step = q + k + v + out (each B*nh*S*hd*2).

DECODE (query length 1):
  kernel = jax.experimental.pallas.ops.tpu.paged_attention.paged_attention
    (q [B,nh,hd], k_pages / v_pages [nkv, num_pages, page_size, hd],
     lengths i32[B], page_indices i32[B, pages_per_seq],
     pages_per_compute_block=PPCB, megacore_mode=None (v6e: 1 TensorCore),
     inline_seq_dim=True)
  GQA nh = 32, nkv = 8, page_size 16, pages laid out SEQUENTIALLY per
  sequence (page_indices[b] = b*pps + arange(pps); K1 covers shuffled
  layouts), lengths[b] = S.
  Chain: carry = q [B,nh,hd]; step: out = paged_attention(q, ...);
    q_next = q + (out * 1e-3).astype(bf16); return sum(final q).
  bytes per step = KV bytes read = B*S*nkv*hd*2*2 (+ q, out);
  FLOPs = 4*B*S*nh*hd.
  Note (kernel contract): with nh/nkv = 4 groups (not a multiple of 8) the
  kernel reshapes q to [B,nh,1,hd] and launches it in f32; the output is
  cast back to bf16. q/out bytes are counted at the API dtype (bf16); they
  are < 0.1 % of the KV bytes at every cell.

Chain honesty (both modes): the scan carry is the live data the next step
depends on, every kernel output element influences it, the jit returns a
full-array reduction of the final carry, and per point the bytes the kernel
must move and the FLOPs it must do are computed independently of the
measurement. A row above 1.05x peak (918 TF/s) or 1.05x plate (1638 GB/s)
is refused (SANITY FAIL, not written) UNLESS its working set fits VMEM
(< 128 MiB), in which case it is written with vmem_resident=1 -- a
legitimate VMEM-bandwidth reading (E1 found 2.2-4.5 TB/s for such carries).

Every kernel call goes through a wrapper that logs the exact call
signature used. --probe-api prints the resolved import paths, signatures
and a shape smoke test (jax.eval_shape through the kernels' own validation)
without executing anything; it runs on CPU JAX. The run mode fails loudly
with the exception text; there is NO silent fallback. --fallback-xla
(documented, off by default) substitutes a plain XLA attention
(jax.nn.dot_product_attention for prefill, an einsum GQA for decode over
the same paged KV gathered contiguously) ONLY when the Pallas kernel raises,
and records kernel=xla_* in the row.

With --trace DIR one XProf trace per point (a single chained call).
Resumable: points whose key fields are already in --out are skipped.

Usage: a1_attention.py [--mode prefill|decode|both] [--out a1_attention.csv]
         [--trace DIR] [--dry-run] [--probe-api] [--chain 8]
         [--block 512[,256,...]] [--ppcb 16[,8,...]] [--page-size 16] [--fallback-xla]
--block / --ppcb accept comma lists (the config is part of the CSV key), so
the block-size choice can be swept: the whole default 15-point session is
< 1 s of device time at peak/plate, so a sweep is cheap.
"""
import argparse
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


# ----------------------------------------------------------------------------
# Budgets (pure Python; used by --dry-run, the CSV and the sanity gate)
# ----------------------------------------------------------------------------

def below_or_on_diag(r, r_blk, c, c_blk):
    """Same predicate as flash_attention.below_or_on_diag (kv block c runs for
    q block r iff the block's bottom-left corner is on/below the diagonal)."""
    return ((r + 1) * r_blk - 1) > (c * c_blk)


def prefill_budget(S, B, block_q, block_k_major):
    nq = math.ceil(S / block_q)
    nk = S // block_k_major
    n_run = sum(1 for r in range(nq) for c in range(nk)
                if below_or_on_diag(r, block_q, c, block_k_major))
    flops_full = 4.0 * B * S * S * NH * HD
    flops_done = 4.0 * B * NH * HD * block_q * block_k_major * n_run
    arr = B * NH * S * HD * 2
    return {
        "flops_full": flops_full, "flops_done": flops_done,
        "causal_frac": flops_done / flops_full, "blocks_run": n_run, "blocks_total": nq * nk,
        "bytes": 4 * arr, "kv_bytes": 2 * arr, "q_bytes": arr,
    }


def decode_budget(S, B, page_size):
    kv = B * S * NKV_DECODE * HD * 2 * 2
    q = B * NH * HD * 2
    return {
        "flops_full": 4.0 * B * S * NH * HD, "flops_done": 4.0 * B * S * NH * HD,
        "causal_frac": 1.0, "blocks_run": B * NKV_DECODE * (S // page_size), "blocks_total": B * NKV_DECODE * (S // page_size),
        "bytes": kv + 2 * q, "kv_bytes": kv, "q_bytes": q,
        "pages_per_seq": S // page_size, "num_pages": B * (S // page_size),
    }


def expected_step_s(bud):
    t_peak = bud["flops_done"] / (PEAK_TFLOPS * 1e12)
    t_plate = bud["bytes"] / (PLATE_GBS * 1e9)
    return t_peak, t_plate, max(t_peak, t_plate)


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


def points_for(mode, args):
    """One point per (cell x kernel config). --block / --ppcb accept comma
    lists so the block-size choice can be swept in the same (cheap) session;
    the config is part of the resumability key."""
    pts = []
    if mode in ("prefill", "both"):
        for (S, B) in PREFILL_CELLS:
            seen = set()
            for blk in _int_list(args.block):
                b = prefill_blocks(S, blk)
                if b in seen:      # min(block, S) collapses e.g. 512 and 1024 at S=512
                    continue
                seen.add(b)
                pts.append(("prefill", S, B,
                            dict(block_q=b, block_k_major=b, block_k=b, block_b=1,
                                 fa_path="single_step" if b == S else "online"),
                            prefill_budget(S, B, b, b)))
    if mode in ("decode", "both"):
        for (S, B) in DECODE_CELLS:
            pps = S // args.page_size
            for ppcb in _int_list(args.ppcb):
                if S % args.page_size or pps % ppcb:
                    raise ValueError(f"S={S}: page_size {args.page_size} must divide S and "
                                     f"pages_per_compute_block {ppcb} must divide pages_per_seq {pps}")
                pts.append(("decode", S, B, dict(page_size=args.page_size, ppcb=ppcb, pages_per_seq=pps),
                            decode_budget(S, B, args.page_size)))
    return pts


def key_fields(mode, S, B, cfg, chain):
    """Resumability key: the cell plus the kernel config that produced it."""
    k = {"mode": mode, "S": S, "B": B, "nh": NH, "nkv": nkv_of(mode), "hd": HD, "chain": chain}
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
        print(f"[kernel-call] {msg}", flush=True)


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

def make_prefill_chain(chain, block_sizes, sm_scale, tag, use_xla=False):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def jchain(q0, k, v):
        def step(q, _):
            if use_xla:
                out = call_xla_prefill(q, k, v, sm_scale=sm_scale, tag=tag)
            else:
                out = call_flash(q, k, v, block_sizes=block_sizes, sm_scale=sm_scale, tag=tag)
            r = jnp.sum(out, axis=-1, dtype=jnp.float32)            # every element of out
            q_next = q + (r * 1e-3).astype(q.dtype)[..., None]      # feeds every element of next q
            return q_next, None
        qf, _ = jax.lax.scan(step, q0, None, length=chain)
        return jnp.sum(qf.astype(jnp.float32))
    return jchain


def make_decode_chain(chain, ppcb, megacore_mode, tag, use_xla=False):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def jchain(q0, k_pages, v_pages, lengths, page_indices):
        def step(q, _):
            if use_xla:
                out = call_xla_decode(q, k_pages, v_pages, lengths, page_indices, tag=tag)
            else:
                out = call_paged(q, k_pages, v_pages, lengths, page_indices,
                                 ppcb=ppcb, megacore_mode=megacore_mode, tag=tag)
            q_next = q + (out * 1e-3).astype(q.dtype)               # every element of out
            return q_next, None
        qf, _ = jax.lax.scan(step, q0, None, length=chain)
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


def decode_inputs(S, B, page_size, key):
    import jax
    import jax.numpy as jnp
    import numpy as np
    kq, kk, kv = jax.random.split(key, 3)
    pps = S // page_size
    num_pages = B * pps
    q = (jax.random.normal(kq, (B, NH, HD), jnp.float32) * 0.1).astype(jnp.bfloat16)
    k_pages = (jax.random.normal(kk, (NKV_DECODE, num_pages, page_size, HD), jnp.float32) * 0.1).astype(jnp.bfloat16)
    v_pages = (jax.random.normal(kv, (NKV_DECODE, num_pages, page_size, HD), jnp.float32) * 0.1).astype(jnp.bfloat16)
    lengths = jnp.full((B,), S, dtype=jnp.int32)
    # SEQUENTIAL layout: sequence b owns pages [b*pps, (b+1)*pps) in order.
    page_indices = jnp.asarray(np.arange(num_pages, dtype=np.int32).reshape(B, pps))
    return q, k_pages, v_pages, lengths, page_indices


# ----------------------------------------------------------------------------
# --dry-run / --probe-api
# ----------------------------------------------------------------------------

def dry_run(points, args):
    print(f"A1 attention probe: chain {args.chain}, hd {HD}, nh {NH}, "
          f"prefill nkv {NH} (MHA), decode nkv {NKV_DECODE} (GQA), page_size {args.page_size}, "
          f"pages_per_compute_block {args.ppcb}, block {args.block}")
    print(f"peak {PEAK_TFLOPS:.0f} TF/s, plate {PLATE_GBS:.0f} GB/s, VMEM {VMEM_BYTES / 2**20:.0f} MiB")
    total = 0.0
    hdr = (f"{'mode':7s} {'S':>5s} {'B':>3s} {'nkv':>3s}  {'kernel':24s} {'cfg':28s} "
           f"{'flops_full':>11s} {'flops_done':>11s} {'frac':>5s} {'bytes_MB':>9s} {'kv_MB':>8s} "
           f"{'t_peak_us':>10s} {'t_plate_us':>10s} {'exp_us':>10s} {'vmem?':>5s}")
    print(hdr)
    for (mode, S, B, cfg, bud) in points:
        t_peak, t_plate, t_exp = expected_step_s(bud)
        kernel = KERNEL_FLASH if mode == "prefill" else KERNEL_PAGED
        cfgs = (f"bq={cfg['block_q']} bk={cfg['block_k']} {cfg['fa_path']}" if mode == "prefill"
                else f"ps={cfg['page_size']} ppcb={cfg['ppcb']} pps={cfg['pages_per_seq']}")
        fits = bud["bytes"] < VMEM_BYTES
        print(f"{mode:7s} {S:5d} {B:3d} {nkv_of(mode):3d}  {kernel:24s} {cfgs:28s} "
              f"{bud['flops_full']:11.3e} {bud['flops_done']:11.3e} {bud['causal_frac']:5.3f} "
              f"{bud['bytes'] / 1e6:9.1f} {bud['kv_bytes'] / 1e6:8.1f} "
              f"{t_peak * 1e6:10.1f} {t_plate * 1e6:10.1f} {t_exp * 1e6:10.1f} {'yes' if fits else 'no':>5s}")
        total += t_exp * args.chain * (args.reps + 1)
    print(f"{len(points)} points; expected device time at peak/plate for (reps+1)*chain calls: "
          f"{total:.1f} s (reps {args.reps}); traces and compile add to that")


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
    print("shape smoke test (jax.eval_shape through each kernel's own validation; nothing executes):")
    ok = True
    for (mode, S, B, cfg, bud) in points:
        tag = f"{mode}_{S}_{B}"
        try:
            if mode == "prefill":
                bs = fa.BlockSizes(block_q=cfg["block_q"], block_k_major=cfg["block_k_major"],
                                   block_k=cfg["block_k"], block_b=cfg["block_b"])
                sm = HD ** -0.5
                spec = jax.ShapeDtypeStruct((B, NH, S, HD), jnp.bfloat16)
                o = jax.eval_shape(lambda q, k, v: call_flash(q, k, v, block_sizes=bs, sm_scale=sm, tag=tag),
                                   spec, spec, spec)
                ch = make_prefill_chain(args.chain, bs, sm, tag)
                r = jax.eval_shape(ch, spec, spec, spec)
                assert o.shape == (B, NH, S, HD) and o.dtype == jnp.bfloat16, o
            else:
                pps = cfg["pages_per_seq"]
                qs = jax.ShapeDtypeStruct((B, NH, HD), jnp.bfloat16)
                ks = jax.ShapeDtypeStruct((NKV_DECODE, B * pps, cfg["page_size"], HD), jnp.bfloat16)
                ls = jax.ShapeDtypeStruct((B,), jnp.int32)
                ps = jax.ShapeDtypeStruct((B, pps), jnp.int32)
                o = jax.eval_shape(lambda q, k, v, l, p: call_paged(q, k, v, l, p, ppcb=cfg["ppcb"],
                                                                     megacore_mode=args.megacore, tag=tag),
                                   qs, ks, ks, ls, ps)
                ch = make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag)
                r = jax.eval_shape(ch, qs, ks, ks, ls, ps)
                assert o.shape == (B, NH, HD) and o.dtype == jnp.bfloat16, o
            assert r.shape == () and r.dtype == jnp.float32, r
            print(f"  OK   {mode:7s} S={S:5d} B={B:3d}  kernel out {o.dtype}{list(o.shape)}  chain -> {r.dtype}[]")
        except Exception as e:  # noqa: BLE001 - report every failing cell, then exit non-zero
            ok = False
            print(f"  FAIL {mode:7s} S={S:5d} B={B:3d}  {type(e).__name__}: {e}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both", choices=("prefill", "decode", "both"))
    ap.add_argument("--out", default="a1_attention.csv")
    ap.add_argument("--trace", default=None, help="capture one XProf trace per point under DIR/A1_<mode>_S<S>_B<B>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-api", action="store_true",
                    help="print resolved import paths, signatures and a shape smoke test; no kernel executes")
    ap.add_argument("--chain", type=int, default=8)
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
    key = jax.random.PRNGKey(0)

    for (mode, S, B, cfg, bud) in points:
        nkv = nkv_of(mode)
        tag = f"{mode}_{S}_{B}"
        kernel = KERNEL_FLASH if mode == "prefill" else KERNEL_PAGED
        if already_done(args.out, key_fields(mode, S, B, cfg, args.chain)):
            print(f"skip (done) {tag} {cfg}", flush=True)
            continue

        if mode == "prefill":
            bs = fa.BlockSizes(block_q=cfg["block_q"], block_k_major=cfg["block_k_major"],
                               block_k=cfg["block_k"], block_b=cfg["block_b"])
            sm = HD ** -0.5
            q, k, v = prefill_inputs(S, B, key)
            jchain = make_prefill_chain(args.chain, bs, sm, tag)
            fn = lambda: jchain(q, k, v)  # noqa: E731
            fb = (lambda: make_prefill_chain(args.chain, bs, sm, tag + "_xla", use_xla=True), KERNEL_XLA_PREFILL)
            cfg_cols = {"block_q": cfg["block_q"], "block_k_major": cfg["block_k_major"], "block_k": cfg["block_k"],
                        "fa_path": cfg["fa_path"], "page_size": "", "ppcb": ""}
        else:
            q, k_pages, v_pages, lengths, page_indices = decode_inputs(S, B, cfg["page_size"], key)
            jchain = make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag)
            fn = lambda: jchain(q, k_pages, v_pages, lengths, page_indices)  # noqa: E731
            fb = (lambda: make_decode_chain(args.chain, cfg["ppcb"], args.megacore, tag + "_xla", use_xla=True),
                  KERNEL_XLA_DECODE)
            cfg_cols = {"block_q": "", "block_k_major": "", "block_k": "", "fa_path": "",
                        "page_size": cfg["page_size"], "ppcb": cfg["ppcb"]}

        try:
            r = time_op(fn, reps=args.reps)
        except Exception as e:  # noqa: BLE001
            print(f"KERNEL FAILURE at {tag} ({kernel}):\n{traceback.format_exc()}", flush=True)
            if not args.fallback_xla:
                print("no fallback enabled (--fallback-xla); aborting", flush=True)
                raise
            print(f"--fallback-xla: retrying {tag} with the XLA attention", flush=True)
            mk, kernel = fb
            jchain = mk()
            if mode == "prefill":
                fn = lambda: jchain(q, k, v)  # noqa: E731
            else:
                fn = lambda: jchain(q, k_pages, v_pages, lengths, page_indices)  # noqa: E731
            r = time_op(fn, reps=args.reps)

        per_step = r["median_s"] / args.chain
        gbs = bud["bytes"] / per_step / 1e9
        tflops = bud["flops_done"] / per_step / 1e12
        fits_vmem = bud["bytes"] < VMEM_BYTES
        over = tflops > PEAK_TFLOPS * 1.05 or gbs > PLATE_GBS * 1.05
        if over and not fits_vmem:
            print(f"SANITY FAIL {tag}: {tflops:.0f} TF/s / {gbs:.0f} GB/s exceeds peak/plate with a "
                  f"{bud['bytes'] / 1e6:.0f} MB working set (> VMEM) -- work was elided; row NOT written", flush=True)
            continue
        vmem = int(over and fits_vmem)
        row = {"mode": mode, "S": S, "B": B, "nh": NH, "nkv": nkv, "hd": HD, "chain": args.chain,
               "kernel": kernel, **r, "per_step_us": per_step * 1e6,
               "kv_mb": bud["kv_bytes"] / 1e6, "bytes_mb": bud["bytes"] / 1e6, "gbs": gbs,
               "tflops": tflops, "mfu": tflops / PEAK_TFLOPS, "vmem_resident": vmem,
               "flops_full": bud["flops_full"], "flops_done": bud["flops_done"],
               "causal_frac": bud["causal_frac"], **cfg_cols}
        csv_append(args.out, row)
        print(f"{mode:7s} S={S:5d} B={B:3d} nkv={nkv:2d} {kernel:24s} {row['per_step_us']:10.2f} us/step  "
              f"{tflops:7.1f} TF/s  {gbs:7.0f} GB/s{'  [VMEM-resident]' if vmem else ''}", flush=True)

        if args.trace:
            d = os.path.join(args.trace, f"A1_{mode}_S{S}_B{B}")
            os.makedirs(d, exist_ok=True)
            jax.profiler.start_trace(d)
            jax.block_until_ready(fn())
            jax.profiler.stop_trace()
        if mode == "prefill":
            del q, k, v
        else:
            del q, k_pages, v_pages, lengths, page_indices


if __name__ == "__main__":
    main()
