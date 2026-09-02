#!/usr/bin/env python3
"""K1: the paged-KV gather derate (fidelity spec 3.1 cell K1), device-side.

Decode attention (one query token per sequence), GQA nh=32 / nkv=8, head_dim
128, bf16, cells S in {512, 2048, 8192} x B in {8, 32}. Six timed variants
per cell, every one of them CHAIN kernel invocations inside ONE jit
(lax.scan) so the ~113 us host floor is amortized, and every one with the
same full-output carry discipline (spec 5.2: a step that returned a slice
let XLA collapse a GEMM to one dot product and report 100 PFLOP/s):

  paged_seq        jax.experimental.pallas.ops.tpu.paged_attention.paged_attention
                   with every sequence's pages laid out contiguously
                   (page_indices[b, p] = b*pps + p).
  paged_shuffled   the same kernel, page_indices = a fixed, seeded random
                   permutation of ALL pages of all sequences (the realistic
                   block-table gather); the KV pages are physically placed
                   by the inverse permutation so the attention result is
                   the same problem as paged_seq (checked against dense).
  dense            plain XLA reference on contiguous K/V [B, nkv, S, hd]
                   with q grouped [B, nkv, nh/nkv, hd]: einsum scores ->
                   softmax over S (f32) -> einsum with V. No paging.
  gather_only      the raw gather: read the same KV bytes through the
                   shuffled page table with jnp.take and reduce ALL
                   elements into the carry (no attention math).
  gather_seq       control: the same jnp.take instruction stream with
                   sequential page indices (identity table, rotated by the
                   step counter) -- isolates page-order randomness from
                   the gather machinery itself.
  contiguous_read  the same bytes read in their natural order (no gather)
                   and reduced the same way.
  So t_gather/t_contig is the pure-gather derate, t_gather/t_gather_seq
  its random-order component, and t_shuffled/t_dense, t_seq/t_dense the
  kernel-level derates.

Carry discipline. For the attention variants the carry is q itself and the
next step's query IS the previous step's attention output (q_{t+1} = o_t,
[B, nh, hd]): every element of every kernel output feeds the next kernel
call and the jit returns the full-array sum of the final q. For the read
variants the carry is (acc, cnt); the gather's page table is rotated by the
loop-carried counter ((perm + cnt) mod total_pages is still a permutation
of all pages) and the raw bf16 read is scaled by a loop-carried factor
before the f32 reduction, so no instruction in the loop body is
loop-invariant (XLA's while-LICM hoists any invariant instruction: a fixed
gather, a bf16->f32 convert of K/V -- observed on CPU, where the loop then
streamed a hidden f32 copy -- or a reduce). Because a backend can still
insert and hoist its own converts, every compiled program is checked
(hoist_check): the ENTRY computation must hold nothing but the while and
trivial glue; any instruction over 1M elements outside the loop is printed
as HOIST WARNING and the row is written with hoisted=1 and excluded from
the derates. The three read variants must also reduce to the same value
(they read the same pages), checked per cell.

Per point: per-step us, KV bytes moved (K + V, bf16), GB/s, TF/s; rows
above 1.05x the HBM plate (1638 GB/s) are kept only when the KV working set
fits VMEM (< 128 MiB; flagged vmem_resident=1 -- a VMEM-bandwidth reading)
and refused otherwise (SANITY FAIL); rows above 1.05x MXU peak (918 TF/s)
are always refused. --trace DIR captures one XProf trace (a single chained
call) per point. Resumable: (variant, S, B, nh, nkv, hd, page_size,
pages_per_block, chain) rows already in the CSV are skipped.

Kernel API (resolved from the installed source, jax 0.11.1; --probe-api
prints it live): paged_attention(q[B, nh, hd], k_pages[nkv, total_pages,
page_size, hd], v_pages[same], lengths i32[B], page_indices i32[B,
pages_per_sequence], *, pages_per_compute_block, mask_value,
attn_logits_soft_cap=None, megacore_mode=None, inline_seq_dim=True).
Constraints: nh % nkv == 0; pages_per_sequence % pages_per_compute_block
== 0; lengths int32; k/v same shape; nh/nkv % 8 != 0 makes the kernel
launch q as f32 [B, nh, 1, hd]. One DMA per (page, kv head) of
page_size*hd*2 bytes (4 KiB at page 16), double-buffered per compute block
of pages_per_compute_block pages, grid (1, B, nkv) with an in-kernel
fori_loop over the blocks. megacore_mode stays None (v6e: one TensorCore).
Nothing here falls back silently: a kernel failure prints the exception
and aborts unless --skip-failed-variants is given, in which case the
variant is skipped with the traceback printed and no row written.

Usage: k1_kv_gather.py [--dry-run] [--probe-api] [--out k1_kv_gather.csv]
                       [--trace DIR] [--page-size 16] [--pages-per-block 16]
                       [--variants a,b,...] [--cells 512x8,...] [--no-check]
                       [--skip-failed-variants]
"""
import argparse
import math
import os
import re
import traceback

CHAIN = 8
NH, NKV, HD = 32, 8, 128
CELLS = [(s, b) for s in (512, 2048, 8192) for b in (8, 32)]
VARIANTS = ("paged_seq", "paged_shuffled", "dense", "gather_only", "gather_seq", "contiguous_read")
READ_VARIANTS = ("gather_only", "gather_seq", "contiguous_read")
PEAK_TFLOPS = 918.0
PLATE_GBS = 1638.0
VMEM_BYTES = 128 * 1024 * 1024
PERM_SEED = 0x4B31      # fixed: the shuffled layout of a cell is a pure function of (seed, total_pages)
DATA_SEED = 0x4B1D
DTYPE_BYTES = 2         # bf16
HOIST_ELEMS = 1 << 20   # anything this big outside the while loop is hoisted work


# ----------------------------------------------------------------------------
# budgets (pure python; used by --dry-run and by the sanity checks)
# ----------------------------------------------------------------------------
def kv_bytes(s, b):
    return 2 * b * NKV * s * HD * DTYPE_BYTES          # K and V


def attn_flops(s, b):
    return 4.0 * b * NH * s * HD                        # QK^T and PV, one query token per sequence


def budget(variant, s, b):
    """(bytes moved per step, flops per step) the variant must do."""
    kvb = kv_bytes(s, b)
    if variant in READ_VARIANTS:
        return kvb, 0.0
    qb = 2 * b * NH * HD * DTYPE_BYTES                  # q in, o out (negligible)
    return kvb + qb, attn_flops(s, b)


def layout_params(s, page_size, pages_per_block):
    if s % page_size:
        raise ValueError(f"S={s} is not a multiple of page_size={page_size}")
    pps = s // page_size
    ppb = min(pages_per_block, pps)
    if pps % ppb:
        raise ValueError(f"pages_per_sequence={pps} not divisible by pages_per_compute_block={ppb}")
    return pps, ppb


def permutation(total_pages):
    """The fixed shuffled block table: seeded numpy, identical on every rep/rerun."""
    import numpy as np
    return np.random.default_rng(PERM_SEED).permutation(total_pages).astype(np.int32)


# ----------------------------------------------------------------------------
# --dry-run
# ----------------------------------------------------------------------------
def dry_run(args, cells, variants):
    reps = int(os.environ.get("PROBE_REPS", "20"))
    total_s = 0.0
    print(f"K1 paged-KV gather: nh={NH} nkv={NKV} hd={HD} bf16, page_size={args.page_size}, "
          f"pages_per_block<={args.pages_per_block}, chain {CHAIN}, perm_seed=0x{PERM_SEED:X}")
    print(f"{'variant':16s} {'S':>5s} {'B':>3s} {'pages':>6s} {'ppb':>4s} {'kv_MB':>8s} {'GFLOP':>8s} "
          f"{'t_plate_us':>10s} {'t_peak_us':>9s} {'bound':>5s} {'vmem_fit':>8s}")
    for (s, b) in cells:
        pps, ppb = layout_params(s, args.page_size, args.pages_per_block)
        for v in variants:
            by, fl = budget(v, s, b)
            t_plate = by / (PLATE_GBS * 1e9)
            t_peak = fl / (PEAK_TFLOPS * 1e12)
            t = max(t_plate, t_peak)
            total_s += t * CHAIN * (reps + 1)
            print(f"{v:16s} {s:5d} {b:3d} {b * pps:6d} {ppb:4d} {by / 1e6:8.1f} {fl / 1e9:8.2f} "
                  f"{t_plate * 1e6:10.1f} {t_peak * 1e6:9.2f} {'mem' if t_plate >= t_peak else 'mxu':>5s} "
                  f"{'yes' if kv_bytes(s, b) < VMEM_BYTES else 'no':>8s}")
    print(f"{len(cells) * len(variants)} points, chain {CHAIN}, {reps} reps (+1 discarded); "
          f"device time at plate/peak ~{total_s:.2f} s total (compile + data generation dominate).")


# ----------------------------------------------------------------------------
# kernel wrapper: the ONLY call site of the Pallas kernel; logs the exact signature
# ----------------------------------------------------------------------------
_LOGGED = set()


def call_paged_attention(q, k_pages, v_pages, lengths, page_indices, *, pages_per_compute_block, tag):
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
    key = (tag, q.shape, k_pages.shape, page_indices.shape, pages_per_compute_block)
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[kernel call] {tag}: paged_attention(q{tuple(q.shape)}:{q.dtype}, "
              f"k_pages{tuple(k_pages.shape)}:{k_pages.dtype}, v_pages{tuple(v_pages.shape)}:{v_pages.dtype}, "
              f"lengths{tuple(lengths.shape)}:{lengths.dtype}, page_indices{tuple(page_indices.shape)}:"
              f"{page_indices.dtype}, pages_per_compute_block={pages_per_compute_block}, "
              f"megacore_mode=None, inline_seq_dim=True, attn_logits_soft_cap=None)", flush=True)
    return paged_attention(q, k_pages, v_pages, lengths, page_indices,
                           pages_per_compute_block=pages_per_compute_block)


# ----------------------------------------------------------------------------
# the chained programs (traced under jit; kernel calls happen inside lax.scan)
# ----------------------------------------------------------------------------
def make_programs(ppb):
    import jax
    import jax.numpy as jnp

    def attn_chain(step_fn):
        def chain(q0, *rest):
            def step(q, _):
                o = step_fn(q, *rest)                    # [B, nh, hd] bf16
                return o.astype(q.dtype), None          # next query = full output
            qf, _ = jax.lax.scan(step, q0, None, length=CHAIN)
            return jnp.sum(qf.astype(jnp.float32))      # full-array reduction of the carry
        return chain

    def paged_step(tag):
        def f(q, k_pages, v_pages, lengths, page_indices):
            return call_paged_attention(q, k_pages, v_pages, lengths, page_indices,
                                        pages_per_compute_block=ppb, tag=tag)
        return f

    def dense_step(q, k, v):
        b, nh, hd = q.shape
        g = nh // NKV
        qg = q.reshape(b, NKV, g, hd)
        s = jnp.einsum("bkgd,bksd->bkgs", qg, k, preferred_element_type=jnp.float32) * (1.0 / math.sqrt(hd))
        p = jax.nn.softmax(s, axis=-1)                   # over ALL S positions
        o = jnp.einsum("bkgs,bksd->bkgd", p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        return o.reshape(b, nh, hd)

    def read_chain(mode):
        def chain(k_pages, v_pages, perm):
            total = k_pages.shape[1]

            def step(carry, _):
                acc, cnt = carry
                # Loop-carried scale (exactly representable in bf16), applied to
                # the RAW bf16 read before any convert, so nothing in the body
                # has only loop-invariant operands (see the module docstring).
                scale = (1.0 + 0.125 * ((cnt + (acc > 0).astype(jnp.int32)) % 4)).astype(k_pages.dtype)
                if mode == "contiguous":
                    kk, vv = k_pages, v_pages
                else:
                    base = perm if mode == "shuffled" else jnp.arange(total, dtype=jnp.int32)
                    idx = (base + cnt) % total           # rotated: still every page exactly once
                    kk = jnp.take(k_pages, idx, axis=1, mode="clip", unique_indices=True)
                    vv = jnp.take(v_pages, idx, axis=1, mode="clip", unique_indices=True)
                tot = (jnp.sum((kk * scale).astype(jnp.float32))
                       + jnp.sum((vv * scale).astype(jnp.float32)))   # every element
                return (acc + tot, cnt + 1), None

            (accf, cntf), _ = jax.lax.scan(step, (jnp.float32(0.0), jnp.int32(0)), None, length=CHAIN)
            return accf + cntf.astype(jnp.float32)
        return chain

    return {
        "paged_seq": jax.jit(attn_chain(paged_step("paged_seq"))),
        "paged_shuffled": jax.jit(attn_chain(paged_step("paged_shuffled"))),
        "dense": jax.jit(attn_chain(dense_step)),
        "gather_only": jax.jit(read_chain("shuffled")),
        "gather_seq": jax.jit(read_chain("sequential")),
        "contiguous_read": jax.jit(read_chain("contiguous")),
        "_dense_step": jax.jit(dense_step),
        "_paged_step": jax.jit(paged_step("check")),
    }


# ----------------------------------------------------------------------------
# compiled-HLO honesty check: nothing big may live outside the scan's while loop
# ----------------------------------------------------------------------------
_TRIVIAL = {"parameter", "while", "get-tuple-element", "tuple", "constant", "bitcast"}


def hoist_check(hlo_text):
    """Return [(name, opcode, shape)] of ENTRY instructions with >= HOIST_ELEMS
    elements that are not the while loop itself -- i.e. loop-invariant work
    XLA hoisted out of the scan (or a copy of K/V it inserted). Empty = OK."""
    entry = hlo_text[hlo_text.index("ENTRY"):]
    bad = []
    for line in entry.splitlines()[1:]:
        if line.strip() == "}":
            break
        m = re.match(r"\s*(?:ROOT )?%(\S+) = (.*?) (\S+)\(", line)
        if not m:
            continue
        name, shape, opcode = m.groups()
        if opcode in _TRIVIAL:
            continue
        elems = 0
        for dims in re.findall(r"\w+\[([\d,]*)\]", shape):
            n = 1
            for d in filter(None, dims.split(",")):
                n *= int(d)
            elems = max(elems, n)
        if elems >= HOIST_ELEMS:
            bad.append((name, opcode, shape))
    return bad


def check_compiled(name, prog, fn_args):
    """Compile once, run hoist_check on the optimized HLO and print. Returns
    (compiled executable, hoisted flag); the caller times THAT executable."""
    compiled = prog.lower(*fn_args).compile()
    bad = hoist_check(compiled.as_text())
    if bad:
        print(f"HOIST WARNING: {name}: {len(bad)} instruction(s) outside the scan loop -- the timed loop does "
              f"not move the bytes it is charged with:", flush=True)
        for (n, op, shape) in bad:
            print(f"    {n}: {op} {shape}", flush=True)
        return compiled, 1
    print(f"hoist check {name:16s}: OK (ENTRY holds only the while + glue)", flush=True)
    return compiled, 0


# ----------------------------------------------------------------------------
# data: one cell's K/V in the three layouts + the block tables
# ----------------------------------------------------------------------------
def build_cell(s, b, page_size, pps, xp, random_fn):
    """xp is numpy (probe-api smoke test) or jax.numpy (run). Returns a dict of arrays.

    K/V dense: [B, nkv, S, hd].  Sequential pages: k_pages_seq[h, b*pps+p] =
    K[b, h, p*page:(p+1)*page].  Shuffled: k_pages_shuf[:, perm[j]] =
    k_pages_seq[:, j], i.e. k_pages_shuf = k_pages_seq[:, argsort(perm)], and
    page_indices_shuf[b, p] = perm[b*pps + p] -- so both paged variants
    attend over exactly the same K/V as the dense reference.
    """
    import numpy as np
    total = b * pps
    q, k, v = random_fn((b, NH, HD)), random_fn((b, NKV, s, HD)), random_fn((b, NKV, s, HD))
    k_seq = xp.transpose(k, (1, 0, 2, 3)).reshape(NKV, total, page_size, HD)
    v_seq = xp.transpose(v, (1, 0, 2, 3)).reshape(NKV, total, page_size, HD)
    perm = permutation(total)
    inv = np.argsort(perm).astype(np.int32)
    pi_seq = xp.asarray(np.arange(total, dtype=np.int32).reshape(b, pps))
    pi_shuf = xp.asarray(perm.reshape(b, pps))
    k_shuf = xp.take(k_seq, xp.asarray(inv), axis=1)
    v_shuf = xp.take(v_seq, xp.asarray(inv), axis=1)
    lengths = xp.full((b,), s, dtype=np.int32)
    return dict(q=q, k=k, v=v, k_seq=k_seq, v_seq=v_seq, k_shuf=k_shuf, v_shuf=v_shuf,
                pi_seq=pi_seq, pi_shuf=pi_shuf, perm=xp.asarray(perm), lengths=lengths)


def variant_args(variant, d):
    if variant == "paged_seq":
        return (d["q"], d["k_seq"], d["v_seq"], d["lengths"], d["pi_seq"])
    if variant == "paged_shuffled":
        return (d["q"], d["k_shuf"], d["v_shuf"], d["lengths"], d["pi_shuf"])
    if variant == "dense":
        return (d["q"], d["k"], d["v"])
    if variant == "gather_only":
        return (d["k_shuf"], d["v_shuf"], d["perm"])
    if variant in ("gather_seq", "contiguous_read"):
        return (d["k_seq"], d["v_seq"], d["perm"])
    raise KeyError(variant)


# ----------------------------------------------------------------------------
# --probe-api: import paths, signature, constraints, layout + abstract-shape smoke test (no kernel execution)
# ----------------------------------------------------------------------------
def probe_api(args):
    import inspect
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas.ops.tpu.paged_attention as pa_pkg
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel as pak
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"paged_attention package : {pa_pkg.__file__}")
    print(f"paged_attention kernel  : {pak.__file__}")
    print(f"flash_attention (A1 ref): {fa.__file__}")
    print(f"paged_attention{inspect.signature(pak.paged_attention)}")
    print("constraints (from source): q[B,nh,hd]; k_pages/v_pages[nkv,total_pages,page_size,hd] same shape; "
          "lengths i32[B]; page_indices i32[B,pages_per_sequence] flattened to b*pps+p; nh%nkv==0; "
          "pages_per_sequence%pages_per_compute_block==0; nh/nkv%8!=0 -> q launched as f32 [B,nh,1,hd]; "
          "grid (1, B, nkv) with an in-kernel fori_loop over blocks (inline_seq_dim); per block "
          "pages_per_compute_block DMAs of [page_size,hd] per K and per V, double-buffered; "
          "megacore_mode None (single TensorCore).")
    s, b = 512, 8
    pps, ppb = layout_params(s, args.page_size, args.pages_per_block)
    rng = np.random.default_rng(DATA_SEED)
    d = build_cell(s, b, args.page_size, pps, np, lambda shp: rng.standard_normal(shp, dtype=np.float32))
    # layout check: the shuffled pages, gathered through the shuffled table, reconstruct K exactly
    for name, pages, pi in (("seq", d["k_seq"], d["pi_seq"]), ("shuf", d["k_shuf"], d["pi_shuf"])):
        rec = np.take(pages, pi.reshape(-1), axis=1).reshape(NKV, b, s, HD).transpose(1, 0, 2, 3)
        ok = np.array_equal(rec, d["k"])
        print(f"layout check {name:4s}: pages[:, page_indices] == K  -> {ok}")
        if not ok:
            raise SystemExit("layout check FAILED")
    perm = np.asarray(d["perm"])
    print(f"perm(seed=0x{PERM_SEED:X}, total_pages={b * pps}): first 8 = {perm[:8].tolist()}, "
          f"is permutation = {np.array_equal(np.sort(perm), np.arange(b * pps))}, "
          f"rotated (perm+3)%N is permutation = {np.array_equal(np.sort((perm + 3) % (b * pps)), np.arange(b * pps))}")
    # abstract-shape smoke test: trace every program (kernel included) without executing
    progs = make_programs(ppb)
    to_bf16 = lambda a: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16 if a.dtype == np.float32 else a.dtype)
    for v in VARIANTS:
        specs = tuple(to_bf16(a) for a in variant_args(v, d))
        out = jax.eval_shape(progs[v], *specs)
        print(f"eval_shape {v:16s} args={[tuple(a.shape) for a in specs]} -> {out.shape} {out.dtype}")
    # the hoist check on this backend's compilation of the XLA-only variants
    # (the Pallas variants only compile on a TPU). On CPU the read variants
    # are EXPECTED to warn: the CPU backend upcasts bf16 elementwise math to
    # f32 and hoists that convert; v6e has native bf16 VPU math.
    print(f"hoist check on backend={jax.default_backend()} (XLA-only variants):")
    for v in ("dense",) + READ_VARIANTS:
        check_compiled(v, progs[v], tuple(to_bf16(a) for a in variant_args(v, d)))
    print("probe-api OK (kernel traced abstractly; execution requires a TPU)")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-api", action="store_true")
    ap.add_argument("--out", default="k1_kv_gather.csv")
    ap.add_argument("--trace", default=None, help="one XProf trace per point under DIR/K1_<variant>_S<S>_B<B>")
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--pages-per-block", type=int, default=16, help="pages_per_compute_block (capped at pages_per_sequence)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--cells", default=",".join(f"{s}x{b}" for s, b in CELLS), help="SxB list")
    ap.add_argument("--no-check", action="store_true", help="skip the per-cell paged-vs-dense output agreement check")
    ap.add_argument("--skip-failed-variants", action="store_true",
                    help="documented fallback: print a failed variant's traceback and continue instead of aborting")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {VARIANTS}")
    cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")]

    if args.dry_run:
        return dry_run(args, cells, variants)
    if args.probe_api:
        return probe_api(args)

    import numpy as np
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done

    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    for (s, b) in cells:
        pps, ppb = layout_params(s, args.page_size, args.pages_per_block)
        key_base = {"S": s, "B": b, "nh": NH, "nkv": NKV, "hd": HD, "page_size": args.page_size,
                    "pages_per_block": ppb, "chain": CHAIN}
        todo = [v for v in variants if not already_done(args.out, {"variant": v, **key_base})]
        if not todo:
            continue
        progs = make_programs(ppb)
        keys = iter(jax.random.split(jax.random.PRNGKey(DATA_SEED), 8))
        d = build_cell(s, b, args.page_size, pps, jnp,
                       lambda shp: jax.random.normal(next(keys), shp, dtype=jnp.bfloat16))
        jax.block_until_ready(d)
        print(f"--- cell S={s} B={b}: {b * pps} pages of {args.page_size}, {ppb} pages/block, "
              f"KV {kv_bytes(s, b) / 1e6:.1f} MB", flush=True)

        if not args.no_check and any(v.startswith("paged") for v in todo):
            # one un-timed call per paged layout: both must agree with the dense reference
            ref = np.asarray(progs["_dense_step"](d["q"], d["k"], d["v"]).astype(jnp.float32))
            for v in ("paged_seq", "paged_shuffled"):
                if v not in todo:
                    continue
                q, kp, vp, ln, pi = variant_args(v, d)
                try:
                    out = np.asarray(progs["_paged_step"](q, kp, vp, ln, pi).astype(jnp.float32))
                except Exception:
                    print(f"K1 KERNEL FAIL ({v}, check call) S={s} B={b}:\n{traceback.format_exc()}", flush=True)
                    if args.skip_failed_variants:
                        todo.remove(v)
                        continue
                    raise
                err = float(np.max(np.abs(out - ref)))
                scale = float(np.max(np.abs(ref))) + 1e-6
                ok = err <= 0.05 * scale
                print(f"check {v:14s}: max|paged-dense|={err:.4g} (max|dense|={scale:.4g}) -> {'OK' if ok else 'MISMATCH'}",
                      flush=True)
                if not ok:
                    msg = f"K1 CHECK FAIL: {v} disagrees with the dense reference (S={s} B={b}); layout or kernel wrong"
                    if args.skip_failed_variants:
                        print(msg, flush=True)
                        todo.remove(v)
                    else:
                        raise SystemExit(msg)

        times, vals = {}, {}
        for v in todo:
            fn_args = variant_args(v, d)
            try:
                compiled, hoisted = check_compiled(v, progs[v], fn_args)
                fn = lambda: compiled(*fn_args)        # the inspected executable is the timed one
                r = time_op(fn)
                vals[v] = float(fn())
            except Exception:
                print(f"K1 VARIANT FAIL ({v}) S={s} B={b}:\n{traceback.format_exc()}", flush=True)
                if args.skip_failed_variants:
                    continue
                raise
            per_step = r["median_s"] / CHAIN
            by, fl = budget(v, s, b)
            gbs = by / per_step / 1e9
            tflops = fl / per_step / 1e12
            vmem = 0
            if tflops > PEAK_TFLOPS * 1.05:
                print(f"SANITY FAIL: {v} S={s} B={b} {tflops:.0f} TF/s exceeds peak -- work elided; row NOT written",
                      flush=True)
                continue
            if gbs > PLATE_GBS * 1.05:
                if kv_bytes(s, b) < VMEM_BYTES:
                    vmem = 1
                else:
                    print(f"SANITY FAIL: {v} S={s} B={b} {gbs:.0f} GB/s exceeds the HBM plate with a "
                          f"{kv_bytes(s, b) / 1e6:.0f} MB working set -- bytes not moved; row NOT written", flush=True)
                    continue
            row = {"variant": v, **key_base, **r, "per_step_us": per_step * 1e6,
                   "kv_mb": kv_bytes(s, b) / 1e6, "gbs": gbs, "vmem_resident": vmem,
                   "tflops": tflops, "hoisted": hoisted,
                   "perm_seed": PERM_SEED if v in ("paged_shuffled", "gather_only") else ""}
            csv_append(args.out, row)
            if not hoisted:
                times[v] = per_step
            print(f"{v:16s} S={s:5d} B={b:3d}  {per_step * 1e6:9.2f} us/step  {gbs:7.0f} GB/s  {tflops:6.2f} TF/s"
                  f"{'  [VMEM-resident]' if vmem else ''}{'  [HOISTED -- excluded from derates]' if hoisted else ''}",
                  flush=True)
            if args.trace:
                td = os.path.join(args.trace, f"K1_{v}_S{s}_B{b}")
                os.makedirs(td, exist_ok=True)
                jax.profiler.start_trace(td)
                jax.block_until_ready(fn())
                jax.profiler.stop_trace()
        # the read variants reduce the same pages: their chain values must agree
        rv = {v: vals[v] for v in READ_VARIANTS if v in vals}
        if len(rv) > 1:
            lo, hi = min(rv.values()), max(rv.values())
            spread = (hi - lo) / (abs(hi) + abs(lo) + 1e-6)
            print(f"read-variant agreement S={s} B={b}: "
                  + "  ".join(f"{k}={x:.6g}" for k, x in rv.items())
                  + f"  -> {'OK' if spread < 1e-3 else 'MISMATCH (a read variant did not cover every page)'}",
                  flush=True)
        report_derates(s, b, times)
        del d, progs

    print("\n=== K1 derates from", args.out, "(hoisted rows excluded) ===")
    summarize(args.out)


def report_derates(s, b, t):
    parts = []
    if "dense" in t:
        for v in ("paged_seq", "paged_shuffled"):
            if v in t:
                parts.append(f"t_{v.split('_')[1]}/t_dense={t[v] / t['dense']:.3f}")
    if "paged_shuffled" in t and "paged_seq" in t:
        parts.append(f"t_shuffled/t_seq={t['paged_shuffled'] / t['paged_seq']:.3f}")
    if "gather_only" in t and "contiguous_read" in t:
        parts.append(f"t_gather/t_contig={t['gather_only'] / t['contiguous_read']:.3f}")
    if "gather_only" in t and "gather_seq" in t:
        parts.append(f"t_gather/t_gather_seq={t['gather_only'] / t['gather_seq']:.3f}")
    if parts:
        print(f"derates S={s} B={b}: " + "  ".join(parts), flush=True)


def summarize(path):
    import csv
    if not os.path.exists(path):
        return
    cells = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("hoisted", "0") == "1":
                continue
            cells.setdefault((int(r["S"]), int(r["B"])), {})[r["variant"]] = float(r["per_step_us"])
    for (s, b), t in sorted(cells.items()):
        report_derates(s, b, t)


if __name__ == "__main__":
    main()
