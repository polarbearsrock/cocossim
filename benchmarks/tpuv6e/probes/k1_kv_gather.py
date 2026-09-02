#!/usr/bin/env python3
"""K1: the paged-KV gather derate (fidelity spec 3.1 cell K1), device-side.

Decode attention (one query token per sequence), GQA nh=32 / nkv=8, head_dim
128, bf16, cells S in {512, 2048, 8192} x B in {8, 32}. Eight timed variants
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
  dma_gather       the raw gather the way the kernel does it: a small Pallas
                   kernel (dma_reduce_kernel below) that walks the SAME
                   block table with the SAME DMA pattern as paged_attention
                   (grid (B, nkv), one pltpu.make_async_copy per page per
                   head for K and for V, double-buffered per compute block)
                   and only sums the pages -- no attention math. Shuffled
                   table.
  dma_seq          the same DMA kernel with the sequential table.
  gather_only      the raw gather in XLA: read the same KV bytes through the
                   shuffled page table with jnp.take and reduce ALL elements
                   into the carry.
  gather_seq       control: the same jnp.take instruction stream with
                   sequential page indices (identity table, rotated by the
                   step counter).
  contiguous_read  the same bytes read in their natural order (no gather)
                   and reduced the same way (XLA).
  So t_shuffled/t_dense and t_seq/t_dense are the kernel-level derates;
  t_dma_gather/t_contig is the pure-gather derate (DMA page stream through
  a random table vs one contiguous XLA stream), t_dma_gather/t_dma_seq its
  page-order-randomness component and t_dma_seq/t_contig the cost of
  4 KiB-page DMA issue itself. t_gather/t_contig and t_gather/t_gather_seq
  are the XLA-gather equivalents; XLA typically MATERIALIZES the gathered
  copy inside the loop (a [nkv, total_pages, page, hd] temp written and
  re-read, 3x the charged traffic) -- the compiled-HLO inspection below
  measures that (body_extra_mb, materialized=1) and the derate line says so.

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
THE KERNEL APPLIES NO SOFTMAX SCALE: its logits are einsum("gd,td->gt", q,
k) unscaled and the API has no sm_scale kwarg (unlike ragged_paged_
attention), so -- exactly as vLLM's TPU backend does -- the probe multiplies
q by 1/sqrt(hd) before every kernel call (a [B, nh, hd] elementwise op,
negligible) and the dense reference scales its logits the same way.
Verified in TPU interpret mode (--interpret, below): unscaled the kernel
disagrees with dense by 9x max|dense|; pre-scaled it agrees to 3e-3.

Carry discipline. For the attention variants the carry is q itself and the
next step's query IS the previous step's attention output (q_{t+1} = o_t,
[B, nh, hd]): every element of every kernel output feeds the next kernel
call and the jit returns the full-array sum of the final q. For the read
variants the carry is (acc, cnt); the page table is rotated by the
loop-carried counter ((perm + cnt) mod total_pages is still a permutation
of all pages) and, in the XLA variants, the raw bf16 read is scaled by a
loop-carried factor before the f32 reduction, so no instruction in the loop
body is loop-invariant (XLA's while-LICM hoists any invariant instruction:
a fixed gather, a bf16->f32 convert of K/V -- observed on CPU, where the
loop then streamed a hidden f32 copy -- or a reduce).

Compiled-HLO inspection (inspect_program), run on every timed executable
(the inspected object is the one timed):
  * ENTRY: every non-trivial instruction whose RESULT OR ANY OPERAND has
    >= 1M elements is hoisted work (a hoisted convert/gather/copy of K/V,
    or a hoisted full reduction -- scalar result, K/V-sized operand) ->
    HOIST WARNING, hoisted=1, row excluded from the derates.
  * the while body must contain at least one instruction that consumes a
    >= 1M-element operand (the loop really reads K/V); if not, hoisted=1.
  * every non-trivial while-body instruction with a >= 1M-element result
    is a materialized intermediate (a gather temp, a layout copy XLA
    inserted for the NT dot, a score matrix): their bytes are summed into
    body_extra_mb; materialized=1 when that exceeds 25% of the KV bytes
    the row is charged with, and the derate lines carry the caveat.
  * a checker self-test compiles a deliberately loop-invariant reduce chain
    (the case a result-shape-only check missed) and must flag it.
The three read variants (XLA) and the two DMA variants must reduce to the
same value (they read the same pages), checked per cell; both paged layouts
are checked against the dense reference before timing.

Per point: per-step us, KV bytes moved (K + V, bf16), GB/s, TF/s; rows
above 1.05x the HBM plate (1638 GB/s) are kept only when the KV working set
fits VMEM (< 128 MiB; flagged vmem_resident=1 -- a VMEM-bandwidth reading)
and refused otherwise (SANITY FAIL); rows above 1.05x MXU peak (918 TF/s)
are always refused. --trace DIR captures one XProf trace (a single chained
call) per point. Resumable: (variant, S, B, nh, nkv, hd, page_size,
pages_per_block, chain) rows already in the CSV are skipped.

Nothing here falls back silently: a kernel failure prints the exception
and aborts unless --skip-failed-variants is given, in which case the
variant is skipped with the traceback printed and no row written.
--interpret runs the full run path on CPU with every Pallas kernel in TPU
interpret mode (simulated HBM/VMEM/DMA/semaphores): all checks execute,
timings are meaningless and NO CSV row is written. Use it on a small cell
(--cells 512x8) before a session.

Usage: k1_kv_gather.py [--dry-run] [--probe-api] [--interpret]
                       [--out k1_kv_gather.csv] [--trace DIR] [--page-size 16]
                       [--pages-per-block 16] [--variants a,b,...]
                       [--cells 512x8,...] [--no-check] [--skip-failed-variants]
"""
import argparse
import functools
import math
import os
import re
import traceback

CHAIN = 8
NH, NKV, HD = 32, 8, 128
CELLS = [(s, b) for s in (512, 2048, 8192) for b in (8, 32)]
VARIANTS = ("paged_seq", "paged_shuffled", "dense", "dma_gather", "dma_seq",
            "gather_only", "gather_seq", "contiguous_read")
READ_VARIANTS = ("dma_gather", "dma_seq", "gather_only", "gather_seq", "contiguous_read")
PEAK_TFLOPS = 918.0
PLATE_GBS = 1638.0
VMEM_BYTES = 128 * 1024 * 1024
PERM_SEED = 0x4B31      # fixed: the shuffled layout of a cell is a pure function of (seed, total_pages)
DATA_SEED = 0x4B1D
DTYPE_BYTES = 2         # bf16
HOIST_ELEMS = 1 << 20   # anything this big outside the while loop is hoisted work
MATERIALIZED_FRAC = 0.25  # body intermediates above this fraction of the charged bytes -> materialized=1


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
# kernel wrappers: the ONLY call sites of the Pallas kernels; each logs the exact signature
# ----------------------------------------------------------------------------
_LOGGED = set()


def _log_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(msg, flush=True)


def call_paged_attention(q, k_pages, v_pages, lengths, page_indices, *, pages_per_compute_block, tag):
    """q is passed PRE-SCALED by 1/sqrt(hd): the kernel does not scale its logits."""
    from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
    _log_once((tag, q.shape, k_pages.shape, page_indices.shape, pages_per_compute_block),
              f"[kernel call] {tag}: paged_attention(q{tuple(q.shape)}:{q.dtype} [pre-scaled by 1/sqrt(hd)], "
              f"k_pages{tuple(k_pages.shape)}:{k_pages.dtype}, v_pages{tuple(v_pages.shape)}:{v_pages.dtype}, "
              f"lengths{tuple(lengths.shape)}:{lengths.dtype}, page_indices{tuple(page_indices.shape)}:"
              f"{page_indices.dtype}, pages_per_compute_block={pages_per_compute_block}, "
              f"megacore_mode=None, inline_seq_dim=True, attn_logits_soft_cap=None)")
    return paged_attention(q, k_pages, v_pages, lengths, page_indices,
                           pages_per_compute_block=pages_per_compute_block)


def dma_reduce_kernel(page_indices_ref, k_hbm_ref, v_hbm_ref, o_ref, k_buf, v_buf, k_sems, v_sems, *,
                      pages_per_sequence, pages_per_compute_block):
    """The paged_attention DMA stream without the math. Grid (b, h): stream every
    page of sequence b for kv head h through the block table -- one async copy
    per page for K and one for V, pages_per_compute_block of each in flight,
    double-buffered (block i+1 is started before block i is consumed), exactly
    like paged_flash_attention_kernel -- and accumulate the page sum into
    o_ref[page_size, hd] (f32), so every element read reaches the output."""
    from jax import lax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    b, h = pl.program_id(0), pl.program_id(1)
    nblk = pages_per_sequence // pages_per_compute_block

    def copies(blk, slot):
        base = b * pages_per_sequence + blk * pages_per_compute_block
        out = []
        for i in range(pages_per_compute_block):
            page = page_indices_ref[base + i]
            out.append(pltpu.make_async_copy(k_hbm_ref.at[h].at[page], k_buf.at[slot].at[i], k_sems.at[slot]))
            out.append(pltpu.make_async_copy(v_hbm_ref.at[h].at[page], v_buf.at[slot].at[i], v_sems.at[slot]))
        return out

    for c in copies(0, 0):
        c.start()
    o_ref[...] = jnp.zeros_like(o_ref)

    def body(blk, carry):
        slot = lax.rem(blk, 2)

        @pl.when(blk + 1 < nblk)
        def _prefetch_next():
            for c in copies(blk + 1, 1 - slot):
                c.start()

        for c in copies(blk, slot):
            c.wait()
        acc = o_ref[...]
        for i in range(pages_per_compute_block):
            acc = acc + k_buf[slot, i].astype(jnp.float32) + v_buf[slot, i].astype(jnp.float32)
        o_ref[...] = acc
        return carry

    lax.fori_loop(0, nblk, body, 0)


def call_dma_reduce(k_pages, v_pages, page_indices, *, pages_per_compute_block, tag):
    """Returns [B, nkv, page_size, hd] f32 page sums (K + V) read through page_indices."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    nkv, total, page_size, hd = k_pages.shape
    b, pps = page_indices.shape
    if k_pages.shape != v_pages.shape:
        raise ValueError(f"k_pages {k_pages.shape} != v_pages {v_pages.shape}")
    if pps % pages_per_compute_block:
        raise ValueError(f"pages_per_sequence={pps} % pages_per_compute_block={pages_per_compute_block} != 0")
    if page_indices.dtype != jnp.int32:
        raise ValueError(f"page_indices must be int32, got {page_indices.dtype}")
    _log_once((tag, k_pages.shape, page_indices.shape, pages_per_compute_block),
              f"[kernel call] {tag}: dma_reduce(k_pages{tuple(k_pages.shape)}:{k_pages.dtype}, "
              f"v_pages{tuple(v_pages.shape)}:{v_pages.dtype}, page_indices{tuple(page_indices.shape)}:"
              f"{page_indices.dtype}, pages_per_compute_block={pages_per_compute_block}) -> "
              f"[{b},{nkv},{page_size},{hd}] f32; grid ({b},{nkv}), {pps // pages_per_compute_block} blocks/seq, "
              f"{2 * pages_per_compute_block} DMAs of {page_size * hd * jnp.dtype(k_pages.dtype).itemsize} B per block")
    kern = functools.partial(dma_reduce_kernel, pages_per_sequence=pps,
                             pages_per_compute_block=pages_per_compute_block)
    return pl.pallas_call(
        kern,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,                      # the flattened block table
            in_specs=[pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec((None, None, page_size, hd), lambda b, h, *_: (b, h, 0, 0)),
            grid=(b, nkv),
            scratch_shapes=[pltpu.VMEM((2, pages_per_compute_block, page_size, hd), k_pages.dtype),
                            pltpu.VMEM((2, pages_per_compute_block, page_size, hd), v_pages.dtype),
                            pltpu.SemaphoreType.DMA((2,)),
                            pltpu.SemaphoreType.DMA((2,))]),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
        out_shape=jax.ShapeDtypeStruct((b, nkv, page_size, hd), jnp.float32),
    )(page_indices.reshape(-1), k_pages, v_pages)


# ----------------------------------------------------------------------------
# the chained programs (traced under jit; kernel calls happen inside lax.scan)
# ----------------------------------------------------------------------------
def make_programs(ppb, b, pps):
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
            q_scaled = q * jnp.asarray(1.0 / math.sqrt(HD), q.dtype)   # the kernel applies no scale
            return call_paged_attention(q_scaled, k_pages, v_pages, lengths, page_indices,
                                        pages_per_compute_block=ppb, tag=tag)
        return f

    def dense_step(q, k, v):
        bb, nh, hd = q.shape
        g = nh // NKV
        qg = q.reshape(bb, NKV, g, hd)
        s = jnp.einsum("bkgd,bksd->bkgs", qg, k, preferred_element_type=jnp.float32) * (1.0 / math.sqrt(hd))
        p = jax.nn.softmax(s, axis=-1)                   # over ALL S positions
        o = jnp.einsum("bkgs,bksd->bkgd", p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        return o.reshape(bb, nh, hd)

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

    def dma_chain(mode, tag):
        def chain(k_pages, v_pages, perm):
            total = k_pages.shape[1]

            def step(carry, _):
                acc, cnt = carry
                base = perm if mode == "shuffled" else jnp.arange(total, dtype=jnp.int32)
                idx = ((base + cnt) % total).reshape(b, pps)   # loop-carried table: every page exactly once
                sums = call_dma_reduce(k_pages, v_pages, idx, pages_per_compute_block=ppb, tag=tag)
                return (acc + jnp.sum(sums), cnt + 1), None    # every output element into the carry

            (accf, cntf), _ = jax.lax.scan(step, (jnp.float32(0.0), jnp.int32(0)), None, length=CHAIN)
            return accf + cntf.astype(jnp.float32)
        return chain

    return {
        "paged_seq": jax.jit(attn_chain(paged_step("paged_seq"))),
        "paged_shuffled": jax.jit(attn_chain(paged_step("paged_shuffled"))),
        "dense": jax.jit(attn_chain(dense_step)),
        "dma_gather": jax.jit(dma_chain("shuffled", "dma_gather")),
        "dma_seq": jax.jit(dma_chain("sequential", "dma_seq")),
        "gather_only": jax.jit(read_chain("shuffled")),
        "gather_seq": jax.jit(read_chain("sequential")),
        "contiguous_read": jax.jit(read_chain("contiguous")),
        "_dense_step": jax.jit(dense_step),
        "_paged_step": jax.jit(paged_step("check")),
    }


# ----------------------------------------------------------------------------
# compiled-HLO honesty check: ENTRY (operand-aware), loop body (materialization)
# ----------------------------------------------------------------------------
_TRIVIAL = {"parameter", "while", "get-tuple-element", "tuple", "constant", "bitcast", "call", "conditional"}
_DTYPE_BYTES = {"pred": 1, "s8": 1, "u8": 1, "f8e5m2": 1, "f8e4m3fn": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2,
                "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8, "c64": 8, "c128": 16}
_INSTR_RE = re.compile(r"^\s*(?:ROOT )?%([^\s=]+) = (.*?) ([a-z][a-z\-]*)\((.*?)\)")
_COMP_RE = re.compile(r"^(ENTRY )?%?([^\s(]+) \(.*\) -> .* \{\s*$")


def _shape_stats(shape):
    """(max elements of any array in the shape, total bytes of all arrays)."""
    elems, nbytes = 0, 0
    for dt, dims in re.findall(r"(\w+)\[([\d,]*)\]", shape):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        elems = max(elems, n)
        nbytes += n * _DTYPE_BYTES.get(dt, 4)
    return elems, nbytes


def parse_hlo(hlo_text):
    """{computation name: [(name, shape, opcode, [operand names], line)]}, plus the ENTRY name."""
    comps, entry, cur = {}, None, None
    for line in hlo_text.splitlines():
        m = _COMP_RE.match(line)
        if m:
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        if line.strip() == "}":
            cur = None
            continue
        if cur is None:
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        name, shape, opcode, ops = m.groups()
        operands = re.findall(r"%([^\s,()]+)", ops)
        comps[cur].append((name, shape, opcode, operands, line))
    if entry is None:
        raise ValueError("no ENTRY computation in the compiled HLO")
    return comps, entry


def _loop_bodies(comps, start, seen=None):
    """Names of every while-body computation reachable from `start` through while/call/conditional."""
    seen = set() if seen is None else seen
    bodies = []
    for (_, _, opcode, _, line) in comps.get(start, []):
        if opcode == "while":
            m = re.search(r"body=%?([^\s,]+)", line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                bodies.append(m.group(1))
                bodies += _loop_bodies(comps, m.group(1), seen)
        elif opcode in ("call", "conditional"):
            for callee in re.findall(r"(?:to_apply|true_computation|false_computation|branch_computations)=\{?%?([^\s,}]+)", line):
                if callee not in seen:
                    seen.add(callee)
                    bodies += _loop_bodies(comps, callee, seen)
    return bodies


def inspect_program(hlo_text):
    """Return dict(hoisted=[...], loop_reads_big=bool, body_extra_bytes=int, materialized=[...], bodies=[...]).

    hoisted: ENTRY instructions (not while/tuple/gte/...) whose result or any
    operand has >= HOIST_ELEMS elements -- work done outside the timed loop,
    including a hoisted full reduction whose result is a scalar.
    loop_reads_big: some loop-body instruction consumes a >= HOIST_ELEMS operand.
    materialized: loop-body instructions with a >= HOIST_ELEMS result (temps
    written and re-read each step); body_extra_bytes sums their sizes."""
    comps, entry = parse_hlo(hlo_text)

    def scan(comp_name, flag_operands, flag_results):
        shapes = {name: shape for (name, shape, _, _, _) in comps[comp_name]}
        flagged, reads_big = [], False
        for (name, shape, opcode, operands, _) in comps[comp_name]:
            if opcode in _TRIVIAL:
                continue
            res_elems, res_bytes = _shape_stats(shape)
            op_elems = max([_shape_stats(shapes[o])[0] for o in operands if o in shapes] or [0])
            if op_elems >= HOIST_ELEMS:
                reads_big = True
            if (flag_results and res_elems >= HOIST_ELEMS) or (flag_operands and op_elems >= HOIST_ELEMS):
                flagged.append((name, opcode, shape, res_bytes, op_elems))
        return flagged, reads_big

    hoisted, _ = scan(entry, flag_operands=True, flag_results=True)
    bodies = _loop_bodies(comps, entry)
    materialized, loop_reads_big = [], False
    for body in bodies:
        m, r = scan(body, flag_operands=False, flag_results=True)
        materialized += m
        loop_reads_big = loop_reads_big or r
    return dict(hoisted=hoisted, loop_reads_big=loop_reads_big,
                body_extra_bytes=sum(x[3] for x in materialized), materialized=materialized, bodies=bodies)


def hoist_check(hlo_text):
    """Backwards-compatible view: [(name, opcode, shape)] of hoisted ENTRY work. Empty = OK."""
    return [(n, op, sh) for (n, op, sh, _, _) in inspect_program(hlo_text)["hoisted"]]


def check_compiled(name, prog, fn_args, charged_bytes=None):
    """Compile once, inspect the optimized HLO, print. Returns (compiled executable,
    hoisted flag, body_extra_bytes, materialized flag); the caller times THAT executable."""
    compiled = prog.lower(*fn_args).compile()
    r = inspect_program(compiled.as_text())
    hoisted = 0
    if r["hoisted"]:
        hoisted = 1
        print(f"HOIST WARNING: {name}: {len(r['hoisted'])} instruction(s) outside the scan loop touch K/V-sized "
              f"data -- the timed loop does not do the work it is charged with:", flush=True)
        for (n, op, shape, _, op_elems) in r["hoisted"]:
            print(f"    {n}: {op} {shape}  (largest operand {op_elems / 1e6:.1f}M elements)", flush=True)
    if not r["bodies"]:
        hoisted = 1
        print(f"HOIST WARNING: {name}: no while loop found in the compiled program", flush=True)
    elif not r["loop_reads_big"]:
        hoisted = 1
        print(f"HOIST WARNING: {name}: no instruction in the loop body consumes a K/V-sized operand", flush=True)
    extra = r["body_extra_bytes"]
    mat = 0
    if r["materialized"]:
        frac = extra / charged_bytes if charged_bytes else float("inf")
        mat = int(frac > MATERIALIZED_FRAC)
        print(f"BODY INTERMEDIATES: {name}: {len(r['materialized'])} materialized instruction(s) per step, "
              f"{extra / 1e6:.1f} MB ({100 * frac:.0f}% of the charged bytes)"
              f"{' -> materialized=1' if mat else ''}:", flush=True)
        for (n, op, shape, nb, _) in r["materialized"]:
            print(f"    {n}: {op} {shape}  ({nb / 1e6:.1f} MB)", flush=True)
    if not hoisted and not r["materialized"]:
        print(f"hoist check {name:16s}: OK (ENTRY holds only the while + glue; loop body reads K/V, "
              f"no materialized intermediate)", flush=True)
    elif not hoisted:
        print(f"hoist check {name:16s}: OK (ENTRY holds only the while + glue; loop body reads K/V)", flush=True)
    return compiled, hoisted, extra, mat


def checker_selftest(k_shape, dtype):
    """Negative control for inspect_program: a chain whose sum(K)+sum(V) is
    loop-invariant. A result-shape-only check passed it (the hoisted reduce is
    a scalar); the operand-aware check must flag it, or -- if this backend does
    not hoist it -- must at least see the loop body consuming K/V."""
    import jax
    import jax.numpy as jnp

    def bad(kp, vp):
        def step(c, _):
            acc, cnt = c
            return (acc + jnp.sum(kp.astype(jnp.float32)) + jnp.sum(vp.astype(jnp.float32)), cnt + 1), None
        (a, c), _ = jax.lax.scan(step, (jnp.float32(0), jnp.int32(0)), None, length=CHAIN)
        return a + c.astype(jnp.float32)

    spec = jax.ShapeDtypeStruct(k_shape, dtype)
    r = inspect_program(jax.jit(bad).lower(spec, spec).compile().as_text())
    flagged = bool(r["hoisted"]) or not r["loop_reads_big"]
    print(f"checker self-test (loop-invariant reduce chain, {k_shape} {jnp.dtype(dtype).name}): "
          f"hoisted instructions={len(r['hoisted'])} loop_reads_big={r['loop_reads_big']} -> "
          f"{'FLAGGED as hoisted (OK)' if flagged else 'NOT flagged: the compiler kept the reduce in the loop'}",
          flush=True)
    if not flagged and not r["loop_reads_big"]:
        raise SystemExit("checker self-test FAILED: parser found neither hoisted work nor an in-loop K/V read")
    return flagged


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
    if variant in ("gather_only", "dma_gather"):
        return (d["k_shuf"], d["v_shuf"], d["perm"])
    if variant in ("gather_seq", "contiguous_read", "dma_seq"):
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
    src = inspect.getsource(pak.paged_flash_attention_kernel)
    unscaled = 'einsum("gd,td->gt", q, k' in src and "sm_scale" not in inspect.signature(pak.paged_attention).parameters
    print(f"softmax scale inside the kernel: {'NONE' if unscaled else 'present?!'} (logits = einsum(q, k) unscaled, "
          f"no sm_scale kwarg) -> the probe pre-multiplies q by 1/sqrt(hd)={1 / math.sqrt(HD):.6f} before every call")
    if not unscaled:
        raise SystemExit("kernel source no longer matches the unscaled-logits assumption; re-check paged_step")
    print("dma_reduce (this file): pl.pallas_call, PrefetchScalarGridSpec(num_scalar_prefetch=1 [flattened block "
          "table]), in_specs K/V in pl.ANY (HBM), grid (B, nkv), scratch VMEM (2, ppb, page, hd) x2 + DMA sems (2,) x2, "
          "out [B, nkv, page, hd] f32 page sums; per block ppb K + ppb V pltpu.make_async_copy, double-buffered")
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
    # abstract-shape smoke test: trace every program (kernels included) without executing
    progs = make_programs(ppb, b, pps)
    to_bf16 = lambda a: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16 if a.dtype == np.float32 else a.dtype)
    for v in VARIANTS:
        specs = tuple(to_bf16(a) for a in variant_args(v, d))
        out = jax.eval_shape(progs[v], *specs)
        print(f"eval_shape {v:16s} args={[tuple(a.shape) for a in specs]} -> {out.shape} {out.dtype}")
    # the checker's negative control, then the inspection of this backend's
    # compilation of the XLA-only variants (the Pallas variants only compile on
    # a TPU). On CPU the read variants are EXPECTED to warn: the CPU backend
    # upcasts bf16 elementwise math to f32 and hoists that convert; v6e has
    # native bf16 VPU math. gather_only/gather_seq are expected to show the
    # materialized jnp.take temp (that is the point of measuring it).
    checker_selftest(d["k_seq"].shape, jnp.bfloat16)
    print(f"hoist check on backend={jax.default_backend()} (XLA-only variants):")
    for v in ("dense", "gather_only", "gather_seq", "contiguous_read"):
        check_compiled(v, progs[v], tuple(to_bf16(a) for a in variant_args(v, d)), charged_bytes=budget(v, s, b)[0])
    print("probe-api OK (kernels traced abstractly; execution requires a TPU, or --interpret on CPU)")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-api", action="store_true")
    ap.add_argument("--interpret", action="store_true",
                    help="CPU validation: run the whole run path with Pallas kernels in TPU interpret mode; "
                         "all checks run, one call per variant, no timing, NO CSV rows written")
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

    import contextlib
    import numpy as np
    import jax
    import jax.numpy as jnp
    from common import time_op, csv_append, already_done

    if args.interpret:
        from jax.experimental.pallas import tpu as pltpu
        interpret_ctx = pltpu.force_tpu_interpret_mode(pltpu.InterpretParams())
        print("INTERPRET MODE: Pallas kernels run in TPU interpret mode on this backend; timings are meaningless "
              "and no CSV row will be written", flush=True)
    else:
        interpret_ctx = contextlib.nullcontext()

    print(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    with interpret_ctx:
        run(args, cells, variants, np, jax, jnp, time_op, csv_append, already_done)
    if not args.interpret:
        print("\n=== K1 derates from", args.out, "(hoisted rows excluded) ===")
        summarize(args.out)


def run(args, cells, variants, np, jax, jnp, time_op, csv_append, already_done):
    selftest_done = False
    for (s, b) in cells:
        pps, ppb = layout_params(s, args.page_size, args.pages_per_block)
        key_base = {"S": s, "B": b, "nh": NH, "nkv": NKV, "hd": HD, "page_size": args.page_size,
                    "pages_per_block": ppb, "chain": CHAIN}
        todo = [v for v in variants if args.interpret or not already_done(args.out, {"variant": v, **key_base})]
        if not todo:
            continue
        progs = make_programs(ppb, b, pps)
        keys = iter(jax.random.split(jax.random.PRNGKey(DATA_SEED), 8))
        d = build_cell(s, b, args.page_size, pps, jnp,
                       lambda shp: jax.random.normal(next(keys), shp, dtype=jnp.bfloat16))
        jax.block_until_ready(d)
        print(f"--- cell S={s} B={b}: {b * pps} pages of {args.page_size}, {ppb} pages/block, "
              f"KV {kv_bytes(s, b) / 1e6:.1f} MB", flush=True)
        if not selftest_done:
            checker_selftest(d["k_seq"].shape, d["k_seq"].dtype)   # the checker must catch a hoisted reduce
            selftest_done = True

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

        times, vals, caveats = {}, {}, {}
        for v in todo:
            fn_args = variant_args(v, d)
            by, fl = budget(v, s, b)
            try:
                compiled, hoisted, extra, mat = check_compiled(v, progs[v], fn_args, charged_bytes=by)
                fn = lambda: compiled(*fn_args)        # the inspected executable is the timed one
                if args.interpret:
                    vals[v] = float(fn())
                    print(f"{v:16s} S={s:5d} B={b:3d}  chain value {vals[v]:.6g}  (interpret mode, not timed)",
                          flush=True)
                    continue
                r = time_op(fn)
                vals[v] = float(fn())
            except Exception:
                print(f"K1 VARIANT FAIL ({v}) S={s} B={b}:\n{traceback.format_exc()}", flush=True)
                if args.skip_failed_variants:
                    continue
                raise
            per_step = r["median_s"] / CHAIN
            gbs = by / per_step / 1e9
            tflops = fl / per_step / 1e12
            vmem = 0
            if tflops > PEAK_TFLOPS * 1.05:
                print(f"SANITY FAIL: {v} S={s} B={b} {tflops:.0f} TF/s exceeds peak -- work elided; row NOT written",
                      flush=True)
                continue
            if gbs > PLATE_GBS * 1.05:
                if kv_bytes(s, b) < VMEM_BYTES and not hoisted:
                    vmem = 1
                else:
                    print(f"SANITY FAIL: {v} S={s} B={b} {gbs:.0f} GB/s exceeds the HBM plate with a "
                          f"{kv_bytes(s, b) / 1e6:.0f} MB working set{' and hoisted work' if hoisted else ''} -- "
                          f"bytes not moved; row NOT written", flush=True)
                    continue
            row = {"variant": v, **key_base, **r, "per_step_us": per_step * 1e6,
                   "kv_mb": kv_bytes(s, b) / 1e6, "gbs": gbs, "vmem_resident": vmem,
                   "tflops": tflops, "hoisted": hoisted, "body_extra_mb": extra / 1e6, "materialized": mat,
                   "perm_seed": PERM_SEED if v in ("paged_shuffled", "gather_only", "dma_gather") else ""}
            csv_append(args.out, row)
            if not hoisted:
                times[v] = per_step
                if mat:
                    caveats[v] = extra / 1e6
            print(f"{v:16s} S={s:5d} B={b:3d}  {per_step * 1e6:9.2f} us/step  {gbs:7.0f} GB/s  {tflops:6.2f} TF/s"
                  f"{'  [VMEM-resident]' if vmem else ''}{'  [HOISTED -- excluded from derates]' if hoisted else ''}"
                  f"{f'  [MATERIALIZED +{extra / 1e6:.0f} MB/step]' if mat else ''}",
                  flush=True)
            if args.trace:
                td = os.path.join(args.trace, f"K1_{v}_S{s}_B{b}")
                os.makedirs(td, exist_ok=True)
                jax.profiler.start_trace(td)
                jax.block_until_ready(fn())
                jax.profiler.stop_trace()
        # the read variants reduce the same pages: their chain values must agree.
        # (The XLA variants scale the raw read by a loop-carried factor, the DMA
        # variants do not, so the two families are compared within themselves.)
        for fam, members in (("xla", ("gather_only", "gather_seq", "contiguous_read")), ("dma", ("dma_gather", "dma_seq"))):
            rv = {v: vals[v] for v in members if v in vals}
            if len(rv) > 1:
                lo, hi = min(rv.values()), max(rv.values())
                spread = (hi - lo) / (abs(hi) + abs(lo) + 1e-6)
                print(f"read-variant agreement ({fam}) S={s} B={b}: "
                      + "  ".join(f"{k}={x:.6g}" for k, x in rv.items())
                      + f"  -> {'OK' if spread < 1e-3 else 'MISMATCH (a read variant did not cover every page)'}",
                      flush=True)
        if "dma_seq" in vals or "dma_gather" in vals:
            # the DMA sums must equal CHAIN * (sum K + sum V) + CHAIN (the counter), independent of the table
            expect = CHAIN * float(jnp.sum(d["k"].astype(jnp.float32)) + jnp.sum(d["v"].astype(jnp.float32))) + CHAIN
            for v in ("dma_gather", "dma_seq"):
                if v in vals:
                    rel = abs(vals[v] - expect) / (abs(expect) + 1e-6)
                    print(f"dma check {v:10s} S={s} B={b}: chain={vals[v]:.6g} expect={expect:.6g} rel={rel:.2e} -> "
                          f"{'OK' if rel < 1e-3 else 'MISMATCH (the DMA kernel did not read every page)'}", flush=True)
        report_derates(s, b, times, caveats)
        del d, progs


def report_derates(s, b, t, caveats=None):
    caveats = caveats or {}
    parts = []

    def ratio(label, num, den):
        if num in t and den in t:
            note = "".join(f" [{x} materialized +{caveats[x]:.0f} MB/step]" for x in (num, den) if x in caveats)
            parts.append(f"{label}={t[num] / t[den]:.3f}{note}")

    ratio("t_seq/t_dense", "paged_seq", "dense")
    ratio("t_shuffled/t_dense", "paged_shuffled", "dense")
    ratio("t_shuffled/t_seq", "paged_shuffled", "paged_seq")
    ratio("t_dma_gather/t_contig", "dma_gather", "contiguous_read")
    ratio("t_dma_gather/t_dma_seq", "dma_gather", "dma_seq")
    ratio("t_dma_seq/t_contig", "dma_seq", "contiguous_read")
    ratio("t_gather/t_contig", "gather_only", "contiguous_read")
    ratio("t_gather/t_gather_seq", "gather_only", "gather_seq")
    if parts:
        print(f"derates S={s} B={b}: " + "  ".join(parts), flush=True)


def summarize(path):
    import csv
    if not os.path.exists(path):
        return
    cells, caveats = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("hoisted", "0") == "1":
                continue
            key = (int(r["S"]), int(r["B"]))
            cells.setdefault(key, {})[r["variant"]] = float(r["per_step_us"])
            if r.get("materialized", "0") == "1":
                caveats.setdefault(key, {})[r["variant"]] = float(r.get("body_extra_mb", 0) or 0)
    for (s, b), t in sorted(cells.items()):
        report_derates(s, b, t, caveats.get((s, b)))


if __name__ == "__main__":
    main()
