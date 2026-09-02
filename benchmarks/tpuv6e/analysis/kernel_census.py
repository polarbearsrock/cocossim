#!/usr/bin/env python3
"""Kernel census v2: XProf trace -> per-class device time, MXU/HBM utilization
and per-step device time, comparable to the simulator's ACCT output.

Wraps xprof's own converters (pip install xprof) so we never parse xplane.pb
ourselves. Two converters are used per trace:

  framework_op_stats  one row per JAX op name (device rows only). Gives the
                      op `type`, `occurrences` (= the MAX occurrences among the
                      HLO ops carrying that name, i.e. the number of program
                      executions, NOT the number of kernel launches) and
                      `total_self_time` (us). Table is capped at 500 rows, so
                      it is used for op types and as a cross-check only.
  op_profile          the byProgram tree: root -> program -> category ->
                      ["<op> and its duplicate(s)" group ->] HLO op / fusion.
                      Every node carries metrics {rawTime (ps), rawFlops,
                      rawBytesAccessedArray[3], bandwidthUtils[3], flops,
                      occurrences}. Time and utilization come from here.

Confirmed semantics (measured on the session-3 v6e traces, exact to 4 digits
on every node of every trace; see README "Census v2"):

  bandwidthUtils[i] = rawBytesAccessedArray[i] / rawTime / PEAK[i]
      PEAK = [1638 GB/s, 23296 GB/s, 16128 GB/s]
      [0] = HBM read+write   (weight-streaming GEMMs sit at 0.8-0.9 here)
      [1] = on-chip (VMEM) read   [2] = on-chip (VMEM) write
      A GEMM whose weight was prefetched into VMEM by an async `copy-start/
      copy-done` pair shows ~0 in [0] and its weight bytes in [1]; the HBM
      traffic then sits on the copy-done op (bucket `data`).
  flops             = rawFlops / PEAK_FLOPS / root.rawTime   (a SHARE of the
                      whole trace's time-at-peak, NOT the op's utilization;
                      the root node divides by its idle-excluded time)
      PEAK_FLOPS = 946.7 TFLOP/s (bf16)
  MXU utilization of an op or class = sum(rawFlops) / sum(rawTime) / PEAK_FLOPS
  HBM utilization of an op or class = sum(rawBytes[0]) / sum(rawTime) / PEAK[0]
  Both peaks are re-derived from the root node of every trace (they are not
  hard-coded); --selftest asserts them.
  Class utilizations are time-weighted means over the ops that have an XProf
  cost model (`cov` column = the share of the class time they cover; printed
  n/a below 50%): the Pallas ragged_paged_attention custom-call and the DMA
  `*-done` ops report flops = bytes = 0, and the `*-start` ops carry the whole
  DMA byte count at ~1 us duration, so they are excluded (their bytes would
  exceed the peak; XProf clamps them to 1.0). The trace-level HBM utilization
  of the root node, which does include the prefetch DMA traffic, is printed
  beside the table.

Trace completeness (session-3 finding): the stored single-prefill traces
(prefill_{512,2048}_{1,8}) contain ONE execution of a model program with
M = 256 x batch GEMM rows regardless of the prompt length (prefill_512_1 and
prefill_2048_1 even share the program id and the FLOP count) - 256 is the KV
page size of the RPA kernel (`RPAm-p_256-...`), i.e. the signature of a
prefix-cache hit that recomputes only the last page, not a profiler cut.
Their attention time does grow with context. The census prints a WARNING
when M x runs < seq x batch parsed from the point name. Decode windows are
complete (63 steps for 64 generated tokens) and their per-step numbers are
trustworthy; their whole-trace totals include the same partial prefill.

Join op_profile -> framework_op_stats: a unit node's `xla.provenance` is the
JAX op name XLA recorded as the fusion's metadata (trailing ':' stripped),
e.g. `jit(run_model_impl)/JaxLinear/mn,np->mp/dot_general`; HLO ops without
provenance (copy-done.N, custom-call.N, IDLE) appear in framework_op_stats
under their HLO name. The framework row's self time is exactly the sum of its
units (cross-checked per op).

Phase buckets (regex on op name/type, first match wins, attention before
gemm because its wrapper ops contain "dot"):
  gemm        SA GEMM jobs            (dot/matmul/einsum/conv/gemm)
  attention   ragged_paged_attention kernel + its reshape/transpose wrappers
  norm        VPU reduce phases       (rms/layer norm, rsqrt-reduce fusions)
  elementwise VPU broadcast phases    (add/mul/silu/gelu/exp residuals ...)
  data        layout/copy/transpose   (incl. XLA async weight prefetch copies)
  idle        XProf's explicit IDLE node
  other       everything else (report loudly; big 'other' = census gap)

--per-class additionally splits `gemm` into the simulator's op classes,
inferred per HLO unit from the op name and the weight shapes in the HLO
expression (see classify_gemm):
  qkv  (q: JaxEinsum TD,DNH->TNH ; kv: TD,DKH->TKH, k and v are separate)
  o    (JaxEinsum TNH,NHD->TD)
  gate_up / down / mlp_fused  (JaxLinear mn,np->mp; a unit reading only a
       [D,F] weight is gate_up, only an [F,D] weight is down; when XLA fuses
       one D->F projection together with the F->D projection into ONE fusion
       (observed on the batch-1 prefill and the decode programs) the unit is
       reported as mlp_fused - gate/up vs down cannot be separated there)
  head (run_compute_logits TD,DV->TV)
  gemm_other (anything in the gemm bucket not matched above)

Per-step device time: for each program, runs = time-weighted mode of the
`occurrences` of the HLO ops inside it (all equal for an unrolled program),
per-run = rawTime / runs. Which programs form "the step" depends on the point
(parsed from a `prefill|decode_<seq>_<batch>` name; without one the generic
rule applies: n_steps = runs of the most-executed jit_run_model_impl program
and every program that ran >= n_steps times is in the step):
  prefill  the forward is ONE step however vLLM chunked it: with
           max_num_batched_tokens = 8192 a 16384-token prompt batch runs the
           model program twice (M = 8192 each), so the step is the whole trace
           minus IDLE (tier-2 Qwen traces, 2026-09-01).
  decode   vLLM pads the decode token count to a bucket (batch 8 runs as an
           M = 16 program) and, while a long context is still being prefilled
           in 8192-token chunks, interleaves mixed steps in which only part of
           the batch decodes (M = 8192 chunk programs ran as often as the
           decode steps in decode_2048_32, and M = 16 partial-batch steps sat
           beside the M = 32 ones). The step is therefore the model program
           with batch <= M <= 2 x batch that ran most (its runs = n_steps),
           plus the LM-head program with M = batch and the sampling glue that
           ran >= n_steps times; chunk programs and partial-batch programs are
           listed as [not in the step]. IDLE is spread over every model run.
Whole-trace device total and the IDLE share are printed beside the step.
--steady restricts the census itself to that per-step view.

Usage:
  kernel_census.py TRACE [TRACE ...] [--per-class] [--steady] [--csv out.csv]
                   [--top 15] [--selftest]
TRACE is a trace dir (contains plugins/profile/<ts>/*.xplane.pb), a parent
dir holding several trace dirs (each becomes a point named after the dir),
or an xplane.pb path. Paths may be relative to the cwd.
"""
import argparse
import collections
import csv as _csv
import glob
import json
import math
import os
import re
import sys

BUCKETS = [  # first match wins - attention before gemm (its wrappers contain dots)
    ("attention", re.compile(r"ragged|paged|attention|flash", re.I)),
    # conv(?!ert): 'convolution' yes, 'convert_element_type' no
    ("gemm", re.compile(r"dot|matmul|einsum|conv(?!ert)|gemm", re.I)),
    ("norm", re.compile(r"rsqrt|norm|reduce|mean|variance|rms", re.I)),
    ("elementwise", re.compile(
        r"add|mul|sub|div|silu|gelu|exp|tanh|sigmoid|max|min|select|compare", re.I)),
    ("data", re.compile(r"copy|async|transpose|reshape|broadcast|concat|slice|pad|"
                        r"gather|scatter|convert|bitcast|tuple", re.I)),
]
# XProf books a DMA's bytes on its *-start op (issue time ~1us), so those bytes
# would exceed the peak by orders of magnitude; they are excluded from class
# utilization (XProf itself clamps their bandwidthUtils to 1.0).
START_OP = re.compile(r"^(copy|async|send|recv|all-\w+|collective-permute)-start", re.I)
MIN_COV = 0.5  # utilization printed only if >= this share of the class time has a cost model
BUCKET_ORDER = ["gemm", "attention", "norm", "elementwise", "data", "idle", "other"]
GEMM_CLASSES = ["qkv", "o", "gate_up", "down", "mlp_fused", "head", "gemm_other"]
GEMM_SUBS = ["q", "kv"]  # breakdown of qkv (kind=gemm_sub in the CSV)

DEFAULT_TRACE_ROOT = "/data2/s2chitni/.tmp/tpuv6e_results/session3/dh_traces"

# fallbacks only if a root node has no bytes/flops (never seen on real traces)
FALLBACK_PEAK_BW = [1638e9, 23296e9, 16128e9]
FALLBACK_PEAK_FLOPS = 946.7e12


# --------------------------------------------------------------------------- io
def discover_points(args):
    """[path,...] -> [(point_name, [xplane paths])]; see module docstring."""
    points = []
    for a in args:
        a = os.path.abspath(a)
        if os.path.isfile(a):  # name the point after the dir that holds plugins/
            d = os.path.dirname(a)
            while d and d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, "plugins")):
                d = os.path.dirname(d)
            points.append((os.path.basename(d if os.path.isdir(os.path.join(d, "plugins")) else os.path.dirname(a)), [a]))
            continue
        if not os.path.isdir(a):
            sys.exit(f"no such trace path: {a}")
        own = sorted(glob.glob(os.path.join(a, "plugins", "profile", "*", "*.xplane.pb")))
        if own:
            points.append((os.path.basename(a.rstrip("/")), own))
            continue
        subs = []
        for d in sorted(os.listdir(a)):
            p = os.path.join(a, d)
            xp = sorted(glob.glob(os.path.join(p, "plugins", "profile", "*", "*.xplane.pb")))
            if xp:
                subs.append((d, xp))
        if not subs:  # last resort: anything below
            xp = sorted(glob.glob(os.path.join(a, "**", "*.xplane.pb"), recursive=True))
            if xp:
                subs.append((os.path.basename(a.rstrip("/")), xp))
        if not subs:
            sys.exit(f"no *.xplane.pb under {a}")
        points.extend(subs)
    return points


def tool_payload(xplanes, tool):
    from xprof.convert import raw_to_tool_data as r
    out = r.xspace_to_tool_data(list(xplanes), tool, {})
    data = out[0] if isinstance(out, tuple) else out
    if isinstance(data, bytes):
        data = data.decode()
    return json.loads(data)


def rows_from_table(table_json):
    """Google-visualization table -> list of dicts."""
    cols = [c["id"] for c in table_json["cols"]]
    for r in table_json.get("rows", []):
        yield dict(zip(cols, [c.get("v") for c in r["c"]]))


def framework_rows(xplanes):
    payload = tool_payload(xplanes, "framework_op_stats")
    table = payload[0] if isinstance(payload, list) else payload
    rows = {}
    for row in rows_from_table(table):
        if str(row.get("host_or_device", "")).lower().startswith("host"):
            continue
        rows[str(row.get("operation") or "")] = row
    return rows


# ------------------------------------------------------------------ op_profile
def _m(node):
    m = node.get("metrics", {})
    b = list(m.get("rawBytesAccessedArray") or [0, 0, 0]) + [0, 0, 0]
    return dict(time=float(m.get("rawTime", 0)), flops=float(m.get("rawFlops", 0)),
                bytes=[float(x) for x in b[:3]], occ=int(m.get("occurrences", 0)),
                share=float(m.get("flops", 0)), bw=list(m.get("bandwidthUtils") or [0, 0, 0]),
                norm_time=float(m.get("normalizedTimePs", 0)))


class Unit:
    """One HLO op / fusion (or a synthetic residual) from the op_profile tree."""
    __slots__ = ("name", "prov", "category", "program", "expr", "time", "flops",
                 "bytes", "occ", "share", "synthetic", "bucket", "gclass", "gsub")

    def __init__(self, name, prov, category, program, expr, met, synthetic=False):
        self.name, self.prov, self.category, self.program, self.expr = name, prov, category, program, expr
        self.time, self.flops, self.bytes, self.occ, self.share = met["time"], met["flops"], met["bytes"], met["occ"], met["share"]
        self.synthetic = synthetic
        self.bucket = self.gclass = self.gsub = None

    @property
    def key(self):  # join key into framework_op_stats
        return self.prov.rstrip(":") if self.prov else self.name


DUP_SUFFIX = " and its duplicate(s)"


def collect_units(root):
    """Walk byProgram: root -> program -> category -> [dup group ->] op.
    Returns (units, programs) where programs = {name: metrics dict}.
    Children lists are capped by xprof (100 per node): the uncovered residual
    of every parent is attributed to a synthetic unit carrying the parent's
    category so that sum(units) == root time exactly."""
    units, programs = [], {}

    def residual(parent, kids, prog, prov, category, expr):
        rest = _m(parent)["time"] - sum(_m(k)["time"] for k in kids)
        if rest > 0.5 * _m(parent)["time"] * 1e-6 + 1:  # > ~0 ps
            met = _m(parent)
            ratio = rest / met["time"] if met["time"] else 0.0
            met = dict(met, time=rest, flops=met["flops"] * ratio,
                       bytes=[b * ratio for b in met["bytes"]], occ=0, share=met["share"] * ratio)
            units.append(Unit(parent["name"] + " [residual]", prov, category, prog, expr, met, synthetic=True))

    def unit_of(node, prog, category):
        x = node.get("xla", {})
        return Unit(node["name"], x.get("provenance", ""), x.get("category", category) or category,
                    prog, x.get("expression", ""), _m(node))

    for prog in root.get("children", []):
        pname = prog["name"]
        programs[pname] = _m(prog)
        if pname == "IDLE" or not prog.get("children"):
            units.append(unit_of(prog, pname, "IDLE" if pname == "IDLE" else ""))
            continue
        for cat in prog["children"]:
            cname = cat["name"]
            kids = cat.get("children") or []
            if "xla" in cat and not kids:  # a category-less op directly under the program
                units.append(unit_of(cat, pname, cname))
                continue
            for node in kids:
                if node["name"].endswith(DUP_SUFFIX):
                    members = node.get("children") or []
                    for mem in members:
                        units.append(unit_of(mem, pname, cname))
                    prov = members[0].get("xla", {}).get("provenance", "") if members else ""
                    residual(node, members, pname, prov, cname, node.get("xla", {}).get("expression", ""))
                else:
                    units.append(unit_of(node, pname, cname))
            residual(cat, kids, pname, "", cname, "")
    return units, programs


def derive_peaks(root):
    """Peak rates implied by the root node (bytes/time/util and flops)."""
    m = _m(root)
    t = m["time"] * 1e-12
    peaks = []
    for i in range(3):
        u = m["bw"][i] if i < len(m["bw"]) else 0
        peaks.append(m["bytes"][i] / t / u if (u > 0 and t > 0) else FALLBACK_PEAK_BW[i])
    nt = m["norm_time"] * 1e-12
    pf = m["flops"] / (m["share"] * nt) if (m["share"] > 0 and nt > 0) else FALLBACK_PEAK_FLOPS
    return peaks, pf


# ------------------------------------------------------------- classification
SHAPE_RX = re.compile(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?")
OPERAND_RX = re.compile(r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+(%[^\s,()]+)")
OPCODE_RX = re.compile(r"\s([a-z][a-z0-9_\-]*)\(")


def parse_expr(expr):
    """HLO expression -> (out_shapes [(dtype, dims)], operands [(dtype, dims, name)])."""
    if " = " not in expr:
        return [], []
    _, rhs = expr.split(" = ", 1)
    mo = OPCODE_RX.search(rhs)
    lhs = rhs[:mo.start()] if mo else rhs
    body = rhs[mo.end():] if mo else ""
    outs = [(d, tuple(int(v) for v in dims.split(",") if v)) for d, dims in SHAPE_RX.findall(lhs)]
    ops = [(d, tuple(int(v) for v in dims.split(",") if v), nm) for d, dims, nm in OPERAND_RX.findall(body)]
    return outs, ops


def rows_of(expr):
    """Leading dim of the first >=2-D output of a GEMM unit (its M)."""
    outs, _ = parse_expr(expr)
    for d, dims in outs:
        if len(dims) >= 2:
            return dims[0]
    return None


def bucket_of(op_type, op_name):
    for name, rx in BUCKETS:
        if rx.search(op_name) or rx.search(op_type):
            return name
    return "other"


def infer_model_dim(units):
    """D from the q/o einsum weights ([D,N,H] / [N,H,D]); fallback: the
    smaller dim of the JaxLinear weights (F > D for every LLM we run)."""
    votes = collections.Counter()
    for u in units:
        k = u.key
        if "TD,DNH->TNH" in k or "TD,DKH->TKH" in k:
            for _, dims, _ in parse_expr(u.expr)[1]:
                if len(dims) == 3:
                    votes[dims[0]] += u.time
        elif "TNH,NHD->TD" in k:
            for _, dims, _ in parse_expr(u.expr)[1]:
                if len(dims) == 3:
                    votes[dims[2]] += u.time
    if votes:
        return votes.most_common(1)[0][0]
    for u in units:
        if "JaxLinear" in u.key or "ParallelLinear" in u.key:
            for w in gemm_weights(u):
                votes[min(w)] += u.time
    return votes.most_common(1)[0][0] if votes else None


MODEL_PROG = ("jit_run_model_impl", "jit_step_fun_impl")          # JAX-native, torch-wrapper
LOGITS_PROG = ("jit_run_compute_logits", "jit_compute_logits_func")
GEMM_KEYS = ("JaxLinear", "JaxEinsum", "run_compute_logits", "ParallelLinear", "compute_logits_func")


def gemm_weights(u):
    """2-D weight operands of a unit: named %state_leaves_* (flax params), or
    any 2-D operand that is not an [M, *] activation."""
    outs, ops = parse_expr(u.expr)
    M = rows_of(u.expr)
    ws = []
    for _, dims, nm in ops:
        if len(dims) != 2:
            continue
        if nm.startswith("%state_leaves") or (dims[0] != M and min(dims) >= 128):
            ws.append(dims)
    return ws


def classify_gemm(u, D):
    """Simulator op class of a gemm-bucket unit (see module docstring).
    Two vLLM model paths are recognised: the native JAX models (Qwen3:
    JaxLinear / JaxEinsum provenance, program jit_run_model_impl) and the
    PyTorch-wrapper path (Mistral-7B ran through it, 2026-09-01: program
    jit_step_fun_impl, provenance VllmQKVParallelLinear /
    MergedColumnParallelLinear (gate+up) / VllmRowParallelLinear (o or down,
    told apart by the weight shape) / compute_logits_func)."""
    k = u.key
    if "run_compute_logits" in k or "TD,DV->TV" in k or "compute_logits_func" in k:
        return "head", "head"
    if "TD,DNH->TNH" in k:
        return "qkv", "q"
    if "TD,DKH->TKH" in k:
        return "qkv", "kv"
    if "TNH,NHD->TD" in k:
        return "o", "o"
    if "QKVParallelLinear" in k:
        return "qkv", "qkv_merged"
    if "MergedColumnParallelLinear" in k:
        return "gate_up", "gate_up"
    if "RowParallelLinear" in k:
        ws = gemm_weights(u)
        if any(w[0] == D and w[1] == D for w in ws):
            return "o", "o"
        if any(w[1] == D and w[0] != D for w in ws):
            return "down", "down"
        return "gemm_other", "row_parallel_unsplit"
    if "JaxLinear" in k or "mn,np->mp" in k:
        ws = gemm_weights(u)
        ins = [w for w in ws if w[0] == D and w[1] != D]
        outs = [w for w in ws if w[1] == D and w[0] != D]
        if ins and outs:
            return "mlp_fused", "mlp_fused"
        if ins:
            return "gate_up", "gate_up"
        if outs:
            return "down", "down"
        return "gemm_other", "mlp_unsplit"
    return "gemm_other", "gemm_other"


# ------------------------------------------------------------------- census
class ClassAcc:
    """Time-weighted accumulator: utilization = sum(work)/sum(covered time)."""
    __slots__ = ("time", "flops", "hbm", "vmem_rd", "occ", "cov_time", "n")

    def __init__(self):
        self.time = self.flops = self.hbm = self.vmem_rd = self.cov_time = 0.0
        self.occ = self.n = 0

    def add(self, u, scale=1.0):
        self.time += u.time * scale
        self.occ += u.occ
        self.n += 1
        if (u.flops > 0 or u.bytes[0] > 0) and not START_OP.match(u.name):
            self.flops += u.flops * scale
            self.hbm += u.bytes[0] * scale
            self.vmem_rd += u.bytes[1] * scale
            self.cov_time += u.time * scale


def program_runs(units, programs):
    """program -> (runs, rows M). runs = time-weighted mode of unit occurrences."""
    occ = collections.defaultdict(collections.Counter)
    rows = collections.defaultdict(collections.Counter)
    for u in units:
        if u.occ > 0:
            occ[u.program][u.occ] += u.time
        if u.expr and any(g in u.key for g in GEMM_KEYS):
            M = rows_of(u.expr)
            if M:
                rows[u.program][M] += u.time
    out = {}
    for p in programs:
        runs = occ[p].most_common(1)[0][0] if occ[p] else 0
        M = rows[p].most_common(1)[0][0] if rows[p] else None
        out[p] = (runs, M)
    return out


def census(point, xplanes, steady=False):
    fw = framework_rows(xplanes)
    op = tool_payload(xplanes, "op_profile")
    root = op["byProgram"]
    peaks, peak_flops = derive_peaks(root)
    units, programs = collect_units(root)
    D = infer_model_dim(units)
    runs = program_runs(units, programs)

    model_progs = [p for p in programs if p.startswith(MODEL_PROG)]
    n_steps = max((runs[p][0] for p in model_progs), default=0)
    idle_time = programs.get("IDLE", {}).get("time", 0.0)
    total_time = _m(root)["time"]
    mo = re.match(r"(prefill|decode)_(\d+)_(\d+)", point or "")
    mode = mo.group(1) if mo else None
    batch = int(mo.group(3)) if mo else None
    step_prog = None            # decode: the padded-batch model program that IS the step
    total_model_runs = sum(runs[p][0] for p in model_progs) or 1
    if mode == "prefill":
        n_steps = 1             # one forward, however many chunks vLLM ran
    elif mode == "decode":
        dec = [p for p in model_progs if runs[p][1] is not None and batch <= runs[p][1] <= 2 * batch]
        if dec:
            step_prog = max(dec, key=lambda p: runs[p][0])
            n_steps = runs[step_prog][0]

    def prog_scale(name):
        """1/runs for programs in the step, 0 for the rest (see module doc)."""
        r, M = runs.get(name, (0, None))
        if mode == "prefill":
            return 1.0
        if step_prog is not None:
            if name.startswith(MODEL_PROG):
                return 1.0 / r if name == step_prog else 0.0
            if name.startswith(LOGITS_PROG):
                return 1.0 / r if (M == batch and r >= n_steps) else 0.0
        return (1.0 / r) if (r and r >= n_steps) else 0.0

    # per-unit classification
    for u in units:
        ftype = str(fw.get(u.key, {}).get("type") or u.category or "")
        u.bucket = "idle" if (u.key == "IDLE" or u.category == "IDLE") else bucket_of(ftype, u.key)
        u.gclass, u.gsub = classify_gemm(u, D) if u.bucket == "gemm" else (None, None)

    def scale_of(u):
        if not steady:
            return 1.0
        if u.bucket == "idle":
            return 1.0 if mode == "prefill" else 1.0 / (total_model_runs if mode == "decode" else (n_steps or 1))
        return prog_scale(u.program)

    buckets = collections.defaultdict(ClassAcc)
    gclasses = collections.defaultdict(ClassAcc)
    gsubs = collections.defaultdict(ClassAcc)
    per_op = collections.defaultdict(ClassAcc)
    for u in units:
        s = scale_of(u)
        if s == 0.0:
            continue
        buckets[u.bucket].add(u, s)
        per_op[(u.bucket, u.key)].add(u, s)
        if u.bucket == "gemm":
            gclasses[u.gclass].add(u, s)
            gsubs[u.gsub].add(u, s)
    grand = sum(a.time for a in buckets.values())

    # join cross-check vs framework_op_stats (whole trace only)
    joined = collections.defaultdict(float)
    for u in units:
        if not u.synthetic:
            joined[u.key] += u.time
    fw_total = sum(float(r.get("total_self_time") or 0) for r in fw.values())
    matched = sum(joined[k] for k in joined if k in fw) * 1e-6
    mism = [(k, joined[k] * 1e-6, float(fw[k]["total_self_time"])) for k in joined
            if k in fw and abs(joined[k] * 1e-6 - float(fw[k]["total_self_time"])) > 1.0]
    synthetic_time = sum(u.time for u in units if u.synthetic)

    return dict(point=point, xplanes=xplanes, root=_m(root), peaks=peaks, peak_flops=peak_flops,
                units=units, programs=programs, runs=runs, n_steps=n_steps, model_progs=model_progs,
                mode=mode, batch=batch, step_prog=step_prog, prog_scale=prog_scale,
                idle_time=idle_time, total_time=total_time, grand=grand, D=D, steady=steady,
                buckets=buckets, gclasses=gclasses, gsubs=gsubs, per_op=per_op, fw=fw,
                join=dict(fw_total_us=fw_total, matched_us=matched, mismatches=mism,
                          unit_time_us=sum(u.time for u in units) * 1e-6,
                          synthetic_us=synthetic_time * 1e-6))


# ------------------------------------------------------------------ reporting
def util(acc, peak, which, min_cov=MIN_COV):
    """Mean utilization over the covered time; None if coverage < min_cov."""
    if acc.time <= 0 or acc.cov_time <= 0 or acc.cov_time < min_cov * acc.time:
        return None
    num = {"mxu": acc.flops, "hbm": acc.hbm, "vmem_rd": acc.vmem_rd}[which]
    return num / (acc.cov_time * 1e-12) / peak


def fmt_u(v):
    return "  n/a" if v is None else f"{v:5.3f}"


def class_rows(res):
    """[(kind, class, acc)] in print/CSV order."""
    rows = [("bucket", b, res["buckets"][b]) for b in BUCKET_ORDER if b in res["buckets"]]
    for b in sorted(res["buckets"]):
        if b not in BUCKET_ORDER:
            rows.append(("bucket", b, res["buckets"][b]))
    rows += [("gemm_class", c, res["gclasses"][c]) for c in GEMM_CLASSES if c in res["gclasses"]]
    rows += [("gemm_sub", c, res["gsubs"][c]) for c in GEMM_SUBS if c in res["gsubs"]]
    for c in sorted(res["gsubs"]):
        if c not in GEMM_SUBS and c not in GEMM_CLASSES and c is not None:
            rows.append(("gemm_sub", c, res["gsubs"][c]))
    return rows


def metrics_of(res, acc):
    mx = util(acc, res["peak_flops"], "mxu")
    hb = util(acc, res["peaks"][0], "hbm")
    cov = acc.cov_time / acc.time if acc.time else 0.0
    return acc.time * 1e-9, (acc.time / res["grand"] if res["grand"] else 0.0), mx, hb, acc.occ, cov


def print_report(res, per_class, top):
    p, r = res["point"], res["root"]
    scope = "per-step (--steady)" if res["steady"] else "whole trace"
    print(f"\n=== {p}  [{scope}]")
    print(f"device total {res['total_time']/1e9:.3f} ms  idle {res['idle_time']/1e9:.3f} ms "
          f"({100*res['idle_time']/res['total_time']:.2f}%)  peaks: HBM {res['peaks'][0]/1e9:.0f} GB/s, "
          f"VMEM rd {res['peaks'][1]/1e9:.0f}, VMEM wr {res['peaks'][2]/1e9:.0f} GB/s, "
          f"MXU {res['peak_flops']/1e12:.1f} TFLOP/s;  D={res['D']}")
    print(f"trace-level (root node, all traffic incl. async prefetch DMAs): HBM util {r['bw'][0]:.3f}, "
          f"MXU util {r['flops']/(r['time']*1e-12)/res['peak_flops']:.3f}")
    print(f"join: {len(res['units'])} op_profile units = {res['join']['unit_time_us']/1e3:.3f} ms "
          f"(residual/synthetic {res['join']['synthetic_us']/1e3:.3f} ms); framework_op_stats "
          f"{res['join']['fw_total_us']/1e3:.3f} ms, matched by name {res['join']['matched_us']/1e3:.3f} ms, "
          f"{len(res['join']['mismatches'])} ops disagree >1us")

    # programs / per-step
    print("programs (per-run = rawTime/runs; runs = occurrences of the HLO ops inside):")
    step = 0.0
    n = res["n_steps"]
    for name, m in sorted(res["programs"].items(), key=lambda kv: -kv[1]["time"]):
        runs, M = res["runs"][name]
        if name == "IDLE":
            print(f"  {'IDLE':56s} total {m['time']/1e9:10.3f} ms   ({100*m['time']/res['total_time']:.2f}% of device time)")
            continue
        per = m["time"] / runs if runs else float("nan")
        s = res["prog_scale"](name) if runs else 0.0
        in_step = s > 0
        if in_step:
            step += m["time"] * s
        tag = "" if in_step else "  [not in the step]"
        if in_step and name == res["step_prog"]:
            tag = "  [THE decode step program]"
        print(f"  {name[:56]:56s} total {m['time']/1e9:10.3f} ms   runs {runs:4d}   per-run {per/1e9:9.3f} ms"
              f"   M={M if M is not None else '-'}{tag}")
    rule = ("prefill: one forward = whole trace minus IDLE" if res["mode"] == "prefill" else
            f"decode: model program with {res['batch']}<=M<=2x{res['batch']} that ran most (n_steps={n}) + its LM head/glue"
            if res["step_prog"] else f"generic: programs with runs>={n}")
    print(f"step: {step/1e9:.3f} ms device time per forward/step [{rule}]; whole trace {res['total_time']/1e9:.3f} ms "
          f"incl. idle {res['idle_time']/1e9:.3f} ms")
    res["step_ms"] = step / 1e9

    # completeness hint from the point name
    mo = re.match(r"(prefill|decode)_(\d+)_(\d+)$", p)
    if mo and res["model_progs"]:
        mode, seq, batch = mo.group(1), int(mo.group(2)), int(mo.group(3))
        best = max(res["model_progs"], key=lambda q: res["runs"][q][0])
        runs_b, M_b = res["runs"][best]
        if mode == "prefill" and M_b is not None and M_b * runs_b < seq * batch:
            print(f"WARNING: GEMM rows in the trace = {M_b} x {runs_b} run(s) < {seq*batch} = seq x batch. "
                  f"This is NOT a full prefill (prefix-cache hit or truncated window); totals are not a forward.")
        if mode == "decode":
            if M_b is not None and M_b < batch:
                print(f"WARNING: decode program rows M={M_b} < batch {batch}")
            for q in res["model_progs"]:
                rq, Mq = res["runs"][q]
                if q != best and Mq is not None and Mq * rq < seq * batch:
                    print(f"note: one-off prefill program {q[:40]} covers {Mq} x {rq} GEMM rows < {seq*batch} "
                          f"= ctx x batch (partial prefill in the window); use the per-step numbers or --steady.")

    # classes
    hdr = f"{'class':14s} {'time_ms':>10s} {'share':>7s} {'MXU':>6s} {'HBM':>6s} {'occ':>7s} {'cov':>5s}"
    print(hdr)
    for kind, cls, acc in class_rows(res):
        if kind != "bucket" and not per_class:
            continue
        t, sh, mx, hb, occ, cov = metrics_of(res, acc)
        ind = {"bucket": "", "gemm_class": "  ", "gemm_sub": "    "}[kind]
        print(f"{ind}{cls:{14-len(ind)}s} {t:10.3f} {100*sh:6.1f}% {fmt_u(mx):>6s} {fmt_u(hb):>6s} {occ:7d} {cov:5.2f}")
    if per_class:
        fused = res["gclasses"].get("mlp_fused")
        if fused and fused.time > 0:
            print("  note: mlp_fused = one D->F projection fused with the F->D projection in a single HLO fusion; "
                  "gate/up vs down are not separable there (gate_up+down+mlp_fused = whole MLP).")
        att = res["buckets"].get("attention")
        if att and att.cov_time < att.time:
            print(f"  note: only {100*att.cov_time/att.time:.0f}% of attention time has an XProf cost model "
                  "(the Pallas ragged_paged_attention custom-call reports flops/bytes = 0); MXU/HBM are n/a "
                  f"below {int(100*MIN_COV)}% coverage. DMA -done ops (weight prefetch) have no HBM bytes either.")

    if top:
        print(f"top {top} ops:")
        for (b, name), acc in sorted(res["per_op"].items(), key=lambda kv: -kv[1].time)[:top]:
            t, sh, mx, hb, occ, cov = metrics_of(res, acc)
            print(f"  {t:9.3f} ms {100*sh:5.1f}%  [{b:11s}] MXU {fmt_u(mx)} HBM {fmt_u(hb)}  {name[:78]}")


CSV_COLS = ["point", "class", "time_ms", "share", "mxu_util", "hbm_util", "occurrences",
            "kind", "costmodel_cov", "steps", "scope"]


def write_csv(path, results, per_class):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(CSV_COLS)
        for res in results:
            for kind, cls, acc in class_rows(res):
                if kind != "bucket" and not per_class:
                    continue
                t, sh, mx, hb, occ, cov = metrics_of(res, acc)
                w.writerow([res["point"], cls, f"{t:.6f}", f"{sh:.6f}",
                            "" if mx is None else f"{mx:.4f}", "" if hb is None else f"{hb:.4f}",
                            occ, kind, f"{cov:.3f}", res["n_steps"], "step" if res["steady"] else "trace"])
    print(f"\nwrote {path}")


# -------------------------------------------------------------------- selftest
def selftest(trace_root):
    """Assertions on the stored prefill_512_1 (and, if present, decode_512_8)."""
    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            raise SystemExit("selftest failed: " + msg)

    pts = dict(discover_points([os.path.join(trace_root, "prefill_512_1")]))
    name, xp = next(iter(pts.items()))
    res = census(name, xp)
    print_report(res, per_class=True, top=0)
    print("selftest assertions (prefill_512_1):")
    shares = sum(a.time for a in res["buckets"].values()) / res["grand"]
    check(abs(shares - 1.0) < 1e-6, f"bucket shares sum to 1 (got {shares:.9f})")
    check(abs(res["join"]["unit_time_us"] * 1e6 - res["total_time"]) < 1e-5 * res["total_time"],
          f"sum of op_profile units ({res['join']['unit_time_us']/1e3:.6f} ms) == root device time "
          f"({res['total_time']/1e9:.6f} ms)")
    check(res["join"]["matched_us"] > 0.99 * res["join"]["fw_total_us"] and not res["join"]["mismatches"],
          f"join covers {100*res['join']['matched_us']/res['join']['fw_total_us']:.2f}% of framework time, no per-op disagreement")
    # bandwidth-triple semantics
    p = res["peaks"]
    check(abs(p[0] - 1638e9) / 1638e9 < 0.01, f"bandwidthUtils[0] denominator = {p[0]/1e9:.0f} GB/s (HBM, 1638)")
    check(abs(p[1] - 23296e9) / 23296e9 < 0.01, f"bandwidthUtils[1] denominator = {p[1]/1e9:.0f} GB/s (VMEM read)")
    check(abs(p[2] - 16128e9) / 16128e9 < 0.01, f"bandwidthUtils[2] denominator = {p[2]/1e9:.0f} GB/s (VMEM write)")
    check(abs(res["peak_flops"] - 946.7e12) / 946.7e12 < 0.01, f"flops share denominator = {res['peak_flops']/1e12:.1f} TFLOP/s")
    # per-node check of the two formulas on the heaviest GEMM unit
    gem = max((u for u in res["units"] if u.bucket == "gemm" and u.flops > 0), key=lambda u: u.time)
    bw0 = gem.bytes[0] / (gem.time * 1e-12) / p[0]
    node_bw = next(iter(_m(nd)["bw"][0] for nd in _walk_nodes(res, gem.name)), None)
    check(node_bw is not None and abs(bw0 - node_bw) < 1e-3,
          f"unit {gem.name}: bytes[0]/time/1638GB/s = {bw0:.3f} == reported bandwidthUtils[0] {node_bw}")
    node_share = gem.share
    calc_share = gem.flops / res["peak_flops"] / (res["total_time"] * 1e-12)
    check(abs(node_share - calc_share) < 1e-6,
          f"unit {gem.name}: `flops` {node_share:.5f} == rawFlops/peak/root_time {calc_share:.5f} (a share, not a utilization)")
    g = res["buckets"]["gemm"]
    gm, gh = util(g, res["peak_flops"], "mxu"), util(g, p[0], "hbm")
    check(gm is not None and gm > 0.3, f"gemm MXU utilization {gm:.3f} > 0.3")
    check(gh is not None and gh > 0.5, f"gemm HBM utilization (entry 0) {gh:.3f} > 0.5 (observed 0.69)")
    wsm = res["gclasses"].get("mlp_fused") or res["gclasses"].get("gate_up")
    e0, e1 = util(wsm, p[0], "hbm"), util(wsm, p[1], "vmem_rd")
    check(e0 is not None and e0 > 0.5 and e1 < 0.1,
          f"weight-streaming MLP GEMMs: entry0 (HBM) {e0:.3f} > 0.5, entry1 (VMEM rd) {e1:.3f} < 0.1")
    qa = res["gsubs"]["q"]
    q0 = util(qa, p[0], "hbm")
    check(q0 < 0.2 and qa.vmem_rd > 10 * qa.hbm,
          f"q projection (weight prefetched into VMEM by copy-done): entry0 (HBM) util {q0:.3f} < 0.2, "
          f"bytes[1] (VMEM rd) {qa.vmem_rd/qa.n/1e6:.1f} MB/call > 10x bytes[0] (HBM) {qa.hbm/qa.n/1e6:.1f} MB/call")
    idle = res["buckets"]["idle"]
    check(idle.flops == 0 and idle.hbm == 0 and idle.time > 0, "idle utilization == 0")
    check(res["n_steps"] == 1, f"prefill: model program ran once (n_steps={res['n_steps']})")
    fo = res["fw"].get("jit(run_model_impl)/JaxLinear/mn,np->mp/dot_general", {})
    check(float(fo.get("occurrences", -1)) == 1.0, "framework_op_stats occurrences(JaxLinear) == 1 == program runs")
    gc = res["gclasses"]
    check(all(gc[c].time > 0 for c in ("qkv", "o", "head")) and (gc.get("gate_up") or gc.get("mlp_fused")),
          "gemm classes qkv/o/head/gate_up|mlp_fused all present")
    check(abs(sum(a.time for a in gc.values()) - g.time) < 1, "gemm classes sum to the gemm bucket")

    dec = os.path.join(trace_root, "decode_512_8")
    if os.path.isdir(dec):
        name, xp = next(iter(dict(discover_points([dec])).items()))
        rd = census(name, xp)
        print_report(rd, per_class=True, top=0)
        print("selftest assertions (decode_512_8):")
        check(rd["n_steps"] == 63, f"decode window: 63 decode steps (n_steps={rd['n_steps']})")
        check(0 < rd["step_ms"] < rd["total_time"] / 1e9 / 63 * 1.05,
              f"per-step {rd['step_ms']:.3f} ms below whole-trace/63 (prefill one-off excluded)")
        rs = census(name, xp, steady=True)
        sh = sum(a.time for a in rs["buckets"].values()) / rs["grand"]
        check(abs(sh - 1.0) < 1e-6, "steady-state bucket shares sum to 1")
        steady_ms = (sum(a.time for a in rs["buckets"].values()) - rs["buckets"]["idle"].time) * 1e-9
        check(abs(steady_ms - rd["step_ms"]) < 1e-3,
              f"--steady class times (minus idle/63) sum to the step ({steady_ms:.3f} vs {rd['step_ms']:.3f} ms)")
    print("selftest passed")


def _walk_nodes(res, name):
    """Yield op_profile nodes with this name (re-reads the tree)."""
    op = tool_payload(res["xplanes"], "op_profile")

    def walk(n):
        if n.get("name") == name:
            yield n
        for c in n.get("children") or []:
            yield from walk(c)
    yield from walk(op["byProgram"])


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="trace dir(s), parent dir, or xplane.pb path(s)")
    ap.add_argument("--per-class", action="store_true", help="split gemm into simulator op classes")
    ap.add_argument("--steady", action="store_true",
                    help="per-step census: divide each program by its run count, drop one-off programs")
    ap.add_argument("--csv", default=None, help="write one row per (point, class)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--trace-root", default=DEFAULT_TRACE_ROOT, help="where --selftest finds prefill_512_1")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.trace_root)
        return
    if not args.traces:
        ap.error("give at least one trace path (or --selftest)")
    results = []
    for name, xp in discover_points(args.traces):
        res = census(name, xp, steady=args.steady)
        print_report(res, args.per_class, args.top)
        results.append(res)
    if args.csv:
        write_csv(args.csv, results, args.per_class)


if __name__ == "__main__":
    main()
