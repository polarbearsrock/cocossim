# TPU v6e measurement harness (spec §4, Plan 2)

Measures a real `v6e-1` to calibrate and validate the COCOSSim TPUv6e model
(`configs/tpuv6e.sh`). Session 0 (2026-08-31) validated the full pipeline:
provision → vLLM offline serve (Qwen3-8B) → JAX profiler trace → kernel
census → teardown, ~$3.

## Layout

- `env.sh` — shared config (project/zone/VM name), `tpu_ssh`/`tpu_scp` helpers
- `provision.sh` / `setup_vm.sh` / `teardown.sh` — VM lifecycle (idempotent)
- `probes/` — raw-JAX microbenchmarks (session 2): C1 array rows, C3
  saturation TFLOPS, C4 stream bandwidth, C5 VMEM cliff, B1 elementwise
  roofline + overhead intercept. All support `--dry-run` locally and are
  resumable (points already in the CSV are skipped). Tier-1 chained probes
  (spec 3.1): `g_sweep.py` (G1-G3), `e1_chained.py` (E1),
  `a1_attention.py` (A1: Pallas `flash_attention` prefill, MHA nh=nkv=32,
  and Pallas `paged_attention` decode, GQA 32/8, page 16, sequential pages;
  `--probe-api` prints the resolved kernel signatures and a CPU shape smoke
  test; `--fallback-xla` is the only fallback and is off by default). Its
  scan carry is the kernel output itself (`--carry out`; `--carry sum` is the
  first version's reduce+broadcast glue, which streams 3 full arrays per
  step outside the kernel and is kept only for comparison). Two byte
  columns: `bytes_mb` is the algorithmic minimum (q+k+v+out, the simulator
  counterpart) and `hbm_bytes_mb` is what the kernel moves --
  `flash_attention`'s kv-innermost grid re-fetches K/V `kv_fetch_factor`
  times per head (2.25x at S=2048, 8.44x at S=8192 at block 512), so
  `hbm_gbs`, the plate gate and the dry-run `exp_us` use it. With `--trace`
  each point's directory carries the kernel config
  (`A1_prefill_S2048_B1_bq512_bk512`) and, when `xprof` is importable, the
  row gets `kernel_us` / `glue_us` / `kernel_gbs` / `kernel_tflops` from the
  trace's `pallas_call` rows; otherwise `--annotate --out CSV --trace DIR`
  fills them offline into `<out>.kernel.csv`. The trace is captured before
  the sanity gate, so rows refused into `<out>.rejected.csv` keep their
  trace directory and `kernel_*` columns (the trace is the arbiter for
  exactly those rows; `--annotate --out <out>.rejected.csv` works too).
  Read the columns as: `per_step_us` and its `gbs` / `tflops` / `mfu` are
  wall-clock per step and include the XLA glue around the kernel;
  `kernel_*` are the kernel's own device time and are primary when present
  (`attribution` = `xprof` vs `wall`). Two switches isolate that glue:
  `--chain-form scan,unroll` runs the chain as `lax.scan` (whose while
  loop copies the [B,nh,S,hd] output into the loop-carry buffer every
  step -- a custom call cannot write over its own operand) and as a Python
  loop of CHAIN calls in one jit (no carry, no copy); their `per_step_us`
  difference is that copy. `--decode-q-dtype f32` (default) generates the
  decode q/carry in float32 so `paged_attention`'s wrapper converts
  (bf16->f32 before the kernel, f32->bf16 after, ~7-9 us launch cost each)
  become no-ops; `--probe-api` prints the per-step primitives of each
  chain body so the remaining glue (two reshapes) is visible before the
  session.
  Also `k1_kv_gather.py`, i.e. `k1_kv_gather.py` (K1: `jax.experimental.pallas.ops.tpu.paged_attention`
  sequential vs shuffled block table vs dense XLA vs raw `jnp.take`
  gather/contiguous read; `--probe-api` prints the resolved kernel
  signature and a layout/shape smoke test without a TPU; every compiled
  program is checked for work hoisted out of the scan loop).
- `holdout/dh_offline.py` — fixed-shape Qwen3-8B points via vLLM offline
  mode; maps 1:1 onto simulator `Transformer` runs. `--trace-dir` captures an
  xplane per point.
- `analysis/kernel_census.py` — xplane trace → phase-class breakdown
  (gemm/attention/norm/elementwise/data/idle/other), per-class MXU/HBM
  utilization, simulator op classes (`--per-class`), per-step device time,
  CSV export; see "Census v2" below. Local venv:
  `/data2/s2chitni/venvs/tpu-analysis` (`pip install xprof`).

## Session runbook

```
source benchmarks/tpuv6e/env.sh
benchmarks/tpuv6e/provision.sh --if-needed
# push probes and run (example: session 2)
tpu_scp benchmarks/tpuv6e/probes "$TPU_NAME":~/probes
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && python c1_msweep.py"
...
tpu_scp "$TPU_NAME":~/probes/'*.csv' "$RESULTS_DIR"/
benchmarks/tpuv6e/teardown.sh        # ALWAYS - on-demand bills ~$3/hr
```

## Hard-won facts (do not rediscover)

- Home dir is quota-full: gcloud config + SSH key live under
  `/data2/s2chitni/.gcloud` (`env.sh` sets both). ssh known-hosts writes fail
  harmlessly.
- Quota map (project `rta-tpu-research`): on-demand v6e in **us-east5** only;
  us-east1-d is spot-only (spot pool preempted 2x/40min on 2026-08-31);
  europe-west4 blocked by org policy. User ruling: on-demand for all sessions.
- The TPU VM's system Python 3.10 cannot build `vllm-tpu`; use `uv` + py3.12
  venv (`setup_vm.sh`).
- vLLM v1 engine owns the TPU in a subprocess: profiling requires
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in `dh_offline.py`). A shutdown
  `AttributeError` after `stop_trace` is cosmetic.
- Launching detached over ssh needs `< /dev/null` + `disown` or the session
  hangs.

## Census v2 (`analysis/kernel_census.py`, spec S5)

```
source /data2/s2chitni/venvs/tpu-analysis/bin/activate
kernel_census.py $RESULTS_DIR/dh_traces --per-class --csv census.csv   # every point under a parent dir
kernel_census.py $RESULTS_DIR/dh_traces/decode_512_8 --per-class --steady   # per-step view of a decode window
kernel_census.py --selftest                                             # asserts on the stored prefill_512_1 (+decode_512_8)
```

Inputs: XProf's `framework_op_stats` (op types, cross-check) and `op_profile`
byProgram tree (all time and utilization). Trace paths may be relative to the
cwd; a parent dir yields one point per sub-dir.

**Table / CSV columns** (`--csv` writes one row per point × class):
`point, class, time_ms, share, mxu_util, hbm_util, occurrences, kind,
costmodel_cov, steps, scope`. `kind` is `bucket` (gemm/attention/norm/
elementwise/data/idle/other, shares sum to 1), `gemm_class` (qkv, o,
gate_up, down, mlp_fused, head, gemm_other — sum to the gemm bucket) or
`gemm_sub` (q, kv — breakdown of qkv); filter on `kind` before summing.
`occurrences` = HLO-op executions in the class over the trace. `mxu_util` =
Σflops/Σtime/946.7 TFLOP/s and `hbm_util` = Σbytes[0]/Σtime/1638 GB/s over
the ops that have an XProf cost model; `costmodel_cov` is the share of the
class time those ops cover and the utilization is blank below 50 %. The
Pallas `ragged_paged_attention` custom-call has no cost model (flops = bytes
= 0), so attention utilization is never available from XProf. `scope` is
`trace` (whole window) or `step` (`--steady`: each program divided by its run
count, one-off programs dropped, idle/n_steps).

**Bandwidth triple** (`op_profile` `bandwidthUtils[3]`), confirmed on every
node of all six stored traces: `bandwidthUtils[i] = rawBytesAccessedArray[i]
/ rawTime / peak_i` with peak = [1638 GB/s, 23296 GB/s, 16128 GB/s] — index 0
is **HBM read+write**, 1 is on-chip (VMEM) read, 2 is on-chip write. Evidence:
weight-streaming MLP fusions (100–200 MB of weights per call) sit at 0.81–0.92
in entry 0 and < 0.01 in entry 1, whereas the q projection, whose 33.5 MB
weight XLA prefetches into VMEM with an async `copy-start/copy-done` pair,
shows 0.04–0.06 in entry 0 and its weight bytes in entry 1 (the prefetch's
HBM bytes are booked on the ~1 µs `copy-start`, which is why `*-start` ops
are excluded from class utilization). The `flops` field is **not** a
utilization: it is `rawFlops / 946.7 TFLOP/s / root time` (a share of the
whole trace), so per-op MXU utilization has to be recomputed from
`rawFlops/rawTime`. Both peaks are re-derived from the root node of each
trace and asserted by `--selftest` (which also checks: bucket shares sum to
1 ± 1e-6, Σunits == device total, join agreement with `framework_op_stats`,
gemm MXU > 0.3 (observed 0.31 on prefill_512_1), gemm HBM entry 0 > 0.5
(observed 0.69), MLP entry 0 > 0.5 with entry 1 < 0.1, q entry 0 < 0.2 with
entry 1 > entry 0, idle utilization == 0, program run counts).

**Join** `op_profile` → `framework_op_stats`: a unit (an HLO op or fusion,
the first node below a category; `X and its duplicate(s)` groups are
expanded) carries `xla.provenance`, the JAX op name XLA stored as the
fusion's metadata (e.g. `jit(run_model_impl)/JaxLinear/mn,np->mp/
dot_general`, trailing `:` stripped); ops without provenance (`copy-done.N`,
`async-done.N`, `IDLE`) are listed under their HLO name. The framework row's
self time equals the sum of its units to the µs. xprof caps children at 100
per node, so categories with more distinct ops (`async-done` on decode
windows, ~1.5 % of time) get a synthetic `[residual]` unit carrying the
category's class so that Σunits == device total exactly.

**Gemm classes** are inferred per unit: `run_compute_logits`/`TD,DV->TV` →
head; `JaxEinsum TD,DNH->TNH` → q and `TD,DKH->TKH` → kv (k and v are
separate fusions; together `qkv`); `TNH,NHD->TD` → o; `JaxLinear mn,np->mp`
by the weight shapes in the fusion (D from the q/o weights): only a `[D,F]`
weight → gate_up, only `[F,D]` → down, both → `mlp_fused`. XLA fuses one
D→F projection with the F→D projection into a single fusion on the batch-1
prefill and on the decode programs, so gate/up vs down **cannot** be
separated there — compare gate_up + down + mlp_fused against the
simulator's gate_up + down. The batched-prefill program keeps them apart
(gate_up = 2 fusions/layer, down = 1). RMSNorm and the SiLU·up product are
epilogues of these fusions on silicon (`norm`/`elementwise` ≈ 0.1 %).

**Per-step device time**: for every program, runs = time-weighted mode of
the `occurrences` of the HLO ops inside it (`framework_op_stats.occurrences`
is the same number: the max over an op's HLO instances, i.e. program runs,
not kernel launches), per-run = rawTime/runs. n_steps = runs of the
most-executed `jit_run_model_impl`; the step = Σ per-run over programs with
runs ≥ n_steps (model + LM head + sampling + glue). One-off programs (the
prefill that opens a decode window) and the IDLE share are printed beside
the whole-trace total. decode_512_8: 63 steps, 14.570 ms/step (model 13.451,
head 1.087) out of 972.3 ms; decode_2048_32: 23.030 ms/step.

**Caveat — what the stored session-3 traces contain.** Every single-prefill
trace (`prefill_{512,2048}_{1,8}`) holds exactly one execution of a model
program with M = 256 × batch GEMM rows, whatever the prompt length:
prefill_512_1 and prefill_2048_1 share the program id and the FLOP count
(3.57 TFLOP = 256 tokens of Qwen3-8B), the batch-8 pair share M = 2048.
256 is the RPA kernel's KV page size (`RPAm-p_256-…`), i.e. the signature of
a prefix-cache hit that recomputes only the last page — not a profiler cut
(the 972 ms / 14k-op decode window is captured whole). Only the attention
time grows with context. So the prefill totals are **not** a forward and the
whole-trace totals of the decode windows include the same partial prefill;
trust only the per-step (occurrences/avg_time, `--steady`) numbers, and
re-capture prefill points as short windows with prefix caching verified off
(spec §2). The census prints a WARNING whenever M × runs < seq × batch.

## Measurement discipline (spec §4)

XProf device time for per-op numbers; wall clock only end-to-end. Compile
iteration discarded; medians over ≥20 reps with p10/p90 recorded. HLO/kernel
names captured via the trace (the census keys on them). Raw traces + CSVs
land in `$RESULTS_DIR`, never in git; the extracted CSVs that feed the fit
get committed under `benchmarks/tpuv6e/results/` once a session is accepted.
