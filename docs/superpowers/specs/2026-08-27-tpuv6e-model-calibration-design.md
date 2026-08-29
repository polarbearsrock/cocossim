# TPU v6e Model and Calibration for COCOSSim — Design

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Context:** Instrument work for the ISCA 2027 fused/morphing systolic–vector fabric study
(see `$TMPDIR/isca2027_cocossim/BRIEF.md`). This design covers Exp 1 (calibrated
TPUv6e-like baseline) and produces the ground truth for Exp 2 (phase-length
distributions). Exp 3 (morph-latency sweep) builds on the calibrated baseline and gets
its own design later.

## 1. Goal and scope

Build a structural single-chip TPU v6e (Trillium) model in COCOSSim and calibrate it
against real v6e hardware on GCP, with a small set of physically interpretable fitted
parameters and held-out end-to-end validation on transformer inference (prefill and
decode).

**Out of scope:** int8 (bf16 only), ICI/multi-chip, SparseCore, scalar core, morph
modeling (Exp 3), and any learned correction models.

## 2. Locked decisions and rationale

1. **Structural TensorCore model** — N MXUs + N VPU lanes as separate scheduler units
   sharing HBM, not one monolithic core. The type-queue scheduler (fixed 2026-08)
   distributes anonymous jobs across units, so this is nearly free, and per-unit phase
   behavior is exactly what Exp 2 needs.
2. **Few-parameter fit + holdout** — spec-set everything possible, fit a small named
   parameter set on microbenchmarks, validate on held-out end-to-end runs. No learned
   correction layers: they cannot extrapolate to morphed architectures and weaken the
   mechanistic-simulator claim.
3. **Workloads: prefill + decode, one LLM family** — microbenchmark sweeps for fitting;
   one transformer family (Llama-3.1-8B-class via MaxText) in both prefill and decode as
   holdout. Decode is where the morphing thesis lives; calibrating only prefill would
   extrapolate uncalibrated into the paper's key regime.
4. **Model knowledge lives in the C++ frontend** — a composite `Transformer` keyword
   expands inside `frontends/standard`, wiring the residual DAG internally. The text
   format stays a one-line interface; no Python workload generator. Python survives only
   on the measurement side (XProf parsing, fit driver).
5. **Deadline ordering** — ISCA'27 submission ~Nov 2026, modest GCP budget (single
   `v6e-1`, tens of chip-hours). Hardware measurement is front-loaded so structural
   surprises arrive in week 2, not week 8.

## 3. The v6e model in COCOSSim

### 3.1 Geometry (hypothesis, not assumption)

Public facts: 918 bf16 TFLOPS peak, 32 GB HBM at ~1.64 TB/s, 256×256-generation MXU.
Clock and MXU count are unpublished, but peak pins their product:
918e12 / (2 · 256² ops/cycle) ≈ 7.0e9 MXU-cycles/s. Four MXUs at ~1.75 GHz is the only
decomposition consistent with the v5e lineage (v5e's published peak works out to
4 × 128×128 at ~1.5 GHz; 2 MXUs would need an implausible 3.5 GHz).

**Default model: `-c 4 -sa_sz 256 -f 1.75`.** Phase C probes (§4) confirm or falsify
this; the clock is a fitted parameter, never a constant. If probes cannot fully
disambiguate, use the decomposition that matches small-M behavior and state the
ambiguity in the paper.

### 3.2 New runtime flags (defaults preserve current behavior)

| Flag | Replaces | Default |
|---|---|---|
| `-dram_ini <path>` | hard-coded `HBM2_8Gb_x128.ini` in `src/memory.cc` | current path |
| `-buf_mb <int>` | `buffer_size_bytes` in `global.h` | 8 |
| `-dram_enq <int>` | `dram_enq_per_cycle` in `global.h` | 9 |
| `-job_overhead <cycles>` | (new) fixed cost added at job init | 0 |
| `-fuse_epilogue <0\|1>` | (new) residual add as separate VPU job vs absorbed into GEMM | 0 |

All existing examples and regression tests must pass unchanged with defaults.

### 3.3 HBM model

Author `dramsim3/configs/HBM2e_v6e.ini`. The DRAMSim3 tree stops at HBM2 (~256–410
GB/s/stack), so scale channels/clock within the HBM2 format to ~1.64 TB/s aggregate.
Validated twice: (a) a DRAMSim3-only streaming test before it touches calibration, and
(b) against the *measured* stream bandwidth from Phase C — the ini targets achievable
bandwidth, not the marketing number. Once matched, the ini is frozen; it is not a knob
in the fitting loop.

VMEM capacity (`-buf_mb`) is not cleanly published for v6e. Nominal hypothesis 128 MB,
refined by the Phase C working-set probe (capacity cliffs in a size sweep).

### 3.4 Frontend extension (`frontends/standard`)

- **Binary elementwise (Gate 0):** `VecUnitJob` phases gain a read-operand multiplier so
  a phase can account two input tensors' read traffic. New layer `Add M N`.
- **`RMSNorm M N`:** same reduce+broadcast phase shape as LayerNorm, its own constants.
- **RoPE:** modeled as unary elementwise via existing Activation machinery.
- **Composite keyword:**
  `Transformer n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch`
  (mode = prefill | decode), expanding in C++ to the per-block job DAG: QKV projections,
  per-head score and AV matmuls (GQA-aware; decode accounts KV-cache reads as DRAM
  traffic), output projection, RMSNorms, SiLU MLP, residual adds. Residual adds are
  wired to **both** true parents (block input and MLP/attention output) so the simulator
  is allowed the same SA/VPU overlap the hardware has — a linear chain would overstate
  serialization and bias Exp 2 toward the paper's own thesis.
- Every shape/tiling decision in the expansion is parameterized by flags, not compile
  constants, so calibration loops don't require recompiles.

### 3.5 Honest utilization stats

`pct_active` counts a rounded-up-to-array-size job as fully busy (an M=1 decode GEMM on
a 256-wide array reads as 100% active). Two additions, both in the stats output:

- **Effective FLOP utilization**, per unit: true MACs ÷ (`sa_sz`² × busy cycles). The
  stat commensurable with XProf's FLOP utilization and the one Exp 2 figures use.
- **Cycle accounting**, per unit: every simulated cycle attributed to exactly one of
  {busy, busy-but-underfilled, stalled-on-memory, idle-no-ready-work}, reported as a
  per-unit breakdown. The state machine already distinguishes memory stalls internally
  (the VCD `IDLE_FROM_MEMORY` signal); this summarizes what today is only visible by
  reading waveforms. The four causes have different architectural remedies, and
  separating dependency starvation from bandwidth starvation is the load-bearing
  distinction for the morphing argument.

`pct_active` is retained. Calibration itself never compares utilization — only time.

### 3.6 Pinned config artifact

`configs/tpuv6e.sh`: the canonical flag set, committed, so "the v6e model" is a
reviewable file. Updated once when fitted values are frozen.

### 3.7 Testing

Extend `tests/regression.sh` in the existing T1–T5 style: Transformer expansion job
counts for a tiny config; every residual-add job has exactly two parents; dimensional
consistency of the expansion; new-flag validation (bad values rejected cleanly);
defaults produce byte-identical behavior on the existing examples; DRAMSim3 stream
smoke test for the new ini; cycle-accounting invariant (the four categories sum to
total cycles for every unit, and a memory-starved workload attributes idle cycles to
stalled-on-memory, not idle-no-ready-work).

## 4. Measurement campaign (GCP, single `v6e-1`)

**Discipline (applies everywhere):** JAX + XProf; `block_until_ready`; discard compile
iteration; medians over ≥20 reps; **XProf device time** for per-op numbers, wall clock
only end-to-end; dump HLO (`--xla_dump_to`) for every benchmark — never assume the op
written is the kernel that ran. Random weights (timing is weight-value-independent), so
no checkpoint logistics. Harness versioned in `benchmarks/tpuv6e/`; raw traces and
extracted CSVs in `$TMPDIR/isca2027_cocossim/results/`.

- **Phase A — GEMM sweep** (SA fitting set): bf16, M ∈ {1, 2, 4, 8, 16, 32, 64, 128,
  256, 512, 1024, 4096} × K, N ∈ {256 … 16384}, plus awkward sizes (100, 300, 1000,
  5120) to expose padding/tiling. Pins decode-regime (small-M) fidelity.
- **Phase B — elementwise/VPU sweep** (VPU fitting set): add, mul, exp, rsqrt, softmax,
  RMSNorm over ~64 KB–256 MB. Small sizes expose per-kernel overhead (`-job_overhead`
  intercept); large sizes give the HBM-bound roofline (sets `-vu_sz` and the achievable
  bandwidth target).
- **Phase C — geometry probes:** (i) fine M-sweep (1–512) at large fixed K, N — cycle
  quantization steps reveal effective array rows; same on N for columns; (ii)
  saturation: sustained TFLOPS on a huge GEMM pins clock × MXU-count independent of the
  spec sheet; M=256 vs M=1024 throughput shows whether tiles spread across MXUs as the
  4-unit model assumes; (iii) stream kernel: achievable HBM bandwidth (ini target);
  (iv) working-set sweep: capacity cliffs locate effective VMEM size.
- **Phase D — holdout (never used in fitting):** Llama-3.1-8B-class via MaxText.
  Per-layer and full-model; prefill seq ∈ {128, 512, 2048, 4096}; decode batch ∈
  {1, 8, 32, 64} × context ∈ {512, 2048}. XProf ops bucketed into phase classes (GEMM /
  elementwise / norm / softmax) — this same data is Exp 2's ground truth. Riding
  checklist: (a) is the residual add a separate kernel or a fused GEMM epilogue (sets
  `-fuse_epilogue`)? (b) do elementwise and GEMM kernels overlap in the timeline
  (validates the DAG wiring)?

**Cost:** sweeps are seconds per point; tens of chip-hours total across a few sessions.
Phases A + C run in the first session.

## 5. Calibration pipeline

**Fitted parameters (all physically interpretable):**

| Parameter | Meaning | Prior / source |
|---|---|---|
| `-f` | clock | ~1.75 GHz (§3.1); Phase C |
| `-dram_enq` | memory issue width per cycle | current value 9 |
| `-job_overhead` | fixed cycles per job dispatch | Phase B small-size intercept |
| `-vu_sz` | effective VPU width | Phase B roofline |

Set structurally, **not** fitted: geometry (`-c`, `-sa_sz`) from Phase C evidence;
`-buf_mb` from the working-set probe; the HBM ini frozen after matching measured stream
bandwidth. Keeping the DRAM model out of the fitting loop stops it from absorbing
unrelated error.

**Procedure:** two stages to prevent cross-talk — fit SA-side parameters on Phase A
only, then VPU-side parameters on Phase B with SA parameters frozen. Objective: MAPE
between simulated time and measured XProf device time. Optimizer: coarse grid, then
coordinate descent (4 parameters, seconds-per-simulation — brute force is reviewable).
Driver: Python script generating tiny layer files, running `perf_model`, caching by
config hash; the full fit reruns from scratch in about an hour.

**Sanity gate:** a fitted clock far from ~1.75 GHz means the *structure* is wrong —
return to Phase C evidence; never accept an uninterpretable value because it lowers
MAPE. Every fitted value appears in the paper next to its prior.

**Validation:** calibrated simulator runs the Phase D configs (generated by the
`Transformer` keyword, `-fuse_epilogue` set from the trace). Report per-layer MAPE,
full-model MAPE, prefill/decode split, and the simulated-vs-measured phase-distribution
figure (the bridge to Exp 2). All comparisons on time; utilization only qualitative via
the effective-FLOP stat.

**Success thresholds:** clearly beat SCALE-Sim TPU's 32% GEMM MAPE on the fitting set;
target 10–20% holdout end-to-end MAPE (COCOSSim's v3 validation was 13%). Tripwire: if
holdout MAPE exceeds ~30%, a mechanism is missing — rank residuals, identify the
mechanism behind the worst cluster, make the smallest simulator change that captures
it, refit from scratch, log the iteration. The log becomes the methodology section.

## 6. Risks and mitigations

1. **XLA fusion breaks op↔job correspondence** — HLO dumps accompany every measurement;
   `-fuse_epilogue` covers the biggest case; holdout compares at phase-class level,
   robust to fusion detail.
2. **Geometry ambiguity** (clock × count confounded) — Phase C probes; else match
   small-M behavior and disclose.
3. **Simulation throughput** — worst case (full 32-layer 4096-token prefill,
   ~10⁸ hardware cycles/layer) estimated tractable; measure simulator throughput in
   week 1; fall back to per-layer simulation and composition (layers are identical).
4. **HBM2-format approximation of v6e memory** — ini validated against measured
   bandwidth; holdout error reported split by memory-bound vs compute-bound points so
   memory-model error is visible, not smeared.
5. **Decode timing noise** (µs-scale steps) — time a compiled loop of N steps and
   divide; medians; device time only.
6. **Access hiccups** — everything scripted before the first session; A + C
   front-loaded; queued resources as fallback.

### 6.7 Known modeling gaps feeding the Plan-2/3 mechanism list

Identified during implementation review; none block M0/M2, but each is a candidate
mechanism for the calibration loop (§5) and should be weighed against Phase D
residuals before being dismissed as noise.

- **Decode attention KV-cache reads.** The `Transformer` expansion creates `nh`
  score/AV jobs per layer, each charging one S×head_dim K/V panel. Hardware reads
  batch×nkv panels (one KV cache per sequence, ideal intra-group reuse). Modeled/true
  factor = nh/(batch·nkv): the model over-charges when batch < nh/nkv and
  under-charges when batch > nh/nkv — 2–16× under at Phase-D decode batches for
  Llama-8B-class shapes (nh=32, nkv=8). Candidate mechanism: a per-job weight-read
  multiplier set to `batch` on score/AV jobs.
- **RoPE and binary-elementwise jobs never chunk to the buffer.** RoPE and
  binary-elementwise ops (residual add, SiLU) are emitted as one unsplit
  `VecUnitJob` regardless of working-set size, unlike RMSNorm/Softmax/LayerNorm,
  which chunk to the buffer. At calibration scale this yields a single tens-of-MB
  memstall block instead of pipelined chunks and can distort the phase-length
  distributions Exp 2 depends on.
- **`SysArrayJob`'s DRAM address range under-allocates for decode shapes.**
  Allocation uses m·m·n-style sizing (`src/units/standard/SysArray.cc:188`), which
  is smaller than post-Task-4b read volumes for decode shapes, so concurrent units'
  read cursors alias each other's address ranges. Traffic totals are still correct;
  the address stream is not bank/row-realistic. Revisit only if bank-level effects
  show up in calibration residuals.

## 7. Milestones (~10 weeks to mid-November)

- **M0 (wk 1):** commit the 2026-08 fix series; promote the four hardware flags of
  §3.2 (`-fuse_epilogue` lands with M2); author + validate the HBM2e ini.
- **M1 (wks 1–2, parallel):** GCP harness written; Phases A/B/C executed in the first
  session → geometry verdict + all fitting data.
- **M2 (wks 2–5):** frontend extension (§3.4) + utilization stats (§3.5), TDD'd,
  regression suite extended (§3.7). Critical path.
- **M3 (wks 4–6):** fit pipeline + calibration on A/B; sanity gates.
- **M4 (wks 6–8):** Phase D campaign; holdout validation; iteration loop if the
  tripwire fires.
- **M5 (wks 8–10):** Exp 2 on the calibrated instrument (reuses Phase D ground truth);
  buffer to deadline.

## 8. Deliverables

Simulator changes + tests (committed); `HBM2e_v6e.ini`; `configs/tpuv6e.sh`;
`benchmarks/tpuv6e/` measurement harness; fit driver + frozen fitted parameters;
results CSVs and measurement report under `$TMPDIR/isca2027_cocossim/results/`; this
spec and its implementation plan.
