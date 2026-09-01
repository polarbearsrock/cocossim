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

### 3.1 Geometry (amended 2026-08-30; original hypothesis falsified)

Public facts: 918 bf16 TFLOPS peak (1836 int8 TOPs), 32 GB HBM at 1638 GB/s, and —
per Google's own v6e documentation — **one TensorCore per chip with 2 MXUs, one
vector unit, and one scalar unit**, with a 256×256 MXU (scaling-book corroboration).
The original 4-MXU/1.75 GHz decomposition below the line was falsified by those docs.

With 2 × 256² at 1.75 GHz, the published peak requires **2 packed bf16 MACs/PE/cycle**
(2 · 256² · 2 MACs · 2 FLOP · 1.75e9 = 918e12; the int8 peak at exactly 2× bf16 is
consistent with packed-arithmetic PEs). MACs/PE and the VPU width are the remaining
unpublished quantities — both are hypotheses Phase C probes confirm or falsify.

**Default model: `-c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1.75`.**
The clock is a fitted parameter with a physical prior; it must never absorb structural
factors (that is what `-mxu_macs_per_pe` exists for).

> Superseded original (kept for the record): "Four MXUs at ~1.75 GHz is the only
> decomposition consistent with the v5e lineage; 2 MXUs would need an implausible
> 3.5 GHz." The overlooked possibility was 2 MACs/PE/cycle at 1.75 GHz.

### 3.2 New runtime flags (defaults preserve current behavior)

| Flag | Replaces | Default |
|---|---|---|
| `-dram_ini <path>` | hard-coded `HBM2_8Gb_x128.ini` in `src/memory.cc` | current path |
| `-buf_mb <int>` | `buffer_size_bytes` in `global.h` | 8 |
| `-dram_enq <int>` | `dram_enq_per_cycle` in `global.h` | 9 |
| `-job_overhead <cycles>` | (new) fixed cost added at job init | 0 |
| `-fuse_epilogue <0\|1>` | (new) residual add as separate VPU job vs absorbed into GEMM | 0 |
| `-mxu_macs_per_pe <int>` | (new) OS accumulation throughput per PE | 1 |
| `-n_vpu <int>` | (new) vector-unit count, decoupled from `-c` | match `-c` |

All existing examples and regression tests must pass unchanged with defaults, with
three documented exceptions: the two under-filled-tile changes (true-M dims and the
weight/KV read term, from the implementation plan), and — amended 2026-08-30 — the
`-mxu_macs_per_pe` physics fix: OS accumulation now costs ceil(K/macs) cycles instead
of K·systolic_fpu_latency, so DEFAULT OS timing is ~2× faster than the original code
(the old 2-cycle K-step modeled no real machine; `systolic_fpu_latency` is demoted to
WS fill/drain only). Stats files declare `SCHEMA 2`: SA eff_util capacity is
macs·sz² and full tiles read ~1.0, vs the schema-1 ceiling of ~0.5 — numbers are not
comparable across schemas.

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

**Amendments (2026-09-01, Plan 2 brainstorm + session 0):** Phase D's vehicle
is **vLLM offline mode (tpu-inference) serving Qwen3-8B**, superseding
"Llama-3.1-8B-class via MaxText" — vLLM gives production Pallas kernels
(ragged paged attention) and a two-tier design: offline fixed-shape points
for the holdout fit, serving-mode sweeps for characterization. Qwen3-8B's
attention geometry matches the Llama-8B assumptions (4096/32/8); dims are
pinned from the HF config in the harness. Session 0 validated the pipeline
on a v6e-1 (~$3): traces are kernel-legible (ragged_paged_attention,
dot_general, add_rsqrt fusions), the kernel census runs on xprof's own
converters, and the first census already shows the LM head at 7.3% of decode
device time (the model's prediction was ~7%). Capacity ruling: on-demand
only (us-east5; us-east1-d is spot-only and preempted 2x/40min). Harness:
`benchmarks/tpuv6e/`. Probe (ii)'s "4-unit model" phrasing below is
superseded by the 2-MXU geometry (§3.1).

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
| `-f` | clock | physical ~1.75 GHz (§3.1); Phase C |
| `-dram_enq` | memory issue width per cycle | current value 9 |
| `-job_overhead` | fixed cycles per job dispatch | Phase B small-size intercept |
| `-vu_sz` | effective VPU width | prior 512 (§3.1, unpublished); Phase B roofline |

Set structurally, **not** fitted: geometry (`-c 2`, `-n_vpu 1`, `-sa_sz 256` from
Google docs; `-mxu_macs_per_pe 2` from the peak decomposition, Phase C-confirmable);
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

- **Decode attention KV-cache reads (ADDRESSED 2026-08-31, commit bfa9cdb: n_weight_streams = batch on score/AV jobs, GQA groups share a weight_tag; V20 pins both halves).** The `Transformer` expansion creates `nh`
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

- **(added 2026-09-01; ADDRESSED same day, accounting fix — no flag — tests
  V31a + re-derived V18/V18b/V24/V25a/V29a; benchmark spec S4a) OS
  write-back charged at 1/64 of its bytes.** The OS `shift` stage passed
  `beats_per_wb` (= sz·sz·dtw/64, a BEAT count) through `state_transfer`'s
  BYTE argument, so every column tile wrote 1/`bytes_per_tx` of its output
  block (2 beats instead of 128 at sz 64, 32 instead of 2048 at sz 256), and
  `sys_job_alloc_bytes` mirrored the double division on purpose (window/walk
  lockstep). Both sites now charge `output_tile_bytes` =
  min(sz,M)·min(sz,N)·dtw·batch per column tile (`fused_out` suppression
  kept; WS mode already wrote the true M×N). Hand re-derivations, all
  landing beat-exact: V18 66048 → **98304** (writes 512 → 32768 = 16 jobs ×
  16 tiles × 128 beats; `-vmem_reuse 0` 557568 → 589824, now pinned exactly,
  V18b likewise), V24 2363392/790528 → **2621440/1048576** (writes 4096 →
  262144; ratio 2.99 → 2.5, now pinned exactly), V25a delta 6272 → **8192**
  (the four suppressed score write-backs are 512 beats each, not 32). V31a
  pins a single `Matmul 256 64 256` at 1024 reads + 2048 writes = 3072.
  Prefetch credit, fusion deltas and the V27/V29/V30 invariance assertions
  did not move. Known timing limit exposed by the true write volume (NOT
  fixed, documented in V29a): with `-act_share 0` both cores fetch their own
  activation panel at every job start, tile i+1's pre-issued reads throttle
  tile i's write-back in DRAMSim3's read-preferring 32-entry write buffer,
  and `-dbuf_tile 1` comes out slower than 0 on 4096³ (`-dbuf 0`: 399846 vs
  362109). Under the pinned `-act_share 1` the ordering holds (353650 vs
  355523). The first version of this fix shipped an unflagged output-side
  gate (`State::writes_gate`/`hold_writes`: a non-last tile's write-back
  streamed under the next tile's compute) claiming to cure that inversion;
  the review showed it did not (386644 vs 362109 with `-act_share 0`), that
  V29a's ordering is restored by S4b alone, and that it was an unflagged
  timing mechanism outside the accounting scope — it was removed (fix
  round 1). Its removal costs cycles on the pinned config (CMDs invariant):
  4096³ 331538 → 345903 (+4.3%), one Llama-8B prefill layer
  (`Transformer 1 4096 32 8 14336 2048 0 1`) 2861904 → 3088716 (+7.9%), one
  decode layer (`... 512 1 8`) 554805 → 551329 (−0.6%). An output-side
  double buffer is a legitimate mechanism candidate for a separate,
  flagged, default-off task with its own isolating test. Pinned config for
  S4a + S4b together, relative to the pre-S4 tree (cycles / DRAM CMDs):
  4096³ 329454 → 345903 (+5.0%), 1581056 → 1572864 — the +516096 write
  beats and S4b's −524288 activation beats nearly cancel, the coincidence
  the benchmark spec anticipated. The busy/memstall split, not the totals,
  is what the fidelity matrix judges.
- **(added 2026-09-01; ADDRESSED same day, flag `-act_share` default 1,
  tests V31b + V29a CMD pin; benchmark spec S4b) Activation panels read once
  per MXU instead of once into shared VMEM.** `createSAJobs` splits N across
  cores and every core's row-block job charged its own min(sz,M)×K
  activation panel (at init and at each row advance); hardware stages the
  tile once and both MXUs consume it — residual (b) of the VMEM-residency
  entry below, a fixed ~2×A on every 2-core GEMM. Under `-act_share 1` only
  core 0's row-block job charges the panel; the same row block's jobs on
  cores ≥ 1 are constructed `act_resident` (the existing fusion endpoint
  flag: no activation reads at init/row advance, no activation prefetch
  entry, window formula already mirrored). Exact on traffic; an
  approximation on timing — core 1 may start computing before core 0's
  fetch has landed (no cross-core wait is modeled). Attention jobs (per-head
  Q slices, group-pinned) are untouched. `-act_share 0` restores the
  per-MXU reads for ablation; `configs/tpuv6e.sh` pins 1. V31b pins
  `Matmul 256 256 512` at -c 2: 12288 → 10240 CMDs, exactly the one
  suppressed 2048-beat panel. V29a's total becomes derivable and is pinned:
  1572864 (core 0 1048576, core 1 524288; 2097152 with `-act_share 0`).
  Side effect worth knowing: it also removes the lockstep collision of the
  two cores' job-start panel fetches that dominated the `-dbuf 0` control
  run in V29a.
- **(added 2026-09-01; ADDRESSED same day, flag `-fuse_vpu`, tests V30a–d)
  VPU ops fused into GEMM prologues/epilogues.** The silicon kernel census
  puts RMSNorm/RoPE/SiLU/residual at 0.1% of device time: XLA fuses them, so
  they cost no HBM round trip and overlap with the MXU. The model kept them
  as separate VPU jobs with full traffic (~360 MB/layer at prefill-2048, 16%
  of the layer) on the dependency chain. Under `-fuse_vpu 1` they remain
  VPU jobs (attribution intact, VPU busy unchanged) but run traffic-free as
  SIDECARS off the chain: each waits for its true inputs while the consumer
  GEMM depends on those inputs' producers directly (q/k/v on the block
  input, scores on q/k, gate/up on o, down on gate/up, the LM head on the
  last block). Subsumes `-fuse_epilogue`; `configs/tpuv6e.sh` pins 1. V30a
  pins the beat-exact traffic delta (20480 on the small prefill config),
  V30b the chain structure (VPU→GEMM edges 3 → 0), V30c the decode delta
  (2432) — on that tiny 2-layer config fusion also removes six HBM
  read→write latency links per layer (−8.5%), <1% at 36 layers. Holdout:
  prefill-2048 63.9 → 55.0 ms (−6.1% vs silicon), prefill-512 16.1 → 13.6
  (−29%), decode unchanged; MAPE 20.7% → 23.3%. The headline rises because
  every point now sits on the SAME side: all four are under-predicted by the
  unmodeled per-kernel fixed cost, host gaps, `data` ops and paged-KV gather
  — the "too slow" physics terms are exhausted (prefill-2048 demand-idle 87%,
  VPU memstall 0), which is the state the Plan-3 fit was designed for.
- **(added 2026-09-01; ADDRESSED same day, flag `-dbuf_tile`, tests
  V29a–c) Within-op double buffering.** The OS state machine declared a
  tile's reads only when its read stage began, so the shift/write stages
  and every job-start activation fetch were exposed; silicon streams tile
  i+1's operands under tile i's compute and stages the next row block's
  activation panel ahead. Two pieces: (1) `-dbuf_tile 1` issues the next
  tile's reads the moment the current tile's reads drain and lets the
  shift/write stages run while they stream (`State::reads_gate` /
  `hold_reads`; the write→read transition re-arms the gate instead of
  re-declaring the reads); (2) the cross-op prefetcher also stages a READY
  job's first activation panel (`prefetchable_act_beats`; producers done,
  so the data exists), credited exactly like weights. Traffic invariant
  (V29a/b pin CMDs equal). Device-side GEMM acceptance (pinned config):
  8192³ 1448 → 1352 µs vs silicon 1409 (−4%); 4096³ 213 → 188 vs 168
  (+12%, from +27%); 256×8192² 98 vs 96 (+2%). Holdout: prefill-2048
  66.0 → 63.9 ms (+9.1% vs silicon); MAPE 21.7% → 20.7%.
  Residual identified by the same runs: GEMMs with only 2–4 row-block jobs
  (512×8192², 1024×8192²) stay +26–33% because the FIRST row-block job
  fetches the whole weight slice tile by tile while later jobs run
  compute-only on the resident copy — a fetch-heavy phase then DRAM-idle
  phases. Silicon keeps the activations resident, streams each weight tile
  once and computes all rows against it, spreading one weight stream over
  the whole op's compute. That is the loop order `createSAJobs` chooses
  (row-block jobs × all column tiles), not a pipeline gap; inside a layer
  the cross-op prefetcher pre-stages the first job's weights during the
  previous op, so the layer-level cost is smaller than the single-op
  microbenchmark shows. Prefill-2048 DRAM demand-idle is now 77%: the
  layer is serialization/VPU-bound, not bandwidth-bound.
- **(added 2026-09-01; ADDRESSED same day, tests V26/V27a–c) DRAM demand
  starvation: the attention barrier, the missing cross-op prefetch, and the
  GQA placement lottery.** After fusion, a new `MEM demand-idle` stat showed
  DRAM starved of offered work for 42% of prefill-2048 wall time while
  sustaining 1,388 GB/s when fed (matching C4's silicon 1,371). Three
  mechanisms, landed in order: (1) attention stages were wired all-to-all
  (`connectJobLists`), a composite-builder artifact — per-head wiring
  (softmax chunk ↔ overlapping head row-ranges, V26, unconditional) freed
  −13.4% on prefill-2048; (2) `-dbuf <MiB>` cross-op weight prefetch models
  XLA's next-operator streaming (B2/C5v2): first-of-tag jobs (never
  VMEM-resident at dispatch, so the credit is exact) stream their weight
  sweeps into otherwise-idle DRAM slots under a byte budget; traffic is
  exactly invariant (V27a pins CMDs equal; a tag-count window was mis-shaped
  — tiny attention tags starved it short of the MLP weights); (3) V27b's
  invariance assertion exposed GQA sibling placement as a scheduling lottery
  worth several percent of decode traffic — siblings are now pinned
  round-robin per group, matching hardware's fetch-once-into-shared-VMEM.
  `configs/tpuv6e.sh` pins `-dbuf 48`. Prefetch coverage was then extended
  from first-of-tag (~1/8 of weight traffic) to EVERY predicted fetch pass:
  core pinning makes per-core dispatch order deterministic, so the list
  build replays the residency automaton per core; V27's CMD-invariance
  assertions guard the replayed prediction against drift. Post-fix
  demand-idle: decode 0.17%, prefill-512 8%, prefill-2048 22% (budget
  sensitivity 48→128 MiB buys only 1978→1910 µs/layer: the remainder is
  structural — activation-panel and VPU tensor reads at op starts, not
  weights).
  Residuals after this set (pure priors, holdout): prefill-2048 +23%
  (starvation floor above); prefill-512 −16% and decode-512×8 −22%, both
  exactly ~86–87 µs/layer of missing constant — the per-kernel dispatch
  overhead B2 measured at 31–35 µs/kernel (`-job_overhead`'s territory);
  decode-2048×32 −34%, whose extra ~160 µs/layer scales with KV volume —
  consistent with paged-attention gather achieving lower effective HBM
  bandwidth than the sim's streaming rate (candidate: per-stream KV derate).
  All three parameters belong to the Plan-3 fit from A/B/C + D-charac
  data, never from the holdout points.
  Review pass (same day, independent reviewer, all verified and fixed;
  tests V27a2/V27d/V27e/V28): prefetch credit is now issued and deducted in
  whole BEATS per tile from the full-formula charge (exact for any byte
  remainder -- the byte version broke invariance at sub-beat panels and its
  floor-of-sum issue count could overrun a fused job's window); the prefetch
  walk carries the same bounds guard as the demand paths; `-dbuf` is forced
  off in WS mode (WS charge sites never consume credit); `MEM demand-idle`
  counts demand only, with the all-traffic idle printed beside it; GQA
  pinning falls back to per-head pinning when nkv < n_cores (MQA) so no MXU
  idles. Calibration shapes are bit-identical before and after. Found in
  passing, out of scope: `MatmulAct`/`ActMatmul` crash under `-ws 1`
  (OS-sized windows walked by the WS state machine) -- pre-existing, filed
  separately.
- **(added 2026-09-01; ADDRESSED same day, flag `-fuse_attn`, tests V25a–c)
  Unfused attention was the dominant prefill gap.** The `Transformer` expansion
  materialized the score matrix S through DRAM four times per layer (QK^T
  write-back, softmax read + rewrite, AV activation read) — ~0.8–1.1 GB/layer of
  phantom traffic at seq 2048 that the real chip never emits: the session-3
  kernel census shows XLA/vLLM always run attention as one fused
  flash-attention-style kernel, whose HBM traffic is exactly read-Q,K,V +
  write-O at any sequence length. Post-fix holdout re-score: prefill-2048
  +87.9% → the fused prediction (see calibration log). `-fuse_attn 1` is pinned
  in `configs/tpuv6e.sh` as structural truth, not a fitted prior. Suppression
  is expressed as job-endpoint flags (`SysArrayJob::fused_out`/`act_resident`,
  `VecUnitJob::fused_out`) — a general on-chip-edge primitive (the same idea as
  `-fuse_epilogue`) usable for future fusion studies; softmax compute and all
  dependencies are retained (conservative: no intra-head pipelining credit).
  This supersedes the 2026-09-01 prefill-2048 diagnosis that blamed residual
  gap (a) (un-overlapped window refetches): with score traffic removed both
  machines sit near their memory floors (silicon ~1.1×, sim ~1.3×), so
  prefetch/double-buffering is second-order, not the +88% mechanism.
- **(added 2026-08-30; ADDRESSED 2026-08-31 by the VMEM residency model — commit
  `ecddd90`, flags `-vmem_reuse`/`-vmem_headroom`, tests V18/V18b/V18c) OS weight
  re-reads ignored VMEM reuse — was the dominant fidelity gap.** Weights now stay
  resident across row-block jobs (and row passes) when the slice fits the per-MXU
  VMEM share. Measured effect: Matmul 4096³ traffic 672→101 MB (672 MB was
  measured on the old 4-MXU geometry, before the v6e config was pinned to 2
  MXUs in commit `b7e120d`; at the pinned 2-MXU config the pre-fix figure is
  604 MB, reverified against `configs/tpuv6e.sh` — the post-fix 101 MB
  stands either way: exactly the true working set), 306→646 TF/s (70% of
  peak, memstall 62%→21%); prefill-512 858→659 µs; decode: `-vmem_reuse 0`
  also disables GQA sibling sharing, since GQA reuse rides the same
  weight_tag/residency machinery as weight staging (`SysArrayState::init`'s
  `weights_resident`/`resident_weight_tag`) — the ablation over-charges
  decode KV reads by nh/nkv, exactly matching the pre-fix staging-free
  model. This coupling is deliberate: `-vmem_reuse` is the single ablation
  switch for VMEM staging (`global.h`), not two independently toggleable
  mechanisms.
  AMENDED 2026-09-01 (commit a6e6f82) and RETRACTED the same day: the
  512-row re-stream window was derived by fitting C5v2's slope law against
  C3's throughputs, which are unchained calls carrying the ~113 µs host
  dispatch floor (B1). Floor-corrected, 4096³ runs at ~819 TF/s in 168 µs on
  device, which cannot contain eight 32 MB weight passes; and the raw C5v2
  slopes (time vs M at fixed footprint) equal pure MXU compute at peak for a
  VMEM-resident 33.5 MB weight — it streams once — while footprints beyond
  VMEM show about one overlapped re-stream per ~2048 rows. `-vmem_rows` is
  kept as an ablation knob, default 0 (`configs/tpuv6e.sh` pins 0
  explicitly). Not modeled: the ~2048-row overlapped re-stream for weights
  that do NOT fit the VMEM share (the model's non-fit path refetches per
  256-row job, an over-count; no Qwen3-8B slice hits it). Rule adopted:
  only device-side (chained or traced) measurements may calibrate device
  mechanisms — every unchained number carries the host floor.
  Residual, deliberately unmodeled: (a) the fetch pass is not double-buffered —
  the first job of a slice exposes its fetch instead of prefetching under the
  previous op's compute, leaving GEMMs ~40% above their compute floor; (b)
  activations are re-read once per MXU (no cross-core sharing, fixed ~2×A)
  — ADDRESSED 2026-09-01 by `-act_share` (entry above); (c) VMEM is a shared capacity *parameter*, not a contended *resource* — each
  consumer checks fit independently (MXUs against buf/n_cores·headroom, VPU
  chunkers against the full buffer), so co-residency can be over-committed near
  the capacity edge; the model is optimistic exactly there. (d) WS mode ignores
  both `n_weight_streams` and VMEM residency entirely — neither
  `sys_job_alloc_bytes`'s `ws` branch nor `SysArrayState`'s WS `init`/`increment`
  reference either mechanism. Out of scope for the v6e model, which is pinned to
  OS mode (`-ws 0`), but a `-ws 1` user should not be surprised that GQA
  KV-stream scaling and weight staging silently do nothing there. Historical
  text:
  OS weight re-reads ignore VMEM reuse — Every row tile re-reads the
  full weight panel, so a `Matmul 4096^3` moves ~6.7x its true bytes and prefill-512
  moves ~2.4x; with honest MXU throughput (`-mxu_macs_per_pe 2`) this phantom
  traffic saturates DRAM (~1300 GB/s) and caps modeled GEMM/prefill throughput at
  ~29-33% of peak where the real chip is compute-bound (e.g. modeled 858 us vs a
  266 us single-read memory floor on prefill-512). Candidate mechanism: charge the
  weight panel once per resident set that fits `-buf_mb`, not once per row tile.
  Until then, prefill/GEMM absolute times are pessimistic; decode (weights read
  once per token anyway) is unaffected and validates well (343 us vs 266 us floor).

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
