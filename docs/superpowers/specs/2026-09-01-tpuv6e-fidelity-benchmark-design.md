# TPU v6e fidelity benchmark — design

Date: 2026-09-01. Predecessor: `2026-08-27-tpuv6e-model-calibration-design.md`
(the model, the harness, and the mechanism history live there; this spec does
not restate them).

## 1. Purpose

Establish, cell by cell, where the simulator's TPU v6e model matches the
hardware, where it falls short and by how much, and whether it can be used
for the fused/morphing-fabric **utilization experiments** — which read
per-unit attribution (systolic-array vs vector-unit busy / underfilled /
memstall / idle shares, per phase) and the *direction* those shares move
across operating points.

Three outputs: a **fidelity map** (verdict per cell), an **error budget**
(mechanism and magnitude behind every non-pass cell), and a **go/no-go**
computed against exactly the cells the utilization experiments depend on.

Non-goals: validating counterfactual fabrics (a different VPU width, MXU
count, or fusion set cannot be measured on v6e — the campaign establishes
that the model reproduces v6e's utilization landscape with *physical*
mechanisms, which is what makes extrapolation principled); serving-mode
prediction (continuous batching is characterized, not scored).

## 2. Ground rules

- **Device-side only calibrates device mechanisms.** Every tier-1 number is
  a chained call (host dispatch amortized) or an XProf device time. Unchained
  wall times carry the ~113 µs host floor and are recorded only as such (the
  512-row residency window was a mis-derivation from exactly that mistake).
- **Fit on tier 1 (+ a declared tier-2 fit subset); everything else is held
  out.** Holdout points are never used to choose a parameter, a mechanism,
  or a threshold.
- **Noise first.** Anchor points are repeated (≥3 repeats within a
  session); a cell's verdict uses the median and counts anything inside the
  p10–p90 band as agreement.
- **Traffic invariance is a unit test, not a hope.** Any simulator change
  made during the campaign carries a beat-exact test (the V-suite pattern).
- **Utilization ground truth is XProf's `op_profile`** (per-op MXU FLOPS
  utilization and HBM bandwidth utilization as fractions of peak, IDLE as an
  explicit node) and `framework_op_stats` (`measured_flop_rate`,
  `measured_memory_bw`, `operational_intensity`, `bound_by`, `occurrences`,
  `avg_time`). Per-step device time comes from `occurrences`/`avg_time` of
  the model program, so traces stay short (no long-run truncation).

## 3. The fidelity matrix

### 3.1 Tier 1 — primitives (device-side, chained, 20-rep medians + p10/p90)

| cell | shapes | isolates | sim counterpart |
|---|---|---|---|
| G1 square GEMM | N ∈ {1024, 2048, 4096, 8192}, chain 16 | compute-bound MXU efficiency, tile-switch cost | `Matmul N N N` |
| G2 memory-bound GEMM | M ∈ {128, 256, 512, 1024, 2048} × (K,N) ∈ {(8192,8192), (4096,4096)}, chain 16 | weight streaming; the first-job loop-order residual | `Matmul M K N` |
| G3 small-M at real shapes | M ∈ {1, 4, 8, 16, 32, 64, 128, 256} × (K,N) ∈ {q (4096,4096), kv (4096,1024), gate/up (4096,12288), down (12288,4096), head (4096,151936); Mistral (4096,14336), (14336,4096), (4096,32768)}, chain 16 | per-kernel fixed cost t₀ and effective BW per shape class — the decode error's source | `Matmul M K N` |
| E1 elementwise | add (2-in), exp (1-in), n ∈ {2¹⁵ … 2²⁸}, chained | HBM streaming rate, VPU floors | `Add n`, `Activation n` |
| A1 attention kernel | context ∈ {512, 2048, 8192} × batch ∈ {1, 8, 32} × query {prefill q=context, decode q=1}, GQA 32/8, head_dim 128, chain 8 | the fused kernel vs per-head jobs; the long-context decode miss | `Transformer` attention sub-DAG (scores/softmax/AV) at the same dims, extracted per-class |
| K1 paged-KV gather | context ∈ {512, 2048, 8192} × batch ∈ {8, 32}, block 16–32 tokens with a block table, vs contiguous read of the same bytes | the KV bandwidth derate, measured directly | KV streams in score/AV jobs |

Per cell: device latency, throughput (TF/s or GB/s), HBM utilization, MXU
utilization — hardware from XProf, simulator from cycles and ACCT.

A1's primary path is vLLM's ragged paged attention kernel called directly
from `tpu_inference` (import path discovered on the VM and recorded in the
README); fallback is `jax.nn.dot_product_attention` (XLA's fused attention,
not paged). Whichever runs is recorded per cell; if only the fallback runs,
A1's KV-gather content moves entirely to K1.

### 3.2 Tier 2 — whole models (vLLM offline, `TokensPrompt`, `ignore_eos`, prefix caching off)

W1 **Qwen3-8B** (36 L, d 4096, 32/8 heads, d_ff 12288, vocab 151936):
- prefill (seq, batch): (256,1) (512,1) (1024,1) (2048,1) (4096,1) (512,4) (512,8) (2048,4) (2048,8)
- decode (context, batch): (512,1) (512,8) (512,32) (2048,8) (2048,32) (2048,64) (8192,8) (8192,32)
- anchors repeated ×3: prefill (512,1), (2048,1); decode (512,8), (2048,32)

W2 **Mistral-7B-v0.3** (32 L, d 4096, 32/8 heads, d_ff 14336, vocab 32768,
no sliding window, no QK-norm; `Transformer 32 4096 32 8 14336 seq mode batch 32768`):
- prefill (512,1) (2048,1) (512,8) (2048,8); decode (512,8) (2048,32) (8192,8); anchor repeat ×3 on decode (2048,32)

W3 **serving characterization** (Qwen3-8B, vLLM online server, 64 concurrent
requests, inputs 512–2048, outputs 128): census and utilization only, not a
scored cell — it bounds what the pinned points miss.

Traces: one XProf trace per point, of a *short* run (prefill: the single
prefill; decode: 8 steps after a warm generate); per-step device time from
program `occurrences`/`avg_time`.

Per point: device time per forward/step, wall time, per-kernel-class time
share (gemm / attention / norm / elementwise / data / idle / other) and
per-class MXU% and HBM% — versus the simulator's per-op-class time and
ACCT shares.

**Tier-2 fit subset F** = Qwen prefill (1024,1) and decode (512,8). Only F
may inform the data-op term (§6). Every other tier-2 point, and all of W2,
is holdout.

## 4. Simulator ↔ hardware mapping

| simulator | hardware (XProf) |
|---|---|
| cycles / 1.75 GHz | program device time (`avg_time` × `occurrences` for the window) |
| ACCT SA busy share | MXU FLOPS utilization (per op class: `op_profile`) |
| ACCT SA memstall + VPU memstall, DRAM CMDs × 64 B / time | HBM bandwidth utilization, `measured_memory_bw` |
| ACCT VPU busy | no direct counterpart — VPU work is fused into GEMM kernels on silicon (census: 0.1% visible); compared only through total time |
| MEM demand-idle | IDLE node + (1 − HBM utilization) during non-idle ops |
| per-op-class time (new, §5) | `framework_op_stats` by class via the census buckets |

Op classes: qkv, o, gate-up, down, head, attention (scores + softmax + AV),
vpu-norm, vpu-elementwise, other.

## 5. Simulator prerequisites (before TPU time; each with a V-test)

- S1 **Per-op accounting.** ACCT counters attributed per op class (job → class
  from its origin in the composite: tag it at creation), printed beside the
  per-unit totals; the VCD timeline as cross-check. Test: class shares sum to
  the unit totals exactly.
- S2 **`-op_overhead`** cycles charged once per op boundary per core (a new
  weight tag or op class on that core), replacing `-job_overhead` as the fit
  parameter; `-job_overhead` kept for legacy. Test: cycles rise by exactly
  n_ops × overhead on a chain of resident-weight jobs.
- S3 **Batched prefill** in the composite: mode 0 with batch > 1 runs the
  GEMMs at M = batch × seq and attention per sequence (batch independent
  score/AV sets, each S = seq). Test: job counts and traffic hand-derived on a
  tiny shape.
- S4 **Legacy accounting fixes**: OS write-back charged at true bytes
  (`beats_per_wb` double division) and activation panels read once into the
  shared VMEM rather than once per MXU. They cancel at prefill by coincidence
  and distort the busy/memstall split the matrix judges. Tests: V18/V24-style
  exact re-derivations.
- S5 **Census upgrade**: read `op_profile` utilization per class, handle
  short windowed traces, emit one CSV row per (point, class).
- S6 **KV-gather derate knob** (`-kv_bw_pct`, effective HBM rate for KV
  streams as a percentage; default 100) and **data-op term**
  (`-data_overhead` cycles per forward/step; default 0) so §6 has something
  to fit. Tests: traffic invariant, cycles move as specified.

Known composite omissions carried into the error budget, not fixed here:
Qwen3's QK-norm (two per-head RMSNorms per layer; fused sidecars on silicon
anyway), embedding gather, KV-cache write of the new token, sampling.

## 6. Fit protocol

Parameters and the tier-1 cells that identify them:

| parameter | identified by |
|---|---|
| `-f` (effective MXU clock) | G1 large-N MFU |
| `-dram_enq`, DRAM ini rate | E1 large-n streaming, G2 |
| `-op_overhead` | G3: t = t₀ + bytes/BW per shape class; t₀ across M |
| `-kv_bw_pct` | K1 (and A1 decode if the kernel path runs) |
| `-vu_sz` | prior unless A1 prefill (softmax-bound regime) identifies it; reported as unidentified otherwise |
| `-data_overhead` | tier-2 fit subset F only |
| `-dbuf` budget, `-vmem_headroom` | priors; sensitivity reported, not fit |

Method: minimize mean absolute log-error over tier-1 cells with the
parameters above (coordinate descent is sufficient; parameter count is
small), report each parameter's sensitivity (Δ objective for ±20%), freeze
into `configs/tpuv6e_fitted.sh`, then score tier 2 once. If a tier-2 miss
exceeds 25% with no mechanism in the error budget, that is a finding (new
mechanism), not a reason to refit.

## 7. Verdicts and go/no-go

Per cell and metric:
- **PASS**: |error| ≤ 10% (latency, throughput, bandwidth) or ≤ 10 points
  absolute (a utilization share), or inside the measured noise band.
- **CONDITIONAL**: 10–25% with a named mechanism and bounded sign in the
  error budget.
- **FAIL**: > 25%, or any size without a mechanism.

Utilization-experiment dependency set D:
- SA busy / VPU busy / idle shares per phase for prefill and decode, at all
  tier-2 batch and context points;
- the direction of change of those shares along the batch sweep (decode
  batch 1→64) and the context sweep (512→8192) — the proxies for shifting the
  fabric balance;
- the prefill-vs-decode contrast of MXU utilization.

**GO** if ≥ 80% of D is PASS and no cell of D is FAIL inside the regime the
experiments claim (state the regime explicitly: model sizes, batch and
context ranges). **CONDITIONAL-GO** otherwise, listing the excluded regime.
**NO-GO** if any D cell fails with an unexplained mechanism.

## 8. Hardware sessions (on-demand v6e-1, us-east5, ≤ $50 each)

| session | content | est. |
|---|---|---|
| H1 | G1, G2, G3, E1 with traces | ~2 h, ~$6 |
| H2 | A1, K1 | ~2 h, ~$6 |
| H3 | W1 (17 points + anchor repeats), traces | ~3 h, ~$9 |
| H4 | W2 (Mistral download ~14.5 GB) + W3 | ~3 h, ~$9 |
| H5 | reserve: re-measure CONDITIONAL cells after the fit, extra repeats | ~2 h, ~$6 |

Every session: provision → run → `scp` results → teardown → three-zone
orphan check → spend reported. Results CSVs and per-point census rows are
force-added under `benchmarks/tpuv6e/results/fidelity/`.

## 9. Deliverables

1. `benchmarks/tpuv6e/fidelity/`: `run_tier1.sh`, `run_tier2.sh` (VM side),
   `sim_matrix.sh` (simulator side, one line per cell), `score_matrix.py`
   (joins hardware and simulator CSVs, computes verdicts, emits the map).
   `score_matrix.py` re-runs from stored traces and CSVs in minutes — no TPU
   time for future model changes.
2. The fidelity map: a matrix figure (cells × metrics, colored by verdict)
   and the table behind it.
3. The error budget: one entry per non-PASS cell — mechanism, sign,
   magnitude, evidence, fix status.
4. `configs/tpuv6e_fitted.sh` with the fit report (parameters, sensitivities).
5. The go/no-go statement with the regime it covers.

## 10. Risks

- vLLM TPU support for `MistralForCausalLM`: expected (Llama architecture);
  verified in H4's first minutes, with Llama-3.1-8B (gated, needs a token) as
  the fallback.
- A1's kernel-direct path may not import cleanly; the fallback is defined.
- XProf per-op utilization semantics (what "peak" the fractions reference;
  the three bandwidth entries) are confirmed against G1/E1 known-rate cells
  before any tier-2 use.
- Noise: single-session repeats bound within-session noise only;
  cross-session drift is checked by re-running two anchors in H5.
- Budget: estimated total ≈ $36; the $50-per-session ceiling stands.

## 11. Order

F0 prerequisites S1–S6 (simulator, no TPU) → H1 + H2 → fit → freeze
`tpuv6e_fitted.sh` → H3 + H4 → score → error budget → H5 → fidelity map,
go/no-go.
