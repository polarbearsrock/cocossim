# TPU v6e VPU-operator calibration — design

Date: 2026-09-02. Predecessors: `2026-08-27-tpuv6e-model-calibration-design.md`
(model, harness, mechanism history) and `2026-09-01-tpuv6e-fidelity-benchmark-design.md`
(tiers, scoring, verdicts; §6.3 results). Budget for this campaign: **$200 of
on-demand v6e-1 time** (≈ $2.7/h; every session below is capped at $50 and
tears down with the orphan check, as before).

## 1. Purpose

Give the vector unit the per-operator calibration the MXU already has, in
the form the ISPASS 2025 paper used for TPU v3 (standalone softmax cells
compared by latency) but device-side, at the real model shapes, and with the
hardware's own vector-unit counters as a second reference:

1. **Per-operator VPU latency** for every vector operator a decoder layer
   contains (softmax, RMSNorm, RoPE, SiLU·up, residual add), so a
   layer → operator decomposition can quote a validated cost per operator.
2. **Measured VPU occupancy** (`VECTOR_ALU_INSTRUCTION_0..3`, `VECTOR_ISSUE`,
   `HOLD_VECTOR_ISSUE`, `VLD/VST`, `XLU_BUSY`) per cell, so the simulator's
   "VPU busy" is checked against a hardware number for the first time.
3. **The VPU work inside GEMMs** (30–56 % ALU-slot occupancy during a pure
   `Matmul` on v6e, 0 % in the simulator) measured as a function of shape,
   so it can be modelled.
4. **Coarse time-resolved unit activity** on v6e (periodic counter sampling)
   at the smallest interval the device tolerates, to bound what the hardware
   can say about "which unit at which time" and to compare against the
   simulator's per-cycle ACCT binned to the same windows.
5. **Fusion cost** on silicon: the same operator standalone vs fused into
   its GEMM neighbour, which is what `-fuse_vpu` claims is free.

Non-goals: counterfactual fabrics (simulator only); training; multi-chip.

## 2. Ground rules (unchanged from the fidelity spec, plus two)

- Device-side chained timing by the slope method with the 2C-program
  consistency check (`t_2C/t_C ∈ [1.80, 2.05]`, else chain-C minus the
  113 µs floor, flagged). ≥ 20 repetitions, medians, p10/p90.
- Every cell records whether its working set is VMEM-resident (< 150 MB) —
  for these operators at model shapes it usually is, which is the intended
  regime (VPU as a compute/issue engine); the HBM-bound regime is E1.
- **Fusion boundary check (new).** A standalone operator is only a cell if
  XLA compiled it as its own fusion inside the chain: the probe counts the
  fusions per scan step in the compiled HLO and refuses the row otherwise
  (the G-sweep's "never reduce to a slice" rule, applied to fusion).
- **Counters travel with every trace (new).** Each cell's trace is captured
  with the session counters; `analysis/kernel_census.py` v3 reads the
  `/device:TPU:0` `counters_0` line and reports per-cell unit occupancy
  normalised by the chain program's device cycles.

## 3. Cells

Shapes are Qwen3-8B's (d 4096, 32/8 heads × 128, d_ff 12288) with Mistral's
d_ff 14336 as holdout, plus the v3 paper's Stable-Diffusion softmax rows for
continuity. `rows` = tokens in the operator's input (prefill: seq × batch;
decode: batch).

| cell | operator (JAX) | shapes | sim line | isolates |
|---|---|---|---|---|
| V1 softmax | `jax.nn.softmax` over the last axis, bf16 in/out, f32 accumulate | rows ∈ {8, 32, 256, 2048} × S ∈ {512, 2048, 8192}; plus causal-masked S=2048 | `Softmax rows S` (exists) | the softmax phase model that dominates the sim's prefill attention |
| V2 RMSNorm | x · rsqrt(mean(x²)+ε) · w, bf16, f32 stats | rows ∈ {1, 8, 32, 256, 2048, 8192} × D 4096 | `RMSNorm rows D` (new) | reduce + broadcast phases, per-row fixed cost |
| V3 RoPE | rotate-half on (rows, 32, 128) q and (rows, 8, 128) k with precomputed cos/sin | rows as V2 | `RoPE rows H HD` (new) | pure elementwise with gather-free layout |
| V4 SiLU·up | silu(g) ⊙ u, bf16 | rows as V2 × F ∈ {12288, 14336} | `SiluMul rows F` (new) | 2-in 1-out elementwise at MLP width |
| V5 residual | a + b, bf16 | rows as V2 × D 4096 | `Add n` (exists) | the E1 stream at model shapes |
| V6 v3 continuity | softmax (8, S) for S ∈ {64, 256, 1024, 4096}; LayerNorm (32, 4096, 320) | `Softmax 8 S`, `LayerNorm` (exist) | the exact cells the ISPASS artifact timed on v3 |
| V7 fusion pairs | RMSNorm→GEMM(4096²) chained vs GEMM alone; SiLU·up→down-GEMM vs GEMM alone; GEMM→residual vs GEMM alone | `Transformer` sub-DAG with `-fuse_vpu 0/1` | what an epilogue/prologue costs on silicon (the `-fuse_vpu` claim) |
| V8 VPU-in-GEMM | `Matmul M K N`, M ∈ {1, 8, 32, 256, 1024, 4096} × (K,N) ∈ {4096², 4096×12288, 8192²}, counters only (latency already in G1–G3) | `Matmul` | ALU-slot and issue occupancy vs shape for the MXU-feeding term |
| V9 periodic sampling | decode step (Qwen 512×8, 2048×32) and prefill (2048×1) with `tpu_enable_periodic_counter_sampling`, TC counters {MXU_BUSY_2, VECTOR_ALU_0..3, VECTOR_ISSUE, HOLD_VECTOR_ISSUE, VLD, VST, XLU_BUSY_0}, `interval_us` ∈ {1000, 200, 50, 10} | sim ACCT binned to the same windows | smallest interval without timing perturbation (> 2 % step time shift = perturbed); per-window unit occupancy series |
| V10 decode-only counters | all 7 Qwen and 3 Mistral decode points, trace started after the first generated token | tier-2 decode l1 runs | clean per-step VPU/MXU occupancy for decode (today's totals include the context prefill) |

Every cell: device time per step (slope), TF/s or GB/s where meaningful,
VMEM-residency flag, fusion-boundary check, and the counter row.

## 4. Simulator side

- **Frontend lines** `RMSNorm M D`, `RoPE M H HD`, `SiluMul M F` that call the
  Transformer composite's existing builders (`makeRMSNormJobs`, the rope
  `VecUnitJob`, `mk_binary_ew`) so a standalone operator is job-for-job the
  in-model one. V-test: the standalone job list equals the corresponding
  slice of a 1-layer Transformer's jobs (dims, phases, byte amounts).
- **Counter column in the census** (`kernel_census.py` v3): per trace, the
  TC counters normalised by the chain program's device cycles; per-op
  timeline join for V9/V10 so windows are labelled by op class.
- **Scorer** `fidelity/score_vpu.py`: latency verdicts per cell (PASS ≤ 10 %,
  CONDITIONAL ≤ 25 %), VMEM flag, and beside them measured mean ALU-slot
  occupancy vs the sim's VPU busy share, and measured `HOLD` vs the sim's
  VPU memstall share. V7 reports the fused-minus-unfused delta on both
  sides. V8 reports ALU occupancy vs M for the fit in §5.

## 5. Fit protocol

Fit set: Qwen shapes of V1–V5 and V8. Holdout: Mistral widths (V4 F=14336),
the v3-continuity rows (V6), all of tier 2.

1. **VPU phase costs.** The sim's `softmax_phases`, RMSNorm and elementwise
   phase tables are cycles-per-element-per-lane constants; fit them to V1/V2/
   V4 at throughput-bound sizes (rows ≥ 256), then check the rows ≤ 32
   cells, which measure the per-op fixed cost. If those miss by > 25 %, add
   a per-VPU-job fixed cost (the v3 rows also missed there: 20–45 %).
2. **MXU-feeding VPU term.** From V8: ALU cycles per systolic tile pass as a
   function of (M, K, N) → a mechanism that charges the VPU (as a sidecar,
   no traffic) during SA jobs. Validate on the tier-2 prefill points where
   the sim is 20–30 points low on VPU occupancy today.
3. **Fusion.** From V7: if an epilogue costs > 10 % of its GEMM on silicon,
   `-fuse_vpu` gets a per-fusion charge; else the sidecar model stands.
4. Re-score tier 2 (prefill attention is the target: sim 1.6–2.2× too slow,
   softmax-dominated) and the A1 prefill cells.

## 6. Sessions and budget

| session | content | TPU time | est. cost |
|---|---|---|---|
| VPU-A | V1–V6, V8 (all chained cells, traces + counters) | ~45 min | ~$3 |
| VPU-B | V7 fusion pairs; V9 sampling-interval search; V10 decode-only windows (Qwen + Mistral models re-downloaded, ~10 min) | ~75 min | ~$5 |
| VPU-C | repeats: anchors of V1/V2/V4 ×3, any refused/flagged rows re-run with alternative chain forms | ~30 min | ~$2 |
| RE-VAL | after the mechanism fixes: full tier-2 grid (both models) + tier-1 anchors re-measured on the same day, one VM per model concurrently | ~2 h | ~$10 |
| reserve | stockouts, re-captures, a second sampling-interval pass | | ~$20 |

Total ≈ $40 of the $200. The remainder is deliberately unallocated: the
largest foreseeable additional spend is a v6e-4 or v6e-8 session if the
paper needs multi-chip data, which is out of this spec's scope.

## 7. Deliverables

- `benchmarks/tpuv6e/probes/v1_vpu_ops.py` (V1–V6, V8), `v7_fusion_pairs.py`,
  `v9_counter_sampling.py`; `holdout/dh_offline.py --trace-after-first-token`
  (V10); `run_vpu.sh` driver.
- Frontend lines + V-tests; census v3 (counters, op-labelled windows);
  `fidelity/score_vpu.py`; `fidelity/sim_vpu.sh`.
- Results under `benchmarks/tpuv6e/results/fidelity/vpu/`; scorecards; a
  VPU panel added to `fidelity_map.py`.
- Fitted VPU phase constants and (if warranted) the MXU-feeding term and the
  fusion charge, pinned in `configs/tpuv6e.sh`; calibration spec §6.7
  entries; fidelity spec §6.4 results; README section.

## 8. Risks

- **XLA fuses the operator with the chain carry** (the G-sweep's elision in
  another form): the fusion-boundary check refuses such rows; the fallback is
  `lax.optimization_barrier` around the operator.
- **Everything is VMEM-resident** at these shapes: intended, flagged; HBM
  behaviour comes from E1 and the whole-model runs.
- **Periodic counter sampling is rejected by this libtpu** or perturbs timing
  at short intervals: V9 starts with a dry run of the option at 1000 µs; the
  interval floor is a result, not an assumption.
- **Decode-only windows** need the runner to start the trace mid-generate;
  vLLM's engine loop makes that a callback, not a flag — one evening of
  harness work before VPU-B.
- **Sum-of-operators over-counts** relative to fused kernels (as it did for
  v3): V7 measures the over-count so the decomposition can state it.

## 9. Order

Probe + frontend lines + V-tests (TDD) → VPU-A → score, fit VPU phase
costs → decode-only harness + V7/V9 probes → VPU-B → MXU-feeding term and
fusion charge (mechanisms with V-tests) → VPU-C repeats → re-score tier 2 →
RE-VAL session → spec/README/config updates.
