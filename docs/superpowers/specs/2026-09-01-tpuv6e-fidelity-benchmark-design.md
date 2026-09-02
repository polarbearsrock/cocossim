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
  the unit totals exactly. (DONE 2026-09-01: `Job::op_class` + `ACCTC` lines,
  stats `SCHEMA 3`, tests V32a–c; class map in the calibration spec §6.7.)
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

### 5.1 Status (2026-09-01, evening)

All six prerequisites landed on `codex/tpuv6-model`: S4 (`6d9017c` +
review fixes `b5c984d`/`6022ed6`/`7bb503d`: true-byte write-back,
`-act_share`, two pre-existing prefetch/window defects found by the review
panel), S1 (`51a9d2c`: ACCTC per-class lines, SCHEMA 3), S5 (`f9fbc3c`:
census v2), S2/S6/S3 (one commit: `-op_overhead` with OPBOUND counts,
`-kv_bw_pct` relative to the DRAM plate, `-data_overhead`, batched
prefill). Suites 66/66 + 5/5.

Post-F0 holdout, pure priors (`configs/tpuv6e.sh`, all fit knobs at their
defaults), now six points thanks to S3:

| point | predicted | measured | error |
|---|---|---|---|
| prefill 512 b1 | 15.06 ms | 19.15 | −21.4% |
| prefill 2048 b1 | 58.01 ms | 58.57 | −1.0% |
| prefill 512 b8 | 85.8 ms | 108.9 | −21.2% |
| prefill 2048 b8 | 425.8 ms | 453.0 | −6.0% |
| decode 512×8 | 11.14 ms/step | 14.10 | −21.0% |
| decode 2048×32 | 17.28 ms/step | 26.20 | −34.0% |

MAPE 17.4% (six points), 19.3% (the original four). The long prefills —
where per-kernel and host costs are amortized — are within noise; the
short-prefill and decode points miss by the near-constant ~21% that the
per-op / data terms exist to fit, and the long-context decode point adds
the KV-gather term. Every point sits on the under-predicted side, as the
physics-only model should.

Two findings from S5 that change §3.2/§6:
- **XProf semantics confirmed**: `bandwidthUtils` = [HBM read+write, VMEM
  read, VMEM write] against peaks [1638, 23296, 16128] GB/s; the `flops`
  field is a share of the trace, not a utilization — per-op MXU utilization
  is recomputed as rawFlops / rawTime / peak. The Pallas ragged-paged-
  attention custom call carries no cost model (flops = bytes = 0), so
  attention MXU/HBM utilization is `n/a` from XProf; A1 (kernel in
  isolation, timed) is the only attention utilization source. Weight
  prefetches are booked on `copy-start` DMA ops, so per-class HBM numbers
  understate real traffic; the root-level HBM utilization includes them.
- **The stored session-3 prefill traces are prefix-cache hits, not full
  prefills**: `prefill_512_1` and `prefill_2048_1` contain the same 256-token
  program (the ragged-paged-attention page size), the signature of a cache
  hit recomputing only the last page. Every tier-2 prefill trace must be
  re-captured with prefix caching verified off (the dh2 *walls* already were;
  the traces predate that fix). Decode traces are whole (63 steps captured).

### 5.2 H1 executed (2026-09-01 evening): first tier-1 fidelity map

Session H1 ran on an on-demand v6e-1 (us-east5-b, ~$4.50 including two
aborted passes): 78 G-sweep and 22 E1 points, chained, 20-rep medians, one
XProf trace per point (`results/fidelity/h1/*.csv`; traces under
`$RESULTS_DIR/fidelity/h1/traces`, 1.3 GB). Two probe defects were found
and fixed on the way and are recorded in `probes/g_sweep.py`: a scan step
that returned one output element let XLA slice the GEMM to a single dot
product (100 PFLOP/s readings), and a redeploy that nested the fixed
scripts into a subdirectory ran the stale ones. Both probes now refuse or
flag rows above peak/plate.

Simulator side: `fidelity/sim_matrix.sh` ran every H1 shape on the pinned
config; `fidelity/score_matrix.py` scored 99 GEMM cells and 21 E1 rows
(`results/fidelity/h1/tier1_scorecard_priors.csv`, and
`_ovh15000.csv` with `-op_overhead 15000` as a first fit coordinate).

Priors (no fit): G3 35 PASS / 19 CONDITIONAL / 10 FAIL; G2 3/6/1; G1
1/2/1; E1 1/2/18. The map decomposes into five mechanisms:

1. **Per-kernel fixed cost t₀ (the dominant miss).** Silicon is flat at
   31.5 µs for the 33.5 MB q projection from M=1 to 64, 15.2 µs for the
   8.4 MB kv projection, 7–9 µs for any elementwise kernel under ~4 MB; the
   sim charges nothing fixed (qwen_q −13…−26%, qwen_kv −55%, 1024³ −44%,
   small E1 −54…−98%). t = t₀ + bytes/BW with t₀ ≈ 9–10 µs (GEMM), ≈ 7 µs
   (elementwise), BW ≈ 1.35 TB/s. With `-op_overhead 15000` (8.6 µs) G3
   goes to 38/23/3 — the q/kv cells pass — while elementwise overshoots
   (+13…+25%): the fit wants a smaller t₀ for VPU ops than for GEMMs, or
   a per-class overhead.
2. **Big weight streams match.** gate/up, down and the Mistral head
   (100–270 MB) are within ±10% at every M: the sim's read streaming rate
   is silicon's.
3. **Loop order / first-job fetch (structural, sim too slow).** M×8192² at
   M ≥ 512: +17…+28%; mistral_down at M ≥ 128: +18%; the effect grows
   with K and with the number of row blocks that run compute-only on a
   resident slice while the first block paid the whole fetch (spec 6.7).
4. **Write-heavy streaming (sim too fast).** The only E1 rows that measure
   HBM (≥ 400 MB moved; every smaller row is a VMEM-resident scan carry on
   silicon, 2.2–4.5 TB/s) show add at 1.09–1.10 TB/s and exp at 1.06 TB/s
   on silicon vs ~1.5 TB/s in the sim (−25…−28%). Read-only weight streams
   agree, so this is the write path: DRAMSim3's write buffering / read-write
   turnaround is more generous than HBM's. Candidate: a write-bandwidth
   derate or an ini timing revisit (tWR/tWTR), validated on an E1b probe
   that separates read-only (reduce) from write-only (fill) streams.
5. **Two singletons.** The Qwen head (1.24 GB, N = 151936, not a multiple
   of 256) runs 17–20% slower on silicon than pure streaming predicts —
   ragged last tile or large-tensor layout effect; probe with N ∈ {131072,
   151936, 152064} in H5. 8192³ sustains 686 TF/s vs 727 at 4096³:
   sustained-power capping the sim does not model (−13%).

E1 design note: a scan carry under VMEM never touches HBM, so the cell's
small and mid sizes measure t₀ + VMEM, not HBM. `score_matrix.py` flags
rows with bytes_moved < 150 MB as `vmem_resident` and scores them
separately; the HBM verdict uses only the large rows. H5 should add a
read-only and a write-only stream probe at ≥ 512 MB.

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

Note on `-dram_enq` (found while reviewing S6): the pinned 32 beats/cycle
issue width is 2.2× the HBM plate at 1.75 GHz (14.6 beats/cycle), so a
decode KV stream can burst 32 beats into the FIFO ahead of weight fetches;
pacing KV issue at the plate (`-kv_bw_pct 99`) measured 5.6% FASTER than
uncapped on the V34a shape. The fit should try `-dram_enq` = plate (15)
before anything else, since an unphysical issue width is an easy way for
the DRAM queue to reorder traffic the hardware never sees.

Method: minimize mean absolute log-error over tier-1 cells with the
parameters above (coordinate descent is sufficient; parameter count is
small), report each parameter's sensitivity (Δ objective for ±20%), freeze
into `configs/tpuv6e_fitted.sh`, then score tier 2 once. If a tier-2 miss
exceeds 25% with no mechanism in the error budget, that is a finding (new
mechanism), not a reason to refit.

### 6.1 First fit (2026-09-01 evening, tier 1 only)

`fidelity/fit_tier1.sh` scanned `-dram_enq` ∈ {32, 15} × `-op_overhead` ∈
{0, 12250, 15750} against the 84 non-VMEM H1 cells (objective = mean
|ln(sim/silicon)|):

| combo | objective | PASS | G1 | G2 | G3 | E1 |
|---|---|---|---|---|---|---|
| priors (32, 0) | 0.190 | 39/84 | 0.218 | 0.134 | 0.186 | 0.303 |
| 15, 0 | 0.194 | 47 | 0.220 | 0.104 | 0.196 | 0.298 |
| 32, 12250 | 0.116 | 39 | 0.153 | 0.187 | 0.087 | 0.283 |
| **15, 12250** | **0.104** | **54** | 0.144 | 0.126 | 0.082 | 0.279 |
| 15, 15750 | 0.116 | 50 | 0.181 | 0.149 | 0.092 | 0.274 |

Frozen as `configs/tpuv6e_fitted.sh`. `-dram_enq 15` (the plate width)
buys G2 exactly as §6's note predicted; `-op_overhead 12250` (7 µs) is the
t₀ compromise between GEMM (~9 µs) and elementwise (~7 µs); E1 stays at
~0.28 because the write-path gap is not a knob.

Holdout scored with the fitted file untouched (six points): prefill 512
b1 −18.2%, 2048 b1 +0.6%, 512 b8 −20.6%, 2048 b8 −4.8%, decode 512×8
−15.7%, decode 2048×32 −27.7%; **MAPE 14.6%** (priors 17.4%). Still all
under-predicted: the per-op term recovers ~0.7 ms/step at decode (7 µs ×
~100 boundaries per core) of a 2.2–7 ms gap, so the remainder is the
paged-KV derate (H2, `-kv_bw_pct`), the `data` ops (tier-2 F,
`-data_overhead`), and whatever the small-M GEMM cells still hide at layer
composition. Note for the fit: op stalls in the sim overlap with `-dbuf`
prefetch streaming (the DRAM stays busy during a launch gap, as XLA's
async copies would), so a stall costs less at layer level than in a
single-op microbenchmark; the tier-1 t₀ may therefore be a lower bound on
the whole-model value.

### 6.2 Correction (2026-09-01, late): H1's chain length did not amortize the host floor

E1's rows under 4 MB are flat at 7.1 µs per step, and 113 µs / 16 = 7.06 µs:
the B1 host-dispatch floor divided by CHAIN. A chained call is one
executable, so its ~113 µs launch + completion cost lands once per call
and leaks floor/CHAIN into every per-step number — 7 µs at chain 16, 14 µs
at chain 8 (which is where C5's "t₀ ≈ 10 µs" came from too). Consequences:
- the small-op FAIL cells of §5.2 (qwen_kv, qwen_q, 1024³, small E1) are
  mostly this leak: qwen_q 31.5 − 7 ≈ 24.5 µs vs sim 24.7 µs;
- the "per-kernel t₀ ≈ 7–10 µs" of §5.2 item 1 is NOT established, and
  `-op_overhead 12250` in `tpuv6e_fitted.sh` largely compensates an
  artifact; the fit is void until re-measured;
- B2's 31–35 µs "per kernel" was host issue throughput on async chains,
  not device time. The device-side per-kernel cost is currently unknown.
Method going forward (all probes): time each cell at CHAIN and 2·CHAIN and
report per_step = (t₂C − tC) / CHAIN (pure device per-step) and intercept =
2·tC − t₂C (the per-call fixed cost, measured); choose CHAIN so a call runs
≥ 1 ms. Session H1b re-runs G1/G2/G3/E1 this way (plus E1b read-only /
write-only streams for the write-path gap) alongside H2.

### 6.3 Re-measurement and tier 2 (2026-09-01, session "tier1" + "tier2_qwen" + "tier2_mistral", three concurrent VMs)

All three data sets were collected concurrently on separate on-demand v6e-1
VMs in us-east5-b (us-east5-a was stocked out); raw CSVs are under
`benchmarks/tpuv6e/results/fidelity/{tier1,tier2_qwen,tier2_mistral}/`.

**Slope method, corrected.** The slope estimator assumes the C- and
2C-step scans compile to the same per-step program. They do not always:
every Mistral G3 shape ran 2.1–2.75× longer at 2C than at C
(`t_2C/t_C` in `g_sweep_slope.csv`), while the chain-16 per-step
reproduced the H1 chained numbers to 1%. `score_matrix.py` therefore uses
the slope only when `t_2C/t_C ∈ [1.80, 2.05]` and otherwise falls back to
the chain-C per-step minus the nominal 113 µs floor (a ≤ 1% correction on
those ≥ 1 ms calls), flagging the row (`method` column; 24 of 120 rows).
The H1 numbers for the ≥ 1 ms shapes stand; the floor leak only mattered
where per-step × chain was small (E1 below 2²⁰ elements, kv rows).

**Tier 1, corrected map** (`tier1/scorecard_tier1_slope.csv`; error =
(sim − silicon)/silicon):

| cell | result |
|---|---|
| G3 decode shapes M ≤ 64 | q/gate_up/down PASS (±8%); kv −11…−15% (8 MB streams: silicon 1.06 TB/s, a ~1 µs per-kernel cost the sim lacks); Qwen head −13…−19% (ragged N = 151936 streams at 1.15 TB/s on silicon vs 1.35 in the sim); Mistral head +5…+16%. |
| G2 / G1 mid-M (512–2048 rows, K = N = 4096/8192) | sim +13…+37% too SLOW: silicon overlaps weight streaming and compute (512×8192² runs at 585 TF/s **and** 1.14 TB/s simultaneously), the sim's SA time is ≈ stream + compute (memstall 0.48). Small squares worse: 1024³ +66%, 2048³ +50% (silicon 572/725 TF/s). 8192³ −11.5% (sim sustains 788 TF/s, silicon 698). |
| E1 HBM rows (≥ 2²⁵ elements) | sim −15…−25% too fast: silicon streams write-heavy elementwise at 1.16–1.17 TB/s, the sim at 1.5. Rows below 2²⁵ elements are VMEM-resident on silicon (up to 9.6 TB/s effective) and get no HBM verdict. |
| A1 prefill (Pallas `flash_attention`) | sim +30% (S 512), +88% (S 2048), +79% (S 8192) at B = 1; +8…+20% at B = 8. Consistent with the in-model census: RPA prefill attention costs 250–290 µs per 2048-token sequence-layer on silicon, the sim ≈ 450–470 µs (head_dim-128 contraction half-fills the 256-deep array). |
| A1 decode (Pallas `paged_attention`, GQA 32/8, page 16) | **Not the models' kernel.** vLLM runs `ragged_paged_attention` (RPA): 166 µs per layer at S 2048 B 32 in the Qwen decode census vs 1470 µs for the probe's kernel. A1 decode and K1 are kernel-relative data only; decode attention is calibrated against the census. |

**Tier 2, priors** (`score_tier2.py`; silicon = per-step device time from
the census, sim = full run or the l1/l2/lh extrapolation, which matched
seven full runs within ±2%). Two vLLM facts shape the census: prefill is
chunked at 8192 tokens (one forward = the whole trace), and the decode
batch is padded to a token bucket (batch 8 runs as an M = 16 program) with
partial-batch steps interleaved while long contexts prefill — the step is
the padded-batch program plus its LM head and glue. Mistral ran through
vLLM's PyTorch-wrapper path (`step_fun_impl`, `*ParallelLinear`), Qwen3
through the native JAX path; both are recognised.

| model | prefill | decode |
|---|---|---|
| Qwen3-8B (fit set) | 256×1 −22%, 512×1 −16%, 512×4 −13%, 512×8 −16%, 1024×1 −6%, 2048×1 +3%, 2048×4 −4%, 4096×1 +21% | 512×1 −27%, 512×8 −28%, 512×32 −31%, 2048×8 −24%, 2048×32 −13%, 4096×16 +1%, 8192×8 +9% |
| Mistral-7B-v0.3 (holdout) | 512×1 −13%, 2048×1 −12%, 512×8 −35% | 512×8 −28%, 2048×32 −15%, 8192×8 +6% |

Priors MAPE: Qwen 15.5% (bias −11%), Mistral 18.1% (bias −16%).

**Where the residual sits** (per-class attribution, census vs sim
`ACCTC`; silicon class times overlap with XLA's async weight prefetch, so
they are indicative, the totals are exact):

1. *In-model weight streaming.* Silicon streams the 15 GB of Qwen weights
   at ≈ 1.15 TB/s inside a decode step (isolated chained GEMMs reach
   1.33–1.38 TB/s; the difference is XLA's `copy`/layout ops — "data",
   ~1 ms per step — and per-op gaps), the sim at ≈ 1.5 TB/s. This is the
   whole −25…−30% at short-context decode and short prefill. A per-op
   boundary stall (`-op_overhead`, section 5) of ~5 µs reproduces the
   in-model loss without touching the isolated-kernel cells.
2. *Decode attention (RPA kernel).* Per layer ≈ 20 µs fixed + ~5 µs per
   sequence at short context (512×32: 184 µs/layer for 67 MB of KV), but
   ≈ 1.6 TB/s effective at long context (2048×32: 166 µs/layer for
   268 MB; 8192×8: 146 µs). The sim is 3.7× too fast at 512×32 and 2×
   too slow at 8192×8: its per-head KV jobs are demand-fetched (KV panels
   are excluded from `-dbuf` prefetch by design) and stream at ≈ 0.9 TB/s.
3. *Prefill attention.* Sim 1.6–2.2× too slow at S ≥ 2048 (item A1
   above); at 4096×1 it is the entire +21%.
4. *Mid-M GEMM overlap* (tier-1 G2): only visible in tier 2 where a
   prefill program is compute-heavy; masked by item 3 at 4096×1 and by
   item 1 at short prompts.

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
