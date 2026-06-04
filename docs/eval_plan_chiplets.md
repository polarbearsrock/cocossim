# Evaluation & Case Study Plan: COCOSSim Multi-Chiplet (ASPLOS)

**Status:** Planning  
**Branch:** `dev-chiplets`  
**Companion docs:** `docs/asplos_chiplets_review.md`, `docs/comparison_with_chiplet_frameworks.md`

---

## System Architecture Overview

### End-to-End Simulation Pipeline

The full system takes a PyTorch model and a parallelism strategy description as input and produces cycle-accurate performance, bandwidth, latency, and energy estimates for a multi-chiplet execution.

```
[Arch + Parallelism Description]   [PyTorch Model]
         (2+ pages)                      |
                                         v
                               [Compiler / Mapper]
                                 (egraph-based)
                                /                \
                               v                  v
                         [Job Graph]          [Collectives]
                               \                  /
                                v                v
                          [COCOSSim + ChipletSim]
                      (BW, Latency, Power — UCIe impl.)
```

**Compiler / Mapper responsibilities:**
1. Parse the PyTorch model into an execution graph (egraph)
2. Given the target parallelism strategy, determine which collective primitives are required at each layer boundary
3. Schedule communication (inject into ChipletSim) and compute (inject into COCOSSim) respecting data dependencies
4. Architect a unified mapper that works across data parallelism, tensor parallelism, pipeline parallelism, expert parallelism, and hybrid combinations

The collectives chosen by the compiler are also used independently to validate against Cascade.

### ChipletSim — What It Models

| Component | Model |
|-----------|-------|
| Link bandwidth | UCIe 1.0/1.1 (configurable: 8GT×8, 16GT×16, 32GT×32) |
| Link latency | Credit-based flow control, ~60-cycle round-trip |
| Power | Per-link dynamic + static (7nm/16nm) |
| Collective primitives | AllReduce, AllGather, AllScatter, Broadcast, OneZone |
| Interconnect topologies | Ring, 2D Mesh |

### Supported Collective Primitives

| Primitive | Parallelism use case |
|-----------|---------------------|
| **AllReduce** | Tensor parallelism (linear layer output synchronization) |
| **AllGather** | Tensor parallelism (weight/activation reconstruction) |
| **AllScatter** (×4 variants) | Distributed attention, sequence parallelism |
| **Broadcast** | Parameter broadcast in data parallelism |
| **OneZone** | Localized communication within a chiplet group |

NCCL (NICO implementation) serves as the reference implementation for collective primitive behavior and bandwidth efficiency benchmarks.

### Comparison Targets

| Target | Role |
|--------|------|
| **Cascade** | Validation of collective primitive behavior; workload comparison for Diffusion/ResNet/GPT |
| **SuperMesh [MICRO'25]** (SCALE-Sim + BookSim) | Primary decoupled simulation baseline |
| **NCCL / NICO** | Collective primitive reference |

---

## Research Questions

The RQs follow a three-level argument structure: **What** (observable error) → **Why** (the coupling mechanism) → **Implication** (wrong design decisions). RQ1 establishes validity; RQ2–RQ4 are the core scientific contribution; RQ5–RQ8 are downstream consequences and design-space questions.

### Validity

| RQ | Question |
|----|----------|
| RQ1 | How accurate is COCOSSim for multi-chiplet workloads? (compositional + spec-level validation) |

### Core Contribution — The Decoupling Problem

| RQ | Level | Question |
|----|-------|----------|
| RQ2 | *What* | How much do decoupled simulators over/underestimate throughput ("the decoupling tax"), and for which workload types is the error largest? |
| RQ3 | *Why* | How does simultaneous DRAM pressure and AllReduce congestion interact — and is this interaction captured by decoupled tools? |
| RQ4 | *Mechanism* | How much compute-communication overlap is actually achievable under HBM2 memory pressure, compared to what decoupled simulators implicitly assume? |

**Connecting narrative:** RQ2 measures the error; RQ3 identifies the cause (DRAM contention and AllReduce traffic co-occur, not sequentially); RQ4 explains the mechanism (overlap is not free when the memory bus is already saturated — decoupled tools assume it is).

### Design-Space Implications

| RQ | Question |
|----|----------|
| RQ5 | Do decoupled simulators make the correct parallelism strategy recommendation (TP vs. PP vs. hybrid), or does the decoupling tax shift the optimal configuration? |
| RQ6 | How does topology choice affect LLM inference latency and bandwidth utilization — and does unified vs. decoupled simulation change the topology recommendation? |
| RQ7 | What UCIe link configuration is actually needed — when does NoP stop being the bottleneck, and does the answer change under realistic memory-network coupling? |
| RQ8 | What is the energy cost of communication vs. compute on a multi-chiplet system? |

---

## Workloads

### Primary

| Name | Type | Parallelism | Why |
|------|------|-------------|-----|
| **GPT / LLaMA-7B prefill (S=4096)** | Attention-heavy, compute-bound | Tensor parallel (TP) | Canonical LLM; stresses Q·Kᵀ and AllReduce for TP; compare to Cascade |
| **GPT / LLaMA-7B decode (S=1, KV-cache)** | Memory-bandwidth-bound | Tensor parallel (TP) | Irregular traffic; hardest for decoupled simulators |
| **LLaMA-70B prefill (S=2048)** | Large model, must be sharded | TP + PP hybrid | Requires ≥4 chiplets; tests pipeline + tensor hybrid |
| **ResNet-50** | Compute-bound, spatial | Data parallel (DP) | Control workload — minimal inter-chiplet traffic; compare to Cascade |
| **Diffusion model (UNet)** | Mixed compute + memory | Data / tensor parallel | Tests non-transformer topology; compare to Cascade; also used in speculative decoding case study |

### Case Study Workloads

| Name | Parallelism | Insight target |
|------|-------------|----------------|
| **MoE (Mixture of Experts)** | Expert parallel | Which collectives dominate; how expert routing interacts with AllReduce; scaling behavior with expert count |
| **Speculative decoding** | Prefill + decode interleaved | Communication pattern when draft and target model run on separate chiplet groups; prefill–decode handoff latency |
| **Pipeline parallelism case study** | PP | Bubble overhead vs. communication savings; when PP beats TP in latency |

### Secondary / Microbenchmarks

| Name | Purpose |
|------|---------|
| **Synthetic: collective benchmarks** | Measure AllReduce / AllGather / AllScatter / Broadcast latency and bandwidth on ring vs. mesh; corroborate against prior publications (due before workload experiments) |
| **Synthetic: AllReduce sweep** | Directly sweep message size × topology × chiplet count; establish reference curves for RQ7 |
| **Synthetic: attention-only** | Isolate attention communication from projection layers |

---

## Baselines

| Baseline | Tool | Notes |
|----------|------|-------|
| **COCOSSim single-chip** | This repo, existing path | Already validated at 13% vs. TPU v3 |
| **SCALE-Sim v3 + BookSim2** | SCALE-Sim v3 + BookSim2 | Primary decoupled baseline — SCALE-Sim v3 uses Ramulator (timing-accurate DRAM), closing the memory model gap; see §SuperMesh below |
| **GEMINI** | Public analytical tool | Secondary baseline — bottleneck model, no network simulation |
| **Simba hardware** | Published NVIDIA numbers | Hardware reference for systolic-array multi-chiplet (see comparison doc) |

---

## Validation Strategy (RQ1)

COCOSSim does not model GPU compute (SIMT/tensor cores), so end-to-end comparison against GPU-based multi-chiplet hardware (MI300X, Ponte Vecchio) is not meaningful and should not be attempted. The validation argument rests on two orthogonal pillars:

### Pillar A — Compositional Validity
The multi-chiplet model is composed of two independently validated components:
- **Compute core**: already validated at 13% error vs. Google TPU v3 (single-chip path, existing results)
- **UCIe links**: validated against UCIe 1.0/1.1 spec numbers (latency, bandwidth, efficiency — see Pillar B)

Claim: if each component is accurate in isolation, the integrated model inherits that accuracy. This is standard methodology for simulation papers.

### Pillar B — UCIe Link Micro-Validation
Validate the UCIe physical layer model independently of compute, using published spec values as ground truth.

**What to compare:**

| Metric | UCIe Spec (1.0, 16GT×16) | COCOSSim Model | Target delta |
|--------|--------------------------|----------------|--------------|
| Round-trip latency | 60–80 cycles | model output | < 10% |
| Effective bandwidth | 28 GB/s (87.5% efficiency) | model output | < 5% |
| Credit return latency | 20–50 cycles | model output | within range |
| Serialization delay (256B packet) | ~9 cycles | model output | exact |

**What to run:** synthetic AllReduce microbenchmark on 2-chiplet linear topology; vary packet sizes (64B, 256B, 1024B); compare latency and bandwidth efficiency against spec table in `include/chiplets/README.md`.

**Note on Simba:** NVIDIA Simba (2019) is a systolic-array multi-chiplet system with published throughput numbers. It is the closest hardware validation target if a full end-to-end comparison is desired later. SIAM used Simba for calibration. This is deferred — see `docs/comparison_with_chiplet_frameworks.md`.

---

## SuperMesh Baseline — Full Implementation Plan

### What the pipeline does
SCALE-Sim v3 models a systolic array with Ramulator as its DRAM backend — it outputs compute cycles and timing-accurate DRAM traces per layer. BookSim2 is a cycle-accurate network simulator: given a topology and traffic injection pattern, it outputs per-packet latency and network utilization. The decoupled pipeline chains them sequentially: compute phase (SCALE-Sim v3) → traffic derivation → network phase (BookSim2), with no feedback between stages.

### Why SCALE-Sim v3 over v2

SCALE-Sim v3 uses Ramulator for memory modeling (same class as COCOSSim's DRAMSim3 — both model bank conflicts, row buffer hit/miss, queue depth). This directly closes the most significant fairness gap from using an older analytical-bandwidth ScaleSim. The remaining differences between the two simulators are then purely about coupling methodology — which is exactly the claim being evaluated.

### Where the tools differ from COCOSSim

| Aspect | SCALE-Sim v3 + BookSim2 | COCOSSim |
|--------|-------------------------|----------|
| Compute model | SCALE-Sim v3 (WS/OS, cycle-accurate per tile) | COCOSSim state machine (same class) |
| Memory model | Ramulator (timing-accurate, bank/row modeling) | DRAMSim3 (timing-accurate, bank/row modeling) |
| Network model | BookSim2 (cycle-accurate router simulation) | UCIe links (credit-based flow control, cycle-accurate) |
| Coupling | **Decoupled: compute → traffic derivation → network** | **Unified: all tick together per cycle** |
| Stall propagation | None — network congestion does not stall compute | UCIe credit exhaustion stalls compute state machine |
| DRAM-network interaction | None — DRAM pressure does not affect network timing | HBM queue depth delays AllReduce injection |

The compute and memory models are now equivalent in class. Any delta in multi-chiplet prediction is attributable to coupling methodology alone. This is a clean, defensible comparison.

**Residual difference to disclose:** Ramulator and DRAMSim3 are both timing-accurate DRAM simulators but different implementations with potentially different timing parameters. Run both on the single-chip path (E1 micro-validation) to quantify any residual DRAM model delta before multi-chiplet experiments.

### Parameter matching — required for fairness

Every architectural parameter must be identical across both simulators.

| Parameter | COCOSSim value | SCALE-Sim v3 config field | BookSim2 config field |
|-----------|---------------|--------------------------|----------------------|
| Array size | `sa_sz=64` (64×64) | `ArrayHeight: 64`, `ArrayWidth: 64` | — |
| Dataflow | WS (`-ws 1`) | `Dataflow: ws` | — |
| Frequency | 1 GHz | `Frequency: 1` | — |
| SRAM / buffer | `buffer_size_bytes` (default 64MB) | `IfmapSramSz`, `FilterSramSz`, `OfmapSramSz` | — |
| DRAM backend | DRAMSim3, HBM2 config | Ramulator, HBM config | — |
| Memory bandwidth | HBM2 spec (set in DRAMSim3 config) | HBM config in Ramulator | — |
| Number of chiplets | configurable | one SCALE-Sim v3 run per chiplet | `k: <num_chiplets>` |
| Topology | `TopologyType` (ring / mesh) | — | `topology: ring` or `mesh` |
| Link bandwidth | `UCIePhyConfig` (e.g. 28 GB/s effective for 16GT×16) | — | `channel_width` equivalent |
| Link latency | ~60 cycles (UCIe spec) | — | `router_latency` + `link_latency` |
| Data type | BF16 / FP16 (2 bytes) | `Precision: 16` | — |

SRAM configuration: divide `buffer_size_bytes` equally across the three SCALE-Sim v3 SRAM pools (ifmap / filter / ofmap) as a first approximation — document the split explicitly in the paper.

DRAM config: use the same HBM2 device spec in both Ramulator and DRAMSim3. Both support HBM2 device configs; use the same channel count, burst length, and timing parameters to minimize residual DRAM model delta.

### Traffic generation bridge (ScaleSim → BookSim)

This is the hardest part and must be implemented carefully. ScaleSim does not directly output inter-chiplet traffic — it outputs DRAM access patterns for a single chiplet. The AllReduce volumes must be derived from the tensor partition strategy.

**For Tensor Parallel (TP=N):**
Each chiplet holds 1/N of the weight matrix (column or row parallel). After each linear layer, an AllReduce is needed over all N chiplets.

```
AllReduce volume per layer = output_tensor_bytes
  = M × (N_out / TP) × data_bytes  (each chiplet holds 1/TP of output cols)
  Ring AllReduce = 2 × (TP-1)/TP × total_output_bytes  (standard formula)
```

For a 4096-dim hidden layer with TP=4, BF16:
```
  output = 4096 × 4096 × 2 bytes = 32 MB
  ring AllReduce = 2 × (3/4) × 32 MB = 48 MB total traffic
```

**Traffic injection into BookSim:**
BookSim accepts traffic as injection rate (flits/cycle) or explicit trace (cycle, src, dst, message_size). Use explicit trace mode for accuracy:
1. From ScaleSim compute timeline, determine the cycle at which each layer finishes on each chiplet
2. At that cycle, inject the AllReduce messages for that layer (ring pattern: chiplet i sends to chiplet (i+1)%N)
3. AllReduce message size = volume / (TP-1) for ring schedule
4. BookSim reports when each message is delivered — this gives AllReduce latency
5. Total decoupled latency = sum of (ScaleSim compute cycles + BookSim AllReduce latency) per layer

**For Pipeline Parallel (PP=N):**
Traffic is activation tensors passed between pipeline stages (chiplet i → chiplet i+1). Volume = activation tensor size per layer boundary. No AllReduce needed; inject as point-to-point messages.

### Scripts to write

| Script | Purpose |
|--------|---------|
| `scripts/cocossim_to_scalesimv3.py` | Convert COCOSSim `.txt` input to SCALE-Sim v3 topology CSV and architecture config |
| `scripts/scalesimv3_to_traffic.py` | Parse SCALE-Sim v3 output, compute AllReduce volumes per layer, emit BookSim2 explicit trace |
| `scripts/booksim_to_latency.py` | Parse BookSim2 output, extract per-message latency, compute total network time |
| `scripts/supermesh_run.py` | End-to-end driver: input file → SCALE-Sim v3 → traffic derivation → BookSim2 → total latency |
| `scripts/compare_simulators.py` | Run both COCOSSim and SCALE-Sim v3 + BookSim2 on same input, output comparison CSV |

### Fairness checklist before running any comparison

- [ ] ScaleSim and COCOSSim use identical `sa_sz`, dataflow, frequency, buffer size
- [ ] BookSim topology matches COCOSSim `TopologyType` exactly (same number of nodes, same connectivity)
- [ ] BookSim link bandwidth matches COCOSSim `UCIePhyConfig` effective bandwidth (not raw GT/s — use 87.5% efficiency factor)
- [ ] BookSim router latency + link latency matches COCOSSim UCIe round-trip latency (~60 cycles for 16GT×16)
- [ ] AllReduce volume formula verified against known values (e.g. for a 4096×4096 matmul TP=4, hand-check the numbers)
- [ ] Both simulators run on identical workload inputs (same layer dimensions, same sequence lengths)
- [ ] ScaleSim SRAM config documented and justified (split of buffer_size_bytes)
- [ ] Memory bandwidth in ScaleSim matches COCOSSim HBM2 spec (128 GB/s per chiplet)

### Known residual gaps to disclose in the paper

With SCALE-Sim v3, the major unfair gap (analytical vs. timing-accurate DRAM) is closed. One residual difference remains:

1. **Ramulator vs. DRAMSim3**: Both are timing-accurate DRAM simulators modeling the same HBM2 device, but they are different implementations. Run both on a single-chip matmul before multi-chiplet experiments and report the delta. If it is <5%, treat them as equivalent and note this in the methodology. If larger, report it as a separate factor in the E2 breakdown.

2. **Credit-based flow control vs. virtual-channel router**: BookSim2 models a router with virtual channels; COCOSSim models UCIe credit-based transport. Both are cycle-accurate but the congestion mechanisms differ at the packet level. At the traffic volumes typical of LLM AllReduce on 4–8 chiplets, this difference is expected to be small — but should be noted in the paper.

### Integration into E2

E2 now runs three variants:
1. **COCOSSim unified** — the proposed simulator
2. **ScaleSim + BookSim decoupled** — the full SuperMesh-style baseline
3. **GEMINI analytical** — the fast analytical lower bound

The comparison table per workload reports predicted cycles/latency for all three, with COCOSSim as ground truth (validated via E1) and the delta to the other two as the main result.

---

## Experiment 1 — UCIe Link Micro-Validation (RQ1)

**Goal:** Establish that the UCIe model matches spec before any multi-chiplet workload runs.

**Setup:** 2-chiplet linear topology, synthetic packet stream, vary packet size {64B, 256B, 1024B, 4096B}, UCIe configs {8GT×8, 16GT×16, 32GT×32}.

**What to measure:** Observed latency (cycles), achieved bandwidth (GB/s), credit utilization.

**Compare against:** UCIe 1.0 spec table (in `include/chiplets/README.md`) and efficiency formula.

**Key figure:** Table matching the validation table in the README — observed vs. spec for each config.

**Dependency:** UCIe link tick working in isolation (does not require full ChipletArch integration).

---

## Experiment 2 — The Decoupling Tax (RQ2, RQ3, RQ4)

**Goal:** Quantify how much decoupled (SuperMesh-style) simulation mispredicts throughput vs. unified simulation, and attribute the error to its root cause (memory-network coupling).

**Setup:**
- 4-chiplet ring, UCIe 16GT×16
- Workloads: LLaMA-7B decode, LLaMA-7B prefill, ResNet-50 (control)
- Three-way comparison: COCOSSim unified | ScaleSim+BookSim decoupled | GEMINI analytical

**What to measure:**
- Total predicted latency (cycles) from all three (RQ2)
- Decoupling tax vs. COCOSSim: (decoupled − unified) / unified × 100%
- Analytical error vs. COCOSSim: (GEMINI − unified) / unified × 100%
- Breakdown of cycles COCOSSim reports that decoupled misses: UCIe stall cycles, DRAM-idle cycles caused by compute stall (RQ3)
- Achieved compute-communication overlap fraction vs. what decoupled tool assumes (RQ4)

**Sweep:** TP degree ∈ {2, 4, 8}, sequence length ∈ {512, 2048, 4096} to vary UCIe load.

**Expected finding:**
- ResNet: ~0% decoupling tax (UCIe lightly loaded; validates tool agreement at low utilization)
- Prefill: moderate tax, grows with TP degree as AllReduce volume increases
- Decode: largest tax, increases with sequence length (KV-cache pressure + AllReduce interleaved)
- Mechanism (RQ4): overlap efficiency drops as DRAM utilization rises — decoupled tools assume full overlap, COCOSSim shows the real curve

**Residual confounder:** Using SCALE-Sim v3 (Ramulator) rather than v2 eliminates the analytical-memory gap. The remaining delta between COCOSSim and SCALE-Sim v3 + BookSim2 is attributable to coupling methodology. Any residual from Ramulator vs. DRAMSim3 is quantified in the single-chip calibration step and reported separately.

**Key figures:**
- Grouped bar chart — three bars per workload (COCOSSim / ScaleSim+BookSim / GEMINI); annotate the gap (RQ2)
- Stacked cycle breakdown showing where decoupled model loses accuracy (RQ3)
- Overlap efficiency vs. DRAM utilization curve (RQ4)

**Dependencies:** ChipletArch integration + AllReduce + full ScaleSim+BookSim pipeline (see §SuperMesh above).

---

## Experiment 3 — Topology Sweep (RQ6)

**Goal:** Which topology wins for which workload — and does the decoupled simulator give the same topology recommendation?

**Sweep axes:**
```
chiplets  ∈ {2, 4, 8}
topology  ∈ {ring, 2D mesh, star}
workload  ∈ {prefill-S4096, decode-S1}
```

**Metrics:** Total latency, UCIe link utilization (%), bisection bandwidth usage.

**Expected finding:**
- Ring wins at 4 chiplets for AllReduce-heavy prefill
- Mesh wins at 8+ chiplets (lower diameter)
- Star suits 2–4 chiplets with an I/O-heavy chiplet
- Decode is largely topology-insensitive (UCIe rarely the bottleneck)
- The decoupled simulator may rank topologies differently because it ignores back-pressure from network congestion into compute stalls

**Key figure:** Heatmap — rows = topologies, columns = chiplet counts, cells = normalized latency. One per workload. Side-by-side: COCOSSim ranking vs. SCALE-Sim+BookSim ranking to expose disagreements.

---

## Experiment 4 — UCIe Configuration Sweep (RQ7)

**Goal:** Find the bandwidth saturation point per workload phase.

**Setup:** 4-chiplet ring (fixed), UCIe configs {8GT×8, 16GT×16, 32GT×16, 32GT×32}, workloads: LLaMA-7B prefill TP=4, LLaMA-7B decode TP=4.

**What to measure:** Total latency vs. UCIe bandwidth, AllReduce fraction of total latency, link utilization.

**Expected finding:** Prefill saturates around 32 GB/s; decode saturates much lower (~8–16 GB/s). Going beyond the knee gives <5% improvement.

**Key figure:** Knee curve — latency vs. UCIe bandwidth, one line per workload phase.

---

## Experiment 5 — Parallelism Strategy Comparison (RQ5)

**Setup:** 4 chiplets, strategies {TP=4, PP=4, TP=2×DP=2}, models {LLaMA-7B, LLaMA-70B}.

**What to measure:** Latency per strategy per phase, communication volume breakdown, per-chiplet memory pressure.

**Expected finding:** TP wins for prefill; PP has lower comms but pipeline bubble cost; hybrid needed for 70B at 4 chiplets.

**Key angle (RQ5):** Run the same sweep through SCALE-Sim+BookSim. If the two simulators rank strategies differently — particularly at the TP=4 setting where AllReduce and DRAM pressure peak simultaneously — that directly demonstrates that the decoupling tax affects practical design decisions, not just headline numbers.

---

## Case Study 1 — The Compute-Interconnect Feedback Loop (RQ3, RQ4)

**Goal:** The "money figure" — a concrete cycle-level demonstration of why unified simulation matters.

**Scenario:** LLaMA-7B decode on 4-chiplet ring, TP=4.

**Narrative to show:**
1. AllReduce for attention output projection saturates UCIe credits
2. Simultaneously, DRAM fetches KV-cache for the next decode step
3. UCIe credit exhaustion stalls compute on all 4 chiplets
4. Stalled compute means no new DRAM requests — HBM goes idle (wasted bandwidth)
5. Decoupled simulator misses steps 3→4: it sees no stall, predicts higher throughput

**What to show:** Annotated cycle trace — compute state, UCIe credit count, DRAM queue depth over the same time window. Highlight the window where the feedback loop occurs.

---

## Case Study 2 — Energy vs. Latency Pareto Frontier (RQ8)

**Setup:** LLaMA-7B prefill on 4 chiplets. Sweep UCIe speed × topology × parallelism strategy.

**What to show:** Pareto frontier of total energy vs. latency. Identify dominated configurations. Practical takeaway: which config is energy-optimal vs. latency-optimal at this scale.

---

## Summary Table

| # | Type | RQ | Workload | Key Figure | Dependency |
|---|------|----|----------|------------|------------|
| E1 | Validation | RQ1 | Synthetic packets | UCIe spec match table | UCIe tick only |
| E2 | Eval | RQ2, RQ3, RQ4 | LLaMA decode + prefill + ResNet | Decoupling tax bar chart + stall breakdown + overlap curve | Full integration + decoupled pipeline |
| E3 | Eval | RQ6 | LLaMA prefill + decode | Topology heatmap (COCOSSim vs. decoupled ranking) | Topology sweep |
| E4 | Eval | RQ7 | LLaMA prefill + decode TP=4 | UCIe saturation knee curve | UCIe config sweep |
| E5 | Eval | RQ5 | LLaMA-7B + 70B | Per-strategy latency breakdown; decoupled vs. unified ranking | TP/PP/hybrid impl |
| CS1 | Case study | RQ3, RQ4 | LLaMA decode cycle trace | Feedback loop trace — compute state + UCIe credits + DRAM queue (money figure) | Cycle-level logging |
| CS2 | Case study | RQ8 | LLaMA-7B prefill | Energy-latency Pareto | Energy model + full sweep |

---

## Prerequisites Before Any Experiment Runs

**COCOSSim / ChipletSim side:**
1. Wire `ChipletArch::tick()` into the main sim loop alongside `dramsim3::tick()`
2. Implement ring AllReduce over `ChipletTopology` UCIe links
3. Implement remaining collective primitives: AllGather, AllScatter, Broadcast, OneZone
4. Add 2D mesh topology to `ChipletTopology` (ring is step 2; mesh extends it)
5. Add per-chiplet cycle logging (compute state, UCIe credit count, DRAM queue depth)

**Compiler / Mapper side:**
6. Build PyTorch → egraph parser (job graph extraction from PyTorch model)
7. Implement collective selection logic: given parallelism strategy + layer boundary, determine which primitive and message size to inject
8. Build mapper that handles TP, PP, DP, expert parallel, and hybrid configurations
9. Validate mapper output (collectives) against Cascade on at least one workload (ResNet or GPT) before using for full simulation experiments

**SCALE-Sim v3 + BookSim2 side:**
10. Configure Ramulator backend with matching HBM2 device spec (step already partially done — `npu_v3.py` written)
11. Verify SCALE-Sim v3 single-layer compute cycles match COCOSSim on a known matmul (tolerate <5% delta)
12. Write `scalesimv3_to_traffic.py` — AllReduce traffic derivation from SCALE-Sim v3 output
13. Write `booksim_to_latency.py` — parse BookSim2 output into total network latency
14. Write `supermesh_run.py` — end-to-end driver chaining all steps
15. Validate parameter matching end-to-end: 2-chiplet matmul through both tools

**E1 (UCIe micro-validation) can run before 1–15 are complete — good early smoke test.**
**Collective microbenchmarks (secondary workloads) can run after steps 1–5, before compiler is ready.**
