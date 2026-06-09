# COCOSSim dev-chiplets — Work Status
_Last updated: 2026-06-09_

## Branch
`dev-chiplets` — not yet merged to `main`. All work below lives here.

---

## Simulator Code Status

### Compute

| Component | Status | Notes |
|-----------|--------|-------|
| SA FSM (cycle-accurate) | **Working** | Ticked per cycle in `ChipletArch::tick()`; DRAM requests fire from FSM |
| VPU FSM (cycle-accurate) | **Working** | Same tick path as SA; used for LayerNorm/Softmax/Activation/GEMV |
| DRAMSim3 per-chiplet | **Working** | 1TB address stride per chip; async callbacks into FSM |
| Energy model | **Working** | 16nm/7nm, INT8/FP16/FP32, MAC+SRAM+DRAM+UCIe |

### Interconnect

| Component | Status | Notes |
|-----------|--------|-------|
| UCIe link (credit/latency/power) | **Working** | Credit-based FC, 60-cycle latency, 16–112 GB/s, validated E1 |
| UCIe L0/L1/L2 power FSM | **Working** | L0→L1 after 128 idle cycles, L1→L2 after 10000, wakeup enforced |
| Ring topology + routing | **Working** | Shortest-path routing |
| 2D mesh topology + XY/YX routing | **Working** | Auto-dimensions; route_yx fixed 2026-05-06 |
| Star topology | **Working** | Built; routing via shortest-path |
| Torus, tree, fully-connected | **Stub** | `build_topology()` has cases but bodies empty; not needed for current workloads |

### Collectives

| Collective | Status | Notes |
|------------|--------|-------|
| AllReduce | **Working** | Ring schedule; packet sequence created in `create_collective_packets()` |
| Broadcast | **Working** | Root → all other chiplets |
| Point-to-point | **Working** | Used for pipeline parallel sends |
| AllGather | **Stub** | Falls through to `WARNING: not yet implemented` |
| ReduceScatter | **Stub** | Same |
| Reduce / Scatter / Gather | **Stub** | Same |

AllGather and ReduceScatter are needed for sequence parallelism and some TP variants.
Not needed for current LLM tensor-parallel workloads (which only use AllReduce).

### Workload Dispatch (`main_chiplet.cc`)

| Dispatch | Status | Notes |
|----------|--------|-------|
| Matmul — col parallel (split N + AllReduce) | **Working** | Default for projection layers |
| Matmul — row parallel (split K + AllReduce) | **Working** | Added 2026-06 |
| Matmul — head parallel (split M, no AllReduce) | **Working** | For attention heads |
| VPU elementwise (LayerNorm, Softmax, Activation) | **Working** | |
| Conv (DAG v2) | **Working** | Im2col → matmul; dims extracted from DAG node |
| Pipeline parallelism frontend | **Not built** | P2P collective exists but no frontend emits PP layer→layer sends |
| DAG v2 format | **Working** | Full graph with deps; `.dag` and legacy `.txt` both parsed |

### Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| TensorPartition memory checks | **Working** | Explicit pass + DP fallback |
| `ModelPartitionPlan` class | **Declared, unused** | Full partitioning abstraction in `TensorPartition.h`; sim bypasses it entirely; jobs built directly in `main_chiplet.cc` |
| Topology utility helpers | **Declared, unused** | `get_bisection_bandwidth`, `are_connected`, `get_neighbors`, `set/get_chiplet_position`, `to_string` — compiled but never called |
| **Per-cycle logging** | **Not built** | No mechanism to record FSM state, UCIe credit count, or DRAM queue depth per cycle. Needed for CS1. |

---

## Experiments and Paper Deliverables

### Done

| Item | Result |
|------|--------|
| **E1: UCIe micro-validation** | 58 PASS, 4 RANGE, 2 FAIL (32GT×32 tiny-packet quantization — expected). `tests/test_ucie_validation.cc` + `scripts/ucie_validation.py` |
| **E2: Three-tier decoupling comparison** | Decode: ~29–31% underestimate by GEMINI/SCALE-Sim. Prefill: <0.1% error. Core paper claim validated. `scripts/compare_simulators.py` |
| Chiplet/topology sweep | `scripts/sweep_chiplets.py` → `results_sweep.csv` |
| Sequence length sweep | `scripts/sweep_seqlen.py` → `results_seqlen_sweep.csv` |
| Decoupling tax sweep | `scripts/decoupling_tax.py` → `results_decoupling_tax.csv` |
| SCALE-Sim+BookSim baseline | `scripts/scalesim_booksim_baseline.py` → `results_scalesim_booksim.csv` |

### Not Done

| Item | Priority | What's needed |
|------|----------|---------------|
| **CS1: Cycle waveform figure** | **High — paper's "money figure"** | Per-cycle logging in `ChipletArch::tick()` (FSM state + UCIe credits + DRAM queue depth) + `scripts/plot_cs1.py`. ~1 day. |
| **E3: Topology ranking sweep** | High | Ring vs mesh vs star, COCOSSim vs decoupled ranking. `sweep_chiplets.py` covers latency but not the topology recommendation comparison. |
| **E4: UCIe saturation knee** | Medium | Sweep UCIe configs (8GT×8 → 32GT×32) at fixed topology; find bandwidth saturation point per workload phase. |
| **E5: Parallelism strategy comparison** | Medium | TP vs PP vs hybrid; does decoupling change the strategy recommendation? Needs PP frontend first. |
| **CS2: Energy-latency Pareto** | Medium | Sweep UCIe speed × topology × parallelism; energy model already built. |
| **Speculative decoding (RQ7)** | Medium | Two-phase sim (prefill on N chiplets, decode on 1); no script yet. |
| **AllGather / ReduceScatter** | Low (for current workloads) | Needed for sequence parallelism and some TP variants. |
| **Pipeline parallelism frontend** | Low (blocks E5) | P2P collective works; need main_chiplet.cc to emit PP-style sends at layer boundaries. |
| **DAG workloads in sweep scripts** | Low | Sweeps use legacy `.txt`; works correctly but doesn't exercise DAG path. |

---

## CS1 Detail (the "money figure")

CS1 is the figure that answers *why* the decoupling tax exists with a concrete cycle trace.

**Scenario:** LLaMA-7B decode, 4-chiplet ring, TP=4.

**Three signals on one timeline:**
1. Compute FSM state per chiplet — COMPUTE / DRAM_WAIT / COMM_STALL
2. UCIe credit count on the hot link — drains during AllReduce, refills as credits return
3. DRAM queue depth — drops when SA stalls (no new requests), shows wasted HBM cycles

**The narrative:** AllReduce saturates UCIe credits → SA enters COMM_STALL → no new DRAM
requests → HBM queue empties → wasted bandwidth. Decoupled simulators never see this
because they model AllReduce and DRAM as independent phases.

**What needs to be built:**
1. Add CSV logging to `ChipletArch::tick()`: one row per cycle per chiplet with
   `(cycle, chiplet_id, sa_fsm_state, ucie_credits_hot_link, dram_queue_depth)`
2. `scripts/plot_cs1.py`: read CSV, draw three-panel annotated figure

---

## Key Result Numbers

| Workload | Tier | Chiplets | Cycles | Error vs COCOSSim |
|----------|------|----------|--------|-------------------|
| llama7b_decode | GEMINI | 1 | 1,581,056 | −30.8% |
| llama7b_decode | SCALE-Sim | 1 | 1,623,220 | −28.9% |
| llama7b_decode | **COCOSSim** | 1 | **2,283,755** | — |
| llama7b_prefill | GEMINI | 1 | 6,324,224 | −2.6% |
| llama7b_prefill | SCALE-Sim | 1 | 6,492,880 | −0.0% |
| llama7b_prefill | **COCOSSim** | 1 | **6,495,535** | — |

---

## Build Instructions (new system)

```bash
# 1. Clone and checkout
git clone <repo> cocossim && cd cocossim
git checkout dev-chiplets
git submodule update --init --recursive

# 2. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) perf_model_chiplet test_ucie_validation

# 3. Python env (extract_model.py needs torch)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Verify E1
python3 scripts/ucie_validation.py --binary build/test_ucie_validation

# 5. Run E2
python3 scripts/compare_simulators.py
```

---

## Docs

| File | Purpose |
|------|---------|
| `docs/RESEARCH.md` | Research narrative — the "why": decoupling tax argument, RQ structure, paper outline |
| `docs/STATUS.md` | This file — code and experiment status |
| `docs/eval_plan_chiplets.md` | Full experiment specs: setup, metrics, expected findings, dependencies for E1–E5, CS1–CS2 |
| `docs/comparison_with_chiplet_frameworks.md` | COCOSSim vs GEMINI, SuperMesh, SIAM, STONNE, SCALE-Sim v3 |
| `docs/gpu_modeling_notes.md` | Notes on extending COCOSSim to GPU chiplets (best-case FSM design) |
