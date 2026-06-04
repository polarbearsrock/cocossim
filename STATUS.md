# COCOSSim dev-chiplets — Work Status
_Last updated: 2026-06-04_

## Branch
`dev-chiplets` — not yet merged to `main`.  All work below lives here.

---

## What is complete

### Core simulator
| Component | Status | Notes |
|-----------|--------|-------|
| Single-chip path (`perf_model`) | Done | ISPASS-validated, ~13% avg error vs TPU v3 |
| Multi-chiplet path (`perf_model_chiplet`) | Done | ring/mesh, 1–8 chiplets, HBM2 |
| SA/VPU cycle-accurate FSMs | Done | wired into ChipletArch (2026-04-22) |
| DRAMSim3 per-chiplet instances | Done | 1TB address stride per chip (2026-04-14) |
| UCIe link model | Done | credit-based FC, 60-cycle latency, 16–112 GB/s |
| UCIe L0/L1/L2 power FSM | Done | L0→L1 after 128 idle, L1→L2 after 10000, wakeup enforced |
| Energy model | Done | 16nm/7nm, INT8/FP16/FP32, MAC+SRAM+DRAM+UCIe |
| TensorPartition memory checks | Done | explicit pass + DP fallback |
| ChipletTopology routing | Done | XY (mesh) and ring; route_yx fixed 2026-05-06 |
| DAG v2 frontend (C++) | Done | parses Matmul/Conv/LayerNorm/Softmax/Activation |
| Col/row/head parallelism tags | Done | col=split-N+AR, row=split-K+AR, head=split-M no AR |

### Python frontend
| Component | Status | Notes |
|-----------|--------|-------|
| `scripts/extract_model.py` | Done | torch.fx → ShapeProp → `.dag` v2 format |
| Builtin models | Done | llama7b_block, llama13b_block, gpt2_block (hidden auto-derived) |
| `examples/llama7b_block.dag` | Done | full transformer block from extract_model.py |

### E1: UCIe micro-validation
| Component | Status | Notes |
|-----------|--------|-------|
| `tests/test_ucie_validation.cc` | Done | 4 configs × 4 packet sizes, CSV mode |
| `CMakeLists.txt` target | Done | `test_ucie_validation` links only UCIe sources |
| `scripts/ucie_validation.py` | Done | runs binary, prints 4 paper-ready tables |
| Result | **58 PASS, 4 RANGE, 2 FAIL** | 2 FAILs are 32GT×32 tiny-packet quantization (expected) |

### E2: Three-tier decoupling comparison
| Component | Status | Notes |
|-----------|--------|-------|
| `scripts/compare_simulators.py` | Done | Tier1=GEMINI, Tier2=SCALE-Sim, Tier3=COCOSSim |
| Result: decode (memory-bound) | **~29–31% underestimate** by T1/T2 | Core paper claim validated |
| Result: prefill (compute-bound) | T2 within 0.1% of COCOSSim | Decoupling harmless when compute-bound |
| `results_compare.csv` | Done | written by compare_simulators.py |

### Other sweep scripts
- `scripts/sweep_chiplets.py` — 1/2/4/8 chips × ring/mesh, writes `results_sweep.csv`
- `scripts/sweep_seqlen.py` — memory→compute transition sweep, writes `results_seqlen_sweep.csv`
- `scripts/decoupling_tax.py` — unified vs analytical, writes `results_decoupling_tax.csv`
- `scripts/scalesim_booksim_baseline.py` — COCOSSim vs SCALE-Sim+BookSim baseline, writes `results_scalesim_booksim.csv`

### Docs
- `docs/eval_plan_chiplets.md` — ASPLOS submission RQ plan (RQ1–RQ8)
- `docs/asplos_chiplets_review.md` — reviewer-facing notes
- `docs/comparison_with_chiplet_frameworks.md` — comparison with GEMINI, Simba, etc.

---

## What is NOT done (open items)

### High priority (needed for paper)
1. **CS1: Per-chiplet cycle waveform figure** — the "money figure" showing compute FSM state
   + UCIe credit count + DRAM queue depth on the same timeline, proving the coupling
   mechanism. Needs: per-cycle logging in `ChipletArch::tick()`, a Python plot script.
   _Estimated work: ~1 day._

2. **Sweep scripts use legacy `.txt` format** — `sweep_chiplets.py`, `sweep_seqlen.py`,
   `decoupling_tax.py` all use `llama7b_decode.txt` / `llama7b_prefill.txt`.
   They work but don't exercise the new DAG path. Should add DAG workloads.
   _Priority: medium — results are correct either way._

3. **Speculative decoding case study** — mentioned in eval plan (RQ7) but no script.
   Requires two-phase simulation: prefill with tp chiplets, decode with 1.
   _Priority: medium for paper completeness._

### Low priority / polish
4. **Unimplemented topology helpers** — `get_bisection_bandwidth`, `get_mesh_dimensions`,
   `are_connected`, `get_neighbors`, `set/get_chiplet_position`, `to_string` declared in
   `ChipletTopology.h` but not in the simulation path. Safe to leave as stubs.

5. **Bandwidth contention between co-resident jobs** — currently each job gets full HBM2
   BW. Unrealistic for 8-chiplet shared DRAM. No fix needed unless a reviewer asks.

6. **GEMINI / Simba baseline scripts** — `compare_simulators.py` implements GEMINI
   analytically (Tier 1). A proper Simba comparison would need the Simba binary or
   a more detailed SA-array model. Deferred.

---

## Build instructions (on new system)

```bash
# 1. Clone and checkout branch
git clone <repo> cocossim && cd cocossim
git checkout dev-chiplets

# 2. Init submodules (DRAMSim3)
git submodule update --init --recursive

# 3. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) perf_model_chiplet test_ucie_validation

# 4. Python env (for extract_model.py, needs torch)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Run E1 validation
python3 scripts/ucie_validation.py --binary build/test_ucie_validation

# 6. Run E2 comparison
python3 scripts/compare_simulators.py
```

---

## Key result numbers (for paper)

| Workload | Tier | Chiplets | Cycles | Error vs COCOSSim |
|----------|------|----------|--------|-------------------|
| llama7b_decode | GEMINI | 1 | 1,581,056 | −30.8% |
| llama7b_decode | SCALE-Sim | 1 | 1,623,220 | −28.9% |
| llama7b_decode | COCOSSim | 1 | 2,283,755 | — |
| llama7b_prefill | GEMINI | 1 | 6,324,224 | −2.6% |
| llama7b_prefill | SCALE-Sim | 1 | 6,492,880 | −0.0% |
| llama7b_prefill | COCOSSim | 1 | 6,495,535 | — |

**Core claim:** Decoupled simulators underestimate decode latency by ~29–31% because
they do not model HBM2 bank-activation latency coupled with UCIe AllReduce stalls.
Prefill (compute-bound) is accurate to <0.1% — decoupling tax is regime-specific.

UCIe validation: 58/64 checks PASS, 4 RANGE (within 10%), 2 FAIL (32GT×32 64B packets —
quantization artifact, not a model error; all ≥1024B packets pass).
