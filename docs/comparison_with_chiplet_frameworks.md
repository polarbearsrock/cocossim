# COCOSSim vs Chiplet Simulation Frameworks

## Framework Comparison Table

| Framework | Cycle-Accurate | Memory Model | Interconnect | Compute-Memory Coupling | Energy Model | Open Source |
|-----------|---------------|--------------|--------------|------------------------|--------------|-------------|
| **COCOSSim** | ✓ (unified loop) | DRAMSim3 (timing-accurate) | Implicit in state machine | Tight (per-cycle callbacks) | ✓ (7nm/16nm) | ✓ |
| **SuperMesh** (ScaleSim+BookSim) | Partial (network only) | Analytical bandwidth | BookSim (cycle-accurate) | Loose (trace-driven phases) | Network only | ✓ |
| **[SIAM](https://github.com/gkrish19/SIAM-Chiplet-based-Scalable-In-Memory-Acceleration-with-Mesh-for-Deep-Neural-Networks)** | ✗ (analytical) | DRAM latency model | BookSim-based NoP | Decoupled engines | ✓ | ✓ |
| **[GEMINI](https://arxiv.org/abs/2312.16436)** | ✗ (aggregated) | Analytical | NoP hop-based | Bottleneck analysis | ✓ | ✓ |
| **[NN-Baton](https://ieeexplore.ieee.org/document/9499743/)** | ✗ (analytical) | Analytical | NoP analytical | Hierarchical framework | ✓ | Partial |
| **[STONNE](https://github.com/stonne-simulator/stonne)** | ✓ (single chip) | Table-based | On-chip networks | Tight | ✓ | ✓ |
| **[SCALE-Sim v3](https://arxiv.org/abs/2504.15377)** | ✓ (single chip) | Ramulator (timing-accurate) | NoP latency model | Moderate | Via Accelergy | ✓ |

---

## Hardware Reference Targets

| System | Compute Style | Chiplet Config | Interconnect | Relevance |
|--------|--------------|----------------|--------------|-----------|
| **Simba (NVIDIA, 2019)** | Systolic array (same class as COCOSSim) | 36 chiplets, mesh NoP | Die-to-die bumps (~20 GB/s) | Best hardware validation target — same compute model; published perf numbers; used by SIAM for calibration |
| **AMD MI300X** | CDNA3 GPU (SIMT) | 8 GCDs + 1 I/O die, star | UCIe-compatible, 896 GB/s HBM | Not directly comparable (GPU compute); useful for interconnect-layer reference only |
| **Intel Ponte Vecchio** | Xe GPU tiles | 47 tiles, Foveros 3D | EMIB + Foveros | Not directly comparable (GPU); topology reference |
| **Google TPU v3** | Systolic array | Single chip (no chiplets) | ICI for pods | Already validated at 13% error (single-chip baseline) |

---

## Key Differentiators by Category

### 1. Simulation Methodology

| Approach | Frameworks | Limitation |
|----------|------------|------------|
| **Unified cycle-accurate** | COCOSSim, STONNE | — |
| **Trace-driven coupling** | SuperMesh, SIAM | Loses fine-grained compute-memory overlap |
| **Analytical/bottleneck** | GEMINI, NN-Baton | Cannot model contention, stalls, or timing variations |

**COCOSSim advantage**: All components (compute, buffers, DRAM) advance in a single simulation loop with per-cycle memory callbacks. SuperMesh and SIAM run compute and network as separate phases.

### 2. Memory System Fidelity

| Framework | DRAM Model | Chiplet Buffer |
|-----------|------------|----------------|
| **COCOSSim** | DRAMSim3 with HBM2 timing | Multi-level hierarchy with access tracking |
| **SuperMesh** | SCALE-Sim analytical BW | Single-level SRAM |
| **SIAM** | DRAM latency engine | Per-chiplet buffer |
| **GEMINI** | Bandwidth constraint | Analytical |
| **NN-Baton** | Analytical | O-L2 buffer model |

**COCOSSim advantage**: DRAMSim3 models bank conflicts, row buffer hits/misses, and queue depths—effects that analytical models miss entirely.

### 3. Compute-Network Synchronization

```
COCOSSim (unified):
  Every cycle: [Compute state machine] ↔ [Buffer access] ↔ [DRAM ClockTick]

SuperMesh/SIAM (decoupled):
  Phase 1: SCALE-Sim computes total cycles (atomic)
  Phase 2: BookSim simulates network messages
  → No mid-layer communication modeling
```

**COCOSSim advantage**: Can model scenarios where memory stalls affect compute, and compute patterns affect memory queueing—critical for realistic chiplet modeling.

### 4. Chiplet Communication Modeling

| Framework | NoP Model | Communication Granularity |
|-----------|-----------|---------------------------|
| **COCOSSim** | State-machine based (extensible) | Per-transaction |
| **SuperMesh** | BookSim cycle-accurate | Message-level (fixed size) |
| **SIAM** | Trace-based BookSim | Aggregated per-layer |
| **GEMINI** | Hop-count analytical | Aggregated per-layer |

**COCOSSim advantage**: Transactions flow through the simulation naturally with callbacks, enabling fine-grained overlap analysis without artificial synchronization barriers.

---

## Comparison with Specific Frameworks

### vs SuperMesh (ScaleSim + BookSim)

| Aspect | COCOSSim | SuperMesh |
|--------|----------|-----------|
| **Integration** | Single codebase, unified loop | Three tools + Python glue |
| **Compute cycle accuracy** | Per-cycle state machine | Aggregate (SCALE-Sim returns total) |
| **Network simulation** | Integrated (extensible) | BookSim (separate phase) |
| **Overlap modeling** | Natural (same loop) | Sequential phases only |
| **Energy** | Full stack (compute + memory) | Network power only |

### vs SIAM (In-Memory Computing)

| Aspect | COCOSSim | SIAM |
|--------|----------|------|
| **Target architecture** | Systolic arrays + vector units | IMC crossbars |
| **Simulation speed** | Cycle-accurate but fast | Analytical (faster) |
| **NoP model** | Unified | BookSim trace-based |
| **Memory model** | DRAMSim3 timing | DRAM latency engine |
| **Calibration** | TPU v3 (13% error) | Simba silicon |

### vs GEMINI

| Aspect | COCOSSim | GEMINI |
|--------|----------|--------|
| **Purpose** | Performance/energy analysis | Mapping + architecture co-exploration |
| **Accuracy** | Cycle-accurate | Bottleneck approximation |
| **Speed** | ~1000× faster than RTL | Very fast (DSE-focused) |
| **Communication** | Per-transaction | Aggregated per-layer |

### vs STONNE

| Aspect | COCOSSim | STONNE |
|--------|----------|--------|
| **Scope** | Multi-chiplet ready | Single accelerator |
| **Memory** | DRAMSim3 (or SST for STONNE) | Table-based |
| **Flexibility** | WS/OS dataflows | Flexible (TPU, MAERI, SIGMA, etc.) |
| **Framework integration** | Standalone | Caffe/PyTorch frontend |

---

## Paper Claims

### Claim 1: "Unified Cycle-Accurate Chiplet Simulation"
> Unlike decoupled approaches (SuperMesh, SIAM) that simulate compute and network in separate phases, COCOSSim provides a unified simulation loop where all components—systolic arrays, buffers, and DRAMSim3—advance together each cycle with memory transaction callbacks.

### Claim 2: "Timing-Accurate Memory Modeling"
> While existing chiplet frameworks (GEMINI, NN-Baton, SIAM) use analytical memory models, COCOSSim integrates DRAMSim3 for cycle-accurate HBM2 timing, capturing bank conflicts, row buffer effects, and queue depths that impact real system performance.

### Claim 3: "Fine-Grained Compute-Memory Overlap"
> Trace-driven simulators (SuperMesh) cannot model mid-layer memory stalls or compute-dependent traffic patterns. COCOSSim's state machine architecture naturally captures these interactions without artificial phase boundaries.

### Claim 4: "Validated Accuracy with Practical Speed"
> COCOSSim achieves 13% average error versus Google TPU v3, providing higher fidelity than analytical tools (GEMINI, NN-Baton) while remaining practical for design space exploration—unlike RTL simulation.

---

## Summary Figure for Paper

```
                    Accuracy
                       ↑
                       │
            RTL ───────┼─────────────────────────── High
                       │
         COCOSSim ─────┼─────────────● (13% vs TPU)
                       │
   SuperMesh/SIAM ─────┼────●
                       │
    GEMINI/NN-Baton ───┼──●
                       │
                       └─────────────────────────→ Speed
                              Fast
```

---

## References

- [SIAM - ACM TECS](https://dl.acm.org/doi/10.1145/3476999)
- [GEMINI - arXiv](https://arxiv.org/abs/2312.16436)
- [NN-Baton - ISCA 2021](https://ieeexplore.ieee.org/document/9499743/)
- [STONNE - GitHub](https://github.com/stonne-simulator/stonne)
- [SCALE-Sim v3 - arXiv](https://arxiv.org/abs/2504.15377)
- [SuperMesh Communication Characterization](https://arxiv.org/html/2410.22262v2)
- [Simba - NVIDIA Research](https://research.nvidia.com/publication/2019-10_simba-scaling-deep-learning-inference-multi-chip-module-based-architecture)
