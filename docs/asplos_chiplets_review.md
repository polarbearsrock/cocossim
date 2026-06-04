# ASPLOS Submission Review: COCOSSim Chiplets

**Prepared:** April 2026  
**Branch:** `dev-chiplets`  
**Scope:** `include/chiplets/`, `src/chiplets/`, `include/chiplets/README.md`

---

## What Has Been Built

The chiplet subsystem is more complete than the README suggests. Four distinct modules exist:

| Module | Files | Purpose |
|--------|-------|---------|
| UCIe Physical Layer | `UCIeConfig`, `UCIeLink`, `UCIePacket` | Cycle-accurate link modeling with credit flow control |
| Topology | `ChipletTopology` | Linear, ring, 2D mesh, torus, star, tree, fully-connected; XY/YX/shortest-path routing |
| Tensor Partitioning | `TensorPartition` | DP, TP, PP, hybrid 2D/3D, expert parallel (Megatron-LM style) |
| System Integration | `ChipletArch` | Chiplet struct + multi-chiplet orchestration shell |

The README only describes Phase 1 (UCIe physical layer) but Phases 2–3 (topology, partitioning) are structurally implemented. The critical missing piece is that `ChipletArch.h` explicitly notes the compute resources are "simplified — will integrate with COCOSSim units later," meaning the chiplet subsystem is not yet wired into the main simulation loop.

---

## Novelty Assessment for ASPLOS

### Strong Points

**1. Unified simulation loop (the real differentiator)**  
COCOSSim advances compute, buffers, and DRAMSim3 together each cycle via memory callbacks. This is structurally true today for the single-chip path. Extending this to chiplets — where UCIe links tick alongside the compute state machine — would be a genuine architectural contribution. No existing open-source chiplet simulator does this. SuperMesh, SIAM, and GEMINI all use decoupled phases.

**2. UCIe + DRAMSim3 co-simulation**  
The combination of UCIe physical layer modeling (with credit-based flow control) and cycle-accurate HBM2 DRAM timing is unique. Competing tools use analytical bandwidth constraints or trace-based DRAM. This matters for workloads like LLM decode where off-chip memory latency and inter-chiplet communication overlap in non-trivial ways.

**3. Tensor partition + topology co-modeling**  
Having both the communication pattern (from TensorPartition) and the physical topology (from ChipletTopology) in the same framework enables joint analysis. Existing tools either model topology (BookSim-based) or parallelism strategies (analytical), not both together with timing-accurate transport.

**4. Validated baseline**  
The 13% average error versus Google TPU v3 for the single-chip path is a credible foundation. ASPLOS reviewers will ask "how do you know your model is right?" — this gives an answer for the compute core, even if multi-chiplet validation is pending.

### Weak Points

**1. Integration gap is submission-blocking**  
`ChipletArch` does not call `gcycles`, does not tick alongside the main sim loop, and has no callbacks into `State.h`. The chiplets and the compute engine are currently two separate programs. This is the most critical gap — ASPLOS requires end-to-end demonstrated results, not infrastructure.

**2. No multi-chiplet silicon validation**  
AMD MI300 and Intel Ponte Vecchio are public enough to validate against published throughput/bandwidth numbers. Without at least one data point on real multi-chiplet hardware, reviewers will flag this. The single-chip TPU validation does not transfer automatically.

**3. Collective operations are absent**  
`TensorPartition` models data partitioning but the actual AllReduce/AllGather communication — critical for TP and DP — is listed as future work. A transformer model cannot complete a tensor-parallel forward pass without these.

**4. No workload-level results yet**  
There are no end-to-end numbers: no latency comparison across topologies, no bandwidth utilization curves, no energy breakdown across chiplets. ASPLOS demands at least 4–6 workloads with multi-dimensional analysis.

---

## Assessment: Is This ASPLOS-Ready?

**Not yet, but the path is clear.**  
The infrastructure is strong. The conceptual argument (unified loop vs. decoupled phases) is genuinely novel and ASPLOS-appropriate. The gap is execution: the subsystem needs to be wired into the sim loop, collectives need to be implemented, and results need to be generated. Given the existing codebase, this is a 2–3 month gap, not a fundamental rethink.

---

## Recommended Storyline

### "The Decoupling Tax: Why Phase-Based Chiplet Simulators Mispredict LLM Inference"

**Core argument:**  
Modern multi-chiplet AI accelerators (MI300, Ponte Vecchio) run workloads where compute stalls trigger memory traffic, which in turn creates inter-chiplet congestion, which extends compute stalls. Existing simulators (SuperMesh, SIAM, GEMINI) break this feedback loop by simulating phases sequentially. COCOSSim's unified loop preserves it. We show this difference leads to systematic X% overestimation of throughput for attention-heavy workloads.

**Story arc:**

1. **Motivation** — LLM inference on multi-chiplet systems is the dominant emerging workload. Architects need accurate simulation to size UCIe links, choose topologies, and select parallelism strategies. Existing simulators are inaccurate by design (decoupled phases).

2. **The feedback loop** — Concretely show a scenario (e.g., attention layer on 4-chiplet mesh with TP=4) where:
   - Compute stalls because the KV cache retrieval from HBM is slow (DRAMSim3 captures this)
   - This changes the timing of AllReduce traffic on UCIe links
   - Which creates congestion that further delays compute
   - A decoupled simulator cannot model this; it will predict higher throughput

3. **COCOSSim's unified model** — Describe the tick-based integration: every cycle, compute units, buffer hierarchy, DRAMSim3, and UCIe links all advance together. Show the code-level mechanism (callbacks, credit returns).

4. **Evaluation** — Four workloads across attention and dense layers:
   - BERT-large (pure attention)
   - LLaMA-7B prefill
   - LLaMA-7B decode (memory-bandwidth-bound)
   - ResNet-50 (compute-bound baseline for comparison)
   
   Sweep: 2–16 chiplets, ring vs. mesh, UCIe 16GT×16 vs. 32GT×32.
   
   Compare against SuperMesh / GEMINI predictions. Show where decoupled tools diverge and why.

5. **Design insights** — Use the simulator to answer: What UCIe bandwidth is actually needed for LLaMA decode with TP=4? Does ring or mesh topology matter more at 4 vs. 8 chiplets? What is the energy cost of AllReduce vs. compute?

---

## Directions to Pursue

### Priority 1 (required for any submission)

- **Wire ChipletArch into the main sim loop.** `main_chiplet.cc` needs to call `chiplet_arch.tick(gcycles)` inside the same loop that calls `dramsim3::tick()`. The UCIe links must participate in the global simulation clock.
- **Implement AllReduce over the topology.** A ring AllReduce over UCIe links is sufficient for TP. This is the minimum collective needed to run a transformer layer end-to-end.
- **Generate one end-to-end result.** Even a single GEMM across 2 chiplets with measured vs. predicted bandwidth is enough to anchor the paper.

### Priority 2 (for a strong submission)

- **Validate against MI300 or Ponte Vecchio published numbers.** AMD and Intel have published peak HBM bandwidth, inter-die bandwidth, and LLM inference throughput. Even rough agreement (within 20%) is publishable.
- **Add the attention workload sweep.** The sweep scripts from `mr+mrs` (`sweep_attention.py`, `sweep_dense.py`) are designed for this. Run them once the integration is complete.
- **Topology vs. workload analysis.** Show ring vs. mesh for different parallelism degrees. This is a concrete, actionable result that architects care about.

### Priority 3 (differentiators for acceptance over rejection)

- **Energy breakdown across chiplets.** The energy model already exists; extending it to per-chiplet UCIe power gives a result no other open-source tool provides.
- **Expert parallel for MoE models.** `ParallelismType::EXPERT_PARALLEL` is defined but not implemented. MoE is the direction LLMs are heading (Mixtral, GPT-4 rumors) and ASPLOS 2025/2026 will have reviewers looking for it.
- **Calibration against the `calibrate_threshold.py` methodology.** Show how to tune the simulator parameters to match a new target architecture — this is a practical contribution for the community.

---

## Comparison Doc Cross-Reference

The `comparison_with_chiplet_frameworks.md` doc has the right four claims (unified loop, DRAMSim3 fidelity, compute-memory overlap, validated accuracy). These map directly onto the storyline above. The summary figure (Accuracy vs. Speed axis) is strong for a paper figure. One addition needed: a row for COCOSSim-chiplets in the main comparison table once multi-chiplet results exist.

---

## Summary

| Dimension | Status |
|-----------|--------|
| Infrastructure completeness | ~70% (UCIe + topology + partitioning built; not integrated) |
| Novelty of core argument | High (unified loop for chiplets is genuinely new) |
| Evaluation readiness | Low (no end-to-end multi-chiplet results yet) |
| ASPLOS-readiness | 2–3 months of focused integration + evaluation work away |
| Strongest venue fit | ASPLOS, ISCA, MICRO (roughly in that order for this story) |
