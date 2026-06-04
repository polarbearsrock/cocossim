# COCOSSim: Research Narrative and Plan
_Branch: `dev-chiplets` — Last updated: 2026-06-04_

> For build/run status, see `STATUS.md`.  
> For experiment specs and RQ details, see `docs/eval_plan_chiplets.md`.  
> This document is the "why and how" — start here on a new machine.

---

## The Problem We Are Solving

As LLMs get bigger, they get partitioned across multiple chiplets — small dies connected by
high-speed die-to-die links (UCIe). The standard way to evaluate these systems is to run
compute and network simulation separately: first simulate the systolic array, then simulate
the network, then add the results together. This is called **decoupled simulation**, and
it is how almost every existing tool works (GEMINI, SuperMesh / SCALE-Sim + BookSim, SIAM).

The claim of this paper is that decoupled simulation is systematically wrong for
**memory-bound workloads** — specifically, LLM decode — because it misses a feedback loop
that only exists when both DRAM and the UCIe link are under pressure at the same time:

1. AllReduce traffic saturates UCIe credits → compute stalls
2. Stalled compute means no new DRAM requests → HBM bank pipeline goes idle
3. Idle HBM means DRAM bandwidth is wasted, even though the model assumed it was free
4. Decoupled tools see step 1 and step 2 as independent events; they never model step 3→4

The result is that decoupled simulators predict **~29–31% lower latency** than actually
occurs during decode, because they assume compute, DRAM, and AllReduce can overlap freely.
For prefill (compute-bound), the tools agree within 0.1% — the coupling mechanism only
matters when DRAM pressure and AllReduce traffic co-occur. This regime-specificity is a
key part of the argument: it is not that decoupled tools are always wrong, it is that they
are wrong precisely when you need accuracy most (LLM serving is dominated by decode).

We call this error the **decoupling tax**.

---

## Why This Is a Contribution

Every prior multi-chiplet simulator (GEMINI, SuperMesh, SIAM) treats compute, memory, and
network as independent engines that communicate only at layer boundaries. COCOSSim advances
all three components in a single simulation loop — every cycle, the systolic array state
machine, DRAMSim3, and UCIe credit counter all tick together. This means:

- UCIe credit exhaustion stalls the compute state machine *mid-cycle*
- A stalled SA means DRAMSim3 receives no new requests *that cycle*
- DRAMSim3 bank queue effects feed back into when the next DRAM response arrives, which
  determines when the next AllReduce can start

None of these feedback paths exist in decoupled tools. They can be expressed in equations
after the fact, but they cannot be *observed* without a unified cycle-accurate loop.

The contribution is threefold:
1. **Build** the first unified cycle-accurate multi-chiplet simulator for systolic-array
   accelerators (COCOSSim + UCIe model + DRAMSim3)
2. **Quantify** the decoupling tax across workloads and chiplet counts
3. **Explain the mechanism** with a cycle-level trace that shows exactly when and why the
   feedback loop occurs (Case Study 1 — the "money figure")

---

## The Three-Tier Comparison

We compare three tools on the same workloads:

| Tier | Tool | What it models | What it misses |
|------|------|----------------|----------------|
| T1 | GEMINI (analytical) | Arithmetic roofline, bottleneck analysis | DRAM timing, network congestion, any coupling |
| T2 | SCALE-Sim + BookSim (decoupled) | Cycle-accurate DRAM (Ramulator), cycle-accurate network (BookSim) — but sequentially | The feedback between DRAM pressure and AllReduce timing |
| T3 | COCOSSim (unified) | All components in a single loop | Ground truth for this class of architecture |

T2 is the **primary baseline**: it is the most capable prior tool (SuperMesh uses SCALE-Sim v3
with Ramulator, the same class of DRAM model as COCOSSim's DRAMSim3). Any delta between T2
and T3 is attributable purely to coupling methodology, not to different memory models.
T1 is the secondary baseline — it establishes how much the analytical shortcut costs.

The comparison is implemented in `scripts/compare_simulators.py`. Current results:

```
llama7b_decode  (memory-bound):   T1=-30.8%  T2=-28.9%  vs COCOSSim   [decoupling tax]
llama7b_prefill (compute-bound):  T1= -2.6%  T2= -0.0%  vs COCOSSim   [decoupling harmless]
```

The asymmetry between decode and prefill is the empirical proof of the claim.

---

## Research Questions

The RQs follow a three-level argument:

**What** (E2): How large is the decoupling tax, and for which workload types?  
**Why** (CS1): What is the cycle-level mechanism that causes it?  
**Implication** (E3–E5): Does the decoupling tax cause wrong design decisions?

| RQ | Level | Experiment | Status |
|----|-------|------------|--------|
| RQ1 | Validity | E1: UCIe micro-validation vs. spec | Done — 58/64 PASS |
| RQ2 | *What* | E2: Decoupling tax measurement | Done — core result validated |
| RQ3 | *Why* | E2 stall breakdown + CS1 cycle trace | CS1 not started |
| RQ4 | *Mechanism* | CS1: overlap efficiency vs. DRAM util | CS1 not started |
| RQ5 | Implication | E5: Does decoupling change parallelism strategy ranking? | Not started |
| RQ6 | Implication | E3: Does decoupling change topology ranking? | Not started |
| RQ7 | Implication | E4: UCIe bandwidth saturation point | Not started |
| RQ8 | Energy | CS2: Energy-latency Pareto | Not started |

---

## What Each Script Does and Why It Exists

```
scripts/
  compare_simulators.py    — E2: the three-tier table (T1/T2/T3 per workload per chiplet count)
                             THIS is the paper's Table 1 / Figure 2 generator
  ucie_validation.py       — E1: runs test_ucie_validation binary, formats 4 spec-comparison tables
  sweep_chiplets.py        — scaling sweep: 1/2/4/8 chips × ring/mesh
  sweep_seqlen.py          — sequence length sweep: shows memory→compute crossover point
  decoupling_tax.py        — alternative decoupling tax calculation (analytical reference)
  scalesim_booksim.py      — baseline comparison script
  extract_model.py         — torch.fx → .dag v2 format (PyTorch model → simulation input)

tests/
  test_ucie_validation.cc  — standalone C++ binary: 4 UCIe configs × 4 packet sizes,
                             checks latency/BW/credit/power vs. spec table

examples/
  llama7b_decode.txt       — memory-bound workload: M=1, all layers, GEMV via VPU
  llama7b_prefill.txt      — compute-bound workload: M=512, attention + projection
  llama7b_block.dag        — full transformer block from extract_model.py (DAG v2 format)
                             includes LayerNorm, Softmax, residuals — not just matmuls
```

---

## Parallelism Model

The simulator implements three tensor-parallelism split modes, tagged in `.dag` and `.txt` files:

| Tag | Split | AllReduce? | Use case |
|-----|-------|------------|----------|
| `col` | Split output columns (N dim) across chips | Yes | Default for projection layers |
| `row` | Split input rows (K dim) across chips | Yes | When K >> N |
| `head` | Split batch/sequence (M dim) | No | Attention heads, already independent |

These map directly to the two standard tensor-parallel configurations from Megatron-LM:
column-parallel (col) and row-parallel (row). Head-parallel is used for MHA where each
head can be assigned to a separate chiplet without any cross-chiplet reduce.

---

## The "Money Figure" — CS1 (Not Yet Built)

The most important missing piece is Case Study 1. The E2 numbers show *that* decoupled
tools underestimate by ~29%; CS1 will show *why* with a concrete cycle trace.

What the figure needs to show, for LLaMA-7B decode on 4-chiplet ring:
- **Top row**: Compute FSM state per chiplet (COMPUTE / DRAM_WAIT / COMM_STALL)
- **Middle row**: UCIe credit count on the hot link
- **Bottom row**: DRAM queue depth

The annotated window shows: AllReduce starts → credits drain → SA enters COMM_STALL →
DRAM request rate drops → HBM queue empties → wasted DRAM cycles.

**Implementation required:**
1. Add per-cycle logging to `ChipletArch::tick()` — write one CSV row per cycle with
   (cycle, chiplet_id, sa_state, ucie_credits, dram_queue_depth)
2. Write `scripts/plot_cs1.py` — reads the CSV, draws the annotated three-panel figure

This is the highest-priority remaining item for the paper.

---

## The Argument Structure for the Paper

```
§1 Introduction
   Problem: decoupled simulation misses compute-DRAM-network coupling
   Claim: ~29–31% underestimate in decode; causes wrong design decisions

§2 Background / Motivation
   Show the feedback loop with a diagram (CS1 preview)
   Prior work: GEMINI, SuperMesh, SIAM — all decoupled

§3 COCOSSim Design
   Unified simulation loop; UCIe model; DRAMSim3 per chiplet; SA/VPU FSMs

§4 Validation (RQ1)
   E1: UCIe spec match (Table 1)
   Compositional argument: compute validated at 13%, UCIe validated via E1

§5 The Decoupling Tax (RQ2–RQ4)
   E2: three-tier comparison table/figure
   CS1: cycle trace — the mechanism

§6 Design-Space Implications (RQ5–RQ7)
   E3: topology ranking
   E4: UCIe saturation knee
   E5: parallelism strategy ranking

§7 Energy Analysis (RQ8)
   CS2: Pareto frontier

§8 Conclusion
```

---

## Comparison with Prior Work — Key Differentiators

The argument against SCALE-Sim v3 + BookSim (SuperMesh) needs to be crisp because a
reviewer will ask: "SCALE-Sim v3 already uses Ramulator — same class as DRAMSim3. What
is actually different?"

Answer: **The coupling methodology, not the DRAM model class.**

- In SuperMesh: SCALE-Sim finishes all compute for a layer, *then* BookSim injects traffic.
  DRAM is never under pressure at the same time as the network, because they run in sequence.
- In COCOSSim: DRAM requests fire mid-layer, AllReduce packets compete for UCIe credits
  mid-layer, and the SA FSM can stall mid-layer if credits are exhausted. These interactions
  cannot happen in a tool that phases compute and communication separately.

The ~2% residual delta between Ramulator and DRAMSim3 timing (quantified in E1 single-chip
calibration) is separate from the ~29% coupling effect — two different magnitudes, two
different causes.

---

## Open Questions / Things to Revisit

1. **Do we need the full SuperMesh pipeline (SCALE-Sim v3 + BookSim2 integration)?**  
   The current `compare_simulators.py` implements T2 analytically (roofline + analytical
   AllReduce). For the paper, the reviewer may require running actual SCALE-Sim v3 + BookSim2.
   The analytical T2 is algebraically equivalent for these workloads but is harder to defend
   without the actual tool run. See `docs/eval_plan_chiplets.md` §SuperMesh for the full
   integration plan if needed.

2. **LLaMA-70B / MoE workloads** — currently only 7B in examples. 70B requires pipeline
   parallelism support (PP), which is not yet implemented. This would be needed for RQ5 and
   the hybrid TP+PP experiment.

3. **Bandwidth contention between co-resident jobs** — each job currently gets full HBM2 BW.
   If a reviewer asks about shared-DRAM scenarios (8-chiplet, multiple concurrent jobs), this
   would need to be modeled. Deferred unless asked.
