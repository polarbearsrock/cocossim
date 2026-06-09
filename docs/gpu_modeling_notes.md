# GPU Modeling in COCOSSim: Background and Best-Case FSM Design
_Notes on extending COCOSSim-style unified simulation to GPU chiplets_

---

## 1. Why This Is Hard: The SA vs GPU Difference

COCOSSim models a systolic array with a small FSM because SA behavior is almost entirely
determined by the **architecture**, not the workload. Given array size, dataflow, and matrix
dimensions, the cycle count is deterministic. The memory access pattern follows directly from
the dataflow (weight-stationary: load weights once, stream activations — fully predictable).

A GPU is different in one fundamental way: its throughput depends on **what the code does**,
not just what the hardware is. Two kernels on the same GPU can achieve 10% or 90% of peak
for architectural reasons that have nothing to do with FLOP count. Understanding why requires
understanding three interacting mechanisms: the SIMT execution model, the warp scheduler,
and the memory hierarchy.

---

## 2. GPU Architecture Background

### 2.1 The SIMT Execution Model

A GPU SM (Streaming Multiprocessor) executes threads in groups of 32 called **warps**.
All threads in a warp execute the same instruction each cycle — this is Single Instruction
Multiple Thread (SIMT). The SM has multiple warp slots (typically 32–64 per SM on modern
hardware) and a warp scheduler that issues one or more warps per cycle.

The critical property: **when a warp stalls** (waiting for memory, a barrier, or a dependent
instruction), the scheduler switches to another warp at zero cost. This is how GPUs hide
memory latency — not with deep out-of-order buffers, but with thread-level parallelism.

A modern H100 SM has:
- 128 CUDA cores (4 × 32-wide SIMT pipelines)
- 4 Tensor Core units (MMA operations, 16×16×16 per cycle)
- 64KB L1 / shared memory (configurable split)
- Up to 64 warps resident per SM
- 256KB register file

### 2.2 The Warp Scheduler

Each SM has 4 warp scheduler units. Each cycle, each scheduler selects one eligible warp
(a warp whose next instruction has all operands ready) and issues it to an execution unit.

**Eligible** = not stalled waiting for: memory, a barrier, a dependent instruction result.

Standard scheduling policies:
- **Round-robin (RR)**: cycle through warps in order — fair, predictable
- **Greedy-then-oldest (GTO)**: keep issuing the same warp until it stalls, then pick
  the oldest eligible warp — better instruction-level cache locality but less latency hiding
- **Two-level**: partition warps into groups; RR within group, GTO across groups

For best-case analysis, round-robin is the right assumption — it maximizes latency hiding
by distributing memory stalls evenly across warps.

### 2.3 Occupancy

**Occupancy** is the ratio of active warps per SM to the hardware maximum:

```
occupancy = active_warps / max_warps_per_SM
```

Active warps are limited by whichever resource runs out first:

| Resource | Limit | Formula |
|----------|-------|---------|
| Registers | 256K per SM (Ampere/Hopper) | `floor(256K / (regs_per_thread × 32))` |
| Shared memory | 164KB per SM (H100) | `floor(shmem_per_SM / shmem_per_block) × warps_per_block` |
| Thread blocks | max 32 per SM | `max_blocks × (threads_per_block / 32)` |
| Architectural | 64 warps max (H100) | hard ceiling |

A kernel using 64 registers/thread on a 256K-register SM supports at most
`256K / (64 × 32) = 128` warps — but the architectural ceiling is 64, so occupancy
ceiling is 64/64 = 100% here. A kernel using 128 registers/thread gets
`256K / (128 × 32) = 64` warps — still 100%. But 192 registers/thread gives
`256K / (192 × 32) = 42` warps → occupancy = 42/64 = 65%.

**Why occupancy matters:** it sets the ceiling on latency hiding. With 200-cycle memory
latency and 20 warps active, the scheduler can cover `20 × issue_interval` cycles of
waiting before running out of eligible warps. Drop to 5 warps and the SM goes idle for
the remainder of each memory stall.

Using Little's Law:
```
sustained_throughput = min(occupancy × issue_rate, memory_bandwidth / bytes_per_access)
```

### 2.4 Latency Hiding

The relationship between occupancy and latency hiding is not binary — it is a curve:

```
warps_needed_to_hide_latency = ceil(memory_latency_cycles / warp_issue_interval)
```

For HBM2 (latency ~200 cycles) and 1-warp-per-cycle issue rate: need ~200 warps to fully
hide latency. But SM max is 64, so you can never fully hide 200-cycle latency with one
outstanding request per warp. In practice, kernels issue multiple memory requests per warp
before stalling (software pipelining / prefetching) to increase effective in-flight requests.

FlashAttention and cuBLAS GEMM both do this explicitly — they structure the tile loop to
have several memory loads in-flight simultaneously per warp, which is why they approach
roofline even though raw occupancy-based hiding is insufficient.

### 2.5 Coalescing

When 32 threads in a warp issue load instructions, the memory controller checks whether
their addresses are contiguous and aligned to a cache line (typically 128 bytes). If yes:
one L2/HBM transaction. If not: up to 32 separate transactions.

Coalescing efficiency:
```
coalescing_efficiency = unique_cache_lines_accessed / threads_in_warp
                      = 1/32 (perfect) to 32/32 (fully uncoalesced)
```

Stride-1 access (row-major matrix, column-parallel threads): fully coalesced.  
Stride-N access (column-major matrix, row-parallel threads): 1 element per cache line → 32×
more HBM transactions than necessary.

For LLM kernels: GEMM is written for coalesced access. Attention requires careful tiling
(FlashAttention) to maintain coalescing. Most production LLM kernels are well-coalesced.

### 2.6 Cache Hierarchy

| Level | Size | Scope | Latency | Notes |
|-------|------|-------|---------|-------|
| Register file | 256KB/SM | Per-warp | 0 cycles | Operands only |
| L1 / Shared mem | 164KB/SM (H100) | Per-SM | 20–30 cycles | Configurable split |
| L2 | 50MB (H100) | Chip-wide | 100–150 cycles | Shared across all SMs |
| HBM3 | 80GB (H100) | Off-chip | 200–400 cycles | Full DRAM timing |

**L1/L2 behavior for simulation:**
- Shared memory (programmer-managed partition of L1): access pattern is deterministic,
  latency is fixed. No simulation needed — it is a scratchpad.
- L1 cache (hardware-managed): hit rate depends on per-SM working set vs. 164KB.
  For tiled GEMM, tile fits in L1 → high hit rate. For decode GEMV (M=1), weight matrix
  row is fetched once → L1 is effectively bypassed (streaming access).
- L2: all SMs compete. For decode, all SMs fetch different weight rows → L2 is a pass-through
  to HBM. For prefill, weight tiles can be reused across sequence positions → L2 can help.

**The cache problem for simulation:** L1/L2 hit rates are determined by the kernel's
tile structure and the ratio of working set to cache size. They cannot be derived from
architecture parameters alone — they require either analytical characterization of the
kernel's access structure or hardware profiling.

### 2.7 Divergence

When threads in a warp take different branches, the warp executes both sides sequentially
with masking — half the threads are idle on each side. Worst case: 32 threads, 32 unique
paths → 32× slowdown on that region.

For LLM kernels: divergence is minimal. GEMM has no branches in the inner loop. Softmax
has a max-reduction but it's structured as a parallel reduction without divergence.
The only significant divergence in LLM inference is in token sampling (top-k/top-p) and
KV-cache index management — neither is in the hot path.

---

## 3. What Accurate GPU Simulation Requires

This is what a full GPU simulator (GPGPU-Sim, Accel-Sim) actually models and why each
piece matters:

| Component | Required for accuracy | Complexity |
|-----------|----------------------|------------|
| Per-warp instruction pipeline | Correct instruction-level latency (data deps) | High — needs per-warp PC, reg tracking |
| Warp scheduler policy | GTO vs RR affects cache locality and latency hiding | Medium |
| Divergence tracking | Correct cycle count for branchy kernels | Medium — needs per-warp branch state |
| Shared memory bank conflicts | 32 banks, same bank = serial access | Medium |
| L1 cache simulation | Hit/miss per access based on address | High — set-associative cache model |
| L2 cache simulation | Cross-SM contention, multi-bank L2 | High |
| Memory coalescing | Address alignment and stride per warp | Medium |
| DRAM timing | HBM bank/row timing (DRAMSim3 handles this) | Already solved |
| TensorCore pipeline | MMA latency + throughput | Low — regular pipeline |
| Register file banking | Port contention on large warps | Low for best-case |

For LLM inference specifically, the high-complexity items (divergence, L1 simulation,
per-warp instruction pipeline) are either negligible (divergence) or approximable with
a single parameter (L1 hit rate ≈ 0 for decode, ≈ tile-size-dependent for prefill).

---

## 4. The Best-Case FSM Model

### 4.1 Core Idea

Best-case assumptions eliminate the hard-to-model dynamic effects:

- **No divergence** — all warps execute identical instruction sequences
- **Perfect coalescing** — one DRAM transaction per warp load instruction
- **No shared memory bank conflicts** — shmem latency = 1 cycle
- **No L1 thrashing** — global memory accesses bypass L1 (streaming); shmem/registers hit always
- **Round-robin scheduling** — maximizes latency hiding, fully deterministic
- **Maximum occupancy** — use the register/shmem ceiling, not a dynamic measurement

With these, warp behavior is **symmetric**: all warps are in the same logical state at the
same relative time. You do not need to track individual warps — only **counts of warps
in each state**.

### 4.2 The Warp Micro-FSM

```
        ┌─────────────────────────────────────────────────────┐
        │                    WARP FSM                         │
        │                                                     │
        │   ┌────────┐  issue load   ┌──────────┐            │
        │   │        │──────────────►│ MEM_WAIT │            │
        │   │        │               └────┬─────┘            │
        │   │ READY  │  DRAMSim3 cb       │                   │
        │   │        │◄───────────────────┘                   │
        │   │        │  issue MMA    ┌──────────┐            │
        │   │        │──────────────►│  TC_BUSY │            │
        │   │        │               └────┬─────┘            │
        │   │        │  TC_LATENCY        │                   │
        │   │        │◄───────────────────┘                   │
        │   │        │  issue ALU    ┌──────────┐            │
        │   │        │──────────────►│ ALU_BUSY │            │
        │   │        │               └────┬─────┘            │
        │   │        │  1–4 cycles        │                   │
        │   │        │◄───────────────────┘                   │
        │   │        │  NVLink/UCIe  ┌──────────┐            │
        │   │        │◄──────stall───│COMM_WAIT │            │
        │   └────────┘               └──────────┘            │
        └─────────────────────────────────────────────────────┘
```

### 4.3 SM-Level FSM

The SM tracks counts, not individual warps:

```
SM state = { n_ready, n_mem_wait, n_tc_busy, n_alu_busy, n_comm_wait }

Invariant: n_ready + n_mem_wait + n_tc_busy + n_alu_busy + n_comm_wait == n_warps
```

Each cycle:
```
1. Tick TC/ALU countdowns → move completions back to n_ready
2. Process DRAMSim3 callbacks → n_mem_wait--, n_ready++
3. Process UCIe callbacks → n_comm_wait--, n_ready++
4. If n_ready > 0:
     sample next instruction type from instruction_mix[]
     n_ready--
     if MEM:   n_mem_wait++;  dramsim3.enqueue(next_address())
     if TC:    n_tc_busy++;   start TC_LATENCY countdown
     if ALU:   n_alu_busy++;  start ALU_LATENCY countdown
     if COMM:  n_comm_wait++; enqueue UCIe packet
5. Else:
     idle_cycles++  ← latency hiding gap
```

### 4.4 Inputs Required

Three inputs, all derivable without running the kernel:

**Occupancy (n_warps)**
```python
n_warps = min(
    floor(regs_per_SM / (regs_per_thread * 32)),
    floor(shmem_per_SM / shmem_per_block) * warps_per_block,
    max_warps_per_SM
)
```
For known LLM kernels, register usage is fixed and published (or measurable once with
`nvcc --ptxas-options=-v`). This is a static pre-computation, not a simulation.

**Instruction mix**
Fraction of instructions that are MEM / TC / ALU. Derivable analytically for regular kernels:

| Kernel | MEM% | TC% | ALU% | Notes |
|--------|------|-----|------|-------|
| GEMM inner loop | ~20% | ~70% | ~10% | TC-dominated |
| LayerNorm | ~75% | 0% | ~25% | Memory-bound |
| Softmax | ~70% | 0% | ~30% | Memory-bound |
| FlashAttention | ~30% | ~55% | ~15% | Mixed, tiled |

For unknown kernels: parse PTX/SASS and count instruction types. One-time cost.

**Memory access sequence**
In best case: sequential, coalesced, following the tile traversal order. For a matmul
with tile size T×T, the access sequence is: load A-tile (T×K_tile consecutive rows),
load B-tile (K_tile×T consecutive columns) — identical to how COCOSSim generates SA
weight/activation access patterns. Feed directly to DRAMSim3.

### 4.5 What Emerges from the FSM

The FSM produces these outputs without additional computation:

| Output | How it emerges |
|--------|---------------|
| **Stall cycles** | Cycles where `n_ready == 0` — the latency hiding gap |
| **Utilization** | `1 - (idle_cycles / total_cycles)` |
| **Memory-bound vs compute-bound** | Whether `n_mem_wait` or `n_tc_busy` is the saturated resource |
| **Effective HBM bandwidth** | DRAMSim3 request rate × bytes_per_request |
| **Decoupling tax** | Same as SA case: DRAM timing variance affects warp stall duration |

The `MEM_WAIT → READY` transition is driven by DRAMSim3, not a fixed latency. Even
under best-case access patterns, DRAMSim3 gives actual row-buffer, bank-activation,
and queue-depth timing. This variance — invisible to analytical models — is what
determines whether n_warps is enough to hide memory latency or not.

### 4.6 Why This is Tighter Than Roofline

```
Roofline:      max(MACs / peak_FLOPS,  bytes / peak_BW)
               ↑ ignores occupancy, DRAM timing, latency hiding

This FSM:      given max occupancy + perfect access patterns,
               DRAMSim3 timing determines actual stall duration
               ↑ captures DRAM variance and latency hiding gap
               ↓ misses: divergence, L1 thrashing, bank conflicts, coalescing inefficiency

Real hardware: all of the above
```

For LLM kernels (GEMM, FlashAttention), the gap between this model and real hardware
is small — these kernels are explicitly written to maximize coalescing, minimize
divergence, and tile for cache locality. The model gives a tight best-case bound that
is also a reasonable approximation.

---

## 5. Fit Into COCOSSim

The integration is clean because the interface is identical to the SA FSM:

```
ChipletArch currently:
  SA_FSM::tick(cycle)
    ├── issues DRAM requests → DRAMSim3
    ├── receives DRAMSim3 callbacks → advances state
    └── stalls on UCIe credit exhaustion → COMM_WAIT

GPU_SM_FSM::tick(cycle)  ← drop-in replacement
    ├── issues DRAM requests (per warp MEM instruction) → DRAMSim3
    ├── receives DRAMSim3 callbacks → n_mem_wait--, n_ready++
    └── stalls on NVLink/UCIe credit exhaustion → n_comm_wait++
```

Everything outside ChipletArch — DRAMSim3 instances, UCIe links, topology, AllReduce
scheduling — is unchanged. The decoupling tax analysis carries over exactly: UCIe
credit exhaustion stalls GPU warps just as it stalls SA rows, and the resulting DRAM
idle cycles are the same class of coupling effect.

---

## 6. What Accurate Modeling Would Still Require

The best-case model gives a ceiling. The gap to accurate modeling, by component:

**Occupancy: small gap**  
Static computation is exact for the best-case ceiling. Real occupancy can be lower due
to launch configuration choices. Fixable by taking occupancy as a measured input
(one `ncu` profile run) rather than computing the ceiling.

**Coalescing: small gap for LLM kernels**  
GEMM and FlashAttention are written for coalesced access. Irregular kernels (sparse
attention, custom ops) can be 10–32× worse. Model coalescing efficiency as a parameter
`c ∈ (0, 1]` multiplying the effective memory bandwidth, default 1.0 for best case.

**L1/L2 cache: medium gap for prefill, small for decode**  
For decode (M=1, streaming access): L1/L2 are effectively bypassed — best-case assumption
(no L1) is actually accurate. For prefill (tiled GEMM): weight tiles reused across
sequence positions. Capturable with a hit rate parameter `h` that reduces effective HBM
traffic: `hbm_bytes = total_bytes × (1 - h)`. For a known tile size and cache size, `h`
is computable analytically (tile fits in L2 → h ≈ 1 - 1/seqlen_tile_ratio).

**Divergence: negligible for LLM kernels**  
GEMM inner loop has no branches. Softmax reduction is structured. Safe to ignore for
the LLM workloads in scope.

**Warp scheduler policy: small gap**  
RR vs GTO changes cache locality but not bandwidth. For memory-bound kernels (decode),
the scheduling policy doesn't affect DRAM traffic — only latency hiding, and the
occupancy-based hiding analysis is the same.

**Shared memory bank conflicts: negligible for standard tiling**  
GEMM with standard 16×16 or 32×32 tiles and `__ldg()` loads is written to avoid bank
conflicts. Capturable as a latency multiplier if needed.

**Summary:** for LLM inference kernels on standard hardware, the best-case FSM model
is expected to be within 10–15% of real hardware for prefill and within 5–10% for decode
(where streaming access and lack of divergence make best-case assumptions accurate).
The remaining gap is coalescing efficiency and L2 hit rate for prefill tiling — both
fixable with one measured parameter per kernel.

---

## 7. References

- Volkov, V. "Better performance at lower occupancy." GPU Technology Conference 2010.
  — The canonical paper showing occupancy is not the only thing that matters; latency
  hiding through ILP can substitute for thread-level parallelism.

- Hong, S. & Kim, H. "An analytical model for a GPU architecture with memory-level and
  thread-level parallelism awareness." ISCA 2009.
  — First formal model relating occupancy, memory latency, and throughput. The
  Little's Law argument for GPUs.

- Luo, Y. et al. "AIME: Enabling Efficient Mapping and Hardware Scalability for Vision
  Transformer Accelerators." DAC 2023.
  — Analytical GPU model with occupancy and cache parameterization; closest prior work
  to the best-case FSM approach described here.

- Nai, L. et al. "GraphPhi: Enabling Partitioned Graph Processing on Emerging
  Throughput-Oriented Architectures." SC 2017 (GPGPU-Sim reference).

- Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness." NeurIPS 2022.
  — Relevant for understanding why FlashAttention approaches best-case: it is
  explicitly designed to maximize coalescing and minimize HBM traffic.
