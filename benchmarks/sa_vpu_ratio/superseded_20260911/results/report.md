# Qwen 2.5 7B: SA / VPU compute-capacity sweep

One decoder layer, prefill, batch **64**, sequence **1024** (65,536 tokens).

Fixed **144 × 32×32 SAs**, varying **128-lane VPUs**, 1.9 GHz. One MAC/cell/cycle; one simple vector operation/lane/cycle. The horizontal axis is **peak MACs/cycle ÷ peak simple vector operations/cycle**. A MAC is counted once, not as two FLOPs.

This is a compute-only COCOSSim extension: original SA/VPU state-machine service times, zero memory latency/traffic/contention, and a custom Qwen layer frontend with a fixed FIFO scheduler. No model weights or numerical tensors are evaluated. These are modeled timings, not measured MTIA latencies.

| MAC:vector-op capacity | VPUs | Layer ms (NL=1) | Change vs 72 VPUs | SA MAC utilization | VPU utilization |
|---:|---:|---:|---:|---:|---:|
| 4:1 | 288 | 59.344 | -0.72% | 94.75% | 0.58% |
| 8:1 | 144 | 59.487 | -0.48% | 94.53% | 1.15% |
| 16:1 | 72 | 59.774 | +0.00% | 94.07% | 2.30% |
| 32:1 | 36 | 60.333 | +0.93% | 93.20% | 4.56% |
| 64:1 | 18 | 61.450 | +2.80% | 91.51% | 8.94% |
| 128:1 | 9 | 65.417 | +9.44% | 85.96% | 16.80% |
| 192:1 | 6 | 70.684 | +18.25% | 79.55% | 23.33% |
| 288:1 | 4 | 78.799 | +31.83% | 71.36% | 31.39% |
| 384:1 | 3 | 87.060 | +45.65% | 64.59% | 37.88% |
| 576:1 | 2 | 103.562 | +73.26% | 54.30% | 47.77% |
| 1152:1 | 1 | 153.090 | +156.11% | 36.73% | 64.62% |

## Nonlinear cost sensitivity

NL is the lane-cycle equivalent cost of exp, reciprocal, and rsqrt; ordinary arithmetic stays at one. These are explicit throughput assumptions, not a claim about a particular SFU pipeline.

| NL cost | 72-VPU reference | Largest tested MAC:VOP ratio within 5% of reference | VPUs | Layer ms |
|---:|---:|---:|---:|---:|
| 1 | 59.774 ms | 64:1 | 18 | 61.450 |
| 4 | 60.200 ms | 64:1 | 18 | 63.160 |

## Work and scheduling assumptions

- Qwen dimensions: hidden 3,584; MLP 18,944; 28 query heads; 4 KV heads; head dimension 128.
- Full decoder layer: two RMSNorms, Q/K/V projections and biases, RoPE, attention, output projection, two residual adds, gate/up projections, SiLU-times-up, and down projection. No embedding or LM head.
- Dense masked causal attention. Both attention GEMMs execute the full S×S shape, including the masked upper triangle. Stable softmax includes scaling, causal-mask addition, max, subtract, exp, sum, reciprocal, and normalization. This is not FlashAttention or a triangular-tile-skipping schedule.
- RoPE uses cached sine/cosine operands and three arithmetic operations per output element, with sign folded into subtraction. SiLU-times-up costs four ordinary operations plus exp and reciprocal per element.
- Reductions retain COCOSSim’s model: one independent row per lane, with serial reduction along that row. The work census therefore counts D reduction steps for a row of length D. Parallel intra-row tree reductions are not modeled.
- SA output tiles are at most 32×32 with full K accumulation and exact tail tiles. Elementwise jobs contain at most 16,384 elements; normalization and attention groups contain at most 128 rows. Job sizes and graph dependencies are identical across VPU counts.
- Global FIFO ready queues for SAs and VPUs. Matrix and vector jobs may overlap when dependencies allow. Projection and MLP boundaries use operator completion; attention pipelines independently per sequence, head, and 128-row group.
- Every configuration keeps all 144 SAs. This is a vector-capacity sensitivity study; area and power are not held constant.

## Validation

- Native SA, elementwise, and reduction tile timings checked against hand-calculated cycles.
- GEMM tail tiling checked with 35×7×67 work, independently of array count.
- A hand-calculated dependency chain and 50 random DAGs compare event advancement with cycle stepping.
- Analytical Qwen MAC and vector-operation censuses are asserted, as are scheduled service/work totals and unit capacity.
- Work and job counts are invariant across every ratio in each nonlinear-cost variant.

## Files

- `sweep.csv`: every ratio, timing, utilization, work count, job count, and queue statistic.
- `operators.csv`: per-category service demand and first/last execution cycles. Category spans can overlap.
- `timeline.csv`: time-binned SA/VPU occupancy and ready-job queue sizes.
- `work_census.csv`, `native_timings.csv`: work accounting and cached native tile measurements.
- `manifest.json`: commands, source revision, source hashes, and assumptions.

## Sources

- [Official Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Qwen2 implementation, Transformers 4.43.4](https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/qwen2/modeling_qwen2.py)
- [COCOSSim](https://github.com/mc186/cocossim)
