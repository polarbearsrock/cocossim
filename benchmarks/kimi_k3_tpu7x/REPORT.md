# Kimi K3 on TPU7x: COCOSSim proxy results

> These are **not measurements from Google TPU hardware**. Raw cycles come from a compute-only COCOSSim proxy; latency and throughput are uncalibrated analytical proxy estimates assembled from published TPU7x lower-bound terms. ICI collectives, XLA fusion/layout, and hardware calibration remain outstanding.

## Configuration

- Scope: Kimi K3 text decoder (vision tower omitted).
- Layer mix: 1 dense KDA + 68 KDA-MoE + 24 MLA-MoE.
- TPU7x slice: 16 chips; 2 TensorCores/chip; 256x256 MXUs.
- Published peaks per chip: 2307 BF16 TFLOP/s, 4614 FP8 TFLOP/s, 7380 GB/s HBM, 1200 GB/s bidirectional ICI.
- K3 precision proxy: routed-expert and latent-MoE projection weights use MXFP4 plus group scales and their GEMMs use FP8 peak; ignored/unquantized GEMMs and cache/state traffic use BF16.

The aligned calibration GEMM 8192x7168x16384 completed in 946,488 COCOSSim cycles (2.03e+06 FLOP/proxy-cycle). This normalizes shape/scheduling efficiency only; it does not infer a TPU clock.

## Primary results

| Phase | B | Query | Context | Modeled FLOPs | Raw full-stack cycles* | eta_CCS | Shape compute (ms) | Compressed-cache HBM (ms) | Expanded-K/V HBM sensitivity (ms) | No-contention proxy (ms) | Bottleneck | Aggregate tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| prefill | 1 | 128 | 128 | 26.4T | 69,083,944 | 0.1883 | 2.822 | 13.116 | 13.117 | 13.116 | HBM | 9,759 |
| prefill | 1 | 1024 | 1024 | 213T | 348,103,152 | 0.3009 | 14.253 | 13.116 | 13.129 | 14.253 | shape-adjusted compute | 71,843 |
| prefill | 1 | 8192 | 8192 | 1.79P | analytic only | 0.2748 | 133.373 | 13.118 | 13.218 | 133.373 | shape-adjusted compute | 61,422 |
| decode | 1 | 1 | 1024 | 210G | 32,612,460 | 0.0031 | 1.356 | 1.105 | 1.117 | 1.356 | shape-adjusted compute | 738 |
| decode | 1 | 1 | 8192 | 221G | 33,139,212 | 0.0032 | 1.399 | 1.106 | 1.207 | 1.399 | shape-adjusted compute | 715 |
| decode | 1 | 1 | 131072 | 402G | 42,344,052 | 0.0046 | 2.035 | 1.135 | 2.741 | 2.035 | shape-adjusted compute | 492 |
| decode | 32 | 1 | 8192 | 7.07T | 48,082,672 | 0.0715 | 2.030 | 8.061 | 11.273 | 8.061 | HBM | 3,970 |
| decode | 256 | 1 | 8192 | 56.5T | 156,929,424 | 0.1753 | 6.626 | 14.564 | 40.262 | 14.564 | HBM | 17,577 |

\* Raw full-stack cycles are extrapolated from three simulated representative blocks using the exact 1/68/24 layer mix. They exclude the final norm and optional LM head, and they are normalized COCOSSim cycles—not TPU7x cycles.

The primary HBM column counts learned weights plus a compressed 512+64 MLA cache and BF16 KDA-state read/write traffic. The expanded-K/V column is a sensitivity for the current reference-style 96-head K/V representation; it is not treated as compulsory traffic. Both assume perfect sharding and exclude contention. The no-contention proxy is `max(shape-adjusted compute, compressed-cache HBM, ideal routed-MoE one-way-injection ICI)`; tensor-parallel collectives and topology effects are not yet included.

## Representative COCOSSim runs

| Scenario | Dense-KDA cycles | KDA-MoE cycles | MLA-MoE cycles | Mean SA scheduled-active | Mean VU scheduled-active |
|---|---:|---:|---:|---:|---:|
| prefill_b1_q128_ctx128_balanced_aggregate | 762,368 | 780,616 | 634,987 | 77.90% | 0.69% |
| prefill_b1_q1024_ctx1024_balanced_aggregate | 2,909,184 | 3,716,136 | 3,854,030 | 80.35% | 0.61% |
| decode_b1_q1_ctx1024_balanced_aggregate_lmhead | 459,816 | 366,273 | 301,920 | 73.19% | 0.84% |
| decode_b1_q1_ctx8192_balanced_aggregate_lmhead | 459,816 | 366,273 | 323,868 | 73.47% | 0.83% |
| decode_b1_q1_ctx131072_balanced_aggregate_lmhead | 459,816 | 366,273 | 707,403 | 77.28% | 0.71% |
| decode_b32_q1_ctx8192_balanced_aggregate_lmhead | 532,352 | 466,096 | 660,658 | 75.08% | 0.78% |
| decode_b256_q1_ctx8192_balanced_aggregate_lmhead | 1,069,056 | 1,199,976 | 3,094,250 | 77.12% | 0.71% |

`scheduled-active` is the fraction of COCOSSim cycles in a non-idle unit state. It is not MXU lane utilization or TPU profiler utilization.

## Capacity and fidelity notes

- The configured K3 checkpoint is 1.56 TB decimal (1452.9 GiB). This 16-chip slice supplies 3072 GiB HBM (1619.1 GiB raw headroom). The raw-fit minimum is 8 chips and does not reserve runtime/cache workspace.
- The manifest labels exact learned GEMM shapes separately from proxies. KDA recurrence/short convolution, RMSNorm and gating, MLA attention, and grouped MoE execution are approximations; AttnRes scoring and the vision tower are omitted.
- Aggregate expert traces model ideal packed expert-token rows. They do not preserve per-expert small-M utilization or routing skew; use `--expert-layout per-expert` for that sensitivity.
- COCOSSim's legacy DRAMSim3 HBM2 model is intentionally bypassed. TPU7x HBM and ICI appear only in the analytical bounds.
- A hardware-calibrated result requires exact-shape KDA, MLA, and grouped-MoE microbenchmarks on TPU7x plus XProf measurements.

## Sources

- [Google TPU7x documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Google Ironwood performance guidance](https://docs.cloud.google.com/tpu/docs/ironwood-performance)
- [Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 official configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
