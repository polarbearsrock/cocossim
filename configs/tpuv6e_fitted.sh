#!/usr/bin/env bash
# TPU v6e-1 model, FITTED (2026-09-01, fidelity spec section 6.3): the pinned
# priors of tpuv6e.sh plus two calibration terms chosen on the Qwen3-8B
# tier-2 device times and checked on the Mistral-7B-v0.3 holdout:
#   -dram_enq 12          sustained HBM bandwidth inside a model step: silicon
#                         streams weights at ~1.15 TB/s in-model (XLA copy
#                         traffic + DMA gaps), the priors' pipeline at ~1.5.
#   -data_overhead 1750000  1 ms per forward for XLA's layout/copy kernels
#                         (the census "data" class: 0.8-1.2 ms per decode step
#                         on both models), which the model has no jobs for.
#                         It is a per-RUN constant meant for whole-model runs
#                         (one Transformer forward or decode step per run);
#                         for single-kernel runs (Matmul/Add cells) append
#                         -data_overhead 0, or the 1 ms swamps a 4 us kernel.
# Qwen fit set: MAPE 14.8% -> 9.2%, bias -10.6% -> -0.2%, 10 PASS / 6 COND /
# 0 FAIL of 16. Mistral holdout: 18.8% -> 15.4%, bias -17% -> -7%.
# Known residuals (not fixable by these knobs): prefill attention at
# S >= 2048 (sim 1.6-2.2x too slow, +21% at 4096x1), decode KV streaming at
# long context (sim +16..+21% at 4096x16 / 8192x8), batched short prefill
# (-12..-15% at 512x4/x8).
# Usage: configs/tpuv6e_fitted.sh WORKLOAD.txt STATS.txt [extra flags...]
exec bash "$(dirname "${BASH_SOURCE[0]}")/tpuv6e.sh" "$@" -dram_enq 12 -data_overhead 1750000
