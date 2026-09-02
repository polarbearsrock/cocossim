#!/usr/bin/env bash
# Canonical TPU v6e model configuration (spec 3.1/3.6).
# Geometry per Google's v6e docs: one TensorCore with 2 MXUs of 256x256 and
# one vector unit. 2 packed bf16 MACs/PE/cycle at a physical 1.75 GHz gives
# the published 918 TFLOPS peak (the MACs/PE value and VPU width are
# hypotheses -- Google publishes neither -- and Phase C confirms or
# falsifies them).
# PRIORS, to be frozen after calibration (spec 5): -f, -dram_enq,
# -job_overhead, -vu_sz. -buf_mb 128 is the nominal VMEM hypothesis.
# Usage: configs/tpuv6e.sh <layer_file> <stats_out> [extra flags...]
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAYERS="$1"; OUT="$2"; shift 2
cd "$REPO/build"
exec ./perf_model \
  -c 2 \
  -n_vpu 1 \
  -sa_sz 256 \
  -vu_sz 512 \
  -mxu_macs_per_pe 2 \
  -f 1.75 \
  -ws 0 \
  -buf_mb 128 \
  -dram_ini ../configs/HBM2e_v6e.ini \
  -dram_enq 32 \
  -job_overhead 0 \
  -fuse_epilogue 0 \
  -fuse_attn 1 \
  -fuse_vpu 1 \
  -dbuf 48 \
  -dbuf_tile 1 \
  -vmem_rows 0 \
  -act_share 1 \
  -attn_group 1 \
  -kv_prefetch 1 \
  -i "$LAYERS" -o "$OUT" "$@"
