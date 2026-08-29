#!/usr/bin/env bash
# Canonical TPU v6e model configuration (spec 3.1/3.6).
# Geometry hypothesis: 4 MXUs of 256x256 at 1.75 GHz (peak-TFLOPS
# decomposition, v5e lineage) - Phase C probes confirm or falsify.
# PRIORS, to be frozen after calibration (spec 5): -f, -dram_enq,
# -job_overhead, -vu_sz. -buf_mb 128 is the nominal VMEM hypothesis.
# Usage: configs/tpuv6e.sh <layer_file> <stats_out> [extra flags...]
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAYERS="$1"; OUT="$2"; shift 2
cd "$REPO/build"
exec ./perf_model \
  -c 4 \
  -sa_sz 256 \
  -vu_sz 256 \
  -f 1.75 \
  -ws 0 \
  -buf_mb 128 \
  -dram_ini ../configs/HBM2e_v6e.ini \
  -dram_enq 32 \
  -job_overhead 0 \
  -fuse_epilogue 0 \
  -i "$LAYERS" -o "$OUT" "$@"
