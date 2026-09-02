#!/usr/bin/env bash
# FITTED TPU v6e configuration (fidelity spec section 6; first tier-1 fit,
# 2026-09-01). The pinned model (configs/tpuv6e.sh) plus the two coordinates
# the H1 scan identified:
#   -dram_enq 15         issue width = HBM plate at 1.75 GHz (14.6 beats/cycle)
#   -op_overhead 12250   ~7 us per op boundary per core (t0 from G3 / E1)
# Objective 0.104 (mean |ln(sim/silicon)| over 84 tier-1 cells, PASS 54/84)
# vs 0.190 / 39 at priors. -kv_bw_pct and -data_overhead await H2 and the
# tier-2 fit subset F. Holdout points are scored with this file untouched.
# Usage: configs/tpuv6e_fitted.sh <layer_file> <stats_out> [extra flags...]
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
  -dram_enq 15 \
  -job_overhead 0 \
  -op_overhead 12250 \
  -fuse_epilogue 0 \
  -fuse_attn 1 \
  -fuse_vpu 1 \
  -dbuf 48 \
  -dbuf_tile 1 \
  -vmem_rows 0 \
  -act_share 1 \
  -i "$LAYERS" -o "$OUT" "$@"
