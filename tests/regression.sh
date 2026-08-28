#!/usr/bin/env bash
# Regression tests for COCOSSim scheduler and layer-generation behavior.
# Each test pins a behavior fixed after the codex/tpuv6-model branch review:
#   T1  anonymous VPU jobs are distributed across all vector units
#   T2  Softmax models every row chunk (work is conserved regardless of VPU count)
#   T3  WS sequential-fallback jobs respect the buffer-fit inequality
#   T4  invalid core counts are rejected with a clear error, not UB
#   T5  the shipped CNN example is dimensionally consistent end to end
#
# Usage: tests/regression.sh   (builds are NOT triggered; build/perf_model must exist)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/build/perf_model"
WORK="${TMPDIR:-/tmp}/cocossim_regression_$$"
mkdir -p "$WORK"
# DRAMSim3 config is resolved relative to the build directory
cd "$REPO/build" || exit 2
[ -x "$BIN" ] || { echo "build/perf_model missing - build first" >&2; exit 2; }

PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

# T1: six independent LayerNorm jobs on -c 4 must reach all four vector units.
printf 'LayerNorm 32768 768\n' > "$WORK/t1.txt"
"$BIN" -c 4 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/t1.txt" -o "$WORK/t1_stats.txt" > "$WORK/t1.log" 2>&1
n_active_vu=$(awk '/^VECTOR_UNIT/ && $2+0 > 0 {n++} END{print n+0}' "$WORK/t1_stats.txt" 2>/dev/null)
if [ "${n_active_vu:-0}" -eq 4 ]; then
  ok "T1 all 4 vector units received work (active VUs: $n_active_vu)"
else
  bad "T1 vector work not distributed: only ${n_active_vu:-0}/4 VUs active"
fi

# T2: Softmax 4096 splits into 4 chunks of 1024 rows; all 4 must become jobs.
printf 'Softmax 4096\n' > "$WORK/t2.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/t2.txt" -o "$WORK/t2_stats.txt" > "$WORK/t2.log" 2>&1
n_vpu_jobs=$(grep -c 'Job Type: 1' "$WORK/t2.log")
if [ "$n_vpu_jobs" -eq 4 ]; then
  ok "T2 Softmax 4096 creates 4 chunk jobs"
else
  bad "T2 Softmax 4096 created $n_vpu_jobs VPU jobs (expected 4: work dropped)"
fi

# T3: WS fallback for Matmul 512x128x8130 (-sa_sz 64): every job must satisfy
# (M*(N + min(K,sa)))*dtype <= 8MiB  =>  N <= 8192 - 64 = 8128, and the split
# must conserve total N (= 8130).
printf 'Matmul 512 128 8130\n' > "$WORK/t3.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -ws 1 -f 1 -i "$WORK/t3.txt" -o "$WORK/t3_stats.txt" > "$WORK/t3.log" 2>&1
read -r max_n sum_n <<< "$(grep -o 'Dims: 512 x 128 x [0-9]*' "$WORK/t3.log" \
  | awk '{n=$NF; if (n>mx) mx=n; s+=n} END{print mx+0, s+0}')"
if [ "${max_n:-0}" -le 8128 ] && [ "${sum_n:-0}" -eq 8130 ]; then
  ok "T3 WS fallback jobs bufferable (max N=$max_n) and work-conserving (sum N=$sum_n)"
else
  bad "T3 WS fallback violates buffer fit or drops work (max N=${max_n:-0} sum N=${sum_n:-0})"
fi

# T4: -c 0 must be rejected with a clear diagnostic, not UB/crash.
printf 'Activation 1024\n' > "$WORK/t4.txt"
"$BIN" -c 0 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/t4.txt" -o "$WORK/t4_stats.txt" > "$WORK/t4.log" 2>&1
rc=$?
if [ "$rc" -eq 1 ] && grep -qi 'core' "$WORK/t4.log" && ! grep -q 'Sanitizer' "$WORK/t4.log"; then
  ok "T4 -c 0 rejected cleanly (exit 1 with message)"
else
  bad "T4 -c 0 not rejected cleanly (exit $rc; see $WORK/t4.log)"
fi

# T5: shipped CNN example must be dimensionally self-consistent (each Conv's
# in_ch/in_hw match the previous layer's output under the parser's defaults
# k=3 s=1 p=1; each Activation's element count equals the preceding Conv's
# output tensor), include a classifier head, and run to completion.
consistency=$(awk '
  function outdim(x, k, s, p) { return int((x + 2*p - k) / s) + 1 }
  /^Conv/ {
    ic=$3; h=$4; w=$5; oc=$6; k=(NF>=7?$7:3); s=(NF>=8?$8:1); p=(NF>=9?$9:1)
    if (have && ic != ch)          { printf "line %d: in_ch %d != prev out_ch %d; ", NR, ic, ch; bad=1 }
    if (have && (h != H || w != W)){ printf "line %d: in %dx%d != prev out %dx%d; ", NR, h, w, H, W; bad=1 }
    ch=oc; H=outdim(h,k,s,p); W=outdim(w,k,s,p); have=1; next
  }
  /^Activation/ {
    prod=1; for (i=2;i<=NF;i++) prod*=$i
    if (have && prod != ch*H*W) { printf "line %d: Activation %d elems != conv out %d; ", NR, prod, ch*H*W; bad=1 }
    next
  }
  END { if (bad) exit 1 }
' "$REPO/examples/cnn_model.txt" 2>&1)
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$REPO/examples/cnn_model.txt" -o "$WORK/t5_stats.txt" > "$WORK/t5.log" 2>&1
rc=$?
if [ -z "$consistency" ] && [ "$rc" -eq 0 ] && grep -q '^processing Softmax' "$WORK/t5.log"; then
  ok "T5 CNN example dimensionally consistent, has classifier, runs (exit 0)"
else
  bad "T5 CNN example broken (exit $rc; ${consistency:-classifier missing or run failed})"
fi

echo "==== $PASS passed, $FAIL failed (outputs in $WORK)"
exit "$FAIL"
