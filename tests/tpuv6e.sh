#!/usr/bin/env bash
# TPUv6e-model tests (spec: docs/superpowers/specs/2026-08-27-tpuv6e-model-calibration-design.md)
# V1  -buf_mb reaches the layer generator (Softmax split count changes)
# V2  -dram_enq throttles memory issue (cycles increase)
# V3  -job_overhead adds fixed cycles per job
# V3b invalid new-flag values are rejected cleanly
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/build/perf_model"
WORK="${TMPDIR:-/tmp}/cocossim_tpuv6e_$$"
mkdir -p "$WORK"
cd "$REPO/build" || exit 2
[ -x "$BIN" ] || { echo "build/perf_model missing - build first" >&2; exit 2; }
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }
cycles_of() { awk '/^Cycles/{print $2; exit}' "$1"; }

# V1: Softmax 4096 with default 8 MiB buffer splits into 4 jobs (regression T2);
# with -buf_mb 1 the split factor is max(ceil(32MiB/1MiB)=32, ceil(4096/1024)=4)=32
# so Mp=128 and 32 jobs are created.
printf 'Softmax 4096\n' > "$WORK/v1.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v1.txt" -o "$WORK/v1_stats.txt" > "$WORK/v1.log" 2>&1
n_jobs=$(grep -c 'Job Type: 1' "$WORK/v1.log")
if [ "$n_jobs" -eq 32 ]; then ok "V1 -buf_mb 1 -> 32 softmax jobs"; else bad "V1 got $n_jobs jobs (want 32)"; fi

# V2: memory-hungry LayerNorm; throttling enqueue to 1 beat/cycle must cost cycles.
printf 'LayerNorm 4096 1024\n' > "$WORK/v2.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v2.txt" -o "$WORK/v2_base.txt" > /dev/null 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_enq 1 -i "$WORK/v2.txt" -o "$WORK/v2_slow.txt" > /dev/null 2>&1
cb=$(cycles_of "$WORK/v2_base.txt"); cs=$(cycles_of "$WORK/v2_slow.txt")
if [ -n "$cb" ] && [ -n "$cs" ] && [ "$cs" -gt "$cb" ]; then
  ok "V2 -dram_enq 1 slows memory-bound run ($cb -> $cs)"
else bad "V2 cycles base=$cb slow=$cs"; fi

# V3: one-job workload; -job_overhead 1000 must add a large, overhead-dominated
# number of extra cycles. Derivation from the job dump (build/perf_model, this
# binary): "Activation 64" is exactly one VECTOR_UNIT job (Dims: 64 x 1, a
# single BROADCAST phase). VecUnitState::init() sets that job's stage-1
# compute duration to div_ru(lin*par*phase_mult, vu_sz) = div_ru(1*64*1,64) =
# 1 cycle, so at -job_overhead 0 the observed 38 cycles are governed by DRAM
# read/write round-trip latency, not compute -- i.e. the memory-bound
# baseline already hides slack that a compute-bound run would not have. Once
# -job_overhead dominates that latency, cycles == overhead + 4 exactly
# (checked by sweeping -job_overhead over 100/500/999/1000/1001/2000/5000:
# every value reproduces overhead+4 on the nose). So -job_overhead 1000 ->
# 1004 cycles, a delta of 1004-38=966 against the memory-bound baseline,
# short of the raw 1000 by the 34 cycles (38-4) that baseline already hid.
# Assert a margin safely under that measured 966-cycle floor.
printf 'Activation 64\n' > "$WORK/v3.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v3.txt" -o "$WORK/v3_base.txt" > /dev/null 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -job_overhead 1000 -i "$WORK/v3.txt" -o "$WORK/v3_ovh.txt" > /dev/null 2>&1
cb=$(cycles_of "$WORK/v3_base.txt"); co=$(cycles_of "$WORK/v3_ovh.txt")
if [ -n "$cb" ] && [ -n "$co" ] && [ $((co - cb)) -ge 900 ]; then
  ok "V3 -job_overhead 1000 adds $((co - cb)) cycles"
else bad "V3 base=$cb overhead=$co"; fi

# V3b: invalid values for each new flag must be rejected (exit 1 + message).
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 0 -i "$WORK/v3.txt" -o "$WORK/v3c.txt" > "$WORK/v3c.log" 2>&1; r1=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_enq 0 -i "$WORK/v3.txt" -o "$WORK/v3d.txt" > "$WORK/v3d.log" 2>&1; r2=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -job_overhead -1 -i "$WORK/v3.txt" -o "$WORK/v3e.txt" > "$WORK/v3e.log" 2>&1; r3=$?
if [ "$r1" -eq 1 ] && grep -q 'buf_mb' "$WORK/v3c.log" \
   && [ "$r2" -eq 1 ] && grep -q 'dram_enq' "$WORK/v3d.log" \
   && [ "$r3" -eq 1 ] && grep -q 'job_overhead' "$WORK/v3e.log"; then
  ok "V3b invalid -buf_mb/-dram_enq/-job_overhead rejected"
else
  bad "V3b rejection exit codes: buf=$r1 enq=$r2 ovh=$r3"
fi

# V4: -dram_ini selects the DRAM config. The GDDR6 ini has bus_width=128
# with BL=16 -> 256-byte requests vs HBM2's 64 (verified by running the
# binary against it during plan review), so REQUEST SIZE BYTES must change;
# a missing file must die cleanly with a message naming the path.
printf 'LayerNorm 1024 1024\n' > "$WORK/v4.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini ../dramsim3/configs/GDDR6_8Gb_x16.ini \
  -i "$WORK/v4.txt" -o "$WORK/v4_stats.txt" > "$WORK/v4.log" 2>&1
rc=$?
req=$(awk '/REQUEST SIZE BYTES/{print $NF; exit}' "$WORK/v4.log")
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini /nonexistent/nope.ini \
  -i "$WORK/v4.txt" -o "$WORK/v4b.txt" > "$WORK/v4b.log" 2>&1
rcb=$?
if [ "$rc" -eq 0 ] && [ -n "$req" ] && [ "$req" -ne 64 ] && [ "$rcb" -eq 1 ] && grep -q 'nope.ini' "$WORK/v4b.log"; then
  ok "V4 -dram_ini honored (req size $req) and missing file rejected"
else
  bad "V4 rc=$rc req=${req:-none} missing-file rc=$rcb"
fi

# V5: HBM2e_v6e ini must deliver >= 800 GB/s achieved on a memory-bound
# streaming workload (target 1638 GB/s x typical DRAM efficiency), and the
# request size must stay 64B. Two caps must be non-binding for the test to
# measure DRAM: enqueue width (-dram_enq 32 at 1.75 GHz = 3.58 TB/s issue)
# and compute (-vu_sz 2048: 200M elems / 2048 lanes = 98k compute cycles,
# far below the ~1.4M memory cycles; at the default vu_sz 64 the run is
# compute-bound and measures nothing about the ini).
# Workload: Activation 200000000 = 200M elems: 400 MB read + 400 MB write.
printf 'Activation 200000000\n' > "$WORK/v5.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 2048 -f 1.75 -dram_enq 32 -dram_ini ../configs/HBM2e_v6e.ini \
  -i "$WORK/v5.txt" -o "$WORK/v5_stats.txt" > "$WORK/v5.log" 2>&1
rc=$?
cyc=$(cycles_of "$WORK/v5_stats.txt")
req=$(awk '/REQUEST SIZE BYTES/{print $NF; exit}' "$WORK/v5.log")
# achieved GB/s = 800e6 bytes / (cycles / 1.75 GHz) / 1e9 = 800*1.75/cycles * 1e6... in awk:
bw=$(awk -v c="${cyc:-0}" 'BEGIN{ if (c>0) printf "%d", 800000000.0 / (c / 1.75) ; else print 0 }')
if [ "$rc" -eq 0 ] && [ "$req" = "64" ] && [ "$bw" -ge 800 ] && [ "$bw" -le 1800 ]; then
  ok "V5 HBM2e_v6e achieves ${bw} GB/s (cycles=$cyc)"
else
  bad "V5 rc=$rc req=${req:-?} bw=${bw:-?} GB/s cycles=${cyc:-?}"
fi

# V6: OS-mode job dims must carry the true M. Matmul 1 256 256 at -sa_sz 64
# must produce exactly one SA job printed as "Dims: 1 x 256 x 256" (the old
# code prints "Dims: 64 x 256 x 256"). Matmul 100 256 256 must produce
# ceil(100/64)=2 jobs: one 64-row and one 36-row.
printf 'Matmul 1 256 256\n' > "$WORK/v6.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v6.txt" -o "$WORK/v6_stats.txt" > "$WORK/v6.log" 2>&1
printf 'Matmul 100 256 256\n' > "$WORK/v6b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v6b.txt" -o "$WORK/v6b_stats.txt" > "$WORK/v6b.log" 2>&1
if grep -q 'Dims: 1 x 256 x 256' "$WORK/v6.log" \
   && grep -q 'Dims: 64 x 256 x 256' "$WORK/v6b.log" \
   && grep -q 'Dims: 36 x 256 x 256' "$WORK/v6b.log"; then
  ok "V6 OS jobs carry true M (1; 64+36)"
else
  bad "V6 job dims wrong (see $WORK/v6.log, $WORK/v6b.log)"
fi

# V6b: a decode-shaped GEMM must read its full weight/KV panel. Matmul 1 64 2048
# at -sa_sz 64: weights = K*N*dtw = 64*2048*2 = 256 KiB = 4096 beats (64 B/beat),
# charged min(sz,N)*K*dtw = 8 KiB per column tile x 32 tiles. Before this fix
# the reads are min(sz,M)*K-scaled: ~2 beats/tile, ~64 beats total, and DRAM
# CMDs (reads+writes+init) sit far below 4000.
printf 'Matmul 1 64 2048\n' > "$WORK/v6c.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v6c.txt" -o "$WORK/v6c_s.txt" > "$WORK/v6c.log" 2>&1
cmds=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v6c.log" | tail -1 | awk '{print $3}')
if [ "${cmds:-0}" -ge 4000 ]; then
  ok "V6b decode-shaped GEMM reads its KV/weight panel (DRAM CMDs=$cmds)"
else
  bad "V6b DRAM CMDs=$cmds (< 4000: weight-side reads missing)"
fi

echo "==== $PASS passed, $FAIL failed (outputs in $WORK)"
exit "$FAIL"
