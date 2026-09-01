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

# V3b: invalid values for each new flag must be rejected (exit 1 + message),
# including -buf_mb's upper bound (buf_mb*1024*1024 must stay in-range for int).
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 0 -i "$WORK/v3.txt" -o "$WORK/v3c.txt" > "$WORK/v3c.log" 2>&1; r1=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_enq 0 -i "$WORK/v3.txt" -o "$WORK/v3d.txt" > "$WORK/v3d.log" 2>&1; r2=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -job_overhead -1 -i "$WORK/v3.txt" -o "$WORK/v3e.txt" > "$WORK/v3e.log" 2>&1; r3=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 2048 -i "$WORK/v3.txt" -o "$WORK/v3f.txt" > "$WORK/v3f.log" 2>&1; r4=$?
if [ "$r1" -eq 1 ] && grep -q 'buf_mb' "$WORK/v3c.log" \
   && [ "$r2" -eq 1 ] && grep -q 'dram_enq' "$WORK/v3d.log" \
   && [ "$r3" -eq 1 ] && grep -q 'job_overhead' "$WORK/v3e.log" \
   && [ "$r4" -eq 1 ] && grep -q 'buf_mb' "$WORK/v3f.log"; then
  ok "V3b invalid -buf_mb/-dram_enq/-job_overhead rejected"
else
  bad "V3b rejection exit codes: buf=$r1 enq=$r2 ovh=$r3 buf_hi=$r4"
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

# V7: cycle-accounting invariant + attribution.
# (a) For every unit: busy+underfilled+memstall+idle == Cycles.
# (b) A memory-bound workload must show memstall > 0 on the vector unit.
#     Activation 50M elems at -vu_sz 1024: compute = 50M/1024 = 49k cycles,
#     memory = 100 MB read + 100 MB write = 3.1M beats at ~4-9 beats/cycle
#     >> compute, so the VPU spends most cycles waiting on DRAM. (At the
#     default vu_sz 64 the VPU demand of 128 B/cycle sits below HBM2's
#     ~256 B/cycle and the run is compute-bound: memstall would be 0.)
# (c) An underfilling GEMM (M=1 on a 64-wide array) must show
#     underfilled > 0 on the systolic array.
printf 'Activation 50000000\n' > "$WORK/v7.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 1024 -f 1 -i "$WORK/v7.txt" -o "$WORK/v7_stats.txt" > "$WORK/v7.log" 2>&1
printf 'Matmul 1 256 256\n' > "$WORK/v7b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v7b.txt" -o "$WORK/v7b_stats.txt" > "$WORK/v7b.log" 2>&1
inv=$(awk '/^Cycles/{c=$2} /^ACCT/{ if ($5+$7+$9+$11 != c) print "bad:" $0 }' "$WORK/v7_stats.txt" "$WORK/v7b_stats.txt")
ms=$(awk '/^ACCT VECTOR_UNIT/{print $9; exit}' "$WORK/v7_stats.txt")
uf=$(awk '/^ACCT SYSTOLIC_ARRAY/{print $7; exit}' "$WORK/v7b_stats.txt")
if [ -z "$inv" ] && [ "${ms:-0}" -gt 0 ] && [ "${uf:-0}" -gt 0 ]; then
  ok "V7 accounting sums to Cycles; memstall=$ms underfilled=$uf attributed"
else
  bad "V7 invariant='$inv' memstall=${ms:-?} underfilled=${uf:-?}"
fi

# V8: effective FLOP utilization must expose under-fill that pct_active
# hides. Matmul 1 256 256 on a 64-wide array: work = 1*256*256 = 65536
# MACs; a full-width job would be 64x. eff_util must be < 0.05 while the
# ACCT work field equals exactly 65536.
printf 'Matmul 1 256 256\n' > "$WORK/v8.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v8.txt" -o "$WORK/v8_stats.txt" > "$WORK/v8.log" 2>&1
read -r wk eu <<< "$(awk '/^ACCT SYSTOLIC_ARRAY/{print $13, $15; exit}' "$WORK/v8_stats.txt")"
low=$(awk -v e="${eu:-1}" 'BEGIN{print (e > 0 && e < 0.05) ? 1 : 0}')
if [ "${wk:-0}" = "65536" ] && [ "$low" -eq 1 ]; then
  ok "V8 work=65536 MACs, eff_util=$eu"
else
  bad "V8 work=${wk:-?} eff_util=${eu:-?}"
fi

# V8b: VPU work must sum every phase's multiplier, not just one pass, or a
# multi-phase job (e.g. Softmax's BROADCAST+REDUCE+BROADCAST) reads as
# under-utilized even at full lane occupancy. Softmax 4096 at default flags
# (no -buf_mb -> 8 MiB buffer): spl=max(ceil(32MiB/8MiB)=4, ceil(4096/1024)=4)
# =4 -> 4 jobs of lin=4096, par=1024 (established by V1). Each job's 3
# phases (softmax_phases, mult 1,1,1) each take exactly
# div_ru(4096*1024,64)=65536 cycles (REDUCE's lin*div_ru(par,sz)=4096*16
# coincides with BROADCAST's div_ru(lin*par,sz) here since 1024/64 divides
# exactly): work = 4*4096*1024*(1+1+1) = 50331648; active =
# busy+underfilled = 4*3*65536 = 786432 (par=1024 >= vu_sz=64, so never
# underfilled); eff_util = 50331648/(64*786432) = 1.0 exactly, memstall
# (present, from the initial unbuffered read) excluded from the ratio by
# design. Verified empirically (build/perf_model, this binary): observed
# ACCT VECTOR_UNIT line is "busy 786432 underfilled 0 memstall 141537 idle 4
# work 50331648 eff_util 1.000000", and the pre-fix code (git-stashed
# for comparison) reported work=16777216 eff_util=0.333333 on this same
# workload -- undercounted by exactly the 3 summed phases, confirming the
# fix.
printf 'Softmax 4096\n' > "$WORK/v8b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v8b.txt" -o "$WORK/v8b_stats.txt" > "$WORK/v8b.log" 2>&1
read -r vwk veu <<< "$(awk '/^ACCT VECTOR_UNIT/{print $13, $15; exit}' "$WORK/v8b_stats.txt")"
if [ "${vwk:-0}" = "50331648" ] && [ "${veu:-0}" = "1.000000" ]; then
  ok "V8b Softmax VPU work=$vwk, eff_util=$veu (multi-phase work summed correctly)"
else
  bad "V8b Softmax VPU work=${vwk:-?} eff_util=${veu:-?} (want 50331648 / 1.000000)"
fi

# V9: Add must read two operands. Same element count as Activation ->
# roughly 2x the read traffic; total DRAM CMDs (reads+writes) must be at
# least 1.4x the Activation run's. Both runs memory-dominated.
printf 'Activation 8000000\n' > "$WORK/v9a.txt"
printf 'Add 8000000\n'        > "$WORK/v9b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v9a.txt" -o "$WORK/v9a_s.txt" > "$WORK/v9a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v9b.txt" -o "$WORK/v9b_s.txt" > "$WORK/v9b.log" 2>&1
ca=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v9a.log" | tail -1 | awk '{print $3}')
cb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v9b.log" | tail -1 | awk '{print $3}')
ratio_ok=$(awk -v a="${ca:-0}" -v b="${cb:-0}" 'BEGIN{print (a>0 && b>=1.4*a) ? 1 : 0}')
if [ "$ratio_ok" -eq 1 ]; then
  ok "V9 Add reads two operands (DRAM CMDs $ca -> $cb)"
else
  bad "V9 DRAM CMDs activation=$ca add=$cb"
fi

# V10: RMSNorm parses, runs, and is cheaper than LayerNorm at equal shape
# (2+1 phase-units/row vs 1+4+1): fewer cycles, same single-chunk job count.
printf 'RMSNorm 512 1024\n'   > "$WORK/v10a.txt"
printf 'LayerNorm 512 1024\n' > "$WORK/v10b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v10a.txt" -o "$WORK/v10a_s.txt" > "$WORK/v10a.log" 2>&1
rc=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v10b.txt" -o "$WORK/v10b_s.txt" > "$WORK/v10b.log" 2>&1
cr=$(cycles_of "$WORK/v10a_s.txt"); cl=$(cycles_of "$WORK/v10b_s.txt")
if [ "$rc" -eq 0 ] && [ -n "$cr" ] && [ -n "$cl" ] && [ "$cr" -lt "$cl" ]; then
  ok "V10 RMSNorm runs and is cheaper than LayerNorm ($cr < $cl)"
else
  bad "V10 rc=$rc rms=$cr ln=$cl"
fi

# V11: tiny prefill Transformer job count + DAG shape.
#   Transformer 1 8 2 2 16 8 0 1  with -sa_sz 4 -vu_sz 4 -c 1, OS mode.
# Derivation (head_dim=8/2=4, M=seq=8, S=8):
#   norm1: 1 (VPU)          qkv: 3 matmuls M8 K8 N8 -> ceil(8/4)=2 jobs each = 6 (SA)
#   rope: 1 (VPU)           scores: n_heads=2 SA jobs (M8 K4 N8)
#   softmax: rows=M*nh=16 len=8 -> 1 job (VPU)
#   av: 2 (SA)              o_proj: M8 K8 N8 -> 2 (SA)
#   res1: 1 (VPU)           norm2: 1 (VPU)
#   gate,up: M8 K8 N16 -> 2 each = 4 (SA)   silu_mul: 1 (VPU)
#   down: M8 K16 N8 -> 2 (SA)               res2: 1 (VPU)
# total = 1+6+1+2+1+2+2+1+1+4+1+2+1 = 25
printf 'Transformer 1 8 2 2 16 8 0 1\n' > "$WORK/v11.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v11.txt" -o "$WORK/v11_s.txt" > "$WORK/v11.log" 2>&1
rc=$?
total=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v11.log" | tail -1 | sed 's|.*/||')
fin=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v11.log" | tail -1 | grep -o '[0-9]*')
# DAG check via jobs.dot: the silu-multiply node is the only "8 x 16" VPU
# node; it must have in-degree 4 (2 gate jobs + 2 up jobs), proving the
# expansion wires real fan-in, not a linear chain.
silu=$(awk -F'[" ]' '/label="8 x 16"/{print $3}' jobs.dot)
indeg=$(grep -c -- "-> ${silu};" jobs.dot)
if [ "$rc" -eq 0 ] && [ "${total:-0}" = "25" ] && [ "$fin" = "$total" ] && [ "${indeg:-0}" -eq 4 ]; then
  ok "V11 Transformer prefill: 25 jobs, all finish, silu fan-in=4"
else
  bad "V11 rc=$rc total=${total:-?} fin=${fin:-?} silu_indeg=${indeg:-?}"
fi

# V12: decode semantics. Transformer 1 8 2 1 16 32 1 4 -sa_sz 4:
#   GQA: K/V projections have N = nkv*head_dim = 1*4 = 4 -> SA job "4 x 8 x 4"
#   scores read the 32-token KV context: SA job "4 x 4 x 32"
#   AV contracts over the context:       SA job "4 x 32 x 4"
#   M = batch = 4 everywhere (no seq_len-sized GEMM rows: no "32 x 8 x" node)
printf 'Transformer 1 8 2 1 16 32 1 4\n' > "$WORK/v12.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v12.txt" -o "$WORK/v12_s.txt" > "$WORK/v12.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v12.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4)?1:0}')
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ] \
   && grep -q 'label="4 x 8 x 4"' jobs.dot \
   && grep -q 'label="4 x 4 x 32"' jobs.dot \
   && grep -q 'label="4 x 32 x 4"' jobs.dot \
   && ! grep -q 'label="32 x 8 x' jobs.dot; then
  ok "V12 decode: GQA K/V, KV-context scores/AV, batch-sized rows"
else
  bad "V12 rc=$rc fin_eq=${fin_eq:-?} (check jobs.dot labels)"
fi

# V13: -fuse_epilogue 1 removes exactly 2 VPU jobs per layer (res1, res2).
# 1-layer tiny prefill config from V11 has 25 jobs -> 23 fused, all finish.
# 2-layer variant (Transformer 2 8 2 2 16 8 0 1) is the multi-layer-wiring
# regression guard: it is the only run where block_in=prev_tail (l>0) and
# prev_tail is itself the fused union from the prior layer's res1/res2, so
# it exercises connectJobLists(prev_tail, norm1) with a multi-element fused
# prev_tail -- duplicate list elements there would double-increment
# rem_deps and deadlock, which the "Jobs finished == total" check below
# catches. Per-layer job count is uniform regardless of l (norm1..res2 are
# all created fresh inside the loop body, same derivation as V11): unfused
# total = 2*25 = 50; fusing removes res1+res2 (2 VPU jobs) per layer:
# 50 - 2*2 = 46.
printf 'Transformer 1 8 2 2 16 8 0 1\n' > "$WORK/v13.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_epilogue 1 -i "$WORK/v13.txt" -o "$WORK/v13_s.txt" > "$WORK/v13.log" 2>&1
rc=$?
total=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v13.log" | tail -1 | sed 's|.*/||')
fin=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v13.log" | tail -1 | grep -o '[0-9]*')
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_epilogue 2 -i "$WORK/v13.txt" -o "$WORK/v13b_s.txt" > "$WORK/v13b.log" 2>&1
rcb=$?
printf 'Transformer 2 8 2 2 16 8 0 1\n' > "$WORK/v13c.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_epilogue 1 -i "$WORK/v13c.txt" -o "$WORK/v13c_s.txt" > "$WORK/v13c.log" 2>&1
rcc=$?
totalc=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v13c.log" | tail -1 | sed 's|.*/||')
finc=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v13c.log" | tail -1 | grep -o '[0-9]*')
if [ "$rc" -eq 0 ] && [ "${total:-0}" = "23" ] && [ "$fin" = "$total" ] && [ "$rcb" -eq 1 ] \
   && [ "$rcc" -eq 0 ] && [ "${totalc:-0}" = "46" ] && [ "$finc" = "$totalc" ]; then
  ok "V13 -fuse_epilogue 1 -> 23 jobs (1L) / 46 jobs (2L, multi-layer wiring); invalid value rejected"
else
  bad "V13 rc=$rc total=${total:-?} fin=${fin:-?} badval_rc=$rcb 2L rc=$rcc total=${totalc:-?} fin=${finc:-?}"
fi

# V14: the pinned config runs a decode workload end-to-end on the v6e model.
# 3 ACCT units = 2 MXUs + 1 shared VPU (Google-documented TensorCore layout).
printf 'Transformer 2 512 8 4 1024 128 1 8\n' > "$WORK/v14.txt"
"$REPO/configs/tpuv6e.sh" "$WORK/v14.txt" "$WORK/v14_s.txt" > "$WORK/v14.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v14.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4)?1:0}')
n_acct=$(grep -c '^ACCT' "$WORK/v14_s.txt")
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ] && [ "${n_acct:-0}" -eq 3 ]; then
  ok "V14 tpuv6e.sh runs decode workload (2 MXU + 1 VPU ACCT units)"
else
  bad "V14 rc=$rc fin_eq=${fin_eq:-?} acct_units=${n_acct:-?}"
fi

# V15: many independent VPU jobs must all finish on a multi-core arch.
# Softmax 512 32 splits into 16 unpinned VecUnitJobs (row_len=512, n_rows=16384
# capped at 1024 rows/job), so with -c >= 2 several vector units run neighbouring
# jobs at the same time. Before the address-window fix each VPU job's write pass
# continued the cursor past its own allocation and landed exactly on the next
# job's read range: job k wrote [base_k+1, base_k+2) == job k+1's inputs. A
# concurrent write and read to one address wedges DRAMSim3's controller -- a full
# write buffer re-arms write_draining_ every tick so the read queue is never
# scheduled, while the head write is held back forever by the R->W dependency
# check against a read that can no longer issue. Both DRAM queues then stay full,
# no transaction is ever accepted, and the run spins forever: observed 4/16 jobs
# at -c 4 and 6/16 at -c 3, with cycles climbing past 40M and "Jobs finished"
# frozen. Fixed runs settle in ~17k cycles (0.13 s), so the timeout below is
# pure hang-detection, not a performance bound -- without it a regression hangs
# the suite instead of failing it.
printf 'Softmax 512 32\n' > "$WORK/v15.txt"
v15_bad=""
for c in 2 3 4 8; do
  timeout 60 "$BIN" -c "$c" -sa_sz 256 -vu_sz 256 -f 1.75 -ws 0 -buf_mb 8 \
    -i "$WORK/v15.txt" -o "$WORK/v15_s_$c.txt" > "$WORK/v15_$c.log" 2>&1
  rc=$?
  fin=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v15_$c.log" | tail -1)
  if [ "$rc" -ne 0 ] || [ "$fin" != "Jobs finished: 16/16" ]; then
    v15_bad="$v15_bad -c$c(rc=$rc ${fin:-no-progress})"
  fi
done
if [ -z "$v15_bad" ]; then
  ok "V15 16 independent VPU jobs all finish at -c 2/3/4/8"
else
  bad "V15 multi-core VPU dispatch stalled:$v15_bad"
fi

# V15b: the allocation guard that keeps V15 fixed must actually be armed --
# every job type has to declare an address window covering its whole walk.
# State::enqueue_reads/enqueue_writes throw if a job steps outside its window,
# so a workload mixing both unit types across cores completes only when every
# declared window is right. Transformer decode exercises SA (GEMM/score/AV) and
# VPU (norms, softmax, residual adds with n_read_operands=2) together.
printf 'Transformer 1 8 2 1 16 32 1 4\n' > "$WORK/v15b.txt"
timeout 120 "$BIN" -c 4 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v15b.txt" \
  -o "$WORK/v15b_s.txt" > "$WORK/v15b.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v15b.log" | tail -1 \
         | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
overran=$(grep -c 'walked past its allocation' "$WORK/v15b.log")
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ] && [ "$overran" -eq 0 ]; then
  ok "V15b mixed SA+VPU decode on -c 4 finishes with no allocation overrun"
else
  bad "V15b rc=$rc fin_eq=${fin_eq:-?} overruns=$overran"
fi

# V16: -mxu_macs_per_pe sets OS accumulation throughput. On a compute-bound
# full-tile GEMM (fast DRAM ini required: at the default HBM2 service rate
# memory ties compute at macs=2 and the ratio collapses to ~1.3), macs=2 must
# cut cycles by ~2x vs the default 1. Not exactly 2x, and eff is NOT exactly
# invariant: active = compute (K/macs per tile, halves) + fixed per-tile
# write/drain cycles (~4150 across this run's 64 tiles, does not halve), so
# measured 37014->20699 cycles (1.79x) and eff 0.888->0.798. The assertions
# pin what a regression would break: (a) ratio in [1.7, 2.05] catches a lost
# formula change (ratio ~1); (b) eff1 > 0.85 proves the schema-1 0.5 ceiling
# is gone; (c) eff2 <= 1.0 catches a lost capacity scale (unscaled sz^2 cap
# would read ~1.6). Invalid value rejected (exit 1 + message); stats file
# declares metric semantics via SCHEMA 2.
printf 'Matmul 512 512 512\n' > "$WORK/v16.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini ../configs/HBM2e_v6e.ini -dram_enq 32 \
  -i "$WORK/v16.txt" -o "$WORK/v16a_s.txt" > "$WORK/v16a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini ../configs/HBM2e_v6e.ini -dram_enq 32 \
  -mxu_macs_per_pe 2 -i "$WORK/v16.txt" -o "$WORK/v16b_s.txt" > "$WORK/v16b.log" 2>&1
c1=$(cycles_of "$WORK/v16a_s.txt"); c2=$(cycles_of "$WORK/v16b_s.txt")
e1=$(awk '/^ACCT SYSTOLIC_ARRAY/{print $15; exit}' "$WORK/v16a_s.txt")
e2=$(awk '/^ACCT SYSTOLIC_ARRAY/{print $15; exit}' "$WORK/v16b_s.txt")
ratio_ok=$(awk -v a="${c1:-0}" -v b="${c2:-1}" 'BEGIN{r=a/b; print (r>=1.7 && r<=2.05)?1:0}')
eff_ok=$(awk -v x="${e1:-0}" -v y="${e2:-0}" 'BEGIN{print (x>0.85 && y>0 && y<=1.0)?1:0}')
head -1 "$WORK/v16a_s.txt" | grep -q '^SCHEMA 2$'; sch=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -mxu_macs_per_pe 0 -i "$WORK/v16.txt" -o "$WORK/v16c_s.txt" > "$WORK/v16c.log" 2>&1; r3=$?
if [ "$ratio_ok" -eq 1 ] && [ "$eff_ok" -eq 1 ] && [ "$sch" -eq 0 ] \
   && [ "$r3" -eq 1 ] && grep -q 'mxu_macs_per_pe' "$WORK/v16c.log"; then
  ok "V16 -mxu_macs_per_pe 2 ~halves compute cycles ($c1 -> $c2), capacity scaled (eff $e1 -> $e2 <= 1), SCHEMA 2"
else
  bad "V16 cycles=$c1/$c2 eff=$e1/$e2 schema_rc=$sch badval_rc=$r3"
fi

# V17: -n_vpu decouples vector-unit count from -c (v6e: 2 MXUs, ONE vector
# unit). -c 2 -n_vpu 1 must build 3 units (3 ACCT lines: 2 SA + 1 VPU) and a
# 16-job vector workload (Softmax 512 32, the old livelock shape) must fully
# serialize onto the single VPU and finish. Default (-n_vpu omitted) keeps
# 1:1 -> 4 ACCT lines at -c 2. -n_vpu 0 rejected (exit 1 + message).
printf 'Softmax 512 32\n' > "$WORK/v17.txt"
timeout 120 "$BIN" -c 2 -n_vpu 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v17.txt" -o "$WORK/v17a_s.txt" > "$WORK/v17a.log" 2>&1
rca=$?
acct1=$(grep -c '^ACCT' "$WORK/v17a_s.txt" 2>/dev/null)
nvpu1=$(grep -c '^ACCT VECTOR_UNIT' "$WORK/v17a_s.txt" 2>/dev/null)
fin1=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v17a.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
timeout 120 "$BIN" -c 2 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v17.txt" -o "$WORK/v17b_s.txt" > "$WORK/v17b.log" 2>&1
acct2=$(grep -c '^ACCT' "$WORK/v17b_s.txt" 2>/dev/null)
"$BIN" -c 2 -n_vpu 0 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v17.txt" -o "$WORK/v17c_s.txt" > "$WORK/v17c.log" 2>&1
rcc=$?
if [ "$rca" -eq 0 ] && [ "${acct1:-0}" -eq 3 ] && [ "${nvpu1:-0}" -eq 1 ] && [ "${fin1:-0}" -eq 1 ] \
   && [ "${acct2:-0}" -eq 4 ] && [ "$rcc" -eq 1 ] && grep -q 'n_vpu' "$WORK/v17c.log"; then
  ok "V17 -n_vpu 1 builds 2 SA + 1 VPU, serializes 16 VPU jobs; default stays 1:1"
else
  bad "V17 rc=$rca acct=$acct1(nvpu=$nvpu1) fin=$fin1 default_acct=$acct2 badval_rc=$rcc"
fi

# V18: VMEM weight residency. Matmul 1024 1024 1024 at sz 64, c 1, default
# 8 MiB buffer: the 2 MiB weight matrix fits VMEM, so the 16 row-block jobs
# (same weight_tag, consecutive on the one core) fetch it ONCE.
# Derivation (verified exact against the run; updated for -vmem_rows 512,
# the C5v2-measured residency window): 16 row-block jobs of 64 rows = two
# 512-row windows -> TWO weight fetch passes (jobs 1 and 9, 34816 beats each:
# combined act+weight first tile + 15 weight tiles) + 14 act-only jobs x 2048
# + 512 write beats (beats_per_wb passes through state_transfer's BYTE
# argument and is divided by 64 again -- pre-existing, mirrored in
# sys_job_alloc_bytes). Total 98816 -> window [90000, 110000]. With
# -vmem_reuse 0 the per-job refetch returns: 557568 -> > 500000.
printf 'Matmul 1024 1024 1024\n' > "$WORK/v18.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v18.txt" -o "$WORK/v18a_s.txt" > "$WORK/v18a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -vmem_reuse 0 -i "$WORK/v18.txt" -o "$WORK/v18b_s.txt" > "$WORK/v18b.log" 2>&1
con=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18a.log" | tail -1 | awk '{print $3}')
coff=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18b.log" | tail -1 | awk '{print $3}')
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -vmem_reuse 2 -i "$WORK/v18.txt" -o "$WORK/v18c_s.txt" > "$WORK/v18c.log" 2>&1; rej=$?
if [ "${con:-0}" -ge 90000 ] && [ "${con:-0}" -le 110000 ] \
   && [ "${coff:-0}" -gt 500000 ] && [ "$rej" -eq 1 ] && grep -q 'vmem_reuse' "$WORK/v18c.log"; then
  ok "V18 VMEM residency: weights fetched once (CMDs $coff -> $con)"
else
  bad "V18 CMDs on=$con off=$coff badval_rc=$rej"
fi

# V18b: -buf_mb gates residency (capacity semantics). Same GEMM with
# -buf_mb 1: the 2 MiB slice no longer fits, so amplified traffic returns
# even with reuse enabled. This is the Phase-C-falsifiable crossover.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v18.txt" -o "$WORK/v18d_s.txt" > "$WORK/v18d.log" 2>&1
csmall=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18d.log" | tail -1 | awk '{print $3}')
if [ "${csmall:-0}" -gt 500000 ]; then
  ok "V18b -buf_mb 1 defeats residency (CMDs $csmall)"
else
  bad "V18b CMDs=$csmall (expected > 500000: slice must not fit a 1 MiB VMEM)"
fi

# V18c: within-job row-pass reuse (attention shape). Prefill scores/AV jobs
# have M > sa_sz (direct SysArrayJobs, multiple row tiles per job) and must
# re-read their K/V panel only on the first row pass when it fits. Tiny
# transformer with seq 64 at sa_sz 4 -> 16 row tiles per score job; reuse
# must strictly reduce traffic vs -vmem_reuse 0.
printf 'Transformer 1 8 2 2 16 64 0 1\n' > "$WORK/v18e.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v18e.txt" -o "$WORK/v18e_s.txt" > "$WORK/v18e.log" 2>&1
r1=$?
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -vmem_reuse 0 -i "$WORK/v18e.txt" -o "$WORK/v18f_s.txt" > "$WORK/v18f.log" 2>&1
r2=$?
ct1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18e.log" | tail -1 | awk '{print $3}')
ct2=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18f.log" | tail -1 | awk '{print $3}')
if [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ] && [ "${ct1:-0}" -gt 0 ] && [ "${ct2:-0}" -gt "${ct1:-0}" ]; then
  ok "V18c within-job row-pass reuse cuts prefill attention traffic ($ct2 -> $ct1)"
else
  bad "V18c rc=$r1/$r2 CMDs reuse=$ct1 noreuse=$ct2"
fi

# V19: model parallelism (ISPASS'25 case study A machinery). -mp N replicates
# the model N times; -mp_par 0 chains replicas sequentially, -mp_par 1 makes
# them independent DAG roots that share the hardware. Two replicas must
# double the job count either way; the parallel schedule must be strictly
# faster than the sequential chaining (overlap exists), and both must finish
# every job. -mp 0 rejected.
#
# V19 extension: the sequential-chaining cross-replica anchor
# (StandardLayer::make_layers, the do_par==false branch) used to set
# jp = lists[0].second -- replica m's FIRST layer's tail -- instead of
# lists.back().second, its LAST layer's tail. At -mp 2 this is invisible:
# jp is (re)computed once per replica m>0 and only ever consumed by the NEXT
# replica, so with just two replicas (m=0,1) the bad value computed at m=1
# is never read. It first bites at -mp 3: replica 2's Matmul jobs get
# anchored on replica 1's Matmul tail instead of replica 1's Softmax tail,
# so replica 2's SA work can start while replica 1's Softmax is still
# running on the VPU -- a real, if partial, overlap (the two live on
# different physical units). On this workload at -c 1 (1 SA + 1 VPU total,
# so the single SA still serializes each replica's own 8 Matmul jobs
# regardless of the bug -- why the overlap is a fraction of a cycle count,
# not a collapse), measured exactly: -mp 1 = 51657 cycles (single replica,
# unaffected by the bug either way). Pre-fix -mp 3 -mp_par 0 = 145194
# cycles, ratio 145194/51657 = 2.8107. Post-fix = 154937 cycles, ratio
# 154937/51657 = 2.9993 (essentially exact 3x chaining, as expected once
# replica 2 truly waits on replica 1's Softmax). A >= 2.8x assertion does
# NOT discriminate here -- the pre-fix ratio already clears 2.8 -- so this
# asserts >= 2.9x instead, which has margin below the post-fix value and
# above the measured pre-fix one. (-mp 2 is untouched by the bug either way:
# measured 103456 post-fix cycles, ratio 2.0029, consistent with the
# "invisible at N=2" analysis above; not asserted since it doesn't
# discriminate.)
printf 'Matmul 512 512 512\nSoftmax 512\n' > "$WORK/v19.txt"
run_mp() { # mp par out
  "$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -mp "$1" -mp_par "$2" -i "$WORK/v19.txt" -o "$WORK/$3_s.txt" > "$WORK/$3.log" 2>&1
}
run_mp 1 0 v19a; run_mp 2 0 v19b; run_mp 2 1 v19c; run_mp 3 0 v19e
j1=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19a.log" | tail -1 | sed 's|.*/||')
j2=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19b.log" | tail -1 | sed 's|.*/||')
f2=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19b.log" | tail -1 | grep -o '^[^/]*' | grep -o '[0-9]*$')
j3=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19c.log" | tail -1 | sed 's|.*/||')
j5=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19e.log" | tail -1 | sed 's|.*/||')
f5=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v19e.log" | tail -1 | grep -o '^[^/]*' | grep -o '[0-9]*$')
cseq=$(cycles_of "$WORK/v19b_s.txt"); cpar=$(cycles_of "$WORK/v19c_s.txt")
c_mp1=$(cycles_of "$WORK/v19a_s.txt"); c_mp3=$(cycles_of "$WORK/v19e_s.txt")
ratio3_ok=$(awk -v a="${c_mp1:-0}" -v b="${c_mp3:-0}" 'BEGIN{print (a>0 && b>=2.9*a) ? 1 : 0}')
ratio3=$(awk -v a="${c_mp1:-1}" -v b="${c_mp3:-0}" 'BEGIN{printf "%.3f", b/a}')
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -mp 0 -i "$WORK/v19.txt" -o "$WORK/v19d_s.txt" > "$WORK/v19d.log" 2>&1; rz=$?
if [ "${j2:-0}" -eq $((j1 * 2)) ] && [ "${j3:-0}" -eq $((j1 * 2)) ] \
   && [ -n "$cseq" ] && [ -n "$cpar" ] && [ "$cpar" -lt "$cseq" ] \
   && [ "${j5:-0}" -eq $((j1 * 3)) ] && [ "$f5" = "$j5" ] && [ "$ratio3_ok" -eq 1 ] \
   && [ "$rz" -eq 1 ] && grep -q 'mp' "$WORK/v19d.log"; then
  ok "V19 -mp 2 doubles jobs ($j1 -> $j2); parallel beats sequential ($cseq -> $cpar); -mp 3 sequential chains fully ($c_mp1 -> $c_mp3, ratio $ratio3)"
else
  bad "V19 jobs=$j1/$j2/$j3/$j5 cyc seq=$cseq par=$cpar mp3=$c_mp3(f=$f5) ratio=$ratio3 badval_rc=$rz"
fi

# V20: decode KV-cache traffic must scale with batch and share within GQA
# groups (spec 6.7). KV caches are PER-SEQUENCE state, not shared weights:
# per layer, hardware reads batch x nkv K/V panel pairs; query heads in one
# GQA group reuse the same pair. Transformer 1 256 8 2 512 1024 1 B at
# -c 1 -sa_sz 64: head_dim=32, panel = 32*1024*2 = 64 KiB, so per unit of
# batch the attention reads grow by nkv*(K+V) = 2*128 KiB = 4096 beats.
# Batch-blind code (pre-fix) shows a near-zero delta between B=1 and B=8.
# The exact floor/ceiling are re-derived at GREEN per the standing rule.
printf 'Transformer 1 256 8 2 512 1024 1 1\n' > "$WORK/v20a.txt"
printf 'Transformer 1 256 8 2 512 1024 1 8\n' > "$WORK/v20b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v20a.txt" -o "$WORK/v20a_s.txt" > "$WORK/v20a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v20b.txt" -o "$WORK/v20b_s.txt" > "$WORK/v20b.log" 2>&1
k1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v20a.log" | tail -1 | awk '{print $3}')
k8=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v20b.log" | tail -1 | awk '{print $3}')
kd=$((${k8:-0} - ${k1:-0}))
if [ "$kd" -ge 25000 ] && [ "$kd" -le 45000 ]; then
  ok "V20 decode KV reads scale with batch (CMDs $k1 -> $k8, delta $kd)"
else
  bad "V20 batch-blind KV traffic: B1=$k1 B8=$k8 delta=$kd (want 25000..45000)"
fi

# V21: LM head via optional 9th Transformer dim (vocab; 0/omitted = off).
# vocab > 0 appends final RMSNorm -> unembedding GEMM (M x d_model x vocab)
# -> vocabulary softmax to the stack tail. Transformer 1 256 8 2 512 128 1 4
# plus vocab 8192 at -c 1 -sa_sz 64 must add exactly 3 jobs (1 norm chunk +
# 1 head GEMM row-block + 1 softmax chunk at the 8 MiB buffer) and the head
# weight read: 256*8192*2 B = 4 MiB = 65536 beats (+ softmax/norm streams).
# Exact floor re-derived at GREEN per the standing rule. A negative vocab
# must be rejected.
printf 'Transformer 1 256 8 2 512 128 1 4\n'      > "$WORK/v21a.txt"
printf 'Transformer 1 256 8 2 512 128 1 4 8192\n' > "$WORK/v21b.txt"
printf 'Transformer 1 256 8 2 512 128 1 4 -1\n'   > "$WORK/v21c.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v21a.txt" -o "$WORK/v21a_s.txt" > "$WORK/v21a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v21b.txt" -o "$WORK/v21b_s.txt" > "$WORK/v21b.log" 2>&1
rb=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v21c.txt" -o "$WORK/v21c_s.txt" > "$WORK/v21c.log" 2>&1
rc_neg=$?
ja=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v21a.log" | tail -1 | sed 's|.*/||')
jb=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v21b.log" | tail -1 | sed 's|.*/||')
fb=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v21b.log" | tail -1 | sed 's|.*: ||;s|/.*||')
ca=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v21a.log" | tail -1 | awk '{print $3}')
cb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v21b.log" | tail -1 | awk '{print $3}')
cd=$((${cb:-0} - ${ca:-0}))
# Serving semantics: prefill computes logits for the LAST token per sequence
# only, so the head GEMM carries M = batch (4), never M = seq (128).
printf 'Transformer 1 256 8 2 512 128 0 4 8192\n' > "$WORK/v21d.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v21d.txt" -o "$WORK/v21d_s.txt" > "$WORK/v21d.log" 2>&1
rd=$?
if [ "$rb" -eq 0 ] && [ "${jb:-0}" -eq $((${ja:-0} + 3)) ] && [ "$fb" = "$jb" ] \
   && [ "$cd" -ge 65000 ] && [ "$cd" -le 80000 ] && [ "$rc_neg" -ne 0 ] \
   && [ "$rd" -eq 0 ] && grep -q 'label="4 x 256 x 8192"' jobs.dot \
   && ! grep -q 'label="128 x 256 x 8192"' jobs.dot; then
  ok "V21 LM head: +3 jobs, weights streamed (CMDs $ca -> $cb), prefill head is last-token"
else
  bad "V21 rc=$rb jobs=$ja->$jb(fin=$fb) cmds_delta=$cd negvocab_rc=$rc_neg prefill_head_rc=$rd"
fi

# V22: address-window sizing bugs that aborted these three inputs with
# "walked past its allocation" at HEAD -- State::enqueue_reads/enqueue_writes'
# livelock-fix guard (src/State.cc check_in_bounds) firing because the job's
# declared window (Job::alloc_size) was smaller than its real walk. Both
# window helpers under-sized relative to their own unit's charging code:
#  - SysArray OS mode (include/units/standard/SysArray.h sys_job_alloc_bytes):
#    the charging code (SysArrayState::init / init_row_loop) floors a row-tile
#    pass's FIRST column tile as ONE division of the combined activation +
#    weight*n_weight_streams bytes; the window floored the two parts
#    SEPARATELY, undercounting by one beat whenever their byte remainders sum
#    to >= bytes_per_tx. Odd-shape decode: Transformer 1 640 8 4 64 1023 1 1
#    at -sa_sz 256 hits exactly that remainder-carry case. Pre-fix: aborted
#    "job 13 (type 0, dims 1 x 1023 x 80) walked past its allocation on
#    writes: alloc_size=167680 offset=167424 beats=5".
#  - VectorUnit (include/units/standard/VectorUnit.h vec_job_alloc_bytes): the
#    window returned raw bytes, but reads and writes are each quantized to a
#    whole bytes_per_tx-byte beat with a 1-beat floor per non-zero transfer
#    (State::state_transfer's SET_READS/SET_WRITES), so any tensor under one
#    beat overflows. Softmax 4: tensor = 4*4*2*1 = 32 bytes, under one beat
#    both ways (read + write), so the raw-byte window (64) was half the true
#    2-beat (128-byte) minimum. Pre-fix: aborted "job 0 (type 1, dims 4 x 4)
#    walked past its allocation on writes: alloc_size=64 offset=64 beats=1".
#    Same shortfall hits the batch-1 tiny Transformer's first RMSNorm job
#    (Transformer 1 8 2 1 16 32 1 1 at -sa_sz 4 -vu_sz 4: lin=8, par=1,
#    tensor=16 bytes). Pre-fix: aborted "job 0 (type 1, dims 1 x 8) walked
#    past its allocation on reads: alloc_size=32 offset=0 beats=1".
# All three must now run to completion (rc=0, every job finished).
printf 'Transformer 1 640 8 4 64 1023 1 1\n' > "$WORK/v22a.txt"
printf 'Softmax 4\n'                        > "$WORK/v22b.txt"
printf 'Transformer 1 8 2 1 16 32 1 1\n'    > "$WORK/v22c.txt"
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -i "$WORK/v22a.txt" -o "$WORK/v22a_s.txt" > "$WORK/v22a.log" 2>&1
ra=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v22b.txt" -o "$WORK/v22b_s.txt" > "$WORK/v22b.log" 2>&1
rb=$?
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v22c.txt" -o "$WORK/v22c_s.txt" > "$WORK/v22c.log" 2>&1
rc3=$?
fa=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v22a.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
fb=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v22b.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
fc=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v22c.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
overran=$(cat "$WORK/v22a.log" "$WORK/v22b.log" "$WORK/v22c.log" 2>/dev/null | grep -c 'walked past its allocation')
if [ "$ra" -eq 0 ] && [ "${fa:-0}" -eq 1 ] \
   && [ "$rb" -eq 0 ] && [ "${fb:-0}" -eq 1 ] \
   && [ "$rc3" -eq 0 ] && [ "${fc:-0}" -eq 1 ] \
   && [ "${overran:-1}" -eq 0 ]; then
  ok "V22 address-window fixes: odd-shape decode / Softmax 4 / batch-1 Transformer all complete"
else
  bad "V22 rc=$ra/$rb/$rc3 fin=${fa:-?}/${fb:-?}/${fc:-?} overran=${overran:-?} (walked-past-allocation regression)"
fi

# V23: a layer narrower than the core count must be rejected cleanly (exit 1
# naming the split), in BOTH dataflows. Pre-fix: the WS path built an N=0 job
# and died on the loop_cols_tiles==0 guard (SIGABRT, initial-commit-era bug);
# the OS path silently dropped the layer's work via max(N/sz, 1) and
# "completed". Matmul 64 64 2 at -c 4: 2 columns across 4 cores.
printf 'Matmul 64 64 2\n' > "$WORK/v23.txt"
"$BIN" -c 4 -ws 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v23.txt" -o "$WORK/v23a_s.txt" > "$WORK/v23a.log" 2>&1
rws=$?
"$BIN" -c 4 -ws 0 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v23.txt" -o "$WORK/v23b_s.txt" > "$WORK/v23b.log" 2>&1
ros=$?
if [ "$rws" -eq 1 ] && grep -q 'cores' "$WORK/v23a.log" \
   && [ "$ros" -eq 1 ] && grep -q 'cores' "$WORK/v23b.log"; then
  ok "V23 N < n_cores rejected cleanly in WS and OS"
else
  bad "V23 ws_rc=$rws os_rc=$ros (want clean exit 1 + message; pre-fix: 134/0)"
fi

# V24: -vmem_rows bounds weight residency to a row window (C5v2 measured
# XLA re-streaming the full weight matrix per ~512-row M-tile even at 34 MB).
# Matmul 2048 4096 4096 at -sa_sz 256: 8 row-block jobs, slice 32 MiB fits.
# Default window 512 rows = 2 jobs -> 4 weight fetches; -vmem_rows 0
# (unlimited, the pre-fix behavior) -> 1 fetch. Weight fetch = 524288 beats,
# activations 262144, writes ~4k: ratio (4 fetches)/(1 fetch) ~ 2.99.
# Negative values rejected.
printf 'Matmul 2048 4096 4096\n' > "$WORK/v24.txt"
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -buf_mb 128 -i "$WORK/v24.txt" -o "$WORK/v24a_s.txt" > "$WORK/v24a.log" 2>&1
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -buf_mb 128 -vmem_rows 0 -i "$WORK/v24.txt" -o "$WORK/v24b_s.txt" > "$WORK/v24b.log" 2>&1
cwin=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v24a.log" | tail -1 | awk '{print $3}')
cinf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v24b.log" | tail -1 | awk '{print $3}')
ratio_ok=$(awk -v a="${cwin:-0}" -v b="${cinf:-1}" 'BEGIN{r=a/b; print (r>=2.4 && r<=3.5)?1:0}')
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -vmem_rows -1 -i "$WORK/v24.txt" -o "$WORK/v24c_s.txt" > "$WORK/v24c.log" 2>&1; rn=$?
if [ "$ratio_ok" -eq 1 ] && [ "$rn" -eq 1 ] && grep -q 'vmem_rows' "$WORK/v24c.log"; then
  ok "V24 -vmem_rows 512 windows residency (CMDs $cinf -> $cwin)"
else
  bad "V24 windowed=$cwin unlimited=$cinf badval_rc=$rn"
fi

# V25a: -fuse_attn keeps the attention score matrix on-chip (flash-attention
# fusion, spec 6.7): QK^T jobs skip their output write-back, the score softmax
# runs prebuffered with no output write, and AV jobs skip their activation-
# panel read (the softmaxed scores). Everything else is charged identically,
# so the fused/unfused DRAM CMD delta is exactly the suppressed transfers.
# Transformer 1 256 4 4 512 128 0 1 (head_dim 64, M=S=128, no GQA sharing:
# nh==nkv so every score/AV job has its own weight_tag and no residency hit
# can differ between the runs) at -c 1 -sa_sz 256:
#   scores x4: 1 column tile each; write charge = beats_per_wb bytes =
#     256*256*2/64 = 2048 B -> 32 beats/job -> 128 beats
#   softmax x1 (128 x 512 rows fits buffer, one job): read 128*512*2 =
#     131072 B -> 2048 beats, write same -> 2048; both suppressed -> 4096
#   av x4: single-tile init read = act(128*128*2=32768) + wgt(64*128*2=16384)
#     = 768 beats; act part suppressed -> 256 beats; delta 512/job -> 2048
#   total delta = 128 + 4096 + 2048 = 6272 CMDs
printf 'Transformer 1 256 4 4 512 128 0 1\n' > "$WORK/v25.txt"
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -i "$WORK/v25.txt" -o "$WORK/v25a_s.txt" > "$WORK/v25a.log" 2>&1
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v25.txt" -o "$WORK/v25b_s.txt" > "$WORK/v25b.log" 2>&1
cunf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25a.log" | tail -1 | awk '{print $3}')
cfus=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25b.log" | tail -1 | awk '{print $3}')
if [ -n "$cunf" ] && [ -n "$cfus" ] && [ $((cunf - cfus)) -eq 6272 ]; then
  ok "V25a -fuse_attn suppresses exactly the score-matrix beats ($cunf -> $cfus)"
else
  bad "V25a unfused=$cunf fused=$cfus delta=$((${cunf:-0} - ${cfus:-0})) (want 6272)"
fi

# V25b: decode invariance. At decode the score matrix is batch x S rows (tiny
# next to KV-cache streams), so fusion must not move decode timing by more
# than noise: cycles within 2%, and fused traffic strictly lower.
printf 'Transformer 2 512 8 4 1024 512 1 4\n' > "$WORK/v25d.txt"
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -i "$WORK/v25d.txt" -o "$WORK/v25c_s.txt" > "$WORK/v25c.log" 2>&1
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v25d.txt" -o "$WORK/v25d_s.txt" > "$WORK/v25d.log" 2>&1
cu=$(cycles_of "$WORK/v25c_s.txt"); cf=$(cycles_of "$WORK/v25d_s.txt")
du=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25c.log" | tail -1 | awk '{print $3}')
df=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25d.log" | tail -1 | awk '{print $3}')
inv_ok=$(awk -v a="${cu:-0}" -v b="${cf:-0}" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; print (a>0 && b>0 && d<=0.02)?1:0}')
if [ "$inv_ok" -eq 1 ] && [ -n "$du" ] && [ -n "$df" ] && [ "$df" -lt "$du" ]; then
  ok "V25b decode cycles move <=2% under -fuse_attn ($cu -> $cf, CMDs $du -> $df)"
else
  bad "V25b cycles $cu -> $cf, CMDs $du -> $df"
fi

# V26: attention stages wire per-head, not all-to-all (the barrier was a
# modeling artifact: head h's softmax needs only head h's scores, and only
# head h's AV consumes it -- all-to-all drained the SA while one VPU chewed
# every softmax chunk serially). Softmax chunk jobs cover contiguous row
# ranges of the M*nh row space; head h owns rows [h*M,(h+1)*M); an edge
# exists iff the ranges overlap.
# Config A (chunk-aligned): Transformer 1 64 4 4 128 512 0 1 -buf_mb 1:
#   n_rows=2048, working set 2 MB > 1 MB -> 2 sm chunks of 1024 rows = 2
#   heads each. scores("512 x 16 x 512")->sm("1024 x 512"): 4 edges (was
#   4x2=8); sm->av("512 x 512 x 16"): 4 (was 8).
# Config B (chunk straddles a head): Transformer 1 24 3 3 48 512 0 1:
#   n_rows=1536 -> 2 chunks of 768 rows = 1.5 heads. scores->sm: chunk0
#   overlaps heads {0,1}, chunk1 {1,2} -> 4 edges (was 6); sm->av: av1
#   depends on both chunks -> 4 (was 6).
edges_between() { # $1 src label, $2 dst label; reads build/jobs.dot
  awk -v src="$1" -v dst="$2" '
    /\[label=/ { name=$1; lab=$0; sub(/^[^"]*"/,"",lab); sub(/".*$/,"",lab); L[name]=lab; next }
    / -> / { a=$1; b=$3; sub(/;$/,"",b); if (L[a]==src && L[b]==dst) n++ }
    END { print n+0 }' jobs.dot
}
printf 'Transformer 1 64 4 4 128 512 0 1\n' > "$WORK/v26a.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v26a.txt" -o "$WORK/v26a_s.txt" > "$WORK/v26a.log" 2>&1
sa=$(edges_between "512 x 16 x 512" "1024 x 512")
aa=$(edges_between "1024 x 512" "512 x 512 x 16")
printf 'Transformer 1 24 3 3 48 512 0 1\n' > "$WORK/v26b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v26b.txt" -o "$WORK/v26b_s.txt" > "$WORK/v26b.log" 2>&1
sb=$(edges_between "512 x 8 x 512" "768 x 512")
ab=$(edges_between "768 x 512" "512 x 512 x 8")
if [ "${sa:-0}" -eq 4 ] && [ "${aa:-0}" -eq 4 ] && [ "${sb:-0}" -eq 4 ] && [ "${ab:-0}" -eq 4 ]; then
  ok "V26 per-head attention wiring (aligned 4/4, straddle 4/4 edges)"
else
  bad "V26 aligned scores->sm=$sa sm->av=$aa straddle scores->sm=$sb sm->av=$ab (want 4/4/4/4)"
fi

# V25c: invalid -fuse_attn rejected cleanly.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -fuse_attn 2 -i "$WORK/v25.txt" -o "$WORK/v25e_s.txt" > "$WORK/v25e.log" 2>&1; rv=$?
if [ "$rv" -eq 1 ] && grep -q 'fuse_attn' "$WORK/v25e.log"; then
  ok "V25c -fuse_attn 2 rejected"
else
  bad "V25c rc=$rv (want 1 + message)"
fi

echo "==== $PASS passed, $FAIL failed (outputs in $WORK)"
exit "$FAIL"
