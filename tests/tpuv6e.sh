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
# ('^ACCT ' with the space: SCHEMA 3 files also carry per-class ACCTC lines, V32.)
inv=$(awk '/^Cycles/{c=$2} /^ACCT /{ if ($5+$7+$9+$11 != c) print "bad:" $0 }' "$WORK/v7_stats.txt" "$WORK/v7b_stats.txt")
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
n_acct=$(grep -c '^ACCT ' "$WORK/v14_s.txt")
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
head -1 "$WORK/v16a_s.txt" | grep -q '^SCHEMA 3$'; sch=$?  # 2 -> 3 when ACCTC lines were added (V32c)
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -mxu_macs_per_pe 0 -i "$WORK/v16.txt" -o "$WORK/v16c_s.txt" > "$WORK/v16c.log" 2>&1; r3=$?
if [ "$ratio_ok" -eq 1 ] && [ "$eff_ok" -eq 1 ] && [ "$sch" -eq 0 ] \
   && [ "$r3" -eq 1 ] && grep -q 'mxu_macs_per_pe' "$WORK/v16c.log"; then
  ok "V16 -mxu_macs_per_pe 2 ~halves compute cycles ($c1 -> $c2), capacity scaled (eff $e1 -> $e2 <= 1), SCHEMA 3"
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
acct1=$(grep -c '^ACCT ' "$WORK/v17a_s.txt" 2>/dev/null)
nvpu1=$(grep -c '^ACCT VECTOR_UNIT' "$WORK/v17a_s.txt" 2>/dev/null)
fin1=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v17a.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3>0)?1:0}')
timeout 120 "$BIN" -c 2 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v17.txt" -o "$WORK/v17b_s.txt" > "$WORK/v17b.log" 2>&1
acct2=$(grep -c '^ACCT ' "$WORK/v17b_s.txt" 2>/dev/null)
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
# Derivation (beat-exact): 16 row-block jobs of 64 rows, 16 column tiles
# each, default -vmem_rows 0 (unlimited: the C5v2 raw slopes show a
# VMEM-resident weight streams ONCE -- the earlier 512-row window was
# mis-derived against host-floor-contaminated C3 throughputs and is
# retracted).
#   weight tile  = min(64,1024)*1024*2 B = 131072 B = 2048 beats
#   act panel    = 64*1024*2 B          = 131072 B = 2048 beats
#   job 1 (fetch pass): init act+weight 4096 + 15 weight tiles x 2048 = 34816
#   jobs 2-16 (resident): act panel only              15 x 2048 = 30720
#   writes: every column tile writes its TRUE 64x64 output block (S4a fix;
#     it used to pass a beat count through state_transfer's byte argument
#     and land at 2 beats/tile): 64*64*2/64 = 128 beats x 16 jobs x 16 tiles
#                                                                 = 32768
#   total 34816 + 30720 + 32768 = 98304   (pre-S4a: 66048 with 512 writes)
# With -vmem_reuse 0 every job is a fetch pass: 16 x 34816 + 32768 = 589824
# (pre-S4a 557568).
printf 'Matmul 1024 1024 1024\n' > "$WORK/v18.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v18.txt" -o "$WORK/v18a_s.txt" > "$WORK/v18a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -vmem_reuse 0 -i "$WORK/v18.txt" -o "$WORK/v18b_s.txt" > "$WORK/v18b.log" 2>&1
con=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18a.log" | tail -1 | awk '{print $3}')
coff=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18b.log" | tail -1 | awk '{print $3}')
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -vmem_reuse 2 -i "$WORK/v18.txt" -o "$WORK/v18c_s.txt" > "$WORK/v18c.log" 2>&1; rej=$?
if [ "${con:-0}" -eq 98304 ] \
   && [ "${coff:-0}" -eq 589824 ] && [ "$rej" -eq 1 ] && grep -q 'vmem_reuse' "$WORK/v18c.log"; then
  ok "V18 VMEM residency: weights fetched once (CMDs $coff -> $con)"
else
  bad "V18 CMDs on=$con (want 98304) off=$coff (want 589824) badval_rc=$rej"
fi

# V18b: -buf_mb gates residency (capacity semantics). Same GEMM with
# -buf_mb 1: the 2 MiB slice no longer fits, so amplified traffic returns
# even with reuse enabled -- the same 589824 beats as -vmem_reuse 0 (V18).
# This is the Phase-C-falsifiable crossover.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v18.txt" -o "$WORK/v18d_s.txt" > "$WORK/v18d.log" 2>&1
csmall=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v18d.log" | tail -1 | awk '{print $3}')
if [ "${csmall:-0}" -eq 589824 ]; then
  ok "V18b -buf_mb 1 defeats residency (CMDs $csmall)"
else
  bad "V18b CMDs=$csmall (want 589824: slice must not fit a 1 MiB VMEM)"
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

# V24: -vmem_rows bounds weight residency to a row window -- kept as an
# ablation knob. The 512-row default it once had is RETRACTED: it was
# derived against C3 throughputs that include the ~113 us host dispatch
# floor; the raw C5v2 slopes (time vs M at fixed weight footprint) equal
# pure MXU compute for a VMEM-resident weight, i.e. it streams once, and a
# 512-row HBM re-stream cannot fit inside silicon's 168 us device time for
# 4096^3. Default is now 0 (unlimited) and must equal an explicit 0.
# Matmul 2048 4096 4096 at -sa_sz 256: 8 row-block jobs, slice 32 MiB fits.
# Explicit -vmem_rows 512 = 2 jobs/window -> 4 weight fetches; 0 -> 1 fetch.
# Derivation (beat-exact, -c 1, no prefetch):
#   weight tile = min(256,4096)*4096*2 B = 2 MiB = 32768 beats; 16 tiles
#   act panel   = 256*4096*2 B           = 2 MiB = 32768 beats
#   fetch-pass job: init act+weight 65536 + 15 x 32768 = 557056
#   resident job:   act panel only                     =  32768
#   writes (S4a true bytes): 256*256*2/64 = 2048 beats x 8 jobs x 16 tiles
#                                                      = 262144
#   -vmem_rows 512: 4 x 557056 + 4 x 32768 + 262144   = 2621440
#   -vmem_rows 0:   1 x 557056 + 7 x 32768 + 262144   = 1048576   (ratio 2.5)
# (pre-S4a the writes were 32 beats/tile = 4096: 2363392 / 790528.)
# Negative values rejected.
printf 'Matmul 2048 4096 4096\n' > "$WORK/v24.txt"
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -buf_mb 128 -vmem_rows 512 -i "$WORK/v24.txt" -o "$WORK/v24a_s.txt" > "$WORK/v24a.log" 2>&1
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -buf_mb 128 -vmem_rows 0 -i "$WORK/v24.txt" -o "$WORK/v24b_s.txt" > "$WORK/v24b.log" 2>&1
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -buf_mb 128 -i "$WORK/v24.txt" -o "$WORK/v24d_s.txt" > "$WORK/v24d.log" 2>&1
cwin=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v24a.log" | tail -1 | awk '{print $3}')
cinf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v24b.log" | tail -1 | awk '{print $3}')
cdef=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v24d.log" | tail -1 | awk '{print $3}')
"$BIN" -c 1 -sa_sz 256 -vu_sz 64 -f 1 -vmem_rows -1 -i "$WORK/v24.txt" -o "$WORK/v24c_s.txt" > "$WORK/v24c.log" 2>&1; rn=$?
if [ "${cwin:-0}" -eq 2621440 ] && [ "${cinf:-0}" -eq 1048576 ] && [ "$rn" -eq 1 ] \
   && grep -q 'vmem_rows' "$WORK/v24c.log" && [ "$cdef" = "$cinf" ]; then
  ok "V24 -vmem_rows ablation (512: $cwin, 0: $cinf) and default == unlimited"
else
  bad "V24 windowed=$cwin (want 2621440) unlimited=$cinf (want 1048576) default=$cdef badval_rc=$rn"
fi

# V25a: -fuse_attn keeps the attention score matrix on-chip (flash-attention
# fusion, spec 6.7): QK^T jobs skip their output write-back, the score softmax
# runs prebuffered with no output write, and AV jobs skip their activation-
# panel read (the softmaxed scores). Everything else is charged identically,
# so the fused/unfused DRAM CMD delta is exactly the suppressed transfers.
# Transformer 1 256 4 4 512 128 0 1 (head_dim 64, M=S=128, no GQA sharing:
# nh==nkv so every score/AV job has its own weight_tag and no residency hit
# can differ between the runs) at -c 1 -sa_sz 256:
#   scores x4 (128 x 64 x 128): 1 column tile each; write charge is the true
#     output block (S4a) min(256,128)*min(256,128)*2 = 32768 B -> 512
#     beats/job -> 2048 beats (pre-S4a: 32 beats/job -> 128)
#   softmax x1 (128 x 512 rows fits buffer, one job): read 128*512*2 =
#     131072 B -> 2048 beats, write same -> 2048; both suppressed -> 4096
#   av x4: single-tile init read = act(128*128*2=32768) + wgt(64*128*2=16384)
#     = 768 beats; act part suppressed -> 256 beats; delta 512/job -> 2048
#   total delta = 2048 + 4096 + 2048 = 8192 CMDs (pre-S4a 6272)
printf 'Transformer 1 256 4 4 512 128 0 1\n' > "$WORK/v25.txt"
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -i "$WORK/v25.txt" -o "$WORK/v25a_s.txt" > "$WORK/v25a.log" 2>&1
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v25.txt" -o "$WORK/v25b_s.txt" > "$WORK/v25b.log" 2>&1
cunf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25a.log" | tail -1 | awk '{print $3}')
cfus=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v25b.log" | tail -1 | awk '{print $3}')
if [ -n "$cunf" ] && [ -n "$cfus" ] && [ $((cunf - cfus)) -eq 8192 ]; then
  ok "V25a -fuse_attn suppresses exactly the score-matrix beats ($cunf -> $cfus)"
else
  bad "V25a unfused=$cunf fused=$cfus delta=$((${cunf:-0} - ${cfus:-0})) (want 8192)"
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

# V27: -dbuf N cross-op weight prefetch (XLA streams the NEXT operator's
# weights under the current op's compute/barrier tail -- B2/C5v2). Only the
# FIRST job of each weight_tag is prefetchable: a fresh tag can never be
# VMEM-resident at dispatch, so every prefetched beat replaces a demand beat
# 1:1 and total traffic is EXACTLY invariant (the load-bearing assertion).
# Prefetch fills only otherwise-idle DRAM slots (issues when the demand
# queue is near-empty), so it can only shift time, not add contention.
# V27a: prefill Transformer with VPU-serial segments -> cycles drop, CMDs equal.
printf 'Transformer 1 512 8 8 2048 512 0 1\n' > "$WORK/v27.txt"
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v27.txt" -o "$WORK/v27a_s.txt" > "$WORK/v27a.log" 2>&1
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -dbuf 8 -i "$WORK/v27.txt" -o "$WORK/v27b_s.txt" > "$WORK/v27b.log" 2>&1
c0=$(cycles_of "$WORK/v27a_s.txt"); c1=$(cycles_of "$WORK/v27b_s.txt")
d0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27a.log" | tail -1 | awk '{print $3}')
d1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27b.log" | tail -1 | awk '{print $3}')
if [ -n "$c0" ] && [ -n "$c1" ] && [ "$c1" -lt "$c0" ] && [ "$d0" = "$d1" ]; then
  ok "V27a -dbuf hides weight streams (cycles $c0 -> $c1, CMDs invariant $d0)"
else
  bad "V27a cycles $c0 -> $c1, CMDs $d0 vs $d1 (want fewer cycles, equal CMDs)"
fi
# V27a2: the offered-load stat must count DEMAND only (prefetch beats used to
# hide the very starvation the stat measures). Both numbers are printed; the
# all-traffic idle can only be <= the demand-only idle.
di=$(grep -ao 'MEM demand-idle: [0-9]*' "$WORK/v27b.log" | grep -o '[0-9]*$')
ti=$(grep -ao 'idle incl. prefetch: [0-9]*' "$WORK/v27b.log" | grep -o '[0-9]*$')
if [ -n "$di" ] && [ -n "$ti" ] && [ "$ti" -le "$di" ]; then
  ok "V27a2 demand-only idle stat ($di demand-idle, $ti incl. prefetch)"
else
  bad "V27a2 demand-idle=${di:-?} idle-incl-prefetch=${ti:-?}"
fi

# V27b: decode gate -- -dbuf must not degrade decode (<= +2% cycles) and
# traffic must be exactly invariant. Exactness here leans on GQA group
# pinning: with siblings pinned to one core, residency sequences are
# deterministic, so prefetch timing shifts cannot flip fetch decisions
# (unpinned, sibling placement was a scheduling lottery worth several
# percent of decode traffic -- this assertion is what caught it).
printf 'Transformer 2 512 8 4 1024 512 1 8\n' > "$WORK/v27d.txt"
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v27d.txt" -o "$WORK/v27c_s.txt" > "$WORK/v27c.log" 2>&1
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -dbuf 8 -i "$WORK/v27d.txt" -o "$WORK/v27d_s.txt" > "$WORK/v27d.log" 2>&1
cu=$(cycles_of "$WORK/v27c_s.txt"); cf=$(cycles_of "$WORK/v27d_s.txt")
du=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27c.log" | tail -1 | awk '{print $3}')
df=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27d.log" | tail -1 | awk '{print $3}')
gate=$(awk -v a="${cu:-0}" -v b="${cf:-0}" 'BEGIN{print (a>0 && b>0 && b<=a*1.02)?1:0}')
if [ "$gate" -eq 1 ] && [ "$du" = "$df" ]; then
  ok "V27b decode gate holds under -dbuf ($cu -> $cf, CMDs invariant $du)"
else
  bad "V27b cycles $cu -> $cf, CMDs $du vs $df"
fi

# V27c: invalid -dbuf rejected cleanly.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dbuf -1 -i "$WORK/v27.txt" -o "$WORK/v27e_s.txt" > "$WORK/v27e.log" 2>&1; rv=$?
if [ "$rv" -eq 1 ] && grep -q 'dbuf' "$WORK/v27e.log"; then
  ok "V27c -dbuf -1 rejected"
else
  bad "V27c rc=$rv (want 1 + message)"
fi

# V27d: -dbuf traffic invariance at shapes whose weight panels are NOT whole
# beats (review finding): credit must be issued and consumed in whole beats
# per tile and deducted from the full-formula charge -- otherwise a sub-beat
# remainder costs an extra beat per tile and the credit leaks. (a) head_dim 4
# at sz 4: panel 32 B < one 64 B beat. (b) tiny fused head (head_dim 1, batch
# 9, S 32): the old floor-of-sum issue count also overran the job's address
# window here (+23% traffic).
printf 'Transformer 1 8 2 2 16 8 0 1\n' > "$WORK/v27f.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -dbuf 0 -i "$WORK/v27f.txt" -o "$WORK/v27f0_s.txt" > "$WORK/v27f0.log" 2>&1
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -dbuf 8 -i "$WORK/v27f.txt" -o "$WORK/v27f8_s.txt" > "$WORK/v27f8.log" 2>&1
fa=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27f0.log" | tail -1 | awk '{print $3}')
fb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27f8.log" | tail -1 | awk '{print $3}')
printf 'Transformer 1 3 3 3 6 32 1 9\n' > "$WORK/v27g.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_attn 1 -dbuf 0 -i "$WORK/v27g.txt" -o "$WORK/v27g0_s.txt" > "$WORK/v27g0.log" 2>&1
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_attn 1 -dbuf 8 -i "$WORK/v27g.txt" -o "$WORK/v27g8_s.txt" > "$WORK/v27g8.log" 2>&1
ga=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27g0.log" | tail -1 | awk '{print $3}')
gb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27g8.log" | tail -1 | awk '{print $3}')
if [ -n "$fa" ] && [ "$fa" = "$fb" ] && [ -n "$ga" ] && [ "$ga" = "$gb" ]; then
  ok "V27d -dbuf invariant at sub-beat panels ($fa) and tiny fused heads ($ga)"
else
  bad "V27d sub-beat $fa vs $fb, tiny-fused $ga vs $gb (want equal pairs)"
fi

# V27e: WS mode never consumes prefetch credit (and MatmulAct/ActMatmul
# build OS-flagged jobs that the WS state machine executes), so -dbuf must
# be forced off with a note: traffic invariant and the note names the flag.
# (MatmulAct itself crashes under -ws 1 today -- OS-sized window, WS walk --
# a pre-existing bug outside this test's scope; plain Matmul is the vehicle.)
printf 'Matmul 64 64 128\n' > "$WORK/v27h.txt"
"$BIN" -c 1 -ws 1 -sa_sz 64 -vu_sz 64 -f 1 -dbuf 0 -i "$WORK/v27h.txt" -o "$WORK/v27h0_s.txt" > "$WORK/v27h0.log" 2>&1
"$BIN" -c 1 -ws 1 -sa_sz 64 -vu_sz 64 -f 1 -dbuf 8 -i "$WORK/v27h.txt" -o "$WORK/v27h8_s.txt" > "$WORK/v27h8.log" 2>&1
ha=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27h0.log" | tail -1 | awk '{print $3}')
hb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v27h8.log" | tail -1 | awk '{print $3}')
if [ -n "$ha" ] && [ "$ha" = "$hb" ] && grep -q 'dbuf' "$WORK/v27h8.log"; then
  ok "V27e -dbuf forced off in WS mode (CMDs invariant $ha, note printed)"
else
  bad "V27e CMDs $ha vs $hb, note=$(grep -c dbuf "$WORK/v27h8.log")"
fi

# V28: GQA group pinning must not idle an MXU when nkv < n_cores (MQA, nkv=1):
# group-indexed pinning sent every score/AV job to core 0. Pin by head in
# that regime (deterministic; the documented 2x KV refetch applies). The
# V12 decode shape is symmetric across 2 cores, so both SA units must
# report identical ACCT work.
printf 'Transformer 1 8 2 1 16 32 1 4\n' > "$WORK/v28.txt"
"$BIN" -c 2 -n_vpu 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v28.txt" -o "$WORK/v28_s.txt" > "$WORK/v28.log" 2>&1
w0=$(awk '$1=="ACCT" && $2=="SYSTOLIC_ARRAY" && $3==0 {for(i=1;i<=NF;i++) if($i=="work") print $(i+1)}' "$WORK/v28_s.txt")
w1=$(awk '$1=="ACCT" && $2=="SYSTOLIC_ARRAY" && $3==1 {for(i=1;i<=NF;i++) if($i=="work") print $(i+1)}' "$WORK/v28_s.txt")
if [ -n "$w0" ] && [ "$w0" = "$w1" ]; then
  ok "V28 MQA attention balanced across MXUs (work $w0 == $w1)"
else
  bad "V28 SA0 work=${w0:-?} SA1 work=${w1:-?} (want equal)"
fi

# V29: within-op double buffering. Silicon streams tile i+1's operands under
# tile i's compute and stages the next row-block's activation panel ahead;
# the OS state machine used to declare a tile's reads only when its read
# stage began, leaving the shift/write stages and every job-start activation
# fetch exposed (4096^3: sim 213 us vs silicon 168 us device, window off).
# V29a: -dbuf_tile 1 issues the next tile's reads at read-completion and lets
# shift/write run while they stream. Matmul 4096^3 on the pinned config
# (cross-op prefetch off to isolate): cycles drop, CMDs invariant, and never
# below the MXU compute floor 2*4096^3 / (2 MXU * 256^2 PE * 2 MAC) = 262144.
# Output side: write-backs are charged at true bytes (V31a, 2048 beats per
# 256x256 tile instead of 32) and the write stage waits for them to land --
# there is no output-side double buffering. Known limit, NOT fixed here:
# with -act_share 0 (both cores fetch their own activation panel at every
# job start) tile i+1's pre-issued reads throttle tile i's write-back in
# DRAMSim3's read-preferring 32-entry write buffer and -dbuf_tile 1 comes
# out SLOWER than 0 on this shape (399846 vs 362109); under the pinned
# -act_share 1 the ordering below holds (355523 -> 353650). An unflagged
# output gate (State::writes_gate) once claimed to fix that inversion; it
# did not (386644 vs 362109 with -act_share 0) and was removed.
# CMDs (beat-exact, -act_share 1 default): per core, 16 row-block jobs of
# 8 column tiles; weight tile = min(256,2048)*4096*2 B = 32768 beats, act
# panel = 256*4096*2 B = 32768 beats, write-back = 256*256*2/64 = 2048.
#   core 0: job 1 (fetch pass) 32768 + 8 x 32768 = 294912; jobs 2-16 act
#           panel only 15 x 32768 = 491520; writes 16 x 8 x 2048 = 262144
#           -> 1048576
#   core 1: act_resident (the panel is staged once, S4b): 8 x 32768 weight
#           tiles on job 1 + 262144 writes -> 524288
#   total 1572864 (2097152 with -act_share 0; pre-S4 tree, per-MXU panels
#   and 32-beat writes: 1572864 + 2 x 16 x 8 x 32 = 1581056, measured).
printf 'Matmul 4096 4096 4096\n' > "$WORK/v29.txt"
"$REPO/configs/tpuv6e.sh" "$WORK/v29.txt" "$WORK/v29a_s.txt" -dbuf 0 -dbuf_tile 0 > "$WORK/v29a.log" 2>&1
"$REPO/configs/tpuv6e.sh" "$WORK/v29.txt" "$WORK/v29b_s.txt" -dbuf 0 -dbuf_tile 1 > "$WORK/v29b.log" 2>&1
t0=$(cycles_of "$WORK/v29a_s.txt"); t1=$(cycles_of "$WORK/v29b_s.txt")
m0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v29a.log" | tail -1 | awk '{print $3}')
m1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v29b.log" | tail -1 | awk '{print $3}')
if [ -n "$t0" ] && [ -n "$t1" ] && [ "$t1" -lt "$t0" ] && [ "$t1" -ge 262144 ] \
   && [ "$m0" = "$m1" ] && [ "${m0:-0}" -eq 1572864 ]; then
  ok "V29a -dbuf_tile hides tile transitions (cycles $t0 -> $t1, CMDs invariant $m0)"
else
  bad "V29a cycles $t0 -> $t1 (floor 262144), CMDs $m0 vs $m1 (want 1572864)"
fi

# V29b: the cross-op prefetcher also stages a READY job's first activation
# panel (its producers are done, so the data exists). On a single GEMM the
# weights are resident after job 1, so before this -dbuf had nothing to
# stream: cycles must now drop with -dbuf 48 vs 0 (tile buffering off to
# isolate), CMDs invariant.
"$REPO/configs/tpuv6e.sh" "$WORK/v29.txt" "$WORK/v29c_s.txt" -dbuf 48 -dbuf_tile 0 > "$WORK/v29c.log" 2>&1
t2=$(cycles_of "$WORK/v29c_s.txt")
m2=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v29c.log" | tail -1 | awk '{print $3}')
if [ -n "$t0" ] && [ -n "$t2" ] && [ "$t2" -lt "$t0" ] && [ "$m0" = "$m2" ]; then
  ok "V29b -dbuf stages ready jobs' activation panels (cycles $t0 -> $t2, CMDs invariant)"
else
  bad "V29b cycles $t0 -> $t2, CMDs $m0 vs $m2"
fi

# V29c: invalid -dbuf_tile rejected cleanly.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dbuf_tile 2 -i "$WORK/v29.txt" -o "$WORK/v29e_s.txt" > "$WORK/v29e.log" 2>&1; rv=$?
if [ "$rv" -eq 1 ] && grep -q 'dbuf_tile' "$WORK/v29e.log"; then
  ok "V29c -dbuf_tile 2 rejected"
else
  bad "V29c rc=$rv (want 1 + message)"
fi

# V30: -fuse_vpu. The silicon kernel census puts RMSNorm/RoPE/SiLU/residual at
# 0.1% of device time: XLA fuses them into GEMM prologues/epilogues, so they
# cost no HBM round trip and overlap with the MXU. The model keeps them as
# VPU jobs (attribution intact) but makes them traffic-free SIDECARS off the
# dependency chain: consumers depend on the op's inputs' producers directly.
# V30a: traffic delta is exactly the suppressed round trips. Transformer
# 1 256 4 4 512 128 0 1 at -c 1 (M=128, d_model 256, d_ff 512, fuse_attn off):
#   norm1, norm2: 128x256x2 B = 1024 beats read + 1024 write -> 2048 each
#   rope: 128x(256+256)x2 B -> 2048 + 2048 = 4096
#   res1, res2 (2 operands): 2048 read + 1024 write -> 3072 each
#   silu_mul (2 operands over 128x512): 4096 read + 2048 write -> 6144
#   total 20480; job count unchanged (sidecars still run).
printf 'Transformer 1 256 4 4 512 128 0 1\n' > "$WORK/v30.txt"
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -i "$WORK/v30.txt" -o "$WORK/v30a_s.txt" > "$WORK/v30a.log" 2>&1
n0=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v30a.log" | tail -1 | sed 's|.*/||')
e0=$(edges_between "128 x 256" "128 x 256 x 256")
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_vpu 1 -i "$WORK/v30.txt" -o "$WORK/v30b_s.txt" > "$WORK/v30b.log" 2>&1
n1=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v30b.log" | tail -1 | sed 's|.*/||')
f1=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v30b.log" | tail -1 | grep -o '[0-9]*')
e1=$(edges_between "128 x 256" "128 x 256 x 256")
cu=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v30a.log" | tail -1 | awk '{print $3}')
cf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v30b.log" | tail -1 | awk '{print $3}')
if [ -n "$cu" ] && [ -n "$cf" ] && [ $((cu - cf)) -eq 20480 ] && [ "$n0" = "$n1" ] && [ "$f1" = "$n1" ]; then
  ok "V30a -fuse_vpu suppresses exactly the VPU round trips ($cu -> $cf, $n1 jobs, all finish)"
else
  bad "V30a CMDs $cu -> $cf delta=$((${cu:-0} - ${cf:-0})) (want 20480), jobs $n0 vs $n1 finished $f1"
fi
# V30b: sidecars leave the chain -- norm1/res nodes ("128 x 256") feed q/k/v
# GEMMs ("128 x 256 x 256") 3 times unfused, never fused.
if [ "${e0:-0}" -eq 3 ] && [ "${e1:-0}" -eq 0 ]; then
  ok "V30b fused VPU ops are off the dependency chain (edges $e0 -> $e1)"
else
  bad "V30b VPU->GEMM edges unfused=$e0 fused=$e1 (want 3 -> 0)"
fi
# V30c: decode. Transformer 2 512 8 4 1024 512 1 4 (M = batch = 4, nkv*hd =
# 256, d_ff 1024, 2 layers, no head): suppressed round trips per layer are
# norm1/norm2 64+64 beats each (256), rope 96+96 (192), res1/res2 128+64
# each (384), silu_mul 256+128 (384) = 1216 -> 2432 over 2 layers, exact.
# Cycles must not rise (each fused op also drops an HBM read->write latency
# link from the chain -- large on this tiny config, <1% at 36 layers).
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -i "$WORK/v25d.txt" -o "$WORK/v30c_s.txt" > "$WORK/v30c.log" 2>&1
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -fuse_vpu 1 -i "$WORK/v25d.txt" -o "$WORK/v30d_s.txt" > "$WORK/v30d.log" 2>&1
du=$(cycles_of "$WORK/v30c_s.txt"); df=$(cycles_of "$WORK/v30d_s.txt")
mu=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v30c.log" | tail -1 | awk '{print $3}')
mf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v30d.log" | tail -1 | awk '{print $3}')
if [ -n "$du" ] && [ -n "$df" ] && [ "$df" -le "$du" ] && [ $((mu - mf)) -eq 2432 ]; then
  ok "V30c decode: exact round-trip delta 2432, cycles $du -> $df"
else
  bad "V30c cycles $du -> $df, CMDs $mu -> $mf delta=$((${mu:-0} - ${mf:-0})) (want 2432)"
fi
# V30d: invalid value rejected.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -fuse_vpu 2 -i "$WORK/v30.txt" -o "$WORK/v30e_s.txt" > "$WORK/v30e.log" 2>&1; rv=$?
if [ "$rv" -eq 1 ] && grep -q 'fuse_vpu' "$WORK/v30e.log"; then
  ok "V30d -fuse_vpu 2 rejected"
else
  bad "V30d rc=$rv (want 1 + message)"
fi

# V31a: OS write-back is charged at the TRUE output bytes (spec S4a). The OS
# 'shift' stage used to pass beats_per_wb (= sz*sz*dtw/64, a BEAT count) as
# state_transfer's BYTE argument, so every column tile wrote 1/64 of its
# output block (2 beats instead of 128 at sz 64), and sys_job_alloc_bytes
# mirrored the double division. Matmul 256 64 256 at -c 1 -sa_sz 64, default
# 8 MiB VMEM (weight slice 64x256x2 = 32 KiB fits, so jobs 2-4 run on the
# resident copy), prefetch/tile double buffering off:
#   4 row-block jobs (256/64) x 4 column tiles (256/64), 1 row tile each
#   reads: job 1 init = act(64*64*2 = 8192 B) + wgt(min(64,256)*64*2 =
#          8192 B) = 16384 B -> 256 beats, then 3 weight-only tiles x 128
#          -> 640; jobs 2-4: resident weights, act panel only -> 128 each
#          -> 384; reads total 1024
#   writes: 4 jobs x 4 tiles x (64*64*2/64 = 128 beats) = 2048
#   total DRAM CMDs = 1024 + 2048 = 3072   (pre-S4a: 1024 + 32 = 1056)
printf 'Matmul 256 64 256\n' > "$WORK/v31a.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v31a.txt" -o "$WORK/v31a_s.txt" > "$WORK/v31a.log" 2>&1
ra=$?
wa=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v31a.log" | tail -1 | awk '{print $3}')
fa=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v31a.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3==4)?1:0}')
if [ "$ra" -eq 0 ] && [ "${fa:-0}" -eq 1 ] && [ "${wa:-0}" -eq 3072 ]; then
  ok "V31a OS write-back charged at true bytes (CMDs $wa = 1024 reads + 2048 writes)"
else
  bad "V31a rc=$ra fin=${fa:-?} CMDs=$wa (want 3072)"
fi

# V31b: activation panels are staged ONCE into the shared VMEM, not once per
# MXU (spec S4b, flag -act_share, default 1; 0 restores per-MXU reads for
# ablation). createSAJobs splits N across cores and every core's row-block
# job used to charge its own min(sz,M) x K activation panel at init and at
# each row advance; hardware stages the tile once for both MXUs. Matmul
# 256 256 512 at -c 2 -sa_sz 256: one job per core (M = 256 = one row
# block, core_n = 256 = one column tile), weight slice 256x256x2 = 128 KiB
# fits the 4 MiB per-core share, no VPU jobs.
#   per job: weight tile 256*256*2 = 131072 B = 2048 beats, activation
#     panel 256*256*2 = 131072 B = 2048 beats (charged together at init:
#     4096 beats), write-back 256*256*2/64 = 2048 beats
#   -act_share 0: 2 x (4096 + 2048)                     = 12288
#   -act_share 1: core 0 4096 + 2048, core 1 (act_resident) 2048 + 2048
#                                                        = 10240
#   delta = the one suppressed panel = 2048 beats, exact. -act_share 2 rejected.
printf 'Matmul 256 256 512\n' > "$WORK/v31b.txt"
"$BIN" -c 2 -sa_sz 256 -vu_sz 64 -f 1 -act_share 0 -i "$WORK/v31b.txt" -o "$WORK/v31b0_s.txt" > "$WORK/v31b0.log" 2>&1
r0=$?
"$BIN" -c 2 -sa_sz 256 -vu_sz 64 -f 1 -act_share 1 -i "$WORK/v31b.txt" -o "$WORK/v31b1_s.txt" > "$WORK/v31b1.log" 2>&1
r1=$?
"$BIN" -c 2 -sa_sz 256 -vu_sz 64 -f 1 -i "$WORK/v31b.txt" -o "$WORK/v31b2_s.txt" > "$WORK/v31b2.log" 2>&1
b0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v31b0.log" | tail -1 | awk '{print $3}')
b1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v31b1.log" | tail -1 | awk '{print $3}')
b2=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v31b2.log" | tail -1 | awk '{print $3}')
f1=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v31b1.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4 && $3==2)?1:0}')
"$BIN" -c 2 -sa_sz 256 -vu_sz 64 -f 1 -act_share 2 -i "$WORK/v31b.txt" -o "$WORK/v31b3_s.txt" > "$WORK/v31b3.log" 2>&1; rv=$?
if [ "$r0" -eq 0 ] && [ "$r1" -eq 0 ] && [ "${f1:-0}" -eq 1 ] \
   && [ "${b0:-0}" -eq 12288 ] && [ "${b1:-0}" -eq 10240 ] && [ "$b2" = "$b1" ] \
   && [ "$rv" -eq 1 ] && grep -q 'act_share' "$WORK/v31b3.log"; then
  ok "V31b -act_share stages the activation panel once (CMDs $b0 -> $b1, default on, 2 rejected)"
else
  bad "V31b rc=$r0/$r1 fin=${f1:-?} CMDs share0=$b0 (want 12288) share1=$b1 (want 10240) default=$b2 badval_rc=$rv"
fi

# V31c: multi-row-tile OS jobs under -dbuf_tile 1 (S4a review finding).
# Every createSAJobs row-block job has ONE row tile; only prefill attention
# jobs have several (scores M x hd x S and AV M x S x hd with M > sz), and
# under -fuse_attn 1 the scores jobs write nothing -- so the write-back that
# CROSSES the row rewind was never exercised at true bytes. Under -dbuf_tile
# the read stage of a row's last column tile rewinds the cursor to addr_hold
# and pre-issues the next row's reads BEFORE that tile's write-back is
# issued, so the write-back lands in the next row's epoch. With 32-beat
# writes the resident-weight slack hid it; at true bytes the LAST epoch
# exceeds the window by up to one output tile W: abort ("walked past its
# allocation" -- check_in_bounds catches every overrun, so an overrun can
# never alias a neighbour silently). sys_job_alloc_bytes now reserves
# that tile for multi-row OS jobs when -dbuf_tile is on -- the walk is
# unchanged. Epochs (rows R > 1, cols C, weights resident from row 2 or
# not): epoch 1 = combined + (C-1) wgt + (C-1) W; epochs 2..R-1 = act
# (+wgt) + W(prev) + (C-1) wgt + (C-1) W <= first pass; epoch R = act
# (+wgt) + W(prev) + (C-1) wgt + C W <= first pass + W. -dbuf_tile 0 and
# fused_out (0-beat writes) reserve nothing extra, so the baseline layout
# is bit-identical.
# Run 1: Transformer 1 256 4 4 512 1024 0 1 at -c 1 -sa_sz 256 -vu_sz 512
# -fuse_attn 0 -fuse_vpu 1 (VPU sidecars traffic-free; softmax remains),
# 128 MiB VMEM (every weight slice is resident after its first row/job),
# prefetch off. All 46 jobs finish and CMDs are identical for -dbuf_tile
# 0/1 and equal the hand total (beats; dtw 2, 64 B/beat):
#   q,k,v,o 1024x256x256: 4 row-block jobs of 256x256x256, 1 col tile;
#     weight 256*256*2/64 = 2048, act 2048: job 1 4096, jobs 2-4 2048 each
#     -> 10240 reads + 4 x 2048 writes = 18432 each, x4 = 73728
#   gate,up 1024x256x512: 4 jobs of 256x256x512, 2 tiles: job 1 4096 +
#     2048, jobs 2-4 2048 -> 12288 reads + 8 x 2048 writes = 28672 each
#     -> 57344
#   down 1024x512x256: 4 jobs of 256x512x256, 1 tile: weight 4096, act
#     4096: job 1 8192, jobs 2-4 4096 -> 20480 reads + 4 x 2048 = 28672
#   scores 1024x64x1024 per head (4 rows x 4 cols, one K tag per head,
#     64x1024x2 = 128 KiB fits): weight tile 256*64*2/64 = 512, act 512.
#     Row 1: 1024 + 3 x 512 = 2560; rows 2-4 act only 512 -> 4096 reads;
#     writes 16 x 2048 = 32768 -> 36864 per head, x4 = 147456
#   AV 1024x1024x64 per head (4 rows x 1 col): weight 64*1024*2/64 = 2048,
#     act 256*1024*2/64 = 8192. Row 1: 10240; rows 2-4 8192 -> 34816 reads;
#     writes 4 x (256*64*2/64 = 512) = 2048 -> 36864 per head, x4 = 147456
#   softmax 4096 rows x 1024 (4 jobs of 1024 rows, 2 MiB each): 4 x 32768
#     read + 4 x 32768 write = 262144
#   total 73728 + 57344 + 28672 + 147456 + 147456 + 262144 = 716800
# Run 2: the pinned config with only -fuse_attn 0 flipped, S = 512 (the
# run that HUNG at 14/44 jobs -- that hang was V31d's write-over-pending-
# prefetch-read deadlock, not this overrun, which the wider window only
# sidestepped by timing; kept as the pinned-config completion + CMD pin).
# -c 2, -act_share 1, -dbuf 48 (prefetch is traffic-invariant), -dbuf_tile
# 0 and 1 must both finish 44/44 at:
#   q,k,v,o 512x256x256, core_n 128: 2 jobs/core of 256x256x128; weight
#     128*256*2/64 = 1024, act 2048. Core 0: 3072 + 2048; core 1
#     (act_resident): 1024 + 0; writes 4 x 1024 -> 10240 each, x4 = 40960
#   gate,up 512x256x512, core_n 256: 2 jobs/core of 256x256x256; core 0
#     4096 + 2048, core 1 2048; writes 4 x 2048 -> 16384 each, 32768
#   down 512x512x256, core_n 128: 2 jobs/core of 256x512x128; weight
#     128*512*2/64 = 2048, act 4096. Core 0: 6144 + 4096, core 1 2048;
#     writes 4 x 1024 -> 16384
#   scores 512x64x512 per head (2 x 2 tiles, heads alternate cores): row 1
#     1024 + 512, row 2 512 -> 2048 reads; writes 4 x 2048 -> 10240, 40960
#   AV 512x512x64 per head (2 rows x 1 col): weight 1024, act 4096: 5120 +
#     4096 = 9216 reads; writes 2 x 512 -> 10240, 40960
#   softmax 2048 rows x 512 (2 jobs of 1024): 32768 read + 32768 write
#     = 65536
#   total 40960 + 32768 + 16384 + 40960 + 40960 + 65536 = 237568
# Run 3: the same S = 512 layer at -c 1 (Run 1's flags), -dbuf_tile 1: the
# 512x64x512 scores job aborted here too (window 1024 + 512 + 2 x 2048 =
# 5632 beats; last epoch 512 + 2048 + 2 x 2048 = 6656). 30 jobs (two
# row-block jobs per GEMM), and the same 237568: each 2-core GEMM above
# splits the weight panel in half per core (same total) and -act_share 1
# charges the activation panel once per row block, exactly as one core
# does -- q,k,v,o 6144 + 4096 = 10240 each; gate,up (4096 + 2048) + 2048 +
# 8192 = 16384 each; down 8192 + 4096 + 4096 = 16384; scores, AV and
# softmax are core-count independent. (Run 1's S = 1024 scores jobs fit
# their window by coincidence: the resident-weight savings after row 1,
# 3 x 512 + 512 = 2048, equal W; they pin the 4 x 4-tile walk regardless.)
finished_all() { grep -o 'Jobs finished: [0-9]*/[0-9]*' "$1" | tail -1 | awk -F'[:/ ]+' -v n="$2" '{print ($3==$4 && $3==n)?1:0}'; }
cmds_of() { grep -o 'DRAM CMDs: [0-9]*' "$1" | tail -1 | awk '{print $3}'; }
printf 'Transformer 1 256 4 4 512 1024 0 1\n' > "$WORK/v31c.txt"
printf 'Transformer 1 256 4 4 512 512 0 1\n' > "$WORK/v31c2.txt"
C1="-c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 -fuse_attn 0 -fuse_vpu 1"
timeout 120 "$BIN" $C1 -dbuf_tile 0 -i "$WORK/v31c.txt" -o "$WORK/v31c0_s.txt" > "$WORK/v31c0.log" 2>&1; rc0=$?
timeout 120 "$BIN" $C1 -dbuf_tile 1 -i "$WORK/v31c.txt" -o "$WORK/v31c1_s.txt" > "$WORK/v31c1.log" 2>&1; rc1=$?
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31c2.txt" "$WORK/v31c2_s.txt" -fuse_attn 0 -dbuf_tile 0 > "$WORK/v31c2.log" 2>&1; rc2=$?
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31c2.txt" "$WORK/v31c3_s.txt" -fuse_attn 0 > "$WORK/v31c3.log" 2>&1; rc3=$?
timeout 120 "$BIN" $C1 -dbuf_tile 1 -i "$WORK/v31c2.txt" -o "$WORK/v31c4_s.txt" > "$WORK/v31c4.log" 2>&1; rc4=$?
c0=$(cmds_of "$WORK/v31c0.log"); c1=$(cmds_of "$WORK/v31c1.log")
c2=$(cmds_of "$WORK/v31c2.log"); c3=$(cmds_of "$WORK/v31c3.log"); c4=$(cmds_of "$WORK/v31c4.log")
if [ "$rc0$rc1$rc2$rc3$rc4" = "00000" ] \
   && [ "$(finished_all "$WORK/v31c0.log" 46)" = 1 ] && [ "$(finished_all "$WORK/v31c1.log" 46)" = 1 ] \
   && [ "$(finished_all "$WORK/v31c2.log" 44)" = 1 ] && [ "$(finished_all "$WORK/v31c3.log" 44)" = 1 ] \
   && [ "$(finished_all "$WORK/v31c4.log" 30)" = 1 ] \
   && [ "${c0:-0}" -eq 716800 ] && [ "$c1" = "$c0" ] && [ "${c2:-0}" -eq 237568 ] && [ "$c3" = "$c2" ] \
   && [ "$c4" = "$c2" ]; then
  ok "V31c multi-row OS jobs complete under -dbuf_tile 1 (CMDs -c 1 S=1024: $c0, S=512 pinned/-c 1: $c2, invariant)"
else
  bad "V31c rc=$rc0/$rc1/$rc2/$rc3/$rc4 CMDs -c 1 S=1024: $c0/$c1 (want 716800) S=512 pinned: $c2/$c3 -c 1: $c4 (want 237568)"
fi

# V31d: a job's write-back must never land on an address whose -dbuf
# PREFETCH read has not landed (S4 fix round 2, review findings 1+2). The
# prefetcher streams a job's weight sweep and (once READY) its activation
# panel into [addr_hold, addr_hold + credit) and books the credit at ISSUE;
# the dispatched job deducts the credit from its demand reads, so its own
# walk -- demand reads from addr_hold, then the write-back right behind
# them -- starts its writes at addr_hold + (formula - credit): INSIDE the
# prefetched span whenever credit > formula - credit. Nothing waited for the
# prefetched beats to land (nullptr owner in the DRAM callback), and at K =
# 256 a tile computes in 128 cycles, shorter than the DRAM queue latency of
# a panel issued just before dispatch. A write to an address with a pending
# read is the DRAMSim3 read/write deadlock Job.h documents: the run freezes.
# This needs no window overrun and no -dbuf_tile (V31c's fix cured the
# S=512 pinned run by timing luck only). Fix: prefetch beats keep their
# owner (Job::prefetch_issued/landed_beats); at dispatch the credited-but-
# unlanded beats become the state's prefetch_read_left, which process_stage
# waits on like demand reads -- the MXU cannot compute on data that has not
# arrived, and no write is issued before every prefetched beat has landed.
# Traffic is untouched: the credit still replaces demand beats 1:1.
# Run A: the pinned config, S = 512, -fuse_attn 0 -act_share 0 -dbuf_tile 0
# (hung at 31/44, CMDs frozen at 229951). 44/44 jobs at V31c Run 2's
# 237568 plus the core-1 activation panels -act_share 0 restores:
#   q,k,v,o: 4 GEMMs x 2 core-1 jobs x act(256*256*2/64 = 2048) = 16384
#   gate,up: 2 x 2 x 2048 = 8192;  down: 2 x act(256*512*2/64 = 4096) = 8192
#   total 237568 + 32768 = 270336
# Run B: Run A at -dbuf 0 (the control that always completed): the same
# 270336 -- prefetch stays exactly traffic-invariant.
# Run C: -c 1 (V31c's C1 flags) with -dbuf 48 -dbuf_tile 1, S = 1024 (hung
# at 33/46): 46/46 at V31c Run 1's 716800.
# Run D: the pinned config, S = 640, -fuse_attn 0 -act_share 0 -dbuf_tile 1
# (the review saw 8+ write-over-pending-prefetch events on its 640x64x640
# scores job and completion by luck): 60/60, CMDs equal to its -dbuf 0 run
# (Run E) -- -dbuf_tile does not protect against the alias, so the fix must
# hold there too.
printf 'Transformer 1 256 4 4 512 640 0 1\n' > "$WORK/v31d640.txt"
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31c2.txt" "$WORK/v31dA_s.txt" -fuse_attn 0 -act_share 0 -dbuf_tile 0 > "$WORK/v31dA.log" 2>&1; rA=$?
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31c2.txt" "$WORK/v31dB_s.txt" -fuse_attn 0 -act_share 0 -dbuf_tile 0 -dbuf 0 > "$WORK/v31dB.log" 2>&1; rB=$?
timeout 120 "$BIN" $C1 -dbuf 48 -dbuf_tile 1 -i "$WORK/v31c.txt" -o "$WORK/v31dC_s.txt" > "$WORK/v31dC.log" 2>&1; rC=$?
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31d640.txt" "$WORK/v31dD_s.txt" -fuse_attn 0 -act_share 0 -dbuf_tile 1 > "$WORK/v31dD.log" 2>&1; rD=$?
timeout 120 "$REPO/configs/tpuv6e.sh" "$WORK/v31d640.txt" "$WORK/v31dE_s.txt" -fuse_attn 0 -act_share 0 -dbuf_tile 1 -dbuf 0 > "$WORK/v31dE.log" 2>&1; rE=$?
dA=$(cmds_of "$WORK/v31dA.log"); dB=$(cmds_of "$WORK/v31dB.log"); dC=$(cmds_of "$WORK/v31dC.log")
dD=$(cmds_of "$WORK/v31dD.log"); dE=$(cmds_of "$WORK/v31dE.log")
if [ "$rA$rB$rC$rD$rE" = "00000" ] \
   && [ "$(finished_all "$WORK/v31dA.log" 44)" = 1 ] && [ "$(finished_all "$WORK/v31dB.log" 44)" = 1 ] \
   && [ "$(finished_all "$WORK/v31dC.log" 46)" = 1 ] \
   && [ "$(finished_all "$WORK/v31dD.log" 60)" = 1 ] && [ "$(finished_all "$WORK/v31dE.log" 60)" = 1 ] \
   && [ "${dA:-0}" -eq 270336 ] && [ "$dB" = "$dA" ] && [ "${dC:-0}" -eq 716800 ] \
   && [ -n "$dD" ] && [ "$dD" = "$dE" ]; then
  ok "V31d write-back waits for unlanded prefetch reads (S=512 -act_share 0: $dA, -c 1 S=1024 -dbuf 48: $dC, S=640: $dD, all -dbuf-invariant)"
else
  bad "V31d rc=$rA/$rB/$rC/$rD/$rE CMDs S=512: $dA/$dB (want 270336) -c 1 S=1024: $dC (want 716800) S=640: $dD/$dE"
fi

# V32: per-op-class accounting (benchmark spec S1). Every Transformer job is
# tagged with an OpClass at creation (Job::op_class: QKV, O, GATE_UP, DOWN,
# HEAD, ATTN = scores + softmax chunks + AV, VPU_NORM = norm1/norm2/
# final_norm, VPU_EW = rope/silu_mul/residual/logits softmax; everything
# else OTHER), and Arch's per-cycle classifier increments a per-(unit,
# class) copy of busy/underfilled/memstall beside the per-unit totals. The
# stats file (SCHEMA 3) prints one 'ACCTC <type> <idx> <class> busy <n>
# underfilled <n> memstall <n>' line per class with a nonzero counter after
# each unit's ACCT line, so XProf's per-op utilization has a counterpart.
# V32a: Transformer 1 512 8 8 1024 512 0 1 1024 on the pinned config (-c 2
# -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2, M = S = 512, head_dim 64, nkv =
# nh = 8 so heads alternate MXUs). For every unit the class sums equal the
# ACCT totals EXACTLY (idle is per-unit, not per class), no OTHER line
# appears (every Transformer job is classified) and no class appears on the
# wrong unit type. The busy/underfilled columns are pinned BEAT-EXACT;
# memstall is DRAM timing and is covered by the sum invariant only.
# Job count: norm1 1 + q/k/v 3 x (2 row blocks x 2 cores = 4) + rope 1 +
# scores 8 + AV 8 + softmax 4 + o 4 + res1 1 + norm2 1 + gate/up 2 x 4 +
# silu_mul 1 + down 4 + res2 1 + final_norm 1 + head (1 row block x 2
# cores) 2 + logits softmax 1 = 58.
# SA cycle model (OS, SysArray.cc): per tile the read stage is programmed
# ceil(K / macs) cycles and the shift stage sz * min(fpu_latency, batch) =
# 256; both are non-memstall (a stall can register only once the stage
# counter has run out). The write stage has no compute: its cycles are
# memstall until the write-back lands (none at all when fused_out), and
# the increment that leaves it lands in the NEXT tile's read state (+1
# non-memstall cycle per intra-job tile transition) or in idle (not
# counted). Per job: tiles x (ceil(K/2) + 256) + (tiles - 1). Busy vs
# underfilled: min(sz,M) * min(sz,N) < sz^2, N per core (createSAJobs
# splits N over the 2 cores). Per MXU:
#   QKV     3 GEMMs x 2 jobs 256x512x256 (1 tile): 256+256 = 512 x 6 = 3072 busy
#   O       2 jobs 256x512x256:                          512 x 2 = 1024 busy
#   GATE_UP 2 GEMMs x 2 jobs 256x512x512 (2 tiles): 2x512+1 = 1025 x 4 = 4100 busy
#   DOWN    2 jobs 256x1024x256 (1 tile, K 1024): 512+256 = 768 x 2 = 1536 busy
#   HEAD    1 job 1x512x512 (M 1 -> underfilled; 2 tiles): 2x512+1 = 1025 underfilled
#   ATTN    4 scores 512x64x512 (2x2 tiles, K 64: 32+256 = 288; fused_out):
#             4x288+3 = 1155 x 4 = 4620 busy
#           4 AV 512x512x64 (N 64 -> underfilled; 2 row tiles): 2x512+1
#             = 1025 x 4 = 4100 underfilled
# VPU cycle model (VectorUnit.cc): init programs the first phase directly,
# REDUCE n costs lin * n * ceil(par / vu_sz), BROADCAST n costs
# ceil(lin * par * n / vu_sz), the write stage is memstall-or-nothing, so
# busy + underfilled = the phase sum exactly; underfilled iff par < 512.
# softmax phases {B1, R1, B1}, rmsnorm {R2, B1}:
#   ATTN     4 softmax chunks (lin 512, par 1024, prebuffered): 1024 + 512x2
#              + 1024 = 3072 x 4 = 12288 busy
#   VPU_NORM norm1, norm2 (lin 512, par 512): 512x2 + 512 = 1536 x 2 = 3072
#              busy; final_norm (par 1): 512x2x1 + ceil(512/512) = 1025 underfilled
#   VPU_EW   rope (lin 1, par 512 x (512+512)): 524288/512 = 1024; silu_mul
#              (lin 1024, par 512): 1024; res1, res2 (512 x 512): 512 each
#              -> 3072 busy; logits softmax (lin 1024, par 1): 2 + 1024 + 2
#              = 1028 underfilled
printf 'Transformer 1 512 8 8 1024 512 0 1 1024\n' > "$WORK/v32a.txt"
timeout 300 "$REPO/configs/tpuv6e.sh" "$WORK/v32a.txt" "$WORK/v32a_s.txt" > "$WORK/v32a.log" 2>&1; r32=$?
acctc_sum_ok() { # prints the number of ACCT units whose class sums mismatch (want 0)
  awk '$1=="ACCT"{k=$2" "$3; b[k]=$5; u[k]=$7; m[k]=$9}
       $1=="ACCTC"{k=$2" "$3; cb[k]+=$6; cu[k]+=$8; cm[k]+=$10}
       END{n=0; for(k in b) if (b[k]!=cb[k]+0 || u[k]!=cu[k]+0 || m[k]!=cm[k]+0) n++; print n}' "$1"
}
mism=$(acctc_sum_ok "$WORK/v32a_s.txt")
n_units=$(grep -c '^ACCT ' "$WORK/v32a_s.txt")
cat > "$WORK/v32a_want.txt" <<'EOW'
SYSTOLIC_ARRAY 0 ATTN 4620 4100
SYSTOLIC_ARRAY 0 DOWN 1536 0
SYSTOLIC_ARRAY 0 GATE_UP 4100 0
SYSTOLIC_ARRAY 0 HEAD 0 1025
SYSTOLIC_ARRAY 0 O 1024 0
SYSTOLIC_ARRAY 0 QKV 3072 0
SYSTOLIC_ARRAY 1 ATTN 4620 4100
SYSTOLIC_ARRAY 1 DOWN 1536 0
SYSTOLIC_ARRAY 1 GATE_UP 4100 0
SYSTOLIC_ARRAY 1 HEAD 0 1025
SYSTOLIC_ARRAY 1 O 1024 0
SYSTOLIC_ARRAY 1 QKV 3072 0
VECTOR_UNIT 2 ATTN 12288 0
VECTOR_UNIT 2 VPU_EW 3072 1028
VECTOR_UNIT 2 VPU_NORM 3072 1025
EOW
awk '$1=="ACCTC"{print $2, $3, $4, $6, $8}' "$WORK/v32a_s.txt" | sort > "$WORK/v32a_got.txt"
if [ "$r32" -eq 0 ] && [ "$(finished_all "$WORK/v32a.log" 58)" = 1 ] && [ "$n_units" -eq 3 ] \
   && [ "$mism" = 0 ] && cmp -s "$WORK/v32a_want.txt" "$WORK/v32a_got.txt"; then
  ok "V32a ACCTC class busy/underfilled pinned per unit and class sums equal ACCT totals on all $n_units units"
else
  bad "V32a rc=$r32 fin=$(finished_all "$WORK/v32a.log" 58) units=$n_units mismatched=$mism diff: $(diff "$WORK/v32a_want.txt" "$WORK/v32a_got.txt" | tr '\n' ' ')"
fi
# V32b: a plain Matmul line is not a Transformer op: its SA cycles all land
# in OTHER (the only ACCTC class present) and the sum invariant still holds.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v31a.txt" -o "$WORK/v32b_s.txt" > "$WORK/v32b.log" 2>&1; rb=$?
mism_b=$(acctc_sum_ok "$WORK/v32b_s.txt")
cls_b=$(awk '$1=="ACCTC"{print $4}' "$WORK/v32b_s.txt" | sort -u | tr '\n' ' ')
if [ "$rb" -eq 0 ] && [ "$mism_b" = 0 ] && [ "$cls_b" = "OTHER " ]; then
  ok "V32b plain Matmul reports only class OTHER"
else
  bad "V32b rc=$rb mismatched=$mism_b classes='$cls_b' (want 'OTHER ')"
fi
# V32c: the stats header declares SCHEMA 3 (ACCTC lines present).
if head -1 "$WORK/v32a_s.txt" | grep -q '^SCHEMA 3$' && head -1 "$WORK/v32b_s.txt" | grep -q '^SCHEMA 3$'; then
  ok "V32c stats files declare SCHEMA 3"
else
  bad "V32c header: $(head -1 "$WORK/v32a_s.txt") / $(head -1 "$WORK/v32b_s.txt") (want SCHEMA 3)"
fi

# V33: -op_overhead (benchmark spec S2). Silicon pays a fixed cost per KERNEL
# (~8 per layer; chained microbenchmarks fit t = t0 + bytes/BW), the model
# has ~86 jobs per layer, so the fit knob is per OP boundary per core: every
# composite call stamps its jobs with one op_id (a Matmul line = one op; the
# Transformer's attention stage = one op per layer, like the fused kernel),
# and a unit entering a job whose op_id differs from the last one it ran
# stalls op_overhead cycles before issuing reads -- a pure serial delay,
# not added to the compute length (that would hide under memory-bound
# fetches). Jobs with no op_id (legacy composites) are each their own op.
# The stats file reports OPBOUND <unit> <idx> <n>: the ops a unit entered --
# the count the knob charges -- which is exact by construction. The cycle
# delta is NOT: a 1000-cycle shift moves every later DRAM access against the
# refresh schedule (measured: 3772 and 3972 on these two shapes), so cycles
# are asserted within +-10% of 4 x 1000.
# V33a: 4 Matmul lines on one core: OPBOUND 4, ~4000 cycles.
printf 'Matmul 512 512 512\nMatmul 512 512 512\nMatmul 512 512 512\nMatmul 512 512 512\n' > "$WORK/v33.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v33.txt" -o "$WORK/v33a_s.txt" > "$WORK/v33a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -op_overhead 1000 -i "$WORK/v33.txt" -o "$WORK/v33b_s.txt" > "$WORK/v33b.log" 2>&1
a0=$(cycles_of "$WORK/v33a_s.txt"); a1=$(cycles_of "$WORK/v33b_s.txt")
ob=$(awk '$1=="OPBOUND" && $2=="SYSTOLIC_ARRAY" && $3==0 {print $4}' "$WORK/v33b_s.txt")
if [ -n "$a0" ] && [ -n "$a1" ] && [ "${ob:-0}" -eq 4 ] && [ $((a1 - a0)) -ge 3600 ] && [ $((a1 - a0)) -le 4400 ]; then
  ok "V33a -op_overhead: 4 op boundaries, +$((a1 - a0)) cycles on one core"
else
  bad "V33a boundaries=${ob:-?} cycles $a0 -> $a1 delta=$((${a1:-0} - ${a0:-0})) (want 4, 3600..4400)"
fi
# V33b: two cores: each sees every op once (OPBOUND 4 on both units) and pays
# in parallel, so the critical path still grows by ~4 x 1000.
"$BIN" -c 2 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v33.txt" -o "$WORK/v33c_s.txt" > "$WORK/v33c.log" 2>&1
"$BIN" -c 2 -sa_sz 64 -vu_sz 64 -f 1 -op_overhead 1000 -i "$WORK/v33.txt" -o "$WORK/v33d_s.txt" > "$WORK/v33d.log" 2>&1
b0=$(cycles_of "$WORK/v33c_s.txt"); b1=$(cycles_of "$WORK/v33d_s.txt")
ob0=$(awk '$1=="OPBOUND" && $2=="SYSTOLIC_ARRAY" && $3==0 {print $4}' "$WORK/v33d_s.txt")
ob1=$(awk '$1=="OPBOUND" && $2=="SYSTOLIC_ARRAY" && $3==1 {print $4}' "$WORK/v33d_s.txt")
if [ -n "$b0" ] && [ -n "$b1" ] && [ "${ob0:-0}" -eq 4 ] && [ "${ob1:-0}" -eq 4 ] && [ $((b1 - b0)) -ge 3600 ] && [ $((b1 - b0)) -le 4400 ]; then
  ok "V33b -op_overhead on two cores: 4 boundaries each, +$((b1 - b0)) cycles"
else
  bad "V33b boundaries=${ob0:-?}/${ob1:-?} cycles $b0 -> $b1 delta=$((${b1:-0} - ${b0:-0}))"
fi
# V33c: decode Transformer, sidecars fused off the chain: per layer the serial
# op sequence on a core is q, k, v, attention, o, gate, up, down = 8 ops, so
# -op_overhead 1000 adds at least 8000 cycles (and far fewer than the ~30
# jobs-as-ops the old per-job knob would charge: upper sanity bound 20000).
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -fuse_vpu 1 -i "$WORK/v25d.txt" -o "$WORK/v33e_s.txt" > "$WORK/v33e.log" 2>&1
"$BIN" -c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_attn 1 -fuse_vpu 1 -op_overhead 1000 -i "$WORK/v25d.txt" -o "$WORK/v33f_s.txt" > "$WORK/v33f.log" 2>&1
c0=$(cycles_of "$WORK/v33e_s.txt"); c1=$(cycles_of "$WORK/v33f_s.txt")
if [ -n "$c0" ] && [ -n "$c1" ] && [ $((c1 - c0)) -ge 16000 ] && [ $((c1 - c0)) -le 40000 ]; then
  ok "V33c decode 2 layers: op boundaries on the critical path (+$((c1 - c0)) cycles)"
else
  bad "V33c cycles $c0 -> $c1 delta=$((${c1:-0} - ${c0:-0})) (want 16000..40000 for 2 layers)"
fi
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -op_overhead -1 -i "$WORK/v33.txt" -o "$WORK/v33g_s.txt" > "$WORK/v33g.log" 2>&1; rv=$?
if [ "$rv" -eq 1 ] && grep -q 'op_overhead' "$WORK/v33g.log"; then
  ok "V33d -op_overhead -1 rejected"
else
  bad "V33d rc=$rv (want 1 + message)"
fi

# V34: fit knobs (benchmark spec S6).
# -kv_bw_pct P: decode score/AV jobs stream the paged KV cache; the kernel
# gathers it through a block table at a lower effective HBM rate than a
# contiguous stream. Those jobs (kv_stream, decode only) issue reads at
# max(1, dram_enq * P / 100) beats per cycle and are never prefetched (a
# paged-attention kernel gathers its own blocks; XLA cannot stream them
# ahead). Traffic invariant. V34a: decode 2048 ctx x batch 32, P=50 vs 100:
# CMDs equal, cycles strictly higher.
printf 'Transformer 1 512 8 4 1024 2048 1 32\n' > "$WORK/v34.txt"
PIN="-c 2 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 -fuse_attn 1 -fuse_vpu 1 -dbuf 48 -dbuf_tile 1"
"$BIN" $PIN -i "$WORK/v34.txt" -o "$WORK/v34a_s.txt" > "$WORK/v34a.log" 2>&1
"$BIN" $PIN -kv_bw_pct 50 -i "$WORK/v34.txt" -o "$WORK/v34b_s.txt" > "$WORK/v34b.log" 2>&1
k0=$(cycles_of "$WORK/v34a_s.txt"); k1=$(cycles_of "$WORK/v34b_s.txt")
m0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34a.log" | tail -1 | awk '{print $3}')
m1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34b.log" | tail -1 | awk '{print $3}')
# The budget is ONE per-cycle token bucket shared by every KV-stream job on
# the chip (review finding: a per-unit cap against the device plate let two
# MXUs stream 2x the intended rate, and P=50 cost only +7%). This shape is
# KV-dominated (67 MB of KV per layer vs ~5 MB of weights), so halving the
# KV rate against a DRAM that serves ~12 beats/cycle must cost >= 30%.
# P=99 vs P=100 (no cap): the default stays bit-identical, but pacing KV
# issue at the plate is NOT a no-op -- -dram_enq 32 lets a KV stream burst
# 32 beats/cycle into the FIFO ahead of weight fetches, and P=99 measured
# 5.6% FASTER than uncapped on this shape. Assert the step stays under 10%
# and leave the remedy to the fit (-dram_enq = plate, spec 6).
"$BIN" $PIN -kv_bw_pct 99 -i "$WORK/v34.txt" -o "$WORK/v34x_s.txt" > "$WORK/v34x.log" 2>&1
k99=$(cycles_of "$WORK/v34x_s.txt")
cont=$(awk -v a="${k0:-0}" -v b="${k99:-0}" 'BEGIN{d=(a-b)/a; if(d<0)d=-d; print (a>0 && b>0 && d<=0.10)?1:0}')
if [ -n "$k0" ] && [ -n "$k1" ] && [ "$m0" = "$m1" ] && [ "$cont" -eq 1 ] \
   && [ "$(awk -v a="$k0" -v b="$k1" 'BEGIN{print (b>=1.3*a)?1:0}')" -eq 1 ]; then
  ok "V34a -kv_bw_pct 50 slows decode KV streams ($k0 -> $k1, +$(( (k1 - k0) * 100 / k0 ))%, P99 $k99, CMDs invariant $m0)"
else
  bad "V34a cycles $k0 -> $k1 (want >= 1.3x), P99=$k99 (want within 10%), CMDs $m0 vs $m1"
fi
# V34b: prefill attention reads the just-computed projections contiguously:
# bit-identical under -kv_bw_pct 50.
printf 'Transformer 1 512 8 4 1024 512 0 1\n' > "$WORK/v34p.txt"
"$BIN" $PIN -i "$WORK/v34p.txt" -o "$WORK/v34c_s.txt" > "$WORK/v34c.log" 2>&1
"$BIN" $PIN -kv_bw_pct 50 -i "$WORK/v34p.txt" -o "$WORK/v34d_s.txt" > "$WORK/v34d.log" 2>&1
p0=$(cycles_of "$WORK/v34c_s.txt"); p1=$(cycles_of "$WORK/v34d_s.txt")
q0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34c.log" | tail -1 | awk '{print $3}')
q1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34d.log" | tail -1 | awk '{print $3}')
if [ -n "$p0" ] && [ "$p0" = "$p1" ] && [ "$q0" = "$q1" ]; then
  ok "V34b prefill bit-identical under -kv_bw_pct 50 ($p0 cycles, $q0 CMDs)"
else
  bad "V34b cycles $p0 vs $p1, CMDs $q0 vs $q1"
fi
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -kv_bw_pct 0 -i "$WORK/v33.txt" -o "$WORK/v34e_s.txt" > "$WORK/v34e.log" 2>&1; r0=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -kv_bw_pct 101 -i "$WORK/v33.txt" -o "$WORK/v34f_s.txt" > "$WORK/v34f.log" 2>&1; r1=$?
if [ "$r0" -eq 1 ] && [ "$r1" -eq 1 ] && grep -q 'kv_bw_pct' "$WORK/v34e.log"; then
  ok "V34c -kv_bw_pct 0 / 101 rejected"
else
  bad "V34c rc=$r0/$r1 (want 1/1 + message)"
fi
# -data_overhead N: layout/copy kernels the model has no jobs for (census:
# 4-7% of device time) as a fixed per-run cost: the clock advances N cycles
# before the first dispatch, nothing else moves. V34d: exactly +777, CMDs equal.
printf 'Matmul 256 256 256\n' > "$WORK/v34g.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v34g.txt" -o "$WORK/v34g_s.txt" > "$WORK/v34g.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -data_overhead 777 -i "$WORK/v34g.txt" -o "$WORK/v34h_s.txt" > "$WORK/v34h.log" 2>&1
d0=$(cycles_of "$WORK/v34g_s.txt"); d1=$(cycles_of "$WORK/v34h_s.txt")
e0=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34g.log" | tail -1 | awk '{print $3}')
e1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34h.log" | tail -1 | awk '{print $3}')
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -data_overhead -5 -i "$WORK/v34g.txt" -o "$WORK/v34i_s.txt" > "$WORK/v34i.log" 2>&1; rd=$?
# The overhead cycles are demand-idle by definition (nothing is dispatched
# yet): the MEM demand-idle numerator must grow by exactly 777 too.
di0=$(grep -ao 'MEM demand-idle: [0-9]*' "$WORK/v34g.log" | grep -o '[0-9]*$')
di1=$(grep -ao 'MEM demand-idle: [0-9]*' "$WORK/v34h.log" | grep -o '[0-9]*$')
if [ -n "$d0" ] && [ -n "$d1" ] && [ $((d1 - d0)) -eq 777 ] && [ "$e0" = "$e1" ] && [ "$rd" -eq 1 ] \
   && grep -q 'data_overhead' "$WORK/v34i.log" && [ $((${di1:-0} - ${di0:-0})) -eq 777 ]; then
  ok "V34d -data_overhead 777 adds exactly 777 cycles ($d0 -> $d1), all demand-idle, negative rejected"
else
  bad "V34d cycles $d0 -> $d1 delta=$((${d1:-0} - ${d0:-0})), demand-idle $di0 -> $di1, CMDs $e0 vs $e1, badval_rc=$rd"
fi
# V34e: 64-bit byte amounts in the VPU (review finding): an unfused binary
# elementwise job over M x cols with 4*M*cols >= 2^31 wrapped its read
# count to 1 beat. Transformer 1 64 2 2 16384 4096 0 8 at -c 1 -sa_sz 256
# (M = 32768): the unfused VPU ops move, beat-exact,
#   silu_mul 32768 x 16384: 2 x 2^30 B read = 33554432 + 2^30 B write =
#     16777216 -> 50331648
#   norm1, norm2: 32768 x 64 x 2 B = 4 MiB read + 4 MiB write = 131072 each
#   rope: 32768 x (64+64) x 2 B = 8 MiB read + write -> 262144
#   res1, res2: 2 x 4 MiB read + 4 MiB write = 196608 each
#   total 51249152 = CMDs(-fuse_vpu 0) - CMDs(-fuse_vpu 1)
# (pre-fix the silu read wrapped: delta 17694720).
printf 'Transformer 1 64 2 2 16384 4096 0 8\n' > "$WORK/v34j.txt"
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -i "$WORK/v34j.txt" -o "$WORK/v34j_s.txt" > "$WORK/v34j.log" 2>&1
"$BIN" -c 1 -n_vpu 1 -sa_sz 256 -vu_sz 512 -mxu_macs_per_pe 2 -f 1 -ws 0 -buf_mb 128 -dram_enq 32 \
  -fuse_vpu 1 -i "$WORK/v34j.txt" -o "$WORK/v34k_s.txt" > "$WORK/v34k.log" 2>&1
bu=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34j.log" | tail -1 | awk '{print $3}')
bf=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v34k.log" | tail -1 | awk '{print $3}')
if [ -n "$bu" ] && [ -n "$bf" ] && [ $((bu - bf)) -eq 51249152 ]; then
  ok "V34e 64-bit VPU byte amounts: unfused VPU traffic exact on a 2^29-element op ($bu - $bf)"
else
  bad "V34e CMDs unfused=$bu fused=$bf delta=$((${bu:-0} - ${bf:-0})) (want 51249152)"
fi

# V35: batched prefill (benchmark spec S3). mode 0 with batch > 1 runs every
# GEMM over T = batch * seq rows (all sequences' tokens) and builds attention
# PER SEQUENCE: an independent set of nh score jobs (M = seq, K = head_dim,
# N = S = seq), softmax chunks over seq*nh rows wired per head, and nh AV jobs
# (M = seq, K = S, N = head_dim), with K/V tags per (sequence, group). All
# sequences' scores depend on q/k, all AV feed the o projection; the LM head
# stays last-token (batch rows). Decode (mode 1) is unchanged.
# V35a: job count. Transformer 1 8 2 2 16 8 0 2 at -c 1 -sa_sz 4 (V11 shape,
# batch 2): T = 16, so every GEMM splits into ceil(16/4) = 4 row-block jobs:
#   norm1 1 | q,k,v 3x4 = 12 | rope 1 | attention 2 seqs x (2 scores +
#   1 softmax (16 rows of 8) + 2 AV) = 10 | o 4 | res1 1 | norm2 1 |
#   gate,up 2x4 = 8 | silu 1 | down 4 | res2 1  -> 44   (batch 1 = 25, V11)
printf 'Transformer 1 8 2 2 16 8 0 2\n' > "$WORK/v35.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v35.txt" -o "$WORK/v35a_s.txt" > "$WORK/v35a.log" 2>&1
rc=$?
total=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v35a.log" | tail -1 | sed 's|.*/||')
fin=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v35a.log" | tail -1 | grep -o '[0-9]*')
if [ "$rc" -eq 0 ] && [ "${total:-0}" = "44" ] && [ "$fin" = "$total" ]; then
  ok "V35a batched prefill: 44 jobs, all finish"
else
  bad "V35a rc=$rc total=${total:-?} fin=${fin:-?} (want 44)"
fi
# V35b: traffic, beat-exact. Two sequences of 64 tokens vs one sequence of
# 128 tokens at d_model 64, 2 heads (head_dim 32), -c 1 -sa_sz 64: T = 128
# in both, so every GEMM, norm, rope and residual moves identical bytes and
# only attention differs.
#   1 x 128: scores (128x32x128): pass 1 combined act+wgt (4096+4096)/64 =
#     128 + tile 2 weights 64, pass 2 act only 64 (weights stay resident) =
#     256 reads + 2 passes x 2 tiles x 128 write beats = 768/job x 2 = 1536;
#     softmax 256 rows x 128: 1024 read + 1024 write = 2048;
#     AV (128x128x32): pass 1 (16384+8192)/64 = 384, pass 2 act 256 = 640
#     reads + 2 x 64 write = 768/job x 2 = 1536.               total 5120
#   2 x 64:  scores (64x32x64): (4096+4096)/64 = 128 + 128 write = 256/job
#     x 2 heads x 2 seqs = 1024; softmax 128 rows x 64 per seq: 256 + 256 =
#     512 x 2 = 1024; AV (64x64x32): (8192+4096)/64 = 192 + 64 write = 256
#     x 2 x 2 = 1024.                                          total 3072
#   delta = 2048 exactly.
printf 'Transformer 1 64 2 2 128 128 0 1\n' > "$WORK/v35b1.txt"
printf 'Transformer 1 64 2 2 128 64 0 2\n' > "$WORK/v35b2.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v35b1.txt" -o "$WORK/v35b1_s.txt" > "$WORK/v35b1.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v35b2.txt" -o "$WORK/v35b2_s.txt" > "$WORK/v35b2.log" 2>&1
x1=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v35b1.log" | tail -1 | awk '{print $3}')
x2=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v35b2.log" | tail -1 | awk '{print $3}')
if [ -n "$x1" ] && [ -n "$x2" ] && [ $((x1 - x2)) -eq 2048 ]; then
  ok "V35b batched-prefill attention traffic exact (1x128: $x1, 2x64: $x2)"
else
  bad "V35b CMDs 1x128=$x1 2x64=$x2 delta=$((${x1:-0} - ${x2:-0})) (want 2048)"
fi
# V35c: the pinned holdout batched-prefill shape runs to completion on the
# canonical config.
printf 'Transformer 1 4096 32 8 12288 512 0 8 151936\n' > "$WORK/v35c.txt"
"$REPO/configs/tpuv6e.sh" "$WORK/v35c.txt" "$WORK/v35c_s.txt" > "$WORK/v35c.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v35c.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4)?1:0}')
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ]; then
  ok "V35c holdout batched prefill (512 x 8) runs on the pinned config"
else
  bad "V35c rc=$rc fin_eq=${fin_eq:-?}"
fi

# V36: per-op-class wall-clock span (fidelity spec section 4, 'per-op-class
# time'): OPSPAN <class> first <cycle> last <cycle> after the ACCTC lines,
# one per class with any job. Convention pinned here: 'first' is gcycles at
# the dispatch of the class's first job (dispatch runs at the top of the
# loop before gcycles++, so the very first dispatch is at data_overhead ==
# 0 by default), 'last' is gcycles at the TO_IDLE_CLEANUP of the class's
# last job (increment() runs after gcycles++, and the loop exits right after
# that completion with Cycles == gcycles), so last == Cycles exactly.
# V36a: a single Matmul is one class (OTHER), first 0, last == Cycles.
printf 'Matmul 512 512 512\n' > "$WORK/v36a.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v36a.txt" -o "$WORK/v36a_s.txt" > "$WORK/v36a.log" 2>&1
n_span=$(grep -c '^OPSPAN ' "$WORK/v36a_s.txt")
cyc=$(cycles_of "$WORK/v36a_s.txt")
sp_first=$(awk '/^OPSPAN OTHER /{print $4}' "$WORK/v36a_s.txt")
sp_last=$(awk '/^OPSPAN OTHER /{print $6}' "$WORK/v36a_s.txt")
if [ "$n_span" -eq 1 ] && [ "${sp_first:-x}" = "0" ] && [ -n "$cyc" ] && [ "${sp_last:-x}" = "$cyc" ]; then
  ok "V36a OPSPAN OTHER first 0 last $sp_last == Cycles $cyc (one line)"
else
  bad "V36a OPSPAN lines=$n_span first=${sp_first:-?} last=${sp_last:-?} Cycles=${cyc:-?}"
fi
# V36b: fused transformer layer has one span per class in dependency order
# (QKV before ATTN before O before GATE_UP before DOWN) and no span ends
# after the run.
printf 'Transformer 1 256 4 4 512 128 0 1\n' > "$WORK/v36b.txt"
"$BIN" -c 1 -sa_sz 256 -f 1 -fuse_attn 1 -fuse_vpu 1 -i "$WORK/v36b.txt" -o "$WORK/v36b_s.txt" > "$WORK/v36b.log" 2>&1
cyc=$(cycles_of "$WORK/v36b_s.txt")
span_first() { awk -v c="$2" '$1=="OPSPAN" && $2==c {print $4}' "$1"; }
missing=""
for cls in QKV ATTN O GATE_UP DOWN VPU_NORM VPU_EW; do
  [ -z "$(span_first "$WORK/v36b_s.txt" "$cls")" ] && missing="$missing $cls"
done
f_qkv=$(span_first "$WORK/v36b_s.txt" QKV); f_attn=$(span_first "$WORK/v36b_s.txt" ATTN)
f_o=$(span_first "$WORK/v36b_s.txt" O); f_gu=$(span_first "$WORK/v36b_s.txt" GATE_UP)
f_dn=$(span_first "$WORK/v36b_s.txt" DOWN)
n_late=$(awk -v c="${cyc:-0}" '$1=="OPSPAN" && ($6 > c || $4 > $6) {n++} END{print n+0}' "$WORK/v36b_s.txt")
if [ -z "$missing" ] && [ "$f_qkv" -lt "$f_attn" ] && [ "$f_attn" -lt "$f_o" ] && \
   [ "$f_o" -lt "$f_gu" ] && [ "$f_gu" -lt "$f_dn" ] && [ "$n_late" -eq 0 ]; then
  ok "V36b OPSPAN classes present, QKV<ATTN<O<GATE_UP<DOWN ($f_qkv<$f_attn<$f_o<$f_gu<$f_dn), all last <= Cycles $cyc"
else
  bad "V36b missing=[$missing] firsts qkv=$f_qkv attn=$f_attn o=$f_o gu=$f_gu dn=$f_dn late=$n_late Cycles=$cyc"
fi
# V36c: -data_overhead shifts the first dispatch, so the span starts there.
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -data_overhead 500 -i "$WORK/v36a.txt" -o "$WORK/v36c_s.txt" > "$WORK/v36c.log" 2>&1
sp_first=$(awk '/^OPSPAN OTHER /{print $4}' "$WORK/v36c_s.txt")
if [ "${sp_first:-x}" = "500" ]; then
  ok "V36c OPSPAN OTHER first == 500 under -data_overhead 500"
else
  bad "V36c OPSPAN OTHER first=${sp_first:-?} (want 500)"
fi

echo "==== $PASS passed, $FAIL failed (outputs in $WORK)"
exit "$FAIL"
