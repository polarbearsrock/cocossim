/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_GLOBAL_H
#define PROSE_COMPILER_GLOBAL_H

#include <vector>
#include <tuple>
#include <cstdint>
#include <cstdio>

#define DSE
#define DEBUG

struct State;

// WS-only pipeline latency (fill/drain/shift). OS accumulation throughput is
// governed by mxu_macs_per_pe, not this constant.
const int systolic_fpu_latency = 2;
const int batch_size = 1;
const int n_mxus = 4;
const int n_vpus = 4;
const int data_type_width = 2;
const int seq_len = 2048;
extern int dram_enq_per_cycle;
extern int buffer_size_bytes;
extern int job_overhead_cycles;
extern int fuse_epilogue;
extern int mxu_macs_per_pe;
// VMEM staging model (spec 6.7): weights stay resident across row-block jobs
// of the same GEMM when the slice fits headroom_pct% of the per-MXU VMEM
// share. vmem_reuse=0 restores per-tile refetch (ablation).
extern int vmem_reuse;
extern int vmem_headroom_pct;
// Residency row window: cross-row-block weight reuse is cut after this many
// rows of M. Ablation knob only; default 0 = unlimited. (A 512-row default
// was once derived from C5v2 by fitting against host-floor-contaminated C3
// throughputs and is retracted: the raw C5v2 slopes equal pure MXU compute
// for a VMEM-resident weight, i.e. it streams once.)
extern int vmem_resident_rows;
// Serve SA memory transactions ahead of VPU ones (ISPASS'25 case study A fix).
extern int mem_prio;
// Flash-attention fusion (spec 6.7): attention score matrices live their whole
// life in VMEM -- QK^T writes, softmax reads/writes, and AV score reads never
// touch HBM. Attention traffic becomes read Q,K,V + write O, matching the
// fused kernels XLA/vLLM always emit. Compute and dependencies are unchanged.
extern int fuse_attn;
// VPU-op fusion (spec 6.7): RMSNorm, RoPE, SiLU-mul and residual adds ride
// in GEMM prologues/epilogues on silicon (kernel census: 0.1% of device
// time). They stay VPU jobs here -- attribution intact -- but run
// traffic-free as SIDECARS off the dependency chain, so consumers depend
// on the op's inputs' producers directly. Subsumes -fuse_epilogue.
extern int fuse_vpu;
// Cross-op weight prefetch (-dbuf, spec 6.7): upcoming weight tags' first
// jobs may stream their weight sweeps into otherwise-idle DRAM slots ahead
// of dispatch, modeling XLA's next-operator prefetch (B2/C5v2). The value
// is the prefetch byte budget in MiB -- issued-but-unconsumed bytes never
// exceed it (VMEM honesty). 0 = off. Traffic is invariant: prefetched
// beats replace demand beats 1:1.
extern int dbuf_lookahead;
// Issued-but-unconsumed prefetch beats (Arch.cc issues,
// Job::take_prefetch_credit_beats consumes).
extern int64_t pf_outstanding_beats;
// Within-op double buffering (-dbuf_tile, spec 6.7): an OS tile's reads are
// issued when the previous tile's reads drain, so they stream under the
// shift/write stages instead of after them. 0 = off (tile transitions and
// every job-start fetch exposed).
extern int dbuf_tile;
// Shared activation staging (-act_share, spec S4b): when a GEMM's N is split
// across cores, hardware stages each activation row block ONCE into the
// shared VMEM and both MXUs consume it. 1 (default): only core 0's row-block
// job charges the panel, its siblings on cores >= 1 are act_resident.
// 0: every core reads its own copy (the legacy per-MXU model, ablation).
extern int act_share;
// Fit knobs (benchmark spec S2/S6).
// -op_overhead: cycles a unit stalls when it enters a job of a new op_id
// (per op boundary per core -- silicon pays per kernel, ~8 per layer, not
// per row-block job). Pure serial delay before the job's reads.
extern int op_overhead_cycles;
// -kv_bw_pct: effective issue rate for decode KV-cache streams (paged
// gather) as a percentage of dram_enq_per_cycle. Traffic invariant.
extern int kv_bw_pct;
// The derate is ONE per-cycle token bucket shared by every KV-stream job on
// the chip (Arch.cc refills kv_budget_acc by kv_issue_rate beats each cycle,
// carrying at most one cycle over; State::enqueue_reads draws from it).
extern double kv_issue_rate;
extern double kv_budget_acc;
// -data_overhead: fixed per-run cycles for layout/copy kernels the model
// has no jobs for (kernel census: 4-7% of device time); the clock advances
// by this much before the first dispatch, nothing else moves.
extern int data_overhead_cycles;
// Decode attention structure (fidelity spec 6.3 item 2, 2026-09-01).
// -attn_group 1: one QK^T and one AV job per KV HEAD (M = group x query
// rows), the way the RPA kernel handles a GQA group in one pass per K/V
// tile; 0: the legacy one-job-per-query-head build, whose sibling jobs ran
// array passes against the resident panel with the DRAM idle.
extern int attn_group;
// -kv_prefetch 1: the KV-stream weight sweep is eligible for -dbuf prefetch
// (the kernel's own DMA pipeline streams the next block during the current
// one); 0: the S6 exclusion (a paged gather XLA cannot stream ahead of).
extern int kv_prefetch;
// Attention kernel floors from the census fit of vLLM's ragged_paged_attention
// decode kernel: t_layer = 15 us + sum over (sequence x kv head x 4096-token
// block) of max(block bytes / BW, 0.6 us).
// -attn_overhead: cycles every unit stalls at an ATTENTION op boundary (the
// -op_overhead mechanism scoped to OP_ATTN; the larger of the two applies).
extern int attn_overhead_cycles;
// -kv_block_latency: minimum cycles per weight stream per -kv_block tokens on
// a KV-stream QK^T job; the job cannot complete before streams x blocks x
// this many cycles have elapsed since its start (booked as ATTN memstall).
extern int kv_block_latency_cycles;
extern int kv_block_tokens;

const int embedding_dim= 768;
const int n_heads = 6;

const int periods = 1;
const int n_threads = 1;

extern char const * rand_chars[];

extern std::vector<std::tuple<uint64_t, bool, int, State *>> to_enqueue;
extern FILE *vcd;
extern int bytes_per_tx;
extern int jobs_finished;
extern int total_jobs;
extern uint64_t gcycles;
extern int alloc_task_idx;
extern int model_parallelism;
extern bool do_par;
extern float freq_sa;


int div_ru(int q, int r);


#endif//PROSE_COMPILER_GLOBAL_H