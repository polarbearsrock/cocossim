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
// Residency row window (C5v2, session 3): XLA re-streams a GEMM's weights
// per ~512-row M-tile even when they fit VMEM, so cross-row-block reuse is
// bounded. 0 = unlimited (the pre-measurement model).
extern int vmem_resident_rows;
// Serve SA memory transactions ahead of VPU ones (ISPASS'25 case study A fix).
extern int mem_prio;

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