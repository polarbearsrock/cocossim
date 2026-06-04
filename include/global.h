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

// Forward declare buffer hierarchy
namespace buffers {
    class BufferHierarchy;
    struct BufferHierarchyConfig;
}

const int systolic_fpu_latency = 1;
const int batch_size = 1;
const int n_mxus = 4;
const int n_vpus = 4;
const int data_type_width = 2;
const int seq_len = 2048;
const int dram_enq_per_cycle = 9;

// Buffer size in bytes (runtime configurable via BUFFER_SIZE_MB env var)
// Default: 64 MB, can be overridden in main.cc
extern int64_t buffer_size_bytes;

// Global buffer hierarchy configuration (initialized in global.cc)
extern buffers::BufferHierarchy* global_buffer_hierarchy;

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
extern uint64_t dram_read_cmds;
extern uint64_t dram_write_cmds;
extern int alloc_task_idx;
extern int model_parallelism;
extern bool do_par;
extern float freq_sa;
extern bool skip_dram_writeback;  // When true, outputs stay in on-chip buffer (no DRAM write)
extern bool activations_in_buffer;  // When true, activations already in buffer (no DRAM fetch)


int div_ru(int q, int r);


#endif//PROSE_COMPILER_GLOBAL_H