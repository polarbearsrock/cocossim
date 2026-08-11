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
#include <initializer_list>

#define DSE
#define DEBUG

struct State;

const int systolic_fpu_latency = 2;
const int n_mxus = 4;
const int n_vpus = 4;
const int seq_len = 2048;
const int dram_enq_per_cycle = 9;

const int embedding_dim= 768;

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

// Runtime workload/storage controls. data_type_bits describes packed storage;
// use bytes_for_elements() instead of rounding each scalar up to a whole byte.
extern int batch_size;
extern int data_type_bits;
extern uint64_t buffer_size_bytes;
extern bool compute_only;

uint64_t div_ru(uint64_t q, uint64_t r);
uint64_t checked_product(std::initializer_list<uint64_t> factors);
uint64_t bytes_for_elements(uint64_t element_count);
uint64_t elements_fitting_in_bytes(uint64_t byte_count);


#endif//PROSE_COMPILER_GLOBAL_H
