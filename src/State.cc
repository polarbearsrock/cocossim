/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "Arch.h"
#include "State.h"
#include "global.h"
#include <stdexcept>

// A job owns exactly [addr_hold, addr_hold + alloc_size). Walking past it
// silently overlaps the next bump-allocated job, and concurrent units then
// issue a read and a write to the same address, which deadlocks DRAMSim3's
// controller (see the alloc_size comment in Job.h). A job type whose declared
// allocation does not cover its real traffic is a modeling bug: fail loudly
// here instead of hanging the run tens of millions of cycles later.
static void check_in_bounds(const Job *j, int n_beats, const char *what) {
  if (j->addr + (uint64_t) n_beats * bytes_per_tx > j->addr_hold + j->alloc_size) {
    throw std::runtime_error(
        "job " + std::to_string(j->job_idx) + " (type " + std::to_string(j->get_type()) +
        ", dims " + j->get_job_dims_string() + ") walked past its allocation on " + what +
        ": alloc_size=" + std::to_string(j->alloc_size) +
        " offset=" + std::to_string(j->addr - j->addr_hold) +
        " beats=" + std::to_string(n_beats));
  }
}

void State::enqueue_writes() {
  // Queue memory write transactions with bandwidth limits
  if (mem_write_left_unqueued > 0) {
    int to_enq = std::min(dram_enq_per_cycle, mem_write_left_unqueued);
    check_in_bounds(j, to_enq, "writes");
    mem_write_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (int i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, true, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::enqueue_reads() {
  // Queue memory read transactions with bandwidth limits
  if (mem_read_left_unqueued > 0) {
    int to_enq = std::min(dram_enq_per_cycle, mem_read_left_unqueued);
    check_in_bounds(j, to_enq, "reads");
    mem_read_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (int i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, false, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::check_idle_from_memory() {
  // Check if unit should be marked idle due to pending memory operations
  if (min_stage_cycles == 0 && !is_idle_from_memory &&
      (mem_read_left > 0 ||
       mem_write_left > 0 ||
       prefetch_read_left > 0)) {
    UPDATE_IDLEMEM(true);
  }
}

bool State::process_stage() {
  // Process current stage: decrement cycle counter and check completion
  if (min_stage_cycles > 0)
    min_stage_cycles--;
  // prefetch_read_left is outside the -dbuf_tile reads_gate on purpose: the
  // credited beats are this job's first-tile operands (see State.h).
  if (min_stage_cycles == 0 && (mem_read_left == 0 || !reads_gate) && mem_write_left == 0 &&
      prefetch_read_left == 0) {
    return true;
  }
  check_idle_from_memory();
  return false;
}

void State::state_transfer(int st, int read_amt_bytes, int write_amt_bytes, int min_cycles) {
  IFVERB(printf("Time(%llu) - Transfer from %s to %s\n", gcycles, to_string(state), to_string(st)));
  UPDATE_STATE(st);
  min_stage_cycles = min_cycles;
  int rmin = read_amt_bytes > 0 ? 1 : 0;
  int wmin = write_amt_bytes > 0 ? 1 : 0;
  if (!hold_reads) {// -dbuf_tile: the next tile's reads are already in flight
    SET_READS(std::max(rmin, read_amt_bytes / bytes_per_tx));
  }
  SET_WRITES(std::max(wmin, write_amt_bytes / bytes_per_tx));
  if (is_idle_from_memory) {
    UPDATE_IDLEMEM(false);
  }
}



static int g_vcd_ctr = 0;

State::State(int memory_priority) {
  core_memory_priority = memory_priority;
  vcd_idx = g_vcd_ctr++;
}

void vcd_stat_init(int vcd_idx, const char *sig_name) {
}
