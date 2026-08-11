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

void State::enqueue_writes() {
  // Queue memory write transactions with bandwidth limits
  if (mem_write_left_unqueued > 0) {
    const auto to_enq = std::min<uint64_t>(dram_enq_per_cycle, mem_write_left_unqueued);
    mem_write_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (uint64_t i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, true, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::enqueue_reads() {
  // Queue memory read transactions with bandwidth limits
  if (mem_read_left_unqueued > 0) {
    const auto to_enq = std::min<uint64_t>(dram_enq_per_cycle, mem_read_left_unqueued);
    mem_read_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (uint64_t i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, false, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::check_idle_from_memory() {
  // Check if unit should be marked idle due to pending memory operations
  if (min_stage_cycles == 0 && !is_idle_from_memory &&
      (mem_read_left > 0 ||
       mem_write_left > 0)) {
    UPDATE_IDLEMEM(true);
  }
}

bool State::process_stage() {
  // Process current stage: decrement cycle counter and check completion
  if (min_stage_cycles > 0)
    min_stage_cycles--;
  if (min_stage_cycles == 0 && mem_read_left == 0 && mem_write_left == 0) {
    return true;
  }
  check_idle_from_memory();
  return false;
}

void State::state_transfer(int st, uint64_t read_amt_bytes, uint64_t write_amt_bytes, uint64_t min_cycles) {
  IFVERB(printf("Time(%llu) - Transfer from %s to %s\n", gcycles, to_string(state), to_string(st)));
  UPDATE_STATE(st);
  min_stage_cycles = min_cycles;
  if (compute_only) {
    SET_READS(0);
    SET_WRITES(0);
  } else {
    SET_READS(div_ru(read_amt_bytes, static_cast<uint64_t>(bytes_per_tx)));
    SET_WRITES(div_ru(write_amt_bytes, static_cast<uint64_t>(bytes_per_tx)));
  }
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
