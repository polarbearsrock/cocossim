/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_MEMORY_H
#define PROSE_COMPILER_MEMORY_H
#include "State.h"
#include "memory_system.h"
#include <deque>


namespace mem {
  bool try_enqueue_tx();
  using mem_ty = dramsim3::JedecDRAMSystem;

  extern dramsim3::Config *dramsim3config;
  // One queued credit per in-flight transaction: DRAMSim3 fires one callback
  // per AddTransaction, and same-address requests each get their own callback,
  // so credits must queue per address (FIFO) — a single State* entry would be
  // overwritten by a same-address re-issue and orphan the first requester
  // (hangs the run). Do not simplify back to State*.
  extern std::unordered_map<uint64_t, std::deque<State *>> address_reads_bkwds_lookup, address_writes_bkwds_lookup;
  // Owner of each in-flight -dbuf prefetch read (the reads lookup carries a
  // nullptr for them). One entry per address suffices: a job's prefetch
  // cursor walks its own window once, and windows never overlap. The read
  // callback pops it to count the landing on the job (Job::prefetch_landed_
  // beats) and, once the job is dispatched, on its state's prefetch_read_left.
  extern std::unordered_map<uint64_t, Job *> prefetch_owner;
  extern mem_ty *mem_sys;

  void setup();
};// namespace mem
#endif//PROSE_COMPILER_MEMORY_H
