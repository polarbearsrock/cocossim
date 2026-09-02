/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_JOB_H
#define PROSE_COMPILER_JOB_H

#include <cstdint>
#include <iostream>
#include <vector>

extern uint64_t alloc_addr;
struct State;

// Op class of a job for per-class utilization accounting (benchmark spec
// S1): the Transformer composite tags every job it creates at construction
// so Arch's per-cycle busy/underfilled/memstall classifier can be split the
// way XProf reports per-op utilization (SCHEMA 3 'ACCTC' lines). Composites
// other than Transformer leave the default, OP_OTHER. Names in
// OP_CLASS_NAMES (Job.cc) are the stats-file spelling.
enum OpClass {
  OP_OTHER = 0,
  OP_QKV,     // q/k/v projections
  OP_O,       // attention output projection
  OP_GATE_UP, // MLP gate and up projections
  OP_DOWN,    // MLP down projection
  OP_HEAD,    // LM head (unembedding) GEMM
  OP_ATTN,    // QK^T scores, softmax chunks, AV
  OP_VPU_NORM,// norm1 / norm2 / final_norm
  OP_VPU_EW,  // rope, silu_mul, residual adds, logits softmax
  N_OP_CLASSES
};
extern const char *const OP_CLASS_NAMES[N_OP_CLASSES];

struct Job {
  [[nodiscard]] virtual int get_type() const = 0;
  bool batched_weights = false;
  int op_class = OP_OTHER;
  uint64_t addr;
  const uint64_t addr_hold;
  // Bytes reserved for this job at [addr_hold, addr_hold + alloc_size). The
  // address cursor `addr` must never leave that window: jobs are bump-allocated
  // back to back, so a job that walks past its own allocation lands on the next
  // job's, and two units running those jobs concurrently then issue a read and a
  // write to one address. DRAMSim3's controller deadlocks on exactly that -- a
  // full write buffer re-arms write_draining_ every tick, so the read queue is
  // never scheduled, while the head write is held back forever by the R->W
  // dependency check against the read that can no longer issue. State's enqueue
  // paths enforce the bound.
  const uint64_t alloc_size;
  int task_idx;
  int core_id = -1;  // Core ID for parallel scheduling (-1 = any core)
  int job_idx;

  std::vector<Job *> children;

  int rem_deps;
  bool is_done = false;
  // Cross-op weight prefetch (-dbuf, spec 6.7). The prefetcher (Arch.cc)
  // streams a future job's weight sweep into otherwise-idle DRAM slots and
  // records the BEATS here; the executing state deducts them from its own
  // full-formula read charge, so every prefetched beat replaces a demand
  // beat 1:1 and total traffic is invariant for any byte remainder (the
  // credit is bounded by floor(weight_bytes / bytes_per_tx) per tile, which
  // the full-formula count always contains). `started` stops further issue
  // the moment the job is dispatched.
  int64_t prefetch_credit_beats = 0;
  bool started = false;
  // Landing bookkeeping for those beats (S4 fix round 2). The credit is
  // booked at ISSUE, but the prefetched addresses sit at the front of this
  // job's window, and the job's own write-back -- which starts right behind
  // its (credit-reduced) demand reads -- lands inside that span whenever the
  // credit exceeds the remaining demand. A write to an address whose read is
  // still pending is the DRAMSim3 deadlock described at alloc_size. So every
  // prefetch beat keeps its owner (mem::prefetch_owner), the DRAM callback
  // counts it here, and at dispatch the issued-but-unlanded remainder becomes
  // the executing state's prefetch_read_left (Arch.cc), which process_stage
  // waits on like demand reads: the unit cannot compute on data that has not
  // arrived, and no write is issued before the last prefetched beat landed.
  // Traffic is untouched -- the credit still replaces demand beats 1:1.
  int64_t prefetch_issued_beats = 0;
  int64_t prefetch_landed_beats = 0;
  State *exec_state = nullptr;// set at dispatch, after prefetch_read_left is programmed
  // Whole beats of weight-side traffic the prefetcher may stream ahead of
  // dispatch: per column tile floor(panel / bytes_per_tx), summed -- exactly
  // what the charge sites deduct, and never more than the job's address
  // window covers. 0 = not prefetchable.
  [[nodiscard]] virtual int64_t prefetchable_weight_beats() const { return 0; }
  // Whole beats of the first activation panel, streamable once the job is
  // READY (rem_deps == 0: its producers are done, so the data exists).
  [[nodiscard]] virtual int64_t prefetchable_act_beats() const { return 0; }
  [[nodiscard]] virtual int prefetch_tag() const { return -1; }
  // Inputs the prefetcher's residency replay needs (Arch.cc): the rows this
  // job consumes under its tag's residency window (also the "is an SA job"
  // predicate: > 0), and whether its slice stays resident. Mirror
  // SysArrayState::init's decision inputs.
  [[nodiscard]] virtual int prefetch_rows() const { return 0; }
  [[nodiscard]] virtual bool prefetch_fits_vmem() const { return false; }
  // Consume up to `want` beats of credit; returns the beats consumed.
  int64_t take_prefetch_credit_beats(int64_t want);
  Job(uint64_t alloc_size);

  void add_child(Job *j) {
    children.push_back(j);
    j->rem_deps += 1;
  }

  void reset() {
    addr = addr_hold;
    for (auto *child: children) {
      child->reset();
    }
  }

  virtual std::string get_job_dims_string() const = 0;
  void printDetails() const {
    std::cout << "Job Type: " << get_type()
              << ", Dims: " << get_job_dims_string()
              << ", Address: 0x" << std::hex << addr
              << ", Task Index: " << std::dec << task_idx
              << ", Core ID: " << core_id
              << ", Remaining Dependencies: " << rem_deps
              << ", Children Count: " << children.size() << std::endl;
  }
};

using JobList = std::vector<Job *>;
using JobPair = std::pair<JobList, JobList>;

void jobs_to_dot(std::vector<Job *> &jobs, const std::string &fname = "jobs.dot");

#endif//PROSE_COMPILER_JOB_H
