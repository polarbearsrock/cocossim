/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_VECTORUNIT_H
#define PERF_MODEL_VECTORUNIT_H
#include "State.h"

#include <algorithm>
#include <cstdint>
#include "frontends/standard/StandardUnits.h"
#include <queue>

namespace VectorUnit {
  enum VPUState {
    idle = 0,
    unbuffered_lin,
    unbuffered_par,
    buffered_lin,
    buffered_par,
    write
  };
  enum VPUPhase {
    REDUCE,
    BROADCAST
  };

  class VecUnitState : public State {
public:
    explicit VecUnitState(int sz);

    void init() override;
    int sz;
    VectorUnit::VPUState state = VectorUnit::idle;

    bool increment(const std::function<void(Job *)> &enqueue_job,
                   int &total_idle,
                   int *n_idle_units) override;

    void set_state(int st) override {
      state = (VectorUnit::VPUState) st;
    }
    int get_state() override {
      return (int) state;
    }

    int get_ty_idx() override {
      return VECTOR_UNIT_IDX;
    }

    std::string get_ty_string() override {
      return VECTOR_UNIT_STRING;
    }

    bool is_underfilled() const override;

private:
    uint8_t idx;
    int beats_per_wb;
  };


  // Bytes a VecUnitJob touches over its whole life, and therefore the size of
  // the address window it must own. VecUnitState::init streams n_read_operands
  // copies of the tensor (none when the inputs are already buffered) and the
  // write phase streams one more; sizing the allocation for the reads alone --
  // as this did before -- pushed every job's write pass onto the next job's
  // input range, which livelocks multi-core runs (see Job::alloc_size).
  inline uint64_t vec_job_alloc_bytes(int lin, int par, bool is_prebuffered, int n_read_operands) {
    const uint64_t tensor = (uint64_t) lin * par * data_type_width * batch_size;
    // Reads and writes are each quantized to whole bytes_per_tx-byte beats
    // with a 1-beat floor per non-zero transfer -- State::state_transfer's
    // SET_READS/SET_WRITES macros (SET_READS(x) = max(read_amt>0?1:0,
    // read_amt/bytes_per_tx), symmetric for writes) never charge less than
    // one beat for a transfer that has any bytes at all. Sizing the window
    // in raw bytes, as this did before, undercounts any tensor smaller than
    // one beat: two non-zero transfers (a read pass + the write pass) then
    // cost at least 2*bytes_per_tx bytes of window, not `tensor`.
    //
    // Derived directly from the two charging sites in VectorUnit.cc:
    //  - VecUnitState::init's first_phase_read = tensor * n_read_operands,
    //    passed to state_transfer as the read amount; 0 when is_prebuffered
    //    (the init() branch sets first_phase_read = 0 in that case), so no
    //    read beats are ever charged.
    //  - The write, issued once all REDUCE/BROADCAST phases are consumed:
    //    state_transfer(write, 0, lin*par*data_type_width*batch_size, 0) --
    //    always exactly `tensor` bytes, one copy of the output, regardless
    //    of n_read_operands or is_prebuffered.
    const uint64_t read_beats =
        is_prebuffered ? 0u
                       : std::max<uint64_t>(tensor * (uint64_t) n_read_operands / bytes_per_tx, 1u);
    const uint64_t write_beats = std::max<uint64_t>(tensor / bytes_per_tx, 1u);
    return (read_beats + write_beats) * (uint64_t) bytes_per_tx;
  }

  struct VecUnitJob : public Job {
    int linearized_dimension;
    int parallel_dimension;
    std::queue<std::pair<VPUPhase, int>> phases;
    bool is_prebuffered;
    int op_latency = 1;
    // Number of input tensors this job streams from memory (1 = unary,
    // 2 = binary elementwise such as residual add). Scales unbuffered reads.
    // Const because the base Job allocation is sized from it at construction
    // (see vec_job_alloc_bytes): raising it afterwards would make the job walk
    // past its own address window and into the next job's.
    const int n_read_operands;

    [[nodiscard]] std::string get_job_dims_string() const override;
    VecUnitJob(int linearizedDimension, int parallelDimension, bool is_prebuffered, const std::queue<std::pair<VPUPhase, int>> &phases, int n_read_operands = 1);
    VecUnitJob(int linearizedDimension, int parallelDimension, bool is_prebuffered, const std::vector<std::pair<VPUPhase, int>> &phases, int n_read_operands = 1);

    int get_type() const override;
  };

  // Approximation: a REDUCE/BROADCAST pass with fewer parallel rows than
  // lanes leaves lanes idle; finer per-phase modeling is not needed for
  // the paper's per-unit attribution.
  inline bool VecUnitState::is_underfilled() const {
    if (j == nullptr) return false;
    return ((VecUnitJob *) j)->parallel_dimension < sz;
  }

};// namespace VectorUnit


#endif//PERF_MODEL_VECTORUNIT_H
