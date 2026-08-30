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
    const uint64_t n_passes = (is_prebuffered ? 0u : (uint64_t) n_read_operands) + 1u;
    return tensor * n_passes;
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
