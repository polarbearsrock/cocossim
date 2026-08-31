/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_SYSARRAY_H
#define PERF_MODEL_SYSARRAY_H
#include "State.h"
#include "frontends/standard/StandardUnits.h"
#include <algorithm>

namespace SystolicArray {
  // ---------------------------------------------------------------------
  // Per-phase memory demand. These are the single source of truth for how
  // many bytes each stage of the state machine moves: SysArrayState::init /
  // init_row_loop / increment call them to program the memory counters, and
  // sys_job_alloc_bytes calls them to size the job's address window. Keeping
  // both users on the same helpers is what stops the window and the traffic
  // from drifting apart -- the drift that produced the multi-core livelock
  // (see Job::alloc_size).
  // ---------------------------------------------------------------------

  // Weight/KV block streamed for one column tile.
  inline int weight_panel_bytes(int K, int N, int sz, bool batched_weights) {
    return std::min(sz, N) * K * (batched_weights ? batch_size : 1) * data_type_width;
  }
  // Activation panel, re-read whenever the row tile advances (output-stationary).
  inline int activation_panel_bytes(int M, int K, int sz) {
    return std::min(sz, M) * K * batch_size * data_type_width;
  }
  // Weight-stationary streams the activations the other way round: a
  // min(sz,K)-deep panel spanning all M rows.
  inline int ws_activation_preload_bytes(int M, int K, int sz) {
    return std::min(sz, K) * M * batch_size * data_type_width;
  }
  // Bytes -> beats, matching the state machine's own rounding (a non-zero
  // demand always costs at least one beat).
  inline int demand_beats(int bytes) {
    return std::max(bytes / bytes_per_tx, 1);
  }

  // Size of the address window a SysArrayJob must own.
  //
  // The cursor rewinds to addr_hold every time the row tile advances, so the
  // window only has to cover the widest single row-tile pass, not the whole
  // job. Both modes are bounded by their first pass plus one full sweep of
  // column tiles plus whichever end-of-sweep transfer is larger.
  inline uint64_t sys_job_alloc_bytes(int M, int K, int N, int sz, bool ws,
                                      bool batched_weights, int beats_per_wb,
                                      int64_t n_weight_streams = 1) {
    // Weight-side reads scale with n_weight_streams (per-sequence KV caches);
    // the window must cover the scaled walk, mirroring the charging math.
    const uint64_t weight_beats = std::max<int64_t>(
        (int64_t) weight_panel_bytes(K, N, sz, batched_weights) * n_weight_streams / bytes_per_tx, 1);
    uint64_t beats;
    if (ws) {
      const int cols = div_ru(N, sz);
      // prefetch pulls the activation panel and the array tile together; each
      // column tile then re-reads the array tile; the last tile of the last
      // row writes the whole M x N output.
      const uint64_t array_tile_beats =
          demand_beats(std::min(sz, N) * std::min(sz, K) * data_type_width);
      const uint64_t preload_beats =
          demand_beats(ws_activation_preload_bytes(M, K, sz) +
                       std::min(sz, N) * std::min(sz, K) * data_type_width);
      const uint64_t out_beats = demand_beats(M * N * data_type_width * batch_size);
      const uint64_t ws_act_beats = demand_beats(ws_activation_preload_bytes(M, K, sz));
      beats = preload_beats + (uint64_t) cols * array_tile_beats +
              std::max(ws_act_beats, out_beats);
    } else {
      const int cols = std::max(N / sz, 1);
      // First column tile of a row tile reads activations + weights; the rest
      // read weights only. Every column tile writes back beats_per_wb (which
      // the state machine passes through state_transfer's byte argument, so it
      // is divided by bytes_per_tx again -- mirrored here on purpose).
      beats = demand_beats(activation_panel_bytes(M, K, sz)) +
              (uint64_t) cols * weight_beats +
              (uint64_t) cols * demand_beats(beats_per_wb);
    }
    return beats * (uint64_t) bytes_per_tx;
  }

  struct SysArrayJob : Job {

    int get_type() const override {
      return SYSTOLIC_ARRAY_IDX;
    }
    int M, K, N;

    // VMEM staging identity (spec 6.7). Jobs sharing a weight_tag read the
    // same weight tensor (e.g. row-block jobs of one GEMM); the executing
    // state skips their weight-side HBM charges while the tag stays resident.
    // -1 = untagged: always refetch and clear residency (safe default).
    // weights_fit_vmem is decided at job creation, where the layer knows the
    // slice size and how many cores' slices must co-reside.
    int weight_tag = -1;
    bool weights_fit_vmem = false;
    // Distinct copies of the weight-side operand this job streams (spec 6.7).
    // 1 for true weights (shared across all rows). Decode attention sets it
    // to the batch: each sequence has its OWN KV cache, so the "weight" side
    // is per-row state and must be read once per sequence.
    int n_weight_streams = 1;

    SysArrayJob(int m, int k, int n, int sz, bool ws, int n_weight_streams = 1);

    [[nodiscard]] std::string get_job_dims_string() const override;
  };

  enum ExState {
    idle = 0,
    prefetch,//only WS
    read,
    shift,
    write
  };

  struct SysArrayState : State {
public:
    explicit SysArrayState(int sz, bool ws);
    bool ws = false;// Flag for weight-stationary systolic array

    void init_row_loop(bool new_row);
    void init() override;
    int sz;
    ExState state = idle;

    // VMEM residency (spec 6.7): the weight_tag whose slice this MXU's VMEM
    // share currently stages (-1 = none). Survives across jobs; a job whose
    // tag matches pays no weight-side HBM traffic. The two bools cache this
    // job's residency decision at init() so tile advances don't re-derive it.
    int resident_weight_tag = -1;
    bool weights_resident = false;
    bool weights_stay_resident = false;

    bool increment(const std::function<void(Job *)> &enqueue_job,
                   int &total_idle,
                   int *n_idle_units) override;

    void set_state(int st) override {
      state = (ExState) st;
    };

    int get_state() override {
      return state;
    }
    

    int get_ty_idx() override {
      return SYSTOLIC_ARRAY_IDX;
    }
    std::string get_ty_string() override {
      return SYSTOLIC_ARRAY_STRING;
    }

    bool is_underfilled() const override {
      if (j == nullptr) return false;
      auto *sj = (SysArrayJob *) j;
      return std::min(sz, sj->M) * std::min(sz, sj->N) < sz * sz;
    }

private:
    int beats_per_wb;
  };

};// namespace SystolicArray


#endif//PERF_MODEL_SYSARRAY_H
