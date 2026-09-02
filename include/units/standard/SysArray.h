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
  // Output block one output-stationary column tile writes back: the
  // min(sz,M) x min(sz,N) accumulator tile, at true bytes (spec S4a). This
  // used to be a BEAT count (sz*sz*dtw/bytes_per_tx) passed through
  // state_transfer's byte argument, i.e. charged at 1/bytes_per_tx of the
  // truth; V31a pins the corrected total.
  inline int output_tile_bytes(int M, int N, int sz) {
    return std::min(sz, M) * std::min(sz, N) * batch_size * data_type_width;
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
  // fused_out / act_resident are the fusion flags (-fuse_attn, spec 6.7): a
  // fused_out job's output stays in VMEM (no write-back beats), an
  // act_resident job's activation operand is already there (no activation-
  // panel reads). Both must shrink the window exactly as they shrink the
  // charges below, or window and walk drift apart (the multi-core livelock).
  // tile_dbuf (-dbuf_tile, OS only): the read stage of a row's last column
  // tile rewinds the cursor and pre-issues the next row's reads BEFORE that
  // tile's write-back is issued, so one output tile of every row but the
  // last lands in the FOLLOWING row's epoch -- the last epoch is then one
  // output tile wider than the first pass (V31c derives the epochs).
  inline uint64_t sys_job_alloc_bytes(int M, int K, int N, int sz, bool ws,
                                      bool batched_weights,
                                      int64_t n_weight_streams = 1,
                                      bool fused_out = false,
                                      bool act_resident = false,
                                      bool tile_dbuf = false) {
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
          demand_beats((act_resident ? 0 : ws_activation_preload_bytes(M, K, sz)) +
                       std::min(sz, N) * std::min(sz, K) * data_type_width);
      // Suppressed transfers are 0 beats outright (state_transfer charges
      // nothing for 0 bytes), so no demand_beats floor here.
      const uint64_t out_beats =
          fused_out ? 0 : demand_beats(M * N * data_type_width * batch_size);
      const uint64_t ws_act_beats =
          act_resident ? 0 : demand_beats(ws_activation_preload_bytes(M, K, sz));
      beats = preload_beats + (uint64_t) cols * array_tile_beats +
              std::max(ws_act_beats, out_beats);
    } else {
      const int cols = std::max(N / sz, 1);
      // The first column tile of a row-tile pass reads activations + weights
      // TOGETHER, floored ONCE -- mirroring SysArrayState::init's combined
      // `rb` (activation_panel_bytes + weight_panel_bytes*n_weight_streams,
      // one division) and init_row_loop's `new_row` branch, which charges
      // every later row-tile pass the identical combined way after the
      // cursor rewinds to addr_hold (so the first pass computed here, with no
      // VMEM residency assumed, is the widest one -- residency only removes
      // weight-side reads, never adds any). Flooring the two parts
      // SEPARATELY, as this used to, undercounts by one beat whenever their
      // byte remainders sum to >= bytes_per_tx, and the walk then exceeds
      // this window. The remaining (cols-1) column tiles of the pass read
      // weights alone (init_row_loop's non-new_row branch), each floored on
      // its own -- that part was already right. Every column tile writes
      // back its true output block (output_tile_bytes, the same helper the
      // shift stage charges), floored to one beat like state_transfer does.
      // Under -dbuf_tile a multi-row job's last epoch also carries the
      // previous row's last write-back (see tile_dbuf above): reserve one
      // more output tile then. Epochs, R rows > 1, C cols, W = out_beats:
      //   1:      combined + (C-1) wgt + (C-1) W
      //   2..R-1: act(+wgt) + W + (C-1) wgt + (C-1) W  <= first pass
      //   R:      act(+wgt) + W + (C-1) wgt + C W      <= first pass + W
      // -dbuf_tile 0 and fused_out reserve nothing extra (bit-identical
      // address layout for the baseline).
      const int rows = std::max(M / sz, 1);
      const int64_t combined_first_tile_bytes =
          (act_resident ? 0 : (int64_t) activation_panel_bytes(M, K, sz)) +
          (int64_t) weight_panel_bytes(K, N, sz, batched_weights) * n_weight_streams;
      const uint64_t combined_first_tile_beats =
          std::max<int64_t>(combined_first_tile_bytes / bytes_per_tx, 1);
      const uint64_t out_beats = fused_out ? 0 : demand_beats(output_tile_bytes(M, N, sz));
      beats = combined_first_tile_beats +
              (uint64_t) (cols - 1) * weight_beats +
              (uint64_t) cols * out_beats +
              ((tile_dbuf && rows > 1) ? out_beats : 0);
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
    // Fusion flags (-fuse_attn, spec 6.7). Const because the base Job
    // allocation is sized from them at construction (sys_job_alloc_bytes):
    // flipping them afterwards would let the walk and the window drift.
    // fused_out: the output is consumed on-chip, no write-back charges
    // (QK^T scores feeding the fused softmax). act_resident: the activation
    // operand is already on-chip, no activation-panel reads (AV jobs whose
    // A matrix is the softmaxed scores).
    const bool fused_out;
    const bool act_resident;
    // Geometry the job was built for, kept for prefetchable_weight_bytes
    // (the executing state has its own copies; these must match it).
    const int sz_cfg;
    const bool ws_cfg;

    SysArrayJob(int m, int k, int n, int sz, bool ws, int n_weight_streams = 1,
                bool fused_out = false, bool act_resident = false);

    // Cross-op weight prefetch (-dbuf): one full column-tile sweep of the
    // weight side in whole beats per tile -- exactly what init() +
    // init_row_loop() deduct over a no-residency pass, so the credit
    // consumes to zero, and never more than sys_job_alloc_bytes reserves
    // (each tile's window term is >= floor(panel / bytes_per_tx)). OS only.
    [[nodiscard]] int64_t prefetchable_weight_beats() const override {
      // kv_stream: a paged-attention kernel gathers its own KV blocks; XLA
      // cannot stream them ahead of the kernel (spec S6).
      if (ws_cfg || weight_tag == -1 || kv_stream) return 0;
      int cols = std::max(N / sz_cfg, 1);
      int64_t panel = (int64_t) weight_panel_bytes(K, N, sz_cfg, batched_weights) * n_weight_streams;
      return (int64_t) cols * (panel / bytes_per_tx);
    }
    [[nodiscard]] int64_t prefetchable_act_beats() const override {
      // kv_stream jobs keep this: their Q rows are an ordinary just-computed
      // activation; only the KV (weight-side) gather is un-prefetchable.
      if (ws_cfg || act_resident) return 0;
      return (int64_t) activation_panel_bytes(M, K, sz_cfg) / bytes_per_tx;
    }
    [[nodiscard]] int prefetch_tag() const override { return weight_tag; }
    [[nodiscard]] int prefetch_rows() const override { return M; }
    [[nodiscard]] bool prefetch_fits_vmem() const override { return weights_fit_vmem; }

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
    // Rows consumed under the resident tag (C5v2 window, -vmem_rows).
    int resident_rows_used = 0;

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
  };

};// namespace SystolicArray


#endif//PERF_MODEL_SYSARRAY_H
