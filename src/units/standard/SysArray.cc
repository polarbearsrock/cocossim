/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "units/standard/SysArray.h"
#include "State.h"
#include "frontends/standard/StandardArch.h"
#include <cstdint>
#include "global.h"


using namespace frontend::standard;

bool SystolicArray::SysArrayState::increment(const std::function<void(Job *)> &enqueue_job, int &total_idle, int *n_idle_units) {
  auto *sj = (SysArrayJob *) j;
  if (op_stall_left > 0) {// -op_overhead: nothing issues, the unit reports idle
    op_stall_left--;
    return false;
  }
  enqueue_reads();
  enqueue_writes();
  if (process_stage()) {
    if (ws) {  // Weight Stationary mode
      switch (state) {
        case prefetch:  // Load weights into systolic array
          state_transfer(read,
                         0,
                         0,
                         sj->M * std::max(systolic_fpu_latency, batch_size));
          break;
        case read:  // Read input activations
          state_transfer(shift,
                         std::min(sz, sj->K) * std::min(sz, sj->N) * data_type_width,
                         0,
                         sz * std::max(systolic_fpu_latency, batch_size));
          break;
          
        case shift: {  // Compute phase: shift data through systolic array
          int amt_to_write = 0;
          int amt_to_read = 0;
          int n_cycles = 0;
          int activation_preload = 0;
          
          // Check if we're at the end of tile computation
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              amt_to_write = sj->fused_out ? 0 : sj->M * sj->N * data_type_width * batch_size;
            } else if (!sj->act_resident) {
              activation_preload = ws_activation_preload_bytes(sj->M, sj->K, sz);
            }
          }
          amt_to_read = activation_preload;
          n_cycles = 0;
          state_transfer(write, amt_to_read, amt_to_write, n_cycles);
        } break;
        case write: {  // Write output data to memory
          int rd_cycles = sj->M * std::max(systolic_fpu_latency, batch_size);
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job completed
              total_work += (uint64_t) sj->M * sj->K * sj->N;
              state_transfer(idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to next row tile
              j->addr = j->addr_hold;
              state_transfer(read, 0, 0, rd_cycles);
              col_i = 1;
              row_i++;
            }
          } else {
            // Move to next column tile
            state_transfer(read, 0, 0, rd_cycles);
            col_i++;
          }
        } break;
        case idle:
          break;
      }
    } else {  // Output Stationary mode
      switch (state) {
        case read: {  // Read weights and activations
          const int shift_cycles = sz * std::min(systolic_fpu_latency, batch_size);
          state_transfer(shift, 0, 0, shift_cycles);
          // -dbuf_tile: double buffering. This tile's reads have all drained
          // (process_stage), so issue the NEXT tile's reads now and let the
          // shift/write stages run while they stream -- what the hardware's
          // operand double buffers do. The write->read transition then
          // re-arms the gate instead of declaring the reads afresh.
          bool last_tile = (col_i == loop_cols_tiles && row_i == loop_row_tiles);
          if (dbuf_tile && !last_tile) {
            bool new_row = (col_i == loop_cols_tiles);
            // Rewinding here puts THIS tile's write-back (issued at the
            // shift->write transition below) into the next row's epoch;
            // sys_job_alloc_bytes reserves that extra output tile for
            // multi-row jobs under -dbuf_tile (V31c).
            if (new_row) j->addr = j->addr_hold;
            init_row_loop(new_row);          // programs the next tile's read counters
            min_stage_cycles = shift_cycles;  // init_row_loop set the compute length; restore
            hold_reads = true;
            reads_gate = false;
          }
        } break;
        case shift:  // Compute and accumulate outputs
          // Write back this column tile's TRUE output block (spec S4a; the
          // window formula in sys_job_alloc_bytes charges the same helper).
          // The write stage waits for it to land (no output-side double
          // buffering: under -dbuf_tile only the next tile's READS are in
          // flight here, so nothing but those reads and this write-back is
          // ever outstanding when the row cursor rewinds).
          state_transfer(write, 0, sj->fused_out ? 0 : output_tile_bytes(sj->M, sj->N, sz), 0);
          break;
        case write:  // Write partial sums back to memory
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job completed
              total_work += (uint64_t) sj->M * sj->K * sj->N;
              state_transfer(SystolicArray::idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to next row tile
              if (hold_reads) {
                hold_reads = false;
                reads_gate = true;
                min_stage_cycles = div_ru(sj->K, mxu_macs_per_pe);
              } else {
                init_row_loop(true);
                j->addr = j->addr_hold;
              }
              UPDATE_STATE(SystolicArray::read);
              if (is_idle_from_memory) {
                UPDATE_IDLEMEM(false);
              }
              col_i = 1;
              row_i++;
            }
          } else {
            if (hold_reads) {
              hold_reads = false;
              reads_gate = true;
              min_stage_cycles = div_ru(sj->K, mxu_macs_per_pe);
            } else {
              init_row_loop(false);
            }
            UPDATE_STATE(SystolicArray::read);
            if (is_idle_from_memory) {
              UPDATE_IDLEMEM(false);
            }
            col_i++;
          }
          break;
        case idle:
          break;
        default:
          std::cerr << "Caught in unexpected state in Output Stationary Systolic Array..." << std::endl;
          throw std::exception();
      }
    }
  }
  return state != SystolicArray::ExState::idle;
}

void SystolicArray::SysArrayState::init_row_loop(bool new_row) {
  auto sj = (SysArrayJob *) j;

  int n_read_bytes = 0;
  int n_read_beats = 0;
  if (ws) {
    throw std::exception();
  } else {
    // OS throughput: each PE retires mxu_macs_per_pe MACs/cycle, so a K-deep
    // accumulation takes ceil(K / macs) cycles (spec 3.1; v6e uses 2 packed
    // bf16 MACs/PE/cycle). systolic_fpu_latency plays no role in OS timing.
    min_stage_cycles = div_ru(sj->K, mxu_macs_per_pe);
    // Weight/KV block for this column tile: K x min(sz, N). Activation panel
    // (min(sz, M) x K) is re-read only when the row tile advances. Weight
    // charges are skipped while the slice is VMEM-resident: either staged by
    // an earlier job of the same tag, or by this job's own first row pass
    // (init_row_loop(true) runs BEFORE row_i++, hence the new_row term).
    bool skip_weights = weights_resident ||
                        (weights_stay_resident && (new_row || row_i > 1));
    // int64: panel * n_weight_streams (one KV cache per sequence) can pass
    // 2^31 at long-context decode shapes.
    int64_t wb = skip_weights ? 0
               : (int64_t) weight_panel_bytes(sj->K, sj->N, sz, j->batched_weights) * sj->n_weight_streams;
    int64_t rb = wb;
    if (new_row && !sj->act_resident) {
      rb += activation_panel_bytes(sj->M, sj->K, sz);
    }
    n_read_beats = rb > 0 ? (int) std::max<int64_t>(rb / bytes_per_tx, 1) : 0;
    // -dbuf: beats already streamed ahead come off the full-formula count,
    // which always contains floor(wb / bytes_per_tx) -- exact for any
    // byte remainder, never negative.
    if (wb > 0) n_read_beats -= (int) j->take_prefetch_credit_beats(wb / bytes_per_tx);
  }
  mem_read_left = mem_read_left_unqueued = n_read_beats;
}

void SystolicArray::SysArrayState::init() {
  auto sj = (SysArrayJob *) j;
  if (j->is_done) {
    std::cerr << "ERROR" << std::endl;
  }
  if (ws) {
    UPDATE_STATE(SystolicArray::prefetch);
    loop_cols_tiles = div_ru(sj->N, sz);
    loop_row_tiles = div_ru(sj->K, sz);
    int sys_array_preload = std::min(sz, sj->N) * std::min(sz, sj->K) * data_type_width;
    int activation_preload = sj->act_resident ? 0 : ws_activation_preload_bytes(sj->M, sj->K, sz);
    state_transfer(SystolicArray::prefetch, activation_preload + sys_array_preload, 0, sz);
    row_i = 1;
    col_i = 1;
  } else {
    UPDATE_STATE(SystolicArray::read);
    // No `* batch_size` here, matching init_row_loop's div_ru(sj->K,
    // mxu_macs_per_pe): batch_size is a compile-time 1 (global.h), so the two
    // were always numerically equal -- harmonized to stop them drifting if
    // batch_size ever becomes configurable.
    min_stage_cycles = div_ru(sj->K, mxu_macs_per_pe);
    // VMEM residency handoff (spec 6.7): a job whose weight_tag is already
    // staged pays no weight-side HBM traffic; otherwise it fetches, and its
    // slice stays resident for successors iff it fits the VMEM share. An
    // untagged job (-1) always fetches and clears residency.
    weights_resident = vmem_reuse && sj->weight_tag != -1 &&
                       sj->weight_tag == resident_weight_tag;
    // C5v2 (session 3): the compiler re-streams weights past a ~512-row
    // window even when the slice fits VMEM. Cross-JOB reuse is bounded by
    // -vmem_rows; within-job row passes (prefill attention) keep full reuse,
    // a documented bounded optimism confined to attention KV panels.
    if (weights_resident && vmem_resident_rows > 0 &&
        resident_rows_used + sj->M > vmem_resident_rows) {
      weights_resident = false;
    }
    weights_stay_resident = vmem_reuse && sj->weights_fit_vmem;
    resident_weight_tag =
        (weights_stay_resident && sj->weight_tag != -1) ? sj->weight_tag : -1;
    resident_rows_used = weights_resident ? resident_rows_used + sj->M : sj->M;
    int64_t wb = weights_resident ? 0
               : (int64_t) weight_panel_bytes(sj->K, sj->N, sz, j->batched_weights) * sj->n_weight_streams;
    int64_t rb = (sj->act_resident ? 0 : (int64_t) activation_panel_bytes(sj->M, sj->K, sz)) + wb;
    // rb can now be 0 (act_resident + resident weights): charge nothing, like
    // init_row_loop. For rb > 0 the max(.,1) floor is unchanged.
    int n_read_beats = rb > 0 ? (int) std::max<int64_t>(rb / bytes_per_tx, 1) : 0;
    // -dbuf: prefetched beats come off the full-formula count (see
    // init_row_loop for the exactness argument). The first tile may hold
    // credit for the weight panel AND the activation panel (a ready job's
    // activations are staged ahead too); the count always contains both
    // floors, so the deduction stays non-negative.
    int64_t credit_cap = (wb > 0 ? wb / bytes_per_tx : 0) +
                         (sj->act_resident ? 0 : (int64_t) activation_panel_bytes(sj->M, sj->K, sz) / bytes_per_tx);
    if (credit_cap > 0) n_read_beats -= (int) j->take_prefetch_credit_beats(credit_cap);
    // -dbuf_tile bookkeeping never carries across jobs.
    hold_reads = false;
    reads_gate = true;
    mem_read_left = mem_read_left_unqueued = n_read_beats;

    loop_cols_tiles = std::max(sj->N / sz, 1);
    loop_row_tiles = std::max(sj->M / sz, 1);

    row_i = 1;
    col_i = 1;
  }
  if (loop_row_tiles == 0)
    throw std::runtime_error("loop_row_tiles == 0");
  if (loop_cols_tiles == 0)
    throw std::runtime_error("loop_cols_tiles == 0");
}

SystolicArray::SysArrayState::SysArrayState(int sz, bool ws) : State(1), sz(sz), ws(ws), state(SystolicArray::idle) {
  State::sz = sz;
}

SystolicArray::SysArrayJob::SysArrayJob(int m, int k, int n, int sz, bool ws, int n_weight_streams,
                                        bool fused_out, bool act_resident)
    : Job(sys_job_alloc_bytes(m, k, n, sz, ws, /*batched_weights=*/false,
                              n_weight_streams, fused_out, act_resident,
                              /*tile_dbuf=*/dbuf_tile != 0)),
      M(m), K(k), N(n), n_weight_streams(n_weight_streams),
      fused_out(fused_out), act_resident(act_resident), sz_cfg(sz), ws_cfg(ws) {}


std::string SystolicArray::SysArrayJob::get_job_dims_string() const {
  return std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N);
}