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
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "global.h"


using namespace frontend::standard;

namespace {
int tile_extent(int total, int tile_size, int one_based_tile) {
  if (total <= 0 || tile_size <= 0 || one_based_tile <= 0) {
    throw std::invalid_argument("tile dimensions and indices must be positive");
  }
  const auto offset = checked_product({static_cast<uint64_t>(one_based_tile - 1),
                                       static_cast<uint64_t>(tile_size)});
  if (offset >= static_cast<uint64_t>(total)) {
    throw std::out_of_range("tile index exceeds tensor dimension");
  }
  return static_cast<int>(std::min<uint64_t>(tile_size,
                                             static_cast<uint64_t>(total) - offset));
}

uint64_t checked_add(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    throw std::overflow_error("byte count exceeds 64-bit range");
  }
  return lhs + rhs;
}

uint64_t tensor_bytes(std::initializer_list<uint64_t> dimensions) {
  return bytes_for_elements(checked_product(dimensions));
}

uint64_t sys_array_allocation_bytes(int m, int k, int n) {
  if (m <= 0 || k <= 0 || n <= 0) {
    throw std::invalid_argument("systolic-array job dimensions must be positive");
  }
  const auto activations = tensor_bytes({static_cast<uint64_t>(m),
                                         static_cast<uint64_t>(k),
                                         static_cast<uint64_t>(batch_size)});
  const auto weights = tensor_bytes({static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(n)});
  const auto outputs = tensor_bytes({static_cast<uint64_t>(m),
                                     static_cast<uint64_t>(n),
                                     static_cast<uint64_t>(batch_size)});
  return checked_add(checked_add(activations, weights), outputs);
}
}

bool SystolicArray::SysArrayState::increment(const std::function<void(Job *)> &enqueue_job, int &total_idle, int *n_idle_units) {
  auto *sj = (SysArrayJob *) j;
  enqueue_reads();
  enqueue_writes();
  if (process_stage()) {
    if (ws) {  // Weight Stationary mode
      switch (state) {
        case prefetch:  // Load weights into systolic array
          state_transfer(read,
                         0,
                         0,
                         checked_product({static_cast<uint64_t>(sj->M),
                                          static_cast<uint64_t>(std::max(systolic_fpu_latency, batch_size))}));
          break;
        case read: {  // Read the current weight tile
          const int k_tile = tile_extent(sj->K, sz, row_i);
          const int n_tile = tile_extent(sj->N, sz, col_i);
          const uint64_t weight_copies = j->batched_weights ? batch_size : 1;
          state_transfer(shift,
                         tensor_bytes({static_cast<uint64_t>(k_tile),
                                       static_cast<uint64_t>(n_tile),
                                       weight_copies}),
                         0,
                         checked_product({static_cast<uint64_t>(std::max(k_tile, n_tile)),
                                          static_cast<uint64_t>(std::max(systolic_fpu_latency, batch_size))}));
        } break;
          
        case shift: {  // Compute phase: shift data through systolic array
          uint64_t amt_to_write = 0;
          uint64_t amt_to_read = 0;
          uint64_t activation_preload = 0;
          
          // Check if we're at the end of tile computation
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              amt_to_write = tensor_bytes({static_cast<uint64_t>(sj->M),
                                            static_cast<uint64_t>(sj->N),
                                            static_cast<uint64_t>(batch_size)});
            } else {
              const int next_k_tile = tile_extent(sj->K, sz, row_i + 1);
              activation_preload = tensor_bytes({static_cast<uint64_t>(next_k_tile),
                                                  static_cast<uint64_t>(sj->M),
                                                  static_cast<uint64_t>(batch_size)});
            }
          }
          amt_to_read = activation_preload;
          state_transfer(write, amt_to_read, amt_to_write, 0);
        } break;
        case write: {  // Write output data to memory
          const auto rd_cycles = checked_product(
              {static_cast<uint64_t>(sj->M),
               static_cast<uint64_t>(std::max(systolic_fpu_latency, batch_size))});
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job completed
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
          const int row_extent = tile_extent(sj->M, sz, row_i);
          const int col_extent = tile_extent(sj->N, sz, col_i);
          state_transfer(shift, 0, 0,
                         checked_product({static_cast<uint64_t>(std::max(row_extent, col_extent)),
                                          static_cast<uint64_t>(std::min(systolic_fpu_latency, batch_size))}));
        } break;
        case shift:  // Compute and accumulate outputs
          state_transfer(write, 0,
                         tensor_bytes({static_cast<uint64_t>(tile_extent(sj->M, sz, row_i)),
                                       static_cast<uint64_t>(tile_extent(sj->N, sz, col_i)),
                                       static_cast<uint64_t>(batch_size)}),
                         0);
          break;
        case write:  // Write partial sums back to memory
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job completed
              state_transfer(SystolicArray::idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to the first column of the next row tile.
              row_i++;
              col_i = 1;
              j->addr = j->addr_hold;
              init_row_loop(true);
              UPDATE_STATE(SystolicArray::read);
              if (is_idle_from_memory) {
                UPDATE_IDLEMEM(false);
              }
            }
          } else {
            col_i++;
            init_row_loop(false);
            UPDATE_STATE(SystolicArray::read);
            if (is_idle_from_memory) {
              UPDATE_IDLEMEM(false);
            }
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

  if (ws) {
    throw std::exception();
  } else {
    const int row_extent = tile_extent(sj->M, sz, row_i);
    const int col_extent = tile_extent(sj->N, sz, col_i);
    min_stage_cycles = checked_product({static_cast<uint64_t>(sj->K),
                                        static_cast<uint64_t>(std::max(systolic_fpu_latency, batch_size))});
    uint64_t n_read_bytes = tensor_bytes({static_cast<uint64_t>(sj->K),
                                          static_cast<uint64_t>(col_extent),
                                          static_cast<uint64_t>(j->batched_weights ? batch_size : 1)});
    if (new_row) {
      n_read_bytes = checked_add(
          n_read_bytes,
          tensor_bytes({static_cast<uint64_t>(row_extent),
                        static_cast<uint64_t>(sj->K),
                        static_cast<uint64_t>(batch_size)}));
    }
    const uint64_t n_read_beats = compute_only
                                      ? 0
                                      : div_ru(n_read_bytes, static_cast<uint64_t>(bytes_per_tx));
    mem_read_left = mem_read_left_unqueued = n_read_beats;
  }
}

void SystolicArray::SysArrayState::init() {
  auto sj = (SysArrayJob *) j;
  if (j->is_done) {
    std::cerr << "ERROR" << std::endl;
  }
  if (ws) {
    UPDATE_STATE(SystolicArray::prefetch);
    loop_cols_tiles = static_cast<int>(div_ru(static_cast<uint64_t>(sj->N), static_cast<uint64_t>(sz)));
    loop_row_tiles = static_cast<int>(div_ru(static_cast<uint64_t>(sj->K), static_cast<uint64_t>(sz)));
    row_i = 1;
    col_i = 1;
    const int k_tile = tile_extent(sj->K, sz, row_i);
    const int n_tile = tile_extent(sj->N, sz, col_i);
    const auto activation_preload = tensor_bytes(
        {static_cast<uint64_t>(k_tile), static_cast<uint64_t>(sj->M),
         static_cast<uint64_t>(batch_size)});
    state_transfer(SystolicArray::prefetch,
                   activation_preload, 0,
                   static_cast<uint64_t>(std::max(k_tile, n_tile)));
  } else {
    UPDATE_STATE(SystolicArray::read);
    loop_cols_tiles = static_cast<int>(div_ru(static_cast<uint64_t>(sj->N), static_cast<uint64_t>(sz)));
    loop_row_tiles = static_cast<int>(div_ru(static_cast<uint64_t>(sj->M), static_cast<uint64_t>(sz)));
    row_i = 1;
    col_i = 1;
    init_row_loop(true);
  }
  if (loop_row_tiles == 0)
    throw std::runtime_error("loop_row_tiles == 0");
  if (loop_cols_tiles == 0)
    throw std::runtime_error("loop_cols_tiles == 0");
}

SystolicArray::SysArrayState::SysArrayState(int sz, bool ws) : State(1), sz(sz), ws(ws), state(SystolicArray::idle) {
  if (sz <= 0) {
    throw std::invalid_argument("systolic-array size must be positive");
  }
  beats_per_wb = 0;
}

SystolicArray::SysArrayJob::SysArrayJob(int m, int k, int n)
    : Job(sys_array_allocation_bytes(m, k, n)), M(m), K(k), N(n) {}


std::string SystolicArray::SysArrayJob::get_job_dims_string() const {
  return std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N);
}
