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
#include "buffers/BufferHierarchy.h"
#include <cstdint>
#include "global.h"


using namespace frontend::standard;

/**
 * Weight Stationary (WS) Dataflow for Matmul M×K×N:
 *
 * Tiling: K into loop_row_tiles, N into loop_cols_tiles
 *
 * For each K-tile (row_i):
 *   1. Load activations[M × tile_k] from HBM → Buffer
 *   2. For each N-tile (col_i):
 *      a. Load weights[tile_k × tile_n] from HBM → Buffer
 *      b. Stream activations through array (REUSE from buffer)
 *      c. Accumulate partial outputs
 *   3. After last N-tile, activations can be evicted
 *
 * Output: Write output[M × N] to HBM at the end
 *
 * Key insight: Activations are reused across all N-tiles within same K-tile
 */

bool SystolicArray::SysArrayState::increment(const std::function<void(Job *)> &enqueue_job, int &total_idle, int *n_idle_units) {
  auto *sj = (SysArrayJob *) j;
  enqueue_reads();
  enqueue_writes();

  if (process_stage()) {
    if (ws) {  // Weight Stationary mode
      switch (state) {
        case prefetch: {
          // Prefetch complete - weights and first activations loaded
          // Now ready to start computation
          state_transfer(read, 0, 0, sj->M * std::max(systolic_fpu_latency, batch_size));
        } break;

        case read: {
          // Read phase complete - begin shift/compute
          // For WS: activations stream through, weights stay stationary
          int tile_k = std::min(sz, sj->K);

          // Record buffer read for activations - they're always in buffer for WS
          // (loaded in init/prefetch for first col, or in previous row's shift for new K-tile)
          if (activation_in_buffer && assigned_buffer) {
            uint64_t act_bytes = (uint64_t)tile_k * sj->M * data_type_width;
            assigned_buffer->record_read(j->addr, act_bytes, "activation_reuse");
          }

          // No DRAM reads here - weights were loaded in prefetch (first tile)
          // or in write->read transition (subsequent tiles)
          state_transfer(shift, 0, 0,
                        sz * std::max(systolic_fpu_latency, batch_size));
        } break;

        case shift: {
          // Compute phase complete
          int tile_m = std::min(sz, sj->M);
          int tile_k = std::min(sz, sj->K);
          int tile_n = std::min(sz, sj->N);

          // Calculate MACs for this tile
          uint64_t tile_macs = (uint64_t)tile_m * tile_k * tile_n * batch_size;
          record_compute_energy(tile_macs);

          // Track array utilization
          int compute_cycles = tile_m * batch_size;
          int active_macs_per_cycle = tile_k * tile_n;
          active_mac_cycles_ += (uint64_t)active_macs_per_cycle * compute_cycles;
          total_mac_capacity_cycles_ += (uint64_t)(sz * sz) * compute_cycles;

          int amt_to_write = 0;
          int amt_to_read = 0;
          int dram_read_bytes = 0;

          if (col_i == loop_cols_tiles) {
            // Finished all N-tiles for this K-tile
            if (row_i == loop_row_tiles) {
              // All tiles complete - output goes to buffer, then optionally to DRAM
              int output_bytes = sj->M * sj->N * data_type_width * batch_size;

              // Always record buffer write for output (compute -> buffer)
              if (assigned_buffer) {
                assigned_buffer->record_write(j->addr, output_bytes, "output");
              }

              // Only write to DRAM if not skipping writeback
              if (!skip_dram_writeback) {
                amt_to_write = output_bytes;
              }
            } else {
              // Moving to next K-tile - need NEW activations from HBM
              // (unless activations_in_buffer flag is set)
              activation_in_buffer = activations_in_buffer;
              if (!activations_in_buffer) {
                int act_bytes = std::min(sz, sj->K) * sj->M * data_type_width;
                dram_read_bytes = act_bytes;

                // Record: HBM → Buffer for new activations only
                // (weights will be loaded in write->read transition, not buffered)
                if (assigned_buffer) {
                  assigned_buffer->record_write(j->addr, act_bytes, "activation_fill");
                }
              }
            }
          }
          // If not at end of N-tiles, activations stay in buffer (no DRAM read needed)

          amt_to_read = dram_read_bytes;
          state_transfer(write, amt_to_read, amt_to_write, 0);
        } break;

        case write: {
          int rd_cycles = sj->M * std::max(systolic_fpu_latency, batch_size);

          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // Job complete - mark output as buffered for consumer jobs
              if (assigned_buffer && j->output_size_bytes > 0) {
                uint64_t buffer_capacity = assigned_buffer->get_config().size_bytes;
                if (j->output_size_bytes <= buffer_capacity) {
                  j->mark_output_buffered(buffer_capacity);
                }
              }
              state_transfer(idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to next K-tile (row)
              // New activations were loaded in shift state, now in buffer
              j->addr = j->addr_hold;
              activation_in_buffer = true;

              // Need to load new weights for first N-tile of new K-tile
              // Weights go directly to array (no buffer)
              int weight_bytes = std::min(sz, sj->N) * std::min(sz, sj->K) * data_type_width;
              bool skip_weight_fetch = arch_config.shared_weights && j->core_id > 0;

              if (!skip_weight_fetch) {
                weight_bytes_read_ += weight_bytes;
              }

              state_transfer(read, skip_weight_fetch ? 0 : weight_bytes, 0, rd_cycles);
              col_i = 1;
              row_i++;
            }
          } else {
            // Move to next N-tile (column) - REUSE activations from buffer
            // Only need to load new weights from DRAM (weights not buffered)
            int weight_bytes = std::min(sz, sj->N) * std::min(sz, sj->K) * data_type_width;
            bool skip_weight_fetch = arch_config.shared_weights && j->core_id > 0;

            if (!skip_weight_fetch) {
              weight_bytes_read_ += weight_bytes;
            }

            // Activations stay in buffer - will be read in next read->shift transition
            activation_in_buffer = true;

            state_transfer(read, skip_weight_fetch ? 0 : weight_bytes, 0, rd_cycles);
            col_i++;
          }
        } break;

        case idle:
          break;
      }
    } else {  // Output Stationary mode
      /**
       * Output Stationary (OS) Dataflow for Matmul M×K×N:
       *
       * Tiling: M into loop_row_tiles, N into loop_cols_tiles
       * K is processed fully in one pass (no K-tiling)
       *
       * Data Reuse:
       * - Activations[tile_m × K]: Reused across all N-tiles in same M-tile row
       * - Weights[K × tile_n]: Loaded fresh for each tile (could be buffered across M-tiles)
       */
      switch (state) {
        case read: {
          // Read phase complete - record buffer read for activations
          // Activations are in buffer (loaded at start of row), weights stream from DRAM
          if (activation_in_buffer && assigned_buffer) {
            int tile_m = std::min(sz, sj->M);
            uint64_t act_bytes = (uint64_t)tile_m * sj->K * data_type_width;
            assigned_buffer->record_read(j->addr, act_bytes, "activation_reuse");
          }
          state_transfer(shift, 0, 0, sz * std::min(systolic_fpu_latency, batch_size));
        } break;

        case shift: {
          // Calculate MACs for this tile
          int tile_m = std::min(sz, sj->M);
          int tile_k = sj->K;
          int tile_n = std::min(sz, sj->N);
          uint64_t tile_macs = (uint64_t)tile_m * tile_k * tile_n * batch_size;
          record_compute_energy(tile_macs);

          // Track array utilization
          int compute_cycles = sj->K * batch_size;
          int active_macs_per_cycle = tile_m * tile_n;
          active_mac_cycles_ += (uint64_t)active_macs_per_cycle * compute_cycles;
          total_mac_capacity_cycles_ += (uint64_t)(sz * sz) * compute_cycles;

          // Output tile goes to buffer (always), then optionally to DRAM
          int output_tile_bytes = tile_m * tile_n * data_type_width * batch_size;
          if (assigned_buffer) {
            assigned_buffer->record_write(j->addr, output_tile_bytes, "output");
          }

          state_transfer(write, 0, skip_dram_writeback ? 0 : beats_per_wb, 0);
        } break;

        case write:
          if (col_i == loop_cols_tiles) {
            if (row_i == loop_row_tiles) {
              // All tiles complete - mark output as buffered for consumer jobs
              if (assigned_buffer && j->output_size_bytes > 0) {
                uint64_t buffer_capacity = assigned_buffer->get_config().size_bytes;
                if (j->output_size_bytes <= buffer_capacity) {
                  j->mark_output_buffered(buffer_capacity);
                }
              }
              state_transfer(SystolicArray::idle, 0, 0, 0);
              TO_IDLE_CLEANUP();
            } else {
              // Move to next M-tile row - need new activations
              init_row_loop(true);
              // NOTE: Don't reset j->addr to addr_hold - this causes address collisions
              // with outstanding DRAM transactions when using multimap callback tracking.
              activation_in_buffer = true;  // New activations now in buffer
              UPDATE_STATE(SystolicArray::read);
              if (is_idle_from_memory) {
                UPDATE_IDLEMEM(false);
              }
              col_i = 1;
              row_i++;
            }
          } else {
            // Move to next N-tile - reuse activations from buffer
            init_row_loop(false);
            activation_in_buffer = true;  // Activations still in buffer
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
    // Output Stationary mode
    min_stage_cycles = sj->K * systolic_fpu_latency;

    int activation_bytes = 0;
    int weight_bytes = std::min(sz, sj->M) * sj->K * data_type_width;

    // Only load new activations if new_row AND not already in buffer
    if (new_row && !activations_in_buffer) {
      activation_bytes = std::min(sz, sj->M) * sj->K * batch_size * data_type_width;
    }

    // Track bytes by type
    activation_bytes_read_ += activation_bytes;

    // Check if we should skip weight fetch (shared weights mode)
    bool skip_weight_fetch = arch_config.shared_weights && j->core_id > 0;

    if (skip_weight_fetch) {
      current_op_type = activations_in_buffer ? "none" : "activation";
      n_read_bytes = activation_bytes;
    } else {
      weight_bytes_read_ += weight_bytes;
      if (activations_in_buffer) {
        current_op_type = "weight";
      } else {
        current_op_type = new_row ? "activation+weight" : "weight";
      }
      n_read_bytes = activation_bytes + weight_bytes;
    }

    // Record buffer fill for ACTIVATIONS ONLY - weights go directly to array
    // Activations are buffered and reused across all N-tiles in this M-tile row
    if (assigned_buffer && activation_bytes > 0) {
      assigned_buffer->record_write(j->addr, activation_bytes, "activation_fill");
    }

    n_read_beats = std::max(n_read_bytes / bytes_per_tx, 1);
  }
  mem_read_left = mem_read_left_unqueued = n_read_beats;
}

void SystolicArray::SysArrayState::init() {
  auto sj = (SysArrayJob *) j;
  if (j->is_done) {
    std::cerr << "ERROR" << std::endl;
  }

  // Check if we should skip weight fetch
  bool skip_weight_fetch = arch_config.shared_weights && j->core_id > 0;

  // Reset buffer state - if activations_in_buffer flag is set, they're already there
  activation_in_buffer = activations_in_buffer;

  if (ws) {
    // Weight Stationary mode
    UPDATE_STATE(SystolicArray::prefetch);
    loop_cols_tiles = div_ru(sj->N, sz);
    loop_row_tiles = div_ru(sj->K, sz);

    // Initial load: weights + first activation tile
    int weight_bytes = std::min(sz, sj->N) * std::min(sz, sj->K) * data_type_width;
    int activation_bytes = std::min(sz, sj->K) * sj->M * data_type_width;

    // Skip activation DRAM fetch if already in buffer
    int dram_activation_bytes = activations_in_buffer ? 0 : activation_bytes;

    // Track bytes (still count for stats even if not fetched from DRAM)
    activation_bytes_read_ += dram_activation_bytes;

    int total_read_bytes;
    if (skip_weight_fetch) {
      current_op_type = activations_in_buffer ? "none" : "activation";
      total_read_bytes = dram_activation_bytes;
    } else {
      weight_bytes_read_ += weight_bytes;
      current_op_type = activations_in_buffer ? "weight" : "activation+weight";
      total_read_bytes = dram_activation_bytes + weight_bytes;
    }

    // Record buffer fill for ACTIVATIONS ONLY - weights go directly to array
    // Skip if activations already in buffer
    if (assigned_buffer && dram_activation_bytes > 0) {
      assigned_buffer->record_write(j->addr, dram_activation_bytes, "activation_fill");
    }

    // After initial load, activations will be in buffer
    activation_in_buffer = true;

    state_transfer(SystolicArray::prefetch, total_read_bytes, 0, sz);
    row_i = 1;
    col_i = 1;
  } else {
    // Output Stationary mode
    UPDATE_STATE(SystolicArray::read);
    min_stage_cycles = sj->K * std::max(systolic_fpu_latency, batch_size);

    int activation_bytes = std::min(sz, sj->M) * sj->K * batch_size * data_type_width;
    int weight_bytes = std::min(sz, sj->M) * sj->K * data_type_width;

    // Skip activation DRAM fetch if already in buffer
    int dram_activation_bytes = activations_in_buffer ? 0 : activation_bytes;

    activation_bytes_read_ += dram_activation_bytes;

    int n_read_bytes;
    if (skip_weight_fetch) {
      current_op_type = activations_in_buffer ? "none" : "activation";
      n_read_bytes = dram_activation_bytes;
    } else {
      weight_bytes_read_ += weight_bytes;
      current_op_type = activations_in_buffer ? "weight" : "activation+weight";
      n_read_bytes = dram_activation_bytes + weight_bytes;
    }

    // Record buffer fill for ACTIVATIONS ONLY - weights go directly to array
    // Skip if activations already in buffer
    if (assigned_buffer && dram_activation_bytes > 0) {
      assigned_buffer->record_write(j->addr, dram_activation_bytes, "activation_fill");
    }

    // After initial load, activations will be in buffer
    activation_in_buffer = true;

    int n_read_beats = n_read_bytes / bytes_per_tx;
    mem_read_left = mem_read_left_unqueued = n_read_beats;

    loop_cols_tiles = div_ru(sj->N, sz);
    loop_row_tiles = div_ru(sj->M, sz);

    row_i = 1;
    col_i = 1;
  }

  if (loop_row_tiles == 0)
    throw std::runtime_error("loop_row_tiles == 0");
  if (loop_cols_tiles == 0)
    throw std::runtime_error("loop_cols_tiles == 0");
}

SystolicArray::SysArrayState::SysArrayState(int sz, bool ws, EnergyModel* energy_model,
                                            ComputePrecision precision)
    : State(1), sz(sz), ws(ws), state(SystolicArray::idle),
      energy_model_(energy_model), precision_(precision), total_compute_energy_mj_(0.0),
      weight_bytes_read_(0), activation_bytes_read_(0),
      active_mac_cycles_(0), total_mac_capacity_cycles_(0) {
  beats_per_wb = std::max((sz * sz * data_type_width * batch_size) / bytes_per_tx, 1);
}

double SystolicArray::SysArrayState::get_effective_utilization() const {
  if (total_mac_capacity_cycles_ == 0) return 0.0;
  return (double)active_mac_cycles_ / (double)total_mac_capacity_cycles_;
}

SystolicArray::SysArrayJob::SysArrayJob(int m, int k, int n)
    : Job(m * m * n * data_type_width * batch_size * 2 + n * m * data_type_width * batch_size), M(m), K(k), N(n) {
  // Set output size for buffer tracking (output is M × N matrix)
  output_size_bytes = (uint64_t)m * n * data_type_width * batch_size;
}


std::string SystolicArray::SysArrayJob::get_job_dims_string() const {
  return std::to_string(M) + " x " + std::to_string(K) + " x " + std::to_string(N);
}

void SystolicArray::SysArrayState::record_compute_energy(uint64_t num_macs) {
  if (energy_model_) {
    double energy_mj = energy_model_->compute_mac_energy_mj(num_macs, precision_);
    total_compute_energy_mj_ += energy_mj;
  }
}
