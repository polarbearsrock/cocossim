/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_RUNTIMESTATS_T_H
#define PROSE_COMPILER_RUNTIMESTATS_T_H
#include <cstdint>

// Forward declarations
namespace buffers {
  struct EnergyBreakdown;
  struct BufferAccessStats;
}

struct RuntimeStats_t {
  uint64_t cycles;
  double *pct_active;

  // Power/energy metrics (optional, only filled if power analysis enabled)
  buffers::EnergyBreakdown* energy;
  buffers::BufferAccessStats* buffer_stats;
  double avg_power_W;

  RuntimeStats_t() : cycles(0), pct_active(nullptr), energy(nullptr),
                     buffer_stats(nullptr), avg_power_W(0.0) {}
};

#endif//PROSE_COMPILER_RUNTIMESTATS_T_H