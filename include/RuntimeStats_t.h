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
  struct BufferAccessStats;
}

struct RuntimeStats_t {
  uint64_t cycles;
  double *pct_active;

  // Buffer access statistics (optional, only filled if buffer tracking enabled)
  buffers::BufferAccessStats* buffer_stats;

  RuntimeStats_t() : cycles(0), pct_active(nullptr), buffer_stats(nullptr) {}
};

#endif//PROSE_COMPILER_RUNTIMESTATS_T_H