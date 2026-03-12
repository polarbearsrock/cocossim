/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_STANDARD_ARCH_H
#define PERF_MODEL_STANDARD_ARCH_H

#include "Arch.h"

namespace frontend::standard {
  struct ArchConfig {
    int n_cores = -1;
    int sa_sz_allo = -1;
    int vu_sz_allo = -1;
    bool ws = false;
    bool shared_weights = false;  // Enable shared weight streaming across cores
    ArchConfig(int n_cores, int sa_sz_allo, int vu_sz_allo, bool ws, bool shared_weights = false)
        : n_cores(n_cores), sa_sz_allo(sa_sz_allo), vu_sz_allo(vu_sz_allo), ws(ws), shared_weights(shared_weights) {}
    ArchConfig() = default;
  };

  extern ArchConfig arch_config;

  struct StandardArch: Arch {
    StandardArch();
  };

  extern StandardArch *arch;
}


#endif//PERF_MODEL_STANDARD_ARCH_H
