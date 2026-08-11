/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/standard/StandardParser.h"

using namespace frontend::standard;

Arch* StandardParser::make_arch() {
  int cores = -1;
  int sa_sz = -1;
  int vu_sz = -1;
  int ws = 0;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws}},
             "-c       number of cores\n"
             "-sa_sz   size of the systolic array\n"
             "-vu_sz   size of the vector unit\n"
             "-ws      weight stationary (1) or output stationary (0)");
  if (cores <= 0 || sa_sz <= 0 || vu_sz <= 0) {
    throw std::runtime_error("-c, -sa_sz, and -vu_sz must be positive integers");
  }
  if (ws != 0 && ws != 1) {
    throw std::runtime_error("-ws must be either 0 or 1");
  }
  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws);
  return new StandardArch;
}
