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
  int cores = 1;
  int sa_sz = 64;
  int vu_sz = 64;
  int ws = 0;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws}},
             "-c       number of cores\n"
             "-sa_sz   size of the systolic array\n"
             "-vu_sz   size of the vector unit\n"
             "-ws      weight stationary (1) or output stationary (0)");
  if (cores < 1) {
    std::cerr << "Error: -c (number of cores) must be >= 1, got " << cores << std::endl;
    exit(1);
  }
  if (sa_sz < 1 || vu_sz < 1) {
    std::cerr << "Error: -sa_sz and -vu_sz must be >= 1, got " << sa_sz << " and " << vu_sz << std::endl;
    exit(1);
  }
  if (ws != 0 && ws != 1) {
    std::cerr << "Error: -ws must be 0 (output stationary) or 1 (weight stationary), got " << ws << std::endl;
    exit(1);
  }
  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws);
  return new StandardArch;
}
