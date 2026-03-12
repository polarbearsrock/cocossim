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
  int cores;
  int sa_sz;
  int vu_sz;
  int ws;
  int sw = 0;  // shared weights flag
  int buf_mb = 64;  // buffer size in MB (default 64 MB)
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-sw", &sw},
              {"-buf_mb", &buf_mb}},
             "-c       number of cores\n"
             "-sa_sz   size of the systolic array\n"
             "-vu_sz   size of the vector unit\n"
             "-ws      weight stationary (1) or output stationary (0)\n"
             "-sw      shared weight stream (1) or independent (0)\n"
             "-buf_mb  on-chip buffer size in MB (default: 64)");

  // Set global buffer size (used by layer analysis in StandardLayers.cc)
  buffer_size_bytes = (int64_t)buf_mb * 1024 * 1024;

  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws, sw != 0);
  if (arch_config.shared_weights) {
    std::cout << "Shared weight streaming ENABLED: weights fetched by core 0, broadcast to others\n";
  }
  return new StandardArch;
}
