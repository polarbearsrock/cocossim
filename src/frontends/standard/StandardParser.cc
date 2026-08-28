/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/standard/StandardParser.h"
#include "memory.h"

using namespace frontend::standard;

Arch* StandardParser::make_arch() {
  int cores = 1;
  int sa_sz = 64;
  int vu_sz = 64;
  int ws = 0;
  int buf_mb = 8;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-buf_mb", &buf_mb},
              {"-dram_enq", &dram_enq_per_cycle},
              {"-job_overhead", &job_overhead_cycles},
              {"-fuse_epilogue", &fuse_epilogue}},
             "-c            number of cores\n"
             "-sa_sz        size of the systolic array\n"
             "-vu_sz        size of the vector unit\n"
             "-ws           weight stationary (1) or output stationary (0)\n"
             "-buf_mb       on-chip buffer size in MiB (default 8)\n"
             "-dram_enq     memory requests issued per cycle (default 9)\n"
             "-job_overhead fixed dispatch overhead per job in cycles (default 0)\n"
             "-fuse_epilogue residual adds fused into GEMM epilogue: 0 off (default), 1 on");
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
  if (buf_mb < 1) {
    std::cerr << "Error: -buf_mb must be >= 1, got " << buf_mb << std::endl;
    exit(1);
  }
  if (dram_enq_per_cycle < 1) {
    std::cerr << "Error: -dram_enq must be >= 1, got " << dram_enq_per_cycle << std::endl;
    exit(1);
  }
  if (job_overhead_cycles < 0) {
    std::cerr << "Error: -job_overhead must be >= 0, got " << job_overhead_cycles << std::endl;
    exit(1);
  }
  if (fuse_epilogue != 0 && fuse_epilogue != 1) {
    std::cerr << "Error: -fuse_epilogue must be 0 or 1, got " << fuse_epilogue << std::endl;
    exit(1);
  }
  buffer_size_bytes = buf_mb * 1024 * 1024;
  mem::setup();
  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws);
  return new StandardArch;
}
