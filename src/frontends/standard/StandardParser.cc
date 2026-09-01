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
  int n_vpu = -1;
  int mp_par = 0;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-buf_mb", &buf_mb},
              {"-dram_enq", &dram_enq_per_cycle},
              {"-job_overhead", &job_overhead_cycles},
              {"-fuse_epilogue", &fuse_epilogue},
              {"-mxu_macs_per_pe", &mxu_macs_per_pe},
              {"-n_vpu", &n_vpu},
              {"-vmem_reuse", &vmem_reuse},
              {"-vmem_headroom", &vmem_headroom_pct},
              {"-mp", &model_parallelism},
              {"-mp_par", &mp_par},
              {"-mem_prio", &mem_prio},
              {"-vmem_rows", &vmem_resident_rows},
              {"-fuse_attn", &fuse_attn},
              {"-dbuf", &dbuf_lookahead},
              {"-dbuf_tile", &dbuf_tile}},
             "-c            number of cores\n"
             "-sa_sz        size of the systolic array\n"
             "-vu_sz        size of the vector unit\n"
             "-ws           weight stationary (1) or output stationary (0)\n"
             "-buf_mb       on-chip buffer size in MiB (default 8)\n"
             "-dram_enq     memory requests issued per cycle (default 9)\n"
             "-job_overhead fixed dispatch overhead per job in cycles (default 0)\n"
             "-fuse_epilogue residual adds fused into GEMM epilogue: 0 off (default), 1 on\n"
             "-mxu_macs_per_pe MACs each PE retires per cycle in OS mode (default 1)\n"
             "-n_vpu        number of vector units (default: match -c)\n"
             "-vmem_reuse   weights stay VMEM-resident across row blocks: 1 on (default), 0 off\n"
             "-vmem_headroom percent of the per-MXU VMEM share usable for weights (default 100)\n"
             "-mp           model-parallel replicas of the input model (default 1)\n"
             "-mp_par       replicas run concurrently (1) or chained sequentially (0, default)\n"
             "-mem_prio     serve SA memory transactions before VPU ones: 0 FIFO (default), 1 on\n"
             "-vmem_rows    weight-residency row window for ablation (default 0 = unlimited)\n"
             "-fuse_attn    attention scores stay on-chip (flash-attention fusion): 0 off (default), 1 on\n"
             "-dbuf         cross-op weight prefetch byte budget in MiB (default 0 = off)\n"
             "-dbuf_tile    within-op tile double buffering: 0 off (default), 1 on");
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
  if (buf_mb < 1 || buf_mb > 1024) {
    std::cerr << "Error: -buf_mb must be in [1, 1024], got " << buf_mb << std::endl;
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
  if (mxu_macs_per_pe < 1) {
    std::cerr << "Error: -mxu_macs_per_pe must be >= 1, got " << mxu_macs_per_pe << std::endl;
    exit(1);
  }
  if (n_vpu != -1 && n_vpu < 1) {
    std::cerr << "Error: -n_vpu must be >= 1 (or omitted to match -c), got " << n_vpu << std::endl;
    exit(1);
  }
  if (vmem_reuse != 0 && vmem_reuse != 1) {
    std::cerr << "Error: -vmem_reuse must be 0 or 1, got " << vmem_reuse << std::endl;
    exit(1);
  }
  if (vmem_headroom_pct < 1 || vmem_headroom_pct > 100) {
    std::cerr << "Error: -vmem_headroom must be in [1, 100], got " << vmem_headroom_pct << std::endl;
    exit(1);
  }
  if (model_parallelism < 1) {
    std::cerr << "Error: -mp must be >= 1, got " << model_parallelism << std::endl;
    exit(1);
  }
  if (mp_par != 0 && mp_par != 1) {
    std::cerr << "Error: -mp_par must be 0 or 1, got " << mp_par << std::endl;
    exit(1);
  }
  if (mem_prio != 0 && mem_prio != 1) {
    std::cerr << "Error: -mem_prio must be 0 or 1, got " << mem_prio << std::endl;
    exit(1);
  }
  if (vmem_resident_rows < 0) {
    std::cerr << "Error: -vmem_rows must be >= 0, got " << vmem_resident_rows << std::endl;
    exit(1);
  }
  if (fuse_attn != 0 && fuse_attn != 1) {
    std::cerr << "Error: -fuse_attn must be 0 or 1, got " << fuse_attn << std::endl;
    exit(1);
  }
  if (dbuf_lookahead < 0) {
    std::cerr << "Error: -dbuf must be >= 0, got " << dbuf_lookahead << std::endl;
    exit(1);
  }
  if (dbuf_tile != 0 && dbuf_tile != 1) {
    std::cerr << "Error: -dbuf_tile must be 0 or 1, got " << dbuf_tile << std::endl;
    exit(1);
  }
  if (ws == 1 && dbuf_lookahead > 0) {
    // The WS charge sites never consume prefetch credit, and MatmulAct /
    // ActMatmul build OS-flagged jobs even under WS, so issued beats would
    // be pure extra traffic until the budget silently stalls the prefetcher.
    std::cerr << "Note: -dbuf is OS-only; ignoring -dbuf " << dbuf_lookahead
              << " in WS mode" << std::endl;
    dbuf_lookahead = 0;
  }
  do_par = mp_par;
  buffer_size_bytes = buf_mb * 1024 * 1024;
  mem::setup();
  arch_config = ArchConfig(cores, sa_sz, vu_sz, ws, n_vpu);
  return new StandardArch;
}
