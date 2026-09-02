/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/Frontend.h"
#include "frontends/standard/StandardLayer.h"
#include "frontends/standard/StandardParser.h"
#include "frontends/standard/StandardUnits.h"

#include "memory.h"
#include <chrono>

std::string layer_file;
std::string ofile;
std::string dram_ini_path = "../dramsim3/configs/HBM2_8Gb_x128.ini";

using namespace frontend::standard;

using MyArchParser = StandardParser;
using MyLayerParser = StandardLayer;

int main(int argc, char **argv) {
  MyArchParser archParser(argc, argv);
  MyLayerParser layerParser;

  auto t1 = std::chrono::high_resolution_clock::now();

#ifdef VCD
  vcd = fopen("out.vcd", "w");
#endif

  Arch *arch = archParser.make_arch();
  arch->init_waveforms();

  std::string line;
  std::vector<LayerConfig> layer_configs;

  std::ifstream layer_stream(layer_file);
  if (!layer_stream.is_open()) {
    throw std::runtime_error("Error: Could not open layer configuration file: " + layer_file);
  }

  // Parse layer configuration file (format: layer_name dim1 dim2 ... dim8)
  int n_layers = 0;
  while (std::getline(layer_stream, line)) {
    const char *nt_buff = line.c_str();
    LayerConfig l_config;
    char nm[64] = {0};
    std::cout << "processing " << line << std::endl;
    
    // Parse up to 9 dimensions per layer (Transformer's optional vocab is
    // the 9th; tokens beyond the format string are silently dropped, so the
    // width here caps every layer grammar).
    std::vector<int> dims(9);
    int successes = sscanf(nt_buff, "%s %d %d %d %d %d %d %d %d %d",
                           nm,
                           &dims[0], &dims[1], &dims[2], &dims[3],
                           &dims[4], &dims[5], &dims[6], &dims[7], &dims[8]);
    if (successes < 2) {
      std::cerr << "failed with '" << nt_buff << "'" << std::endl;
      throw std::exception();
    }
    
    l_config.layer_type = std::string(nm);
    int n_dims = successes - 1;
    dims.resize(n_dims);
    l_config.dimensions = dims;
    layer_configs.push_back(l_config);
    n_layers++;
  }
  layer_stream.close();

  // Setup multi-period simulation with time-based job enqueuing
  TimeBasedEnqueue time_enqueues;
  uint64_t t = 0;
  auto dt = 30000000;  // Time between periods
  std::vector<Job *> period_jobs[periods];
  
  std::cout << "Period: " << periods << std::endl;
  for (int i = 0; i < periods; ++i) {
    // Create jobs for each thread in this period
    for (int j = 0; j < n_threads; ++j) {
      auto network = layerParser.make_layers(layer_configs);
      for (auto &layer: network) {
        period_jobs[i].insert(period_jobs[i].end(), layer.first.begin(), layer.first.end());
      }
      alloc_task_idx++;
    }
    time_enqueues.enqueue_at(t, &period_jobs[i]);
    t += dt;

    std::cout << "Jobs for Period " << i << ":" << std::endl;
    for (auto *job: period_jobs[i]) {
      job->printDetails();
    }
  }

  jobs_to_dot(period_jobs[0]);

  mem::mem_sys->ResetStats();

  auto res = arch->get_cycles(time_enqueues);
  FILE *f = fopen(ofile.c_str(), "w");
  // SCHEMA 2: SA eff_util capacity is mxu_macs_per_pe * sz^2 (full tile ~1.0).
  // Schema-1 files (no SCHEMA line) used sz^2 against 2-cycle K-steps, capping
  // full-tile eff_util at ~0.5 -- not comparable across schemas.
  // SCHEMA 3 (benchmark spec S1): ACCT semantics unchanged; each unit's ACCT
  // line is followed by per-op-class lines
  //   ACCTC <UNIT_TYPE> <idx> <class> busy <n> underfilled <n> memstall <n>
  // one per OpClass (Job.h) with any nonzero counter, whose busy/underfilled/
  // memstall columns sum to the unit's ACCT values exactly (idle is per-unit).
  // Jobs not created by the Transformer composite report as class OTHER.
  // SCHEMA 3 additive line (fidelity benchmark spec section 4, per-op-class
  // time): after the ACCT/ACCTC/OPBOUND block, one line per OpClass that ran
  // at least one job,
  //   OPSPAN <class> first <cycle> last <cycle>
  // where first is gcycles at the dispatch of the class's first job (the
  // first dispatch of a run is at -data_overhead, 0 by default) and last is
  // gcycles at the completion of its last job; the class that finishes the
  // run has last == Cycles exactly (State.h). The span is wall-clock, so
  // overlapping classes overlap here too; it is the simulator-side quantity
  // compared with per-kernel wall time, not a sum of busy cycles.
  fprintf(f, "SCHEMA 3\n");
  for (int p = 0; p < periods; ++p) {
    fprintf(f, "Cycles %llu\n", res[p].cycles);
    for (int i = 0; i < arch->states.size(); ++i) {
      fprintf(f, "%s %f\n", arch->states[i]->get_ty_string().c_str(), res[p].pct_active[i]);
    }
    // acct_*/total_work are cumulative across periods while Cycles/pct_active are
    // per-phase: these lines are correct only at periods == 1 (the only supported
    // configuration; the V7 sum invariant relies on it).
    for (int i = 0; i < arch->states.size(); ++i) {
      State *s = arch->states[i];
      uint64_t accounted = s->acct_busy + s->acct_underfilled + s->acct_memstall;
      uint64_t active = s->acct_busy + s->acct_underfilled;
      double cap = (s->get_ty_idx() == SYSTOLIC_ARRAY_IDX)
                       ? (double) mxu_macs_per_pe * s->sz * s->sz
                       : (double) s->sz;
      double eff = active > 0 ? (double) s->total_work / (cap * (double) active) : 0.0;
      fprintf(f, "ACCT %s %d busy %llu underfilled %llu memstall %llu idle %llu work %llu eff_util %f\n",
              s->get_ty_string().c_str(), i,
              (unsigned long long) s->acct_busy,
              (unsigned long long) s->acct_underfilled,
              (unsigned long long) s->acct_memstall,
              (unsigned long long) (gcycles - accounted),
              (unsigned long long) s->total_work, eff);
      for (int c = 0; c < N_OP_CLASSES; ++c) {
        uint64_t cb = s->acctc[State::ACCTC_BUSY][c];
        uint64_t cu = s->acctc[State::ACCTC_UNDERFILLED][c];
        uint64_t cm = s->acctc[State::ACCTC_MEMSTALL][c];
        if (cb == 0 && cu == 0 && cm == 0) continue;
        fprintf(f, "ACCTC %s %d %s busy %llu underfilled %llu memstall %llu\n",
                s->get_ty_string().c_str(), i, OP_CLASS_NAMES[c],
                (unsigned long long) cb, (unsigned long long) cu, (unsigned long long) cm);
      }
      // Ops this unit entered (op_id boundaries, spec S2): the count
      // -op_overhead charges, and the per-unit kernel count silicon pays.
      fprintf(f, "OPBOUND %s %d %llu\n", s->get_ty_string().c_str(), i,
              (unsigned long long) s->op_boundaries);
    }
    // Per-op-class wall-clock span (see the SCHEMA comment above). Like the
    // ACCT block, cumulative across periods: exact at periods == 1.
    for (int c = 0; c < N_OP_CLASSES; ++c) {
      if (opspan_jobs[c] == 0) continue;
      fprintf(f, "OPSPAN %s first %llu last %llu\n", OP_CLASS_NAMES[c],
              (unsigned long long) opspan_first[c], (unsigned long long) opspan_last[c]);
    }
  }


  uint64_t last_cycles = res[periods - 1].cycles;
  auto lc = (double) last_cycles;
  auto expected_c = (double)res[0].cycles;
  double ratio = lc / expected_c;
  printf("Drain Ratio: %f\n", ratio);

  fclose(f);
  mem::mem_sys->PrintEpochStats();
#ifdef VCD
  fclose(vcd);
#endif
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}
