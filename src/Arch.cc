/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "Arch.h"
#include "memory.h"
#include "perf_enums.h"
#include "State.h"
#include <set>
#include <unordered_map>

std::unordered_map<int, int> state_updates;

void Arch::init_waveforms() {
#ifdef VCD
#define dec(nm) fprintf(vcd, "$var wire %d %s %s_%d_%s $end\n", \
  WIDTH_ ##nm, \
  rand_chars[((vcd_idx << total_stat_bits) | (STAT_ ##nm ))], \
  state->get_ty_string().c_str(), vcd_idx, #nm);

  fprintf(vcd, "$timescale 1ns $end\n");
  fprintf(vcd, "$scope module top $end\n");
  fprintf(vcd, "$var wire 8 ? phase $end\n");
  for (int vcd_idx = 0; vcd_idx < states.size(); vcd_idx++) {
    auto &state = states.at(vcd_idx);
    dec(STATE);
    dec(IDLE_FROM_MEMORY);
    dec(JOB_IDX);
  }

  fprintf(vcd, "$upscope $end\n");
  fprintf(vcd, "$enddefinitions $end\n");
  fprintf(vcd, "$dumpvars\n");
  fprintf(vcd, "b00000000 ?\n");
  for (int i = 0; i < states.size(); ++i) {
    fprintf(vcd, "b%s %s\n", int_to_binary(0, 3).c_str(), rand_chars[(i << total_stat_bits) | STAT_STATE]);
  }
  fprintf(vcd, "$end\n");
#endif
}


RuntimeStats_t *Arch::get_cycles(TimeBasedEnqueue &time_enqueues) {
  std::set<int> present_types;
  for (const auto &st : states) {
    present_types.insert(st->get_ty_idx());
  }
  int n_types = (int) present_types.size();

  n_idle_units = new int[n_types];
  memset(n_idle_units, 0, sizeof(int) * n_types);
  // Per-core job queues hold jobs pinned to a specific core; per-type queues
  // hold unpinned (core_id == -1) jobs, which any idle core of that type may
  // dispatch. Routing unpinned jobs at dispatch time (not enqueue time) is
  // what lets multiple units of one type share the anonymous workload.
  std::vector<std::vector<Job *>> core_queues(states.size());
  std::unordered_map<int, std::vector<Job *>> type_queues;

  std::function<void(Job *)> enqueue_job = [&](Job *job) -> void {
    if (job->core_id >= 0 && job->core_id < (int) states.size()) {
      core_queues[job->core_id].push_back(job);// Specific core requested
    } else {
      if (present_types.find(job->get_type()) == present_types.end()) {
        // A job no unit can ever execute would otherwise sit in a queue
        // forever and the simulation loop would never terminate.
        throw std::runtime_error("enqueue_job: no unit of type " +
                                 std::to_string(job->get_type()) +
                                 " exists in this architecture");
      }
      type_queues[job->get_type()].push_back(job);
    }
    total_frontier += 1;
  };


  if (time_enqueues.time_points.empty()) return nullptr;
  if (time_enqueues.time_points[0] != 0) {
    throw std::runtime_error("First time point must be 0");
  }
  auto *stats = new RuntimeStats_t[time_enqueues.to_enqueue.size()];
  auto total_states = states.size();
  for (int i = 0; i < time_enqueues.to_enqueue.size(); ++i) {
    stats[i].pct_active = new double[total_states];
  }

  uint64_t phase_cycles = 0;
  gcycles = 0;
  const uint64_t MAX_TIME = 0xFFFFFFFFFFFFFFFF;

  int phase_idx = 0;
  uint64_t next_phase;
  if (time_enqueues.time_points.size() > 1) {
    next_phase = time_enqueues.time_points[1];
  } else {
    next_phase = MAX_TIME;
  }

  auto &first_enqueue = time_enqueues.to_enqueue[0];


  for (auto &i: *first_enqueue) {
    enqueue_job(i);
  }

  int dram_cmds = 0;

  for (auto state: states) {
    n_idle_units[state->get_ty_idx()] += 1;
    total_idle += 1;
  }

  int logged_job_count = -1;


  auto *per_array_act = new uint64_t[states.size()];
  memset(per_array_act, 0, sizeof(uint64_t) * (states.size()));


  std::function<void(int)> write_stats = [&](int phase_idx) -> void {
    stats[phase_idx].cycles = phase_cycles;
    for (int i = 0; i < (states.size()); ++i) {
      stats[phase_idx].pct_active[i] = (double) (per_array_act[i] * 100) / (double) phase_cycles;
    }
  };

  double diff_accumulator_mem = 0;
  const double mem_slow_factor = 1;
  // DRAM ticks per simulator (compute) cycle = sim_cycle_period / dram_cycle_period
  //   = (1/freq_sa) / tCK = 1 / (freq_sa * tCK). The prior form tCK/freq_sa is
  // INVERTED for tCK != 1 (a faster DRAM clock, smaller tCK, must tick MORE per
  // sim cycle). At tCK=1 the average tick RATE matches the old code, but the
  // per-cycle tick phase differs for any -f != 1 (the old loop drained against
  // the differential instead of 1.0), so old cycle counts reproduce exactly
  // only at -f 1.
  const double differential_mem = 1.0 / (freq_sa * mem::dramsim3config->tCK) / mem_slow_factor;
  const double cycle_adjust = 1. / freq_sa;

  while (!(total_idle == states.size() && total_frontier == 0)) {
    if (gcycles >= next_phase) {
      phase_idx++;
      state_updates.at(-1) = -1;
      for (auto *job: *(time_enqueues.to_enqueue[phase_idx])) {
        enqueue_job(job);
      }
      if (phase_idx + 1 < time_enqueues.time_points.size()) {
        next_phase = time_enqueues.time_points[phase_idx + 1];
      } else {
        next_phase = MAX_TIME;
      }
      write_stats(phase_idx - 1);

      phase_cycles = 0;
      memset(per_array_act, 0, sizeof(uint64_t) * (states.size()));
    }

    bool enqueued_job = false;

    // Dispatch: an idle core first takes the next job pinned to it, then
    // falls back to the shared queue for its unit type.
    bool any_job_assigned = true;
    while (any_job_assigned) {
      any_job_assigned = false;
      for (int core_idx = 0; core_idx < states.size(); ++core_idx) {
        State *state = states[core_idx];
        if (state->get_state() != 0) continue;

        Job *job = nullptr;
        auto &cq = core_queues[core_idx];
        if (!cq.empty() && cq.front()->get_type() == state->get_ty_idx()) {
          job = cq.front();
          cq.erase(cq.begin());
        } else {
          auto &tq = type_queues[state->get_ty_idx()];
          if (!tq.empty()) {
            job = tq.front();
            tq.erase(tq.begin());
          }
        }
        if (job) {
          n_idle_units[job->get_type()] -= 1;
          total_idle -= 1;
          total_frontier--;

          state->j = job;
          LOG_TO_WAVEFORM(STAT_ID(JOB_IDX, state->vcd_idx), job->job_idx);
          state->init();
          state->min_stage_cycles += job_overhead_cycles;
          enqueued_job = true;
          any_job_assigned = true;
        }
      }
    }
#ifdef VCD
    bool first_state_update = true;
    for (auto pr: state_updates) {
      const int vcd_id = pr.first;
      const int to_state = pr.second;
      if (first_state_update) {
        first_state_update = false;
        fprintf(vcd, "#%f\n", (float) gcycles * cycle_adjust);
      }
      if (vcd_id >= 0) {
        int vcd_stat = STAT_EXTRACT(vcd_id);
        int width = STAT_WIDTH(vcd_stat);
        fprintf(vcd, "b%s %s\n", int_to_binary(to_state, width).c_str(), rand_chars[vcd_id]);
      } else {
        if (vcd_id == PHASE_STATE_IDX) {
          fprintf(vcd, "b%s ?\n", int_to_binary(phase_idx, 8).c_str());
        } else {
          throw std::exception();
        }
      }
      fflush(vcd);
    }
    state_updates.clear();
#endif

    gcycles++;
    phase_cycles++;

#if !defined(SILENCE) && !defined(DSE) || defined(DEBUG)
    if (logged_job_count != jobs_finished || gcycles % 100000 == 0) {
      logged_job_count = jobs_finished;
      printf("\rPHASE: %d, Cycles: %llu, Jobs finished: %d/%d, DRAM CMDs: %d", phase_idx, gcycles, jobs_finished, total_jobs, dram_cmds);
      mem::mem_sys->PrintStats();
      fflush(stdout);
    }
#endif
    diff_accumulator_mem += differential_mem;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "LoopDoesntUseConditionVariableInspection"
    // Emit one DRAM ClockTick per whole DRAM cycle accumulated; keep the
    // fractional remainder in [0,1). differential_mem is DRAM-ticks-per-sim-cycle.
    while (diff_accumulator_mem >= 1.0) {
      mem::mem_sys->ClockTick();
      diff_accumulator_mem -= 1.0;
    }
#pragma clang diagnostic pop

    for (int i = 0; i < states.size(); ++i) {
      State *s = states[i];
      bool is_active = s->increment(enqueue_job, total_idle, n_idle_units);
      if (is_active) {
        per_array_act[i]++;
        // Precedence is load-bearing: a stalled cycle counts as memstall even if the
        // job is also underfilled — memory starvation must be separated from shape
        // under-fill for the per-unit attribution (spec 3.5).
        if (s->is_idle_from_memory) s->acct_memstall++;
        else if (s->is_underfilled()) s->acct_underfilled++;
        else s->acct_busy++;
      }
    }

    bool successful_enqueue = true;
    for (int j = 0; j < dram_enq_per_cycle && successful_enqueue; ++j) {
      successful_enqueue = mem::try_enqueue_tx();
      dram_cmds += successful_enqueue;
    }
  }

#ifdef VCD
  bool first_state_update = true;
  for (auto pr: state_updates) {
    const int vcd_id = pr.first;
    const int to_state = pr.second;
    if (first_state_update) {
      first_state_update = false;
      fprintf(vcd, "#%f\n", (float) gcycles * cycle_adjust);
    }
    if (vcd_id >= 0) {
      int stat_extract = STAT_EXTRACT(vcd_id);
      fprintf(vcd, "b%s %s\n", int_to_binary(to_state, STAT_WIDTH(stat_extract)).c_str(), rand_chars[vcd_id]);
    } else {
      if (vcd_id == PHASE_STATE_IDX) {
        fprintf(vcd, "b%s ?\n", int_to_binary(phase_idx, 8).c_str());
      } else {
        throw std::exception();
      }
    }
  }
  state_updates.clear();
#endif

  printf("\rPHASE: %d, Cycles: %llu, Time: %fµs Jobs finished: %d/%d, DRAM CMDs: %d", phase_idx, gcycles, double(gcycles) * cycle_adjust / 1000, jobs_finished, total_jobs, dram_cmds);
  mem::mem_sys->PrintStats();
  fflush(stdout);
  write_stats(phase_idx);

  std::cout << std::endl;
  delete[] n_idle_units;
  return stats;
}
