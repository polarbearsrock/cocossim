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
#include <algorithm>
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

  // -data_overhead: the clock starts this many cycles in (global.h). No
  // DRAM ticks are simulated for them: they stand for layout/copy kernels
  // the model has no jobs for, charged as pure time.
  uint64_t phase_cycles = data_overhead_cycles;
  gcycles = data_overhead_cycles;
  const uint64_t MAX_TIME = 0xFFFFFFFFFFFFFFFF;
  // -kv_bw_pct: budget = P% of the DRAM PLATE (channels x bus_width/8 x 2
  // transfers per tCK, per simulator cycle) in beats, shared chip-wide.
  {
    const auto *dc = mem::dramsim3config;
    double plate_beats = (double) dc->channels * dc->bus_width / 8.0 * 2.0
                         / ((double) freq_sa * dc->tCK) / bytes_per_tx;
    kv_issue_rate = (kv_bw_pct < 100) ? plate_beats * kv_bw_pct / 100.0 : 0.0;
    kv_budget_acc = 0.0;
  }

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
  // The -data_overhead cycles precede the first dispatch: demand-idle by
  // definition, so both idle stats start there.
  uint64_t mem_demand_idle = data_overhead_cycles;
  uint64_t mem_idle_all = data_overhead_cycles;

  // ---------------------------------------------------------------------
  // Cross-op weight prefetch (-dbuf, spec 6.7). XLA streams the NEXT
  // operator's weights under the current op's compute/barrier tail (B2,
  // C5v2). Model: the first job of each weight tag -- which can never be
  // VMEM-resident at dispatch, so its charge is unconditional -- may have
  // its full weight sweep issued into otherwise-idle DRAM slots ahead of
  // dispatch. Issued bytes are recorded as the job's prefetch credit and
  // its charge sites deduct them, so total traffic is exactly invariant.
  // Eligible entries are the next `dbuf_lookahead` unstarted tags in job
  // creation order (creation order tracks execution order; the byte cap is
  // the VMEM-honesty bound). Beats are issued only when the demand queue
  // left slack this cycle, so demand always goes first.
  // ---------------------------------------------------------------------
  struct PrefetchEntry {
    Job *job;
    uint64_t next_addr;
    int64_t beats_left;    // weight sweep (streamable any time)
    int64_t act_beats_left;// first activation panel (streamable once READY)
    bool done() const { return job->started || (beats_left == 0 && act_beats_left == 0); }
  };
  std::vector<PrefetchEntry> pf_list;
  size_t pf_cursor = 0;
  if (dbuf_lookahead > 0) {
    std::set<Job *> seen;
    std::set<int> seen_tags;
    std::vector<Job *> to_visit;
    for (auto &phase_jobs: time_enqueues.to_enqueue)
      for (auto *jb: *phase_jobs) to_visit.push_back(jb);
    std::vector<Job *> all;
    while (!to_visit.empty()) {
      Job *jb = to_visit.back();
      to_visit.pop_back();
      if (!seen.insert(jb).second) continue;
      all.push_back(jb);
      for (auto *c: jb->children) to_visit.push_back(c);
    }
    std::sort(all.begin(), all.end(),
              [](Job *a, Job *b) { return a->job_idx < b->job_idx; });
    // Which jobs WILL fetch weights? Core-pinned jobs dispatch in creation
    // order on their core, so replay SysArrayState::init's residency
    // decision (same rules: vmem_reuse, tag match, -vmem_rows window,
    // stay-resident) per core to predict every fetch pass exactly. Unpinned
    // jobs (core_id -1) fall back to first-of-tag, which can never be
    // resident. A wrong prediction cannot corrupt totals silently: a
    // predicted-fetch that dispatches resident leaves issued beats
    // unconsumed, and V27's CMD-invariance assertions trip on that.
    struct ReplayState { int resident_tag = -1; int rows = 0; };
    std::unordered_map<int, ReplayState> replay;
    for (auto *jb: all) {
      if (jb->prefetch_rows() <= 0) continue;// not a systolic-array job
      const int tag = jb->prefetch_tag();
      bool will_fetch;
      if (jb->core_id >= 0) {
        // Every SA job on the core advances the replay, untagged ones
        // included (init() clears residency for them), so the predicted
        // sequence is the executed one.
        auto &rs = replay[jb->core_id];
        bool resident = vmem_reuse && tag != -1 && tag == rs.resident_tag;
        if (resident && vmem_resident_rows > 0 &&
            rs.rows + jb->prefetch_rows() > vmem_resident_rows)
          resident = false;
        bool stay = vmem_reuse && jb->prefetch_fits_vmem();
        rs.resident_tag = (stay && tag != -1) ? tag : -1;
        rs.rows = resident ? rs.rows + jb->prefetch_rows() : jb->prefetch_rows();
        will_fetch = !resident;
      } else {
        will_fetch = tag != -1 && seen_tags.insert(tag).second;
      }
      int64_t beats = will_fetch ? jb->prefetchable_weight_beats() : 0;
      int64_t act = jb->prefetchable_act_beats();
      if (beats > 0 || act > 0) pf_list.push_back({jb, jb->addr_hold, beats, act});
    }
  }

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
    // Refill the chip-wide KV-stream budget; carry at most one cycle over so
    // an idle stretch cannot bank unbounded credit.
    if (kv_bw_pct < 100)
      kv_budget_acc = std::min(kv_budget_acc + kv_issue_rate, 2.0 * kv_issue_rate);
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
          opspan_note_dispatch(job->op_class);// OPSPAN first (State.h)
          job->started = true;// stops -dbuf prefetch issue for this job
          LOG_TO_WAVEFORM(STAT_ID(JOB_IDX, state->vcd_idx), job->job_idx);
          state->init();
          // Prefetched-but-unlanded beats of this job become a wait the
          // state honours like demand reads (Job.h): program the count
          // BEFORE publishing the state to the callback (single-threaded,
          // callbacks only fire inside ClockTick, so no landing can slip
          // between the two statements).
          state->prefetch_read_left =
              (int) (job->prefetch_issued_beats - job->prefetch_landed_beats);
          job->exec_state = state;
          state->min_stage_cycles += job_overhead_cycles;
          // -op_overhead: a new op on this unit (or an unstamped job, which
          // is its own op) stalls before issuing any read (State.h).
          if (job->op_id == -1 || job->op_id != state->last_op_id) {
            state->op_boundaries++;
            // -attn_overhead: the attention kernel's fixed cost (census fit
            // ~15 us per layer) on every unit entering the attention op;
            // the larger of it and -op_overhead applies (global.h).
            // Charged on the MXU side only: the softmax jobs enter the same
            // op on the VPU between QK^T and AV, and a second charge there
            // would put the kernel's one floor on the critical path twice.
            int stall = op_overhead_cycles;
            if (job->op_class == OP_ATTN && attn_overhead_cycles > stall &&
                job->prefetch_rows() > 0)// a systolic-array job (Job.h)
              stall = attn_overhead_cycles;
            if (stall > 0) state->op_stall_left = stall;
          }
          state->last_op_id = job->op_id;
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
        // The per-class split (spec S1) mirrors the same decision, keyed by
        // the running job's op_class (an active unit always holds its job:
        // increment() returns state != idle and TO_IDLE_CLEANUP clears j
        // only on the transition to idle; OP_OTHER is the defensive
        // fallback so the class rows still sum to the unit totals).
        int oc = s->j ? s->j->op_class : OP_OTHER;
        if (s->is_idle_from_memory) { s->acct_memstall++; s->acctc[State::ACCTC_MEMSTALL][oc]++; }
        else if (s->is_underfilled()) { s->acct_underfilled++; s->acctc[State::ACCTC_UNDERFILLED][oc]++; }
        else { s->acct_busy++; s->acctc[State::ACCTC_BUSY][oc]++; }
      }
    }

    bool successful_enqueue = true;
    for (int j = 0; j < dram_enq_per_cycle && successful_enqueue; ++j) {
      successful_enqueue = mem::try_enqueue_tx();
      dram_cmds += successful_enqueue;
    }
    // Offered-load stats. demand-idle: cycles where no UNIT has any read or
    // write outstanding (the starvation -dbuf exists to fill -- counted from
    // the states' own pending counters so prefetch traffic cannot mask it;
    // prefetch beats a DISPATCHED job is waiting on are demand by then).
    // idle incl. prefetch: nothing queued or in flight at all.
    {
      bool demand_pending = false;
      for (auto *s: states)
        if (s->mem_read_left > 0 || s->mem_write_left > 0 || s->prefetch_read_left > 0) {
          demand_pending = true;
          break;
        }
      if (!demand_pending) mem_demand_idle++;
      if (to_enqueue.empty() && mem::address_reads_bkwds_lookup.empty() &&
          mem::address_writes_bkwds_lookup.empty())
        mem_idle_all++;
    }

    // -dbuf issue: fill only the slack the demand enqueue left this cycle
    // (at -mem_prio 0 a beat issued here can still precede demand pushed
    // NEXT cycle by one FIFO slot -- a one-cycle effect). The lookahead is a
    // BYTE budget (VMEM honesty): keep streaming entries in list order while
    // issued-but-unconsumed beats stay under the cap. A tag-count window
    // proved mis-shaped -- 16 tiny attention-group tags sit between the big
    // GEMMs, so a 2-tag window stalled on them and never reached the MLP
    // weights it existed to hide.
    if (dbuf_lookahead > 0) {
      const int64_t pf_cap_beats = ((int64_t) dbuf_lookahead << 20) / bytes_per_tx;
      while (pf_cursor < pf_list.size() && pf_list[pf_cursor].done())
        pf_cursor++;
      int budget = dram_enq_per_cycle - (int) to_enqueue.size();
      for (size_t e = pf_cursor;
           e < pf_list.size() && budget > 0 && pf_outstanding_beats < pf_cap_beats; ++e) {
        auto &ent = pf_list[e];
        if (ent.job->started) continue;
        // Weights first; the activation panel only once the job is ready.
        int64_t *avail = ent.beats_left > 0 ? &ent.beats_left
                       : (ent.act_beats_left > 0 && ent.job->rem_deps == 0) ? &ent.act_beats_left
                       : nullptr;
        if (!avail) continue;
        int n = (int) std::min<int64_t>(std::min<int64_t>(budget, *avail),
                                        pf_cap_beats - pf_outstanding_beats);
        // -kv_bw_pct: a KV-stream job's weight sweep draws from the same
        // chip-wide token bucket whether it is fetched on demand
        // (State::enqueue_reads) or prefetched here (-kv_prefetch), so the
        // knob stays a rate on the KV stream, not on the fetch path (V34a).
        const bool kv_pf = ent.job->kv_stream && kv_bw_pct < 100 && avail == &ent.beats_left;
        if (kv_pf) n = std::max(0, std::min(n, (int) kv_budget_acc));
        if (n <= 0) continue;
        if (kv_pf) kv_budget_acc -= n;
        // Same guard the demand paths apply (State::check_in_bounds): a walk
        // past the window is the DRAMSim3 R/W-aliasing livelock class.
        if (ent.next_addr + (uint64_t) n * bytes_per_tx >
            ent.job->addr_hold + ent.job->alloc_size)
          throw std::runtime_error("prefetch for job " + std::to_string(ent.job->job_idx) +
                                   " would walk past its allocation");
        for (int b = 0; b < n; ++b) {
          // Priority 3 (behind SA/VPU demand for -mem_prio); no owning
          // STATE (nullptr), but an owning JOB: the callback counts the
          // landing on it (Job.h, prefetch_landed_beats).
          mem::prefetch_owner[ent.next_addr] = ent.job;
          to_enqueue.emplace_back(ent.next_addr, false, 3, nullptr);
          ent.next_addr += bytes_per_tx;
        }
        *avail -= n;
        ent.job->prefetch_credit_beats += n;
        ent.job->prefetch_issued_beats += n;
        pf_outstanding_beats += n;
        budget -= n;
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
  printf("\nMEM demand-idle: %llu / %llu cycles (idle incl. prefetch: %llu)\n",
         (unsigned long long) mem_demand_idle, (unsigned long long) gcycles,
         (unsigned long long) mem_idle_all);
  mem::mem_sys->PrintStats();
  fflush(stdout);
  write_stats(phase_idx);

  std::cout << std::endl;
  delete[] n_idle_units;
  return stats;
}
