/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_ARRAYSTATE_H
#define PROSE_COMPILER_ARRAYSTATE_H

#include "Job.h"
#include "perf_enums.h"
#include <functional>
#include <set>

#include "Arch.h"

#ifdef VCD
#define LOG_TO_WAVEFORM(stat_idx, to) \
  state_updates[stat_idx] = int(to);
#define UPDATE_STATE(x)                                  \
  state_updates[STAT_ID(STATE, vcd_idx)] = int(x); \
  set_state(x);
#define UPDATE_IDLEMEM(to)                                      \
  state_updates[STAT_ID(IDLE_FROM_MEMORY, vcd_idx)] = to; \
  is_idle_from_memory = to
#else
#define LOG_TO_WAVEFORM(stat_idx, to)
#define UPDATE_STATE(x) set_state(x)
#define UPDATE_IDLEMEM(to) is_idle_from_memory = to
#endif

#ifdef VERBOSE
#define IFVERB(x) x
#else
#define IFVERB(x)
#endif

#define SET_READS(x) mem_read_left_unqueued = mem_read_left = x
#define SET_WRITES(x) mem_write_left = mem_write_left_unqueued = x

#define TO_IDLE_CLEANUP()                                   \
  jobs_finished++;                                          \
  total_idle++;                                             \
  n_idle_units[get_ty_idx()] += 1;                \
  opspan_note_complete(j->op_class);                        \
  for (auto *child: j->children) {                          \
    child->rem_deps -= 1;                                   \
    if (child->rem_deps == 0) {                             \
      IFVERB(std::cout << "enqueuing child " << std::endl); \
      enqueue_job(child);                                   \
    }                                                       \
  }                                                         \
  j->is_done = true;                                        \
  j = nullptr



using enqueue_job_f_t = std::function<void(Job *)>;

// Per-op-class wall-clock span (fidelity benchmark spec section 4, 'per-op-
// class time'; stats line OPSPAN, main.cc): opspan_first[c] is gcycles at
// the dispatch of the first job of class c (Arch.cc dispatch loop, which
// runs before that cycle's gcycles++ -- so the first job of a run starts at
// data_overhead_cycles), opspan_last[c] is gcycles at the completion
// (TO_IDLE_CLEANUP, which runs inside increment() after gcycles++) of the
// last job of class c. Because the simulation loop exits right after the
// final completion, the last span's 'last' equals the Cycles line exactly.
// opspan_jobs[c] counts completions so main.cc prints only classes that ran.
extern uint64_t opspan_first[N_OP_CLASSES];
extern uint64_t opspan_last[N_OP_CLASSES];
extern uint64_t opspan_jobs[N_OP_CLASSES];
void opspan_note_dispatch(int op_class);
void opspan_note_complete(int op_class);

struct State {
  int sz = 0;      // Size of the functional array
  Job *j = nullptr;// Job being processed by the array

  uint8_t vcd_idx = 0;// Index for VCD tracing

  // All counters must start at zero: increment() runs every cycle on every
  // unit, including units that have never received a job, and enqueue_reads/
  // enqueue_writes act on these fields (an uninitialized positive value
  // dereferences the null job pointer).
  int min_stage_cycles = 0;       // Minimum cycles required to read data / shift / whatever
  int mem_read_left = 0;          // Remaining memory reads to complete
  int mem_write_left = 0;         // Remaining memory writes to complete
  int mem_read_left_unqueued = 0; // Unqueued memory reads left
  int mem_write_left_unqueued = 0;// Unqueued memory writes left
  int mem_queued = 0;             // Memory operations queued
  int core_memory_priority = 0;
  bool is_idle_from_memory = false;

  // Per-unit cycle accounting (spec 3.5): every non-idle cycle is classified
  // in Arch::get_cycles as memstall (waiting on DRAM with no compute left),
  // underfilled (working, but the job cannot fill the unit), or busy.
  uint64_t acct_busy = 0;
  uint64_t acct_underfilled = 0;
  uint64_t acct_memstall = 0;
  // The same three counters split by the running job's op class (Job.h
  // OpClass; benchmark spec S1): acctc[ACCTC_BUSY|UNDERFILLED|MEMSTALL][class].
  // Incremented at the same site with the same precedence, so each row sums
  // to the per-unit counter above exactly. Idle has no job and stays per-unit.
  enum { ACCTC_BUSY = 0, ACCTC_UNDERFILLED = 1, ACCTC_MEMSTALL = 2, N_ACCTC_KINDS = 3 };
  uint64_t acctc[N_ACCTC_KINDS][N_OP_CLASSES] = {};
  uint64_t total_work = 0;// SA: MACs, accumulated at job completion. VPU: lane-ops,
                           // summed across all phase passes (not just once per job).
  virtual bool is_underfilled() const { return false; }

  int loop_row_tiles = 0; // Number of row tiles in the loop
  int loop_cols_tiles = 0;// Number of column tiles in the loop
  int row_i = 0, col_i = 0;// Current row and column indices

  int beats_per_wb = 0;// Number of memory beats per write-back

  bool activation_in_buffer = false;// Enable DRAM-to-buffer flow simulation if false

  // Within-op double buffering (-dbuf_tile). A unit may issue its NEXT
  // tile's reads while the current tile's shift/write stages run: while
  // hold_reads is set, state_transfer leaves the read counters untouched
  // (they belong to the tile in flight), and while reads_gate is clear,
  // process_stage does not wait on them. The next read stage re-arms both.
  bool reads_gate = true;
  bool hold_reads = false;
  // -dbuf prefetch beats of the current job that were issued before dispatch
  // and have not landed yet (Job::prefetch_issued_beats - landed, programmed
  // at dispatch in Arch.cc and decremented by the DRAM read callback).
  // process_stage waits on it unconditionally -- reads_gate does not apply,
  // these beats are the FIRST tile's operands, never the pre-issued next
  // tile's -- so no write-back can be issued over a pending prefetch read.
  int prefetch_read_left = 0;
  // -op_overhead (benchmark spec S2): op_id of the last job this unit ran
  // (-2 = none yet, so the first job is always a boundary) and the pure
  // serial stall still to burn before the current job issues any read --
  // a stalled cycle is neither busy nor memstall; the unit reports idle.
  int last_op_id = -2;
  int op_stall_left = 0;
  uint64_t op_boundaries = 0;// ops this unit entered (stats line OPBOUND)

  virtual ~State() = default;
  State() = delete;
  State(int memory_priority);


  void enqueue_writes();
  void enqueue_reads();
  void check_idle_from_memory();
  bool process_stage();
  // Byte amounts are 64-bit: a batched-prefill elementwise job can exceed
  // 2^31 bytes (review finding V34e); beats stay int (< 2^31 for < 128 GiB).
  void state_transfer(int st, int64_t read_amt, int64_t write_amt, int min_cycles);
  virtual void init() = 0;
  virtual bool increment(const enqueue_job_f_t &, int &total_idle, int *n_idle_units) = 0;
  virtual void set_state(int st) = 0;
  virtual int get_state() = 0;

  virtual int get_ty_idx() = 0;
  virtual std::string get_ty_string() = 0;
};

void vcd_stat_init(int vcd_idx, const char *sig_name);

#endif// PROSE_COMPILER_ARRAYSTATE_H
