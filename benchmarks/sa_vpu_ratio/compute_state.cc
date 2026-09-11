// Compute-only State backend for the unmodified COCOSSim SA/VPU state machines.
// Reads and writes have no latency, bandwidth, capacity, or issue constraints.
// Internal unit stage durations and stage transitions are retained exactly.
#include "State.h"

uint64_t opspan_first[N_OP_CLASSES] = {};
uint64_t opspan_last[N_OP_CLASSES] = {};
uint64_t opspan_jobs[N_OP_CLASSES] = {};
void opspan_note_dispatch(int) {}
void opspan_note_complete(int) {}

State::State(int priority) : core_memory_priority(priority) {}
void State::enqueue_reads() {
  mem_read_left = mem_read_left_unqueued = prefetch_read_left = 0;
}
void State::enqueue_writes() {
  mem_write_left = mem_write_left_unqueued = 0;
}
void State::check_idle_from_memory() { is_idle_from_memory = false; }
bool State::process_stage() {
  if (min_stage_cycles > 0) --min_stage_cycles;
  return min_stage_cycles == 0;
}
void State::state_transfer(int st, int64_t, int64_t, int cycles) {
  set_state(st);
  min_stage_cycles = cycles;
  enqueue_reads();
  enqueue_writes();
  is_idle_from_memory = false;
}
void vcd_stat_init(int, const char *) {}
