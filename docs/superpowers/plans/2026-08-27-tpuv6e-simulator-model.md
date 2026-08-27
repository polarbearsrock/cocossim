# TPUv6e Simulator Model Implementation Plan (Plan 1 of 3: simulator-side)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make COCOSSim able to model a TPU v6e TensorCore: runtime-configurable memory/buffer/overhead knobs, an HBM2e-class DRAM config, honest per-unit cycle accounting, and a `Transformer` composite layer with true DAG dependencies.

**Architecture:** All changes extend the existing standard frontend and core loop; nothing restructures the scheduler or job model. New knobs become runtime flags with defaults that reproduce current behavior exactly. The `Transformer` keyword expands inside C++ (`StandardLayers.cc`) into existing job primitives, wiring residual-add jobs to both true parents.

**Tech Stack:** C++17, CMake, DRAMSim3 (vendored in `dramsim3/`), bash+awk regression tests.

**Spec:** `docs/superpowers/specs/2026-08-27-tpuv6e-model-calibration-design.md` — read it first. This plan implements spec §3 (M0 + M2). The measurement harness (spec §4) and fit pipeline (spec §5) are separate future plans.

## Global Constraints

- Build: `export CCACHE_DIR=/data2/s2chitni/.tmp/ccache && cmake --build /data2/s2chitni/cocossim/build -j8` — NEVER let ccache write under the home directory (quota-full).
- All scratch/temp files go under `$TMPDIR` (`/data2/s2chitni/.tmp`), never `/tmp` or the home directory.
- Never read, access, or reference files under `/data/eda_tools/pdk/`.
- Repo root: `/data2/s2chitni/cocossim`. Binary: `build/perf_model`. Run it from `build/` (the default DRAM ini path is relative: `../dramsim3/configs/...`).
- With no new flags passed, behavior must stay identical to pre-plan behavior (spec §3.2). Exception, by design: Task 5 changes OS-mode job dims for layers whose M is not a multiple of `-sa_sz` (spec §3.5 requires true dims). No shipped example or existing test depends on such an M.
- Before EVERY commit: run BOTH `tests/regression.sh` (must stay 5/5) and `tests/tpuv6e.sh` (all tests so far green).
- Every commit message ends with these two lines:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01Mbj5eyJ5ocdWzCZ8m7SVsM`
- Test-count arithmetic: several tests assert exact job counts, derived in comments. If an observed count differs at the GREEN step, re-derive by hand from the simulator's job dump BEFORE touching the assertion; the derivation comment must be updated to match.

---

### Task 0: Commit the pending fix series

The working tree carries an already-reviewed, already-verified fix series (scheduler type-queues, State init, WS reservation, Softmax conservation, parser validation, CNN example, `tests/regression.sh`). It must be committed before new work so every later commit is clean.

**Files:**
- Modify (commit only, no edits): `examples/cnn_model.txt`, `include/Arch.h`, `include/State.h`, `src/Arch.cc`, `src/frontends/standard/StandardLayers.cc`, `src/frontends/standard/StandardParser.cc`, `tests/regression.sh`

- [ ] **Step 1: Verify the tree is the reviewed state**

Run: `cd /data2/s2chitni/cocossim && git status --short`
Expected: exactly the 6 modified files above plus untracked `tests/` (and untracked `dramsim3/` which is the vendored submodule-like dir — check `git log --oneline -3` shows `8dfaf9d` spec amendment at HEAD).

Note: `dramsim3/` appears untracked. Run `git check-ignore dramsim3 && echo IGNORED`. If it is not ignored and not previously tracked, do NOT add it in this task — commit only the 6 files + `tests/`.

- [ ] **Step 2: Run the regression suite**

Run: `export CCACHE_DIR=/data2/s2chitni/.tmp/ccache && cmake --build build -j8 && tests/regression.sh`
Expected: `==== 5 passed, 0 failed`

- [ ] **Step 3: Commit**

```bash
git add examples/cnn_model.txt include/Arch.h include/State.h src/Arch.cc \
        src/frontends/standard/StandardLayers.cc src/frontends/standard/StandardParser.cc tests/
git commit -m "Fix scheduler distribution, WS reservation, Softmax conservation, State init; add regression suite"
```
(with the two Global Constraints trailer lines)

---

### Task 1: Promote `buffer_size_bytes`, `dram_enq_per_cycle` to flags; add `-job_overhead`

**Files:**
- Modify: `include/global.h:29-31` (consts → externs), `src/global.cc` (definitions), `src/frontends/standard/StandardParser.cc` (flags + validation), `src/Arch.cc` (apply job overhead at dispatch)
- Create: `tests/tpuv6e.sh`

**Interfaces:**
- Consumes: `parse_args({{"-c",&cores},...})` pattern in `StandardParser::make_arch` (`src/frontends/standard/StandardParser.cc:19`); dispatch site `state->init()` in `src/Arch.cc` (inside the `while (any_job_assigned)` loop).
- Produces: globals `int buffer_size_bytes` (bytes), `int dram_enq_per_cycle`, `int job_overhead_cycles` — mutable, defined in `src/global.cc`, declared `extern` in `include/global.h`. Flags `-buf_mb <MiB>` (default 8), `-dram_enq <n>` (default 9), `-job_overhead <cycles>` (default 0). Later tasks read these globals.

- [ ] **Step 1: Create `tests/tpuv6e.sh` with the harness scaffold and the three failing tests**

```bash
#!/usr/bin/env bash
# TPUv6e-model tests (spec: docs/superpowers/specs/2026-08-27-tpuv6e-model-calibration-design.md)
# V1  -buf_mb reaches the layer generator (Softmax split count changes)
# V2  -dram_enq throttles memory issue (cycles increase)
# V3  -job_overhead adds fixed cycles per job
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/build/perf_model"
WORK="${TMPDIR:-/tmp}/cocossim_tpuv6e_$$"
mkdir -p "$WORK"
cd "$REPO/build" || exit 2
[ -x "$BIN" ] || { echo "build/perf_model missing - build first" >&2; exit 2; }
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }
cycles_of() { awk '/^Cycles/{print $2; exit}' "$1"; }

# V1: Softmax 4096 with default 8 MiB buffer splits into 4 jobs (regression T2);
# with -buf_mb 1 the split factor is max(ceil(32MiB/1MiB)=32, ceil(4096/1024)=4)=32
# so Mp=128 and 32 jobs are created.
printf 'Softmax 4096\n' > "$WORK/v1.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -buf_mb 1 -i "$WORK/v1.txt" -o "$WORK/v1_stats.txt" > "$WORK/v1.log" 2>&1
n_jobs=$(grep -c 'Job Type: 1' "$WORK/v1.log")
if [ "$n_jobs" -eq 32 ]; then ok "V1 -buf_mb 1 -> 32 softmax jobs"; else bad "V1 got $n_jobs jobs (want 32)"; fi

# V2: memory-hungry LayerNorm; throttling enqueue to 1 beat/cycle must cost cycles.
printf 'LayerNorm 4096 1024\n' > "$WORK/v2.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v2.txt" -o "$WORK/v2_base.txt" > /dev/null 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_enq 1 -i "$WORK/v2.txt" -o "$WORK/v2_slow.txt" > /dev/null 2>&1
cb=$(cycles_of "$WORK/v2_base.txt"); cs=$(cycles_of "$WORK/v2_slow.txt")
if [ -n "$cb" ] && [ -n "$cs" ] && [ "$cs" -gt "$cb" ]; then
  ok "V2 -dram_enq 1 slows memory-bound run ($cb -> $cs)"
else bad "V2 cycles base=$cb slow=$cs"; fi

# V3: one-job workload; -job_overhead 1000 must add >= 1000 cycles.
printf 'Activation 64\n' > "$WORK/v3.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v3.txt" -o "$WORK/v3_base.txt" > /dev/null 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -job_overhead 1000 -i "$WORK/v3.txt" -o "$WORK/v3_ovh.txt" > /dev/null 2>&1
cb=$(cycles_of "$WORK/v3_base.txt"); co=$(cycles_of "$WORK/v3_ovh.txt")
if [ -n "$cb" ] && [ -n "$co" ] && [ $((co - cb)) -ge 1000 ]; then
  ok "V3 -job_overhead 1000 adds $((co - cb)) cycles"
else bad "V3 base=$cb overhead=$co"; fi

echo "==== $PASS passed, $FAIL failed (outputs in $WORK)"
exit "$FAIL"
```

Run: `chmod +x tests/tpuv6e.sh`

- [ ] **Step 2: Run to verify all three fail for the right reason**

Run: `tests/tpuv6e.sh`
Expected: 3 FAIL. The logs must show `Failed to parse passed flag: '-buf_mb'` (thrown by `parse_args`) — proving the flags don't exist yet, not some other breakage.

- [ ] **Step 3: Implement**

In `include/global.h`, replace the two consts:
```cpp
extern int dram_enq_per_cycle;
extern int buffer_size_bytes;
extern int job_overhead_cycles;
```
(delete `const int dram_enq_per_cycle = 9;` and `const int buffer_size_bytes = 8 * 1024 * 1024;`)

In `src/global.cc` add:
```cpp
int dram_enq_per_cycle = 9;
int buffer_size_bytes = 8 * 1024 * 1024;
int job_overhead_cycles = 0;
```

In `src/frontends/standard/StandardParser.cc`, extend `make_arch`:
```cpp
  int buf_mb = 8;
  parse_args({{"-c", &cores},
              {"-sa_sz", &sa_sz},
              {"-vu_sz", &vu_sz},
              {"-ws", &ws},
              {"-buf_mb", &buf_mb},
              {"-dram_enq", &dram_enq_per_cycle},
              {"-job_overhead", &job_overhead_cycles}},
             "-c            number of cores\n"
             "-sa_sz        size of the systolic array\n"
             "-vu_sz        size of the vector unit\n"
             "-ws           weight stationary (1) or output stationary (0)\n"
             "-buf_mb       on-chip buffer size in MiB (default 8)\n"
             "-dram_enq     memory requests issued per cycle (default 9)\n"
             "-job_overhead fixed dispatch overhead per job in cycles (default 0)");
```
and after the existing validation blocks:
```cpp
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
  buffer_size_bytes = buf_mb * 1024 * 1024;
```

In `src/Arch.cc`, at the dispatch site, add one line directly after `state->init();`:
```cpp
          state->init();
          state->min_stage_cycles += job_overhead_cycles;
```

- [ ] **Step 4: Build and run both suites**

Run: `export CCACHE_DIR=/data2/s2chitni/.tmp/ccache && cmake --build build -j8 && tests/regression.sh && tests/tpuv6e.sh`
Expected: 5/5 and 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add include/global.h src/global.cc src/frontends/standard/StandardParser.cc src/Arch.cc tests/tpuv6e.sh
git commit -m "Promote buffer/dram-enqueue knobs to flags; add -job_overhead"
```

---

### Task 2: `-dram_ini` flag (and fix the setup ordering it requires)

`mem::setup()` currently runs in `main` BEFORE arguments are parsed, and hard-codes the ini path (`src/memory.cc:60`). The DRAM config must move behind a flag, which forces `mem::setup()` to run after `parse_args` — but before `new StandardArch` (the unit constructors read `bytes_per_tx`). Solution: `make_arch` calls `mem::setup()` itself, between validation and arch construction.

**Files:**
- Modify: `include/frontends/ArchParser.h` (parse `-dram_ini`, declare extern), `src/main.cc` (define the global; remove the `mem::setup()` call), `src/memory.cc` (use the global; fail cleanly on a missing file), `src/frontends/standard/StandardParser.cc` (call `mem::setup()`)
- Test: `tests/tpuv6e.sh` (V4)

**Interfaces:**
- Consumes: `mem::setup()` (`src/memory.cc:59`), `layer_file`/`ofile` extern-in-`ArchParser.h`, defined-in-`main.cc` pattern.
- Produces: global `std::string dram_ini_path` (default `"../dramsim3/configs/HBM2_8Gb_x128.ini"`); flag `-dram_ini <path>`. `mem::setup()` is now called from `StandardParser::make_arch`, NOT from `main`. Task 3's ini and Task 12's config script rely on this flag.

- [ ] **Step 1: Append the failing test V4 to `tests/tpuv6e.sh`** (before the final `echo "===="` line; same for every later task)

```bash
# V4: -dram_ini selects the DRAM config. GDDR6 (x16 bus) has a different
# request size than HBM2 (x128), so REQUEST SIZE BYTES must change; a
# missing file must die cleanly with a message naming the path.
printf 'LayerNorm 1024 1024\n' > "$WORK/v4.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini ../dramsim3/configs/GDDR6_8Gb_x16.ini \
  -i "$WORK/v4.txt" -o "$WORK/v4_stats.txt" > "$WORK/v4.log" 2>&1
rc=$?
req=$(awk '/REQUEST SIZE BYTES/{print $NF; exit}' "$WORK/v4.log")
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -dram_ini /nonexistent/nope.ini \
  -i "$WORK/v4.txt" -o "$WORK/v4b.txt" > "$WORK/v4b.log" 2>&1
rcb=$?
if [ "$rc" -eq 0 ] && [ -n "$req" ] && [ "$req" -ne 64 ] && [ "$rcb" -eq 1 ] && grep -q 'nope.ini' "$WORK/v4b.log"; then
  ok "V4 -dram_ini honored (req size $req) and missing file rejected"
else
  bad "V4 rc=$rc req=${req:-none} missing-file rc=$rcb"
fi
```

- [ ] **Step 2: Run to verify V4 fails with the parse error**

Run: `tests/tpuv6e.sh`
Expected: V1-V3 PASS, V4 FAIL; `$WORK/v4.log` contains `Failed to parse passed flag: '-dram_ini'`.

- [ ] **Step 3: Implement**

`include/frontends/ArchParser.h` — add next to the existing externs and the `-o` branch:
```cpp
extern std::string dram_ini_path;
```
```cpp
      } else if (strcmp(argv[i], "-dram_ini") == 0) {
        dram_ini_path = argv[++i];
```
and extend the `-h` help text with `"-dram_ini <file> DRAMSim3 config ini (default ../dramsim3/configs/HBM2_8Gb_x128.ini)\n"`.

`src/main.cc` — next to `std::string ofile;` add:
```cpp
std::string dram_ini_path = "../dramsim3/configs/HBM2_8Gb_x128.ini";
```
and DELETE the line `mem::setup();` (keep the `#include "memory.h"`).

`src/memory.cc` — in `mem::setup()` replace the hard-coded path:
```cpp
#include <fstream>
```
```cpp
void mem::setup() {
  std::ifstream probe(dram_ini_path);
  if (!probe.good()) {
    std::cerr << "Error: DRAM config file not found: " << dram_ini_path << std::endl;
    exit(1);
  }
  probe.close();
  dramsim3config = new dramsim3::Config(dram_ini_path, "./");
  ...
```
(`dramsim3::Config` takes a `std::string`; also add `#include "frontends/ArchParser.h"` for the extern.)

`src/frontends/standard/StandardParser.cc` — add `#include "memory.h"` and, after the validation blocks / `buffer_size_bytes` assignment, before `arch_config = ...`:
```cpp
  mem::setup();
```

- [ ] **Step 4: Build; run both suites**

Expected: regression 5/5 (unchanged default path proves ordering is safe), tpuv6e 4/4.

- [ ] **Step 5: Commit**

```bash
git add include/frontends/ArchParser.h src/main.cc src/memory.cc src/frontends/standard/StandardParser.cc tests/tpuv6e.sh
git commit -m "Add -dram_ini flag; move mem::setup after argument parsing"
```

---

### Task 3: HBM2e_v6e DRAMSim3 config + bandwidth validation

Spec §3.3. Derivation: base `HBM2_8Gb_x128.ini` is 8 channels × 128-bit bus × 2 Gbps (tCK=1 ns) = 256 GB/s, 64 B requests. Target ≈1.64 TB/s aggregate and 32 GB capacity: **32 channels** (power of two, required by the bit-sliced address mapping) at **tCK = 0.625 ns** (3.2 Gbps/pin, within HBM2e silicon): 32 × 16 B × 3.2 GT/s = 1638 GB/s; 32 × 1024 MiB channel_size = 32 GiB.

**Files:**
- Create: `dramsim3/configs/HBM2e_v6e.ini`
- Test: `tests/tpuv6e.sh` (V5)

**Interfaces:**
- Consumes: `-dram_ini` (Task 2), `-dram_enq`, `-f` flags.
- Produces: `dramsim3/configs/HBM2e_v6e.ini` — referenced by Task 12's config script and the future calibration plan.

- [ ] **Step 1: Append failing test V5**

```bash
# V5: HBM2e_v6e ini must deliver >= 800 GB/s achieved on a memory-bound
# streaming workload (target 1638 GB/s x typical DRAM efficiency), and the
# request size must stay 64B. Two caps must be non-binding for the test to
# measure DRAM: enqueue width (-dram_enq 32 at 1.75 GHz = 3.58 TB/s issue)
# and compute (-vu_sz 2048: 200M elems / 2048 lanes = 98k compute cycles,
# far below the ~1.4M memory cycles; at the default vu_sz 64 the run is
# compute-bound and measures nothing about the ini).
# Workload: Activation 200000000 = 200M elems: 400 MB read + 400 MB write.
printf 'Activation 200000000\n' > "$WORK/v5.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 2048 -f 1.75 -dram_enq 32 -dram_ini ../dramsim3/configs/HBM2e_v6e.ini \
  -i "$WORK/v5.txt" -o "$WORK/v5_stats.txt" > "$WORK/v5.log" 2>&1
rc=$?
cyc=$(cycles_of "$WORK/v5_stats.txt")
req=$(awk '/REQUEST SIZE BYTES/{print $NF; exit}' "$WORK/v5.log")
# achieved GB/s = 800e6 bytes / (cycles / 1.75 GHz) / 1e9 = 800*1.75/cycles * 1e6... in awk:
bw=$(awk -v c="${cyc:-0}" 'BEGIN{ if (c>0) printf "%d", 800000000.0 / (c / 1.75) ; else print 0 }')
if [ "$rc" -eq 0 ] && [ "$req" = "64" ] && [ "$bw" -ge 800 ] && [ "$bw" -le 1800 ]; then
  ok "V5 HBM2e_v6e achieves ${bw} GB/s (cycles=$cyc)"
else
  bad "V5 rc=$rc req=${req:-?} bw=${bw:-?} GB/s cycles=${cyc:-?}"
fi
```
(units: `cycles / 1.75` is elapsed nanoseconds at 1.75 GHz, and bytes/ns = GB/s, so the awk expression yields GB/s directly.)

- [ ] **Step 2: Run to verify V5 fails**

Expected: V5 FAIL with the missing-file diagnostic from Task 2 (`HBM2e_v6e.ini` doesn't exist).

- [ ] **Step 3: Create `dramsim3/configs/HBM2e_v6e.ini`**

Copy `dramsim3/configs/HBM2_8Gb_x128.ini` and change exactly these lines (leave `[power]`, `[other]`, and every unmentioned timing line as-is):
```ini
[timing]
tCK = 0.625

[system]
channels = 32
```
Add a header comment at the top of the file:
```ini
; HBM2e-class config approximating TPU v6e aggregate memory: 32 channels x
; 128-bit x 3.2 Gbps (tCK=0.625ns) = 1638 GB/s, 32 x 1 GiB = 32 GiB.
; Timing values are inherited from HBM2_8Gb_x128 in DRAM-cycle units; only
; achieved bandwidth is calibration-relevant (spec 3.3) - latencies get
; validated against the Phase C stream measurement.
```

- [ ] **Step 4: Build not needed (config only); run both suites**

Expected: regression 5/5, tpuv6e 5/5. If V5's bandwidth lands below 800 GB/s, diagnose before touching the threshold: check the `DRAM CMDs` counter in `v5.log` grows ~12.5M (800 MB / 64 B), and check `dram_enq` reached 32 (a typo'd flag falls back to a parse error, not a silent default).

- [ ] **Step 5: Commit**

```bash
git add dramsim3/configs/HBM2e_v6e.ini tests/tpuv6e.sh
git commit -m "Add HBM2e_v6e DRAMSim3 config (~1.64 TB/s, 32 GiB)"
```

---

### Task 4: Preserve true M in OS-mode SA jobs

`createSAJobs` (`src/frontends/standard/StandardLayers.cc:23`) is always called with `m = sa_sz_allo`: a `Matmul 1 K N` becomes a job claiming `M = 64`. This destroys the under-fill information that Task 5's cycle accounting and Task 6's FLOP-utilization stat report, and mis-models decode (M=1) read traffic. Fix: callers pass the layer's true M; `createSAJobs` splits it into `div_ru(M, sa_sz)` jobs, the last one carrying the remainder.

For M an exact multiple of `sa_sz` (every shipped example: cnn conv M=50176=784×64, matmul M=1 → 1 job either way but dims change 64→1), job COUNT is unchanged; only sub-`sa_sz` dims become honest.

**Files:**
- Modify: `src/frontends/standard/StandardLayers.cc` (`createSAJobs` and its call sites in `Matmul`, `Conv`, `MatmulAct`, `ActMatmul`, `SelfAttention`)
- Test: `tests/tpuv6e.sh` (V6)

**Interfaces:**
- Consumes: `SystolicArray::SysArrayJob(int m, int k, int n)`.
- Produces: new signature `JobList createSAJobs(int M, int K, int N, int sa_sz, int n_cores = 1)` — M is the layer's TRUE row count; the function derives job count and per-job m internally. Task 9's `Transformer` consumes this contract indirectly via `Matmul()` (its attention score/AV jobs construct `SysArrayJob` directly).

- [ ] **Step 1: Append failing test V6**

```bash
# V6: OS-mode job dims must carry the true M. Matmul 1 256 256 at -sa_sz 64
# must produce exactly one SA job printed as "Dims: 1 x 256 x 256" (the old
# code prints "Dims: 64 x 256 x 256"). Matmul 100 256 256 must produce
# ceil(100/64)=2 jobs: one 64-row and one 36-row.
printf 'Matmul 1 256 256\n' > "$WORK/v6.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v6.txt" -o "$WORK/v6_stats.txt" > "$WORK/v6.log" 2>&1
printf 'Matmul 100 256 256\n' > "$WORK/v6b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v6b.txt" -o "$WORK/v6b_stats.txt" > "$WORK/v6b.log" 2>&1
if grep -q 'Dims: 1 x 256 x 256' "$WORK/v6.log" \
   && grep -q 'Dims: 64 x 256 x 256' "$WORK/v6b.log" \
   && grep -q 'Dims: 36 x 256 x 256' "$WORK/v6b.log"; then
  ok "V6 OS jobs carry true M (1; 64+36)"
else
  bad "V6 job dims wrong (see $WORK/v6.log, $WORK/v6b.log)"
fi
```

- [ ] **Step 2: Run to verify V6 fails**

Expected: V6 FAIL; `v6.log` shows `Dims: 64 x 256 x 256` (the rounding) and `v6b.log` shows only one job (old `num_jobs = max(1, 100/64) = 1`).

- [ ] **Step 3: Implement**

Replace `createSAJobs` in `src/frontends/standard/StandardLayers.cc`:
```cpp
// OS-mode job creation. M is the layer's TRUE row count: it is split into
// ceil(M / sa_sz) row-block jobs, the last carrying the remainder, so job
// dims preserve under-fill information (spec 3.5). N is split across cores.
JobList createSAJobs(int M, int K, int N, int sa_sz, int n_cores = 1) {
  JobList jobs;
  int core_n = N / n_cores;
  int num_jobs = div_ru(M, sa_sz);
  static std::vector<int> core_task_counters(n_cores, 0);
  for (int core = 0; core < n_cores; ++core) {
    for (int job = 0; job < num_jobs; ++job) {
      int m = std::min(sa_sz, M - job * sa_sz);
      auto sys_job = new SystolicArray::SysArrayJob(m, K, core_n);
      sys_job->core_id = core;
      sys_job->task_idx = core_task_counters[core]++;
      jobs.push_back(sys_job);
    }
  }
  return jobs;
}
```
(Keep the known pre-existing `static core_task_counters` first-call-sizing quirk as-is — it is tracked separately; do not fix it here.)

Update every call site to the new contract (pass true M and the array size; delete the caller-side `num_jobs` computation):
- `Matmul` OS branch: `JobList matmul_layers = createSAJobs(M, K, N, a_config.sa_sz_allo, a_config.n_cores);`
- `Conv` OS branch: same as Matmul.
- `MatmulAct` WS branch: this caller splits K, not M — preserve its job count exactly: `createSAJobs(M, a_config.sa_sz_allo, N, ...)` created `num_jobs = ceil(K/sa)` copies. Recreate that behavior explicitly with the new function by calling it once per K-block:
  ```cpp
  JobList matmul_layers;
  for (int kb = 0; kb < std::max(1, int(std::ceil(float(K) / a_config.sa_sz_allo))); ++kb) {
    JobList part = createSAJobs(M, a_config.sa_sz_allo, N, a_config.sa_sz_allo);
    matmul_layers.insert(matmul_layers.end(), part.begin(), part.end());
  }
  ```
  Note this multiplies job count by `ceil(M/sa)` vs. the old single-job-per-K-block only when M > sa — the old code modeled ONE M=sa job per K block regardless of true M, i.e. it dropped rows; carrying all rows is the honest model. `MatmulAct` is not used by any shipped example (`grep -rl MatmulAct examples/` is empty) so no example output changes.
- `MatmulAct` OS branch: `createSAJobs(M, K, N, a_config.sa_sz_allo)`.
- `ActMatmul` WS branch: same K-block loop as MatmulAct WS; OS branch: `createSAJobs(M, K, N, a_config.sa_sz_allo)`.
- `SelfAttention` OS branch (6 sites): replace each `createSAJobs(a_config.sa_sz_allo, X, Y, num_jobs)` with `createSAJobs(M, X, Y, a_config.sa_sz_allo)` keeping each site's existing K/N arguments; delete the local `num_jobs`.

- [ ] **Step 4: Build; run both suites**

Expected: regression 5/5 (cnn/transformer examples use exact-multiple or M=1 shapes; T1-T5 assert no OS SA dims), tpuv6e 6/6.

- [ ] **Step 5: Commit**

```bash
git add src/frontends/standard/StandardLayers.cc tests/tpuv6e.sh
git commit -m "Preserve true M in OS systolic-array jobs"
```

---

### Task 5: Per-unit cycle accounting

Spec §3.5. Every simulated cycle per unit lands in exactly one of {busy, busy-but-underfilled, stalled-on-memory, idle-no-ready-work}. Prerequisite fix: `UPDATE_IDLEMEM` is a no-op in non-VCD builds (`include/State.h:32`), so `is_idle_from_memory` is never maintained — make it unconditional.

**Files:**
- Modify: `include/State.h` (macro + counters + virtual), `include/units/standard/SysArray.h` + `include/units/standard/VectorUnit.h` (`is_underfilled` overrides), `src/Arch.cc` (classification in the increment loop), `src/main.cc` (stats output)
- Test: `tests/tpuv6e.sh` (V7)

**Interfaces:**
- Consumes: `increment()` return value (`state != idle`) in the `src/Arch.cc` per-cycle loop; `is_idle_from_memory`; job dims.
- Produces: on `State`: `uint64_t acct_busy = 0, acct_underfilled = 0, acct_memstall = 0;` and `virtual bool is_underfilled() const;`. Stats-file lines (one per unit, after the existing per-unit `pct_active` lines):
  `ACCT <ty_string> <unit_idx> busy <n> underfilled <n> memstall <n> idle <n>`
  where idle = total cycles − the other three. Task 6 extends this same line with `work`.

- [ ] **Step 1: Append failing test V7**

```bash
# V7: cycle-accounting invariant + attribution.
# (a) For every unit: busy+underfilled+memstall+idle == Cycles.
# (b) A memory-bound workload must show memstall > 0 on the vector unit.
#     Activation 50M elems at -vu_sz 1024: compute = 50M/1024 = 49k cycles,
#     memory = 100 MB read + 100 MB write = 3.1M beats at ~4-9 beats/cycle
#     >> compute, so the VPU spends most cycles waiting on DRAM. (At the
#     default vu_sz 64 the VPU demand of 128 B/cycle sits below HBM2's
#     ~256 B/cycle and the run is compute-bound: memstall would be 0.)
# (c) An underfilling GEMM (M=1 on a 64-wide array) must show
#     underfilled > 0 on the systolic array.
printf 'Activation 50000000\n' > "$WORK/v7.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 1024 -f 1 -i "$WORK/v7.txt" -o "$WORK/v7_stats.txt" > "$WORK/v7.log" 2>&1
printf 'Matmul 1 256 256\n' > "$WORK/v7b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v7b.txt" -o "$WORK/v7b_stats.txt" > "$WORK/v7b.log" 2>&1
inv=$(awk '/^Cycles/{c=$2} /^ACCT/{ if ($5+$7+$9+$11 != c) print "bad:" $0 }' "$WORK/v7_stats.txt" "$WORK/v7b_stats.txt")
ms=$(awk '/^ACCT VECTOR_UNIT/{print $9; exit}' "$WORK/v7_stats.txt")
uf=$(awk '/^ACCT SYSTOLIC_ARRAY/{print $7; exit}' "$WORK/v7b_stats.txt")
if [ -z "$inv" ] && [ "${ms:-0}" -gt 0 ] && [ "${uf:-0}" -gt 0 ]; then
  ok "V7 accounting sums to Cycles; memstall=$ms underfilled=$uf attributed"
else
  bad "V7 invariant='$inv' memstall=${ms:-?} underfilled=${uf:-?}"
fi
```

- [ ] **Step 2: Run to verify V7 fails**

Expected: V7 FAIL — no `ACCT` lines exist in the stats files yet.

- [ ] **Step 3: Implement**

`include/State.h` — make the non-VCD `UPDATE_IDLEMEM` real:
```cpp
#define UPDATE_IDLEMEM(to) is_idle_from_memory = to
```
(replace the empty non-VCD definition; the VCD definition already assigns.)

Add to `struct State` (after `bool is_idle_from_memory = false;`):
```cpp
  // Per-unit cycle accounting (spec 3.5): every non-idle cycle is classified
  // in Arch::get_cycles as memstall (waiting on DRAM with no compute left),
  // underfilled (working, but the job cannot fill the unit), or busy.
  uint64_t acct_busy = 0;
  uint64_t acct_underfilled = 0;
  uint64_t acct_memstall = 0;
  virtual bool is_underfilled() const { return false; }
```

`include/units/standard/SysArray.h` — add to `SysArrayState` (public):
```cpp
    bool is_underfilled() const override {
      if (j == nullptr) return false;
      auto *sj = (SysArrayJob *) j;
      return std::min(sz, sj->M) * std::min(sz, sj->N) < sz * sz;
    }
```
(add `#include <algorithm>` if not already transitively present).

`include/units/standard/VectorUnit.h` — add to `VecUnitState` (public):
```cpp
    // Approximation: a REDUCE/BROADCAST pass with fewer parallel rows than
    // lanes leaves lanes idle; finer per-phase modeling is not needed for
    // the paper's per-unit attribution.
    bool is_underfilled() const override {
      if (j == nullptr) return false;
      return ((VecUnitJob *) j)->parallel_dimension < sz;
    }
```

`src/Arch.cc` — in the per-cycle loop, replace:
```cpp
    for (int i = 0; i < states.size(); ++i) {
      bool is_active = states[i]->increment(enqueue_job, total_idle, n_idle_units);
      if (is_active) {
        per_array_act[i]++;
      }
    }
```
with:
```cpp
    for (int i = 0; i < states.size(); ++i) {
      State *s = states[i];
      bool is_active = s->increment(enqueue_job, total_idle, n_idle_units);
      if (is_active) {
        per_array_act[i]++;
        if (s->is_idle_from_memory) s->acct_memstall++;
        else if (s->is_underfilled()) s->acct_underfilled++;
        else s->acct_busy++;
      }
    }
```

`src/main.cc` — after the existing per-unit `pct_active` fprintf loop, add:
```cpp
  for (int i = 0; i < arch->states.size(); ++i) {
    State *s = arch->states[i];
    uint64_t accounted = s->acct_busy + s->acct_underfilled + s->acct_memstall;
    fprintf(f, "ACCT %s %d busy %llu underfilled %llu memstall %llu idle %llu\n",
            s->get_ty_string().c_str(), i,
            (unsigned long long) s->acct_busy,
            (unsigned long long) s->acct_underfilled,
            (unsigned long long) s->acct_memstall,
            (unsigned long long) (gcycles - accounted));
  }
```
Note: counters are cumulative across periods; with `periods = 1` (the only supported configuration — multi-period is broken pre-existing) totals equal the single phase.

- [ ] **Step 4: Build; run both suites**

Expected: regression 5/5, tpuv6e 7/7. If V7(b) memstall is 0, first check `is_idle_from_memory` is being set: the non-VCD macro fix is the usual culprit.

- [ ] **Step 5: Commit**

```bash
git add include/State.h include/units/standard/SysArray.h include/units/standard/VectorUnit.h src/Arch.cc src/main.cc tests/tpuv6e.sh
git commit -m "Add per-unit cycle accounting (busy/underfilled/memstall/idle)"
```

---

### Task 6: Effective FLOP utilization

Spec §3.5: true work ÷ capacity, the stat commensurable with XProf. SA capacity is `sz*sz` MACs/cycle; VPU capacity is `sz` lane-ops/cycle.

**Files:**
- Modify: `include/State.h` (counter), `src/units/standard/SysArray.cc` (accumulate at completion, both WS and OS), `src/units/standard/VectorUnit.cc` (accumulate at completion), `src/main.cc` (extend ACCT line)
- Test: `tests/tpuv6e.sh` (V8)

**Interfaces:**
- Consumes: ACCT line from Task 5; job completion sites (`TO_IDLE_CLEANUP()` call sites).
- Produces: `uint64_t total_work = 0;` on `State` (SA: MACs = M·K·N per job; VPU: lane-ops = linearized·parallel per job). ACCT line becomes:
  `ACCT <ty> <idx> busy <n> underfilled <n> memstall <n> idle <n> work <n> eff_util <float>`
  where `eff_util = work / (capacity * (busy + underfilled))`, 0 when never active. The calibration plan consumes `eff_util`.

- [ ] **Step 1: Append failing test V8**

```bash
# V8: effective FLOP utilization must expose under-fill that pct_active
# hides. Matmul 1 256 256 on a 64-wide array: work = 1*256*256 = 65536
# MACs; a full-width job would be 64x. eff_util must be < 0.05 while the
# ACCT work field equals exactly 65536.
printf 'Matmul 1 256 256\n' > "$WORK/v8.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v8.txt" -o "$WORK/v8_stats.txt" > "$WORK/v8.log" 2>&1
read -r wk eu <<< "$(awk '/^ACCT SYSTOLIC_ARRAY/{print $13, $15; exit}' "$WORK/v8_stats.txt")"
low=$(awk -v e="${eu:-1}" 'BEGIN{print (e > 0 && e < 0.05) ? 1 : 0}')
if [ "${wk:-0}" = "65536" ] && [ "$low" -eq 1 ]; then
  ok "V8 work=65536 MACs, eff_util=$eu"
else
  bad "V8 work=${wk:-?} eff_util=${eu:-?}"
fi
```

- [ ] **Step 2: Run to verify V8 fails** — the ACCT line has no `work`/`eff_util` fields yet (awk prints empty).

- [ ] **Step 3: Implement**

`include/State.h` — add below the acct counters:
```cpp
  uint64_t total_work = 0;// SA: MACs; VPU: lane-ops. Accumulated at job completion.
```

`src/units/standard/SysArray.cc` — in BOTH job-completion branches (the WS `case write:` and the OS `case write:` blocks that call `TO_IDLE_CLEANUP()`), add immediately BEFORE the `state_transfer(... idle ...)` / `TO_IDLE_CLEANUP()` pair:
```cpp
              total_work += (uint64_t) sj->M * sj->K * sj->N;
```

`src/units/standard/VectorUnit.cc` — in `case VectorUnit::VPUState::write:` before `TO_IDLE_CLEANUP()`:
```cpp
        total_work += (uint64_t) sj->linearized_dimension * sj->parallel_dimension;
```

`src/main.cc` — extend the ACCT fprintf: compute capacity per unit type and append two fields:
```cpp
  for (int i = 0; i < arch->states.size(); ++i) {
    State *s = arch->states[i];
    uint64_t accounted = s->acct_busy + s->acct_underfilled + s->acct_memstall;
    uint64_t active = s->acct_busy + s->acct_underfilled;
    double cap = (s->get_ty_idx() == SYSTOLIC_ARRAY_IDX)
                     ? (double) s->sz * s->sz
                     : (double) s->sz;
    double eff = active > 0 ? (double) s->total_work / (cap * (double) active) : 0.0;
    fprintf(f, "ACCT %s %d busy %llu underfilled %llu memstall %llu idle %llu work %llu eff_util %f\n",
            s->get_ty_string().c_str(), i,
            (unsigned long long) s->acct_busy,
            (unsigned long long) s->acct_underfilled,
            (unsigned long long) s->acct_memstall,
            (unsigned long long) (gcycles - accounted),
            (unsigned long long) s->total_work, eff);
  }
```
`State::sz` exists (`include/State.h:63`) but the derived classes shadow it with their own `sz` member — set the BASE member too, or simpler: in `SysArrayState` and `VecUnitState` constructors add `State::sz = sz;`. Add `#include "frontends/standard/StandardUnits.h"` to `main.cc` for `SYSTOLIC_ARRAY_IDX`.

Update V7's awk field positions? No — V7 reads fields 5/7/9/11 which keep their positions (new fields append). Verify V7 still passes.

- [ ] **Step 4: Build; run both suites** — regression 5/5, tpuv6e 8/8.

- [ ] **Step 5: Commit**

```bash
git add include/State.h src/units/standard/SysArray.cc src/units/standard/VectorUnit.cc src/main.cc tests/tpuv6e.sh
git commit -m "Add effective FLOP/lane utilization to per-unit accounting"
```

---

### Task 7: Binary elementwise — `n_read_operands` and the `Add` layer

Spec §3.4, Gate 0. A `VecUnitJob` gains a read-operand multiplier; unbuffered first-phase reads scale by it. New frontend keyword `Add M N` (elementwise sum of two M×N tensors).

**Files:**
- Modify: `include/units/standard/VectorUnit.h` (field), `src/units/standard/VectorUnit.cc` (`init()` read amount), `src/frontends/standard/StandardLayers.cc` (Add layer + registration)
- Test: `tests/tpuv6e.sh` (V9)

**Interfaces:**
- Consumes: `VecUnitJob` constructor; `getLayerLambda` registry.
- Produces: `int n_read_operands = 1;` public field on `VecUnitJob` (set after construction); layer keyword `Add` with dims `{N}` or `{M, N}` (element count = product). Tasks 9-10 create Add-style jobs directly with `n_read_operands = 2`.

- [ ] **Step 1: Append failing test V9**

```bash
# V9: Add must read two operands. Same element count as Activation ->
# roughly 2x the read traffic; total DRAM CMDs (reads+writes) must be at
# least 1.4x the Activation run's. Both runs memory-dominated.
printf 'Activation 8000000\n' > "$WORK/v9a.txt"
printf 'Add 8000000\n'        > "$WORK/v9b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v9a.txt" -o "$WORK/v9a_s.txt" > "$WORK/v9a.log" 2>&1
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v9b.txt" -o "$WORK/v9b_s.txt" > "$WORK/v9b.log" 2>&1
ca=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v9a.log" | tail -1 | awk '{print $3}')
cb=$(grep -o 'DRAM CMDs: [0-9]*' "$WORK/v9b.log" | tail -1 | awk '{print $3}')
ratio_ok=$(awk -v a="${ca:-0}" -v b="${cb:-0}" 'BEGIN{print (a>0 && b>=1.4*a) ? 1 : 0}')
if [ "$ratio_ok" -eq 1 ]; then
  ok "V9 Add reads two operands (DRAM CMDs $ca -> $cb)"
else
  bad "V9 DRAM CMDs activation=$ca add=$cb"
fi
```

- [ ] **Step 2: Run to verify V9 fails** — `Add` is an unknown layer type: `v9b.log` shows `Unknown layer type: Add`.

- [ ] **Step 3: Implement**

`include/units/standard/VectorUnit.h`, in `VecUnitJob`:
```cpp
    // Number of input tensors this job streams from memory (1 = unary,
    // 2 = binary elementwise such as residual add). Scales unbuffered reads.
    int n_read_operands = 1;
```

`src/units/standard/VectorUnit.cc`, in `VecUnitState::init()`, the unbuffered branch:
```cpp
    first_phase_read = sj->linearized_dimension * sj->parallel_dimension * batch_size * data_type_width * sj->n_read_operands;
```

`src/frontends/standard/StandardLayers.cc` — add after `Activation`:
```cpp
JobPair Add(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;
  auto job = new VectorUnit::VecUnitJob(1, sz, false, {{VectorUnit::VPUPhase::BROADCAST, 1}});
  job->n_read_operands = 2;
  return {{job}, {job}};
}
```
and register in `getLayerLambda`:
```cpp
  if (layer_type == "Add")
    return Add;
```

- [ ] **Step 4: Build; run both suites** — regression 5/5, tpuv6e 9/9.

- [ ] **Step 5: Commit**

```bash
git add include/units/standard/VectorUnit.h src/units/standard/VectorUnit.cc src/frontends/standard/StandardLayers.cc tests/tpuv6e.sh
git commit -m "Add binary-elementwise read modeling and Add layer (Gate 0)"
```

---

### Task 8: `RMSNorm` layer + reusable softmax-job helper

Spec §3.4. RMSNorm = one square-accumulate reduction plus one scale broadcast: phases `{{REDUCE, 2}, {BROADCAST, 1}}` (vs LayerNorm's `{{REDUCE,1},{REDUCE,4},{BROADCAST,1}}`); constants are calibration targets, the shape is what matters. Also extract the row-chunking logic from `Softmax` into `makeSoftmaxJobs(row_len, n_rows)` so Task 9 can build attention softmax over non-square score matrices.

**Files:**
- Modify: `src/frontends/standard/StandardLayers.cc`
- Test: `tests/tpuv6e.sh` (V10)

**Interfaces:**
- Consumes: `LayerNorm`'s buffer-chunking pattern; `Softmax`'s split logic; `softmax_phases`.
- Produces: layer keyword `RMSNorm` (dims like LayerNorm: `{lin}` or `{par, lin}`); file-local helpers `JobList makeRMSNormJobs(int lin_dim, int par_dim)` and `JobList makeSoftmaxJobs(int row_len, int n_rows)`. `Softmax` behavior unchanged (T2 guards it). Tasks 9-10 call both helpers.

- [ ] **Step 1: Append failing test V10**

```bash
# V10: RMSNorm parses, runs, and is cheaper than LayerNorm at equal shape
# (2+1 phase-units/row vs 1+4+1): fewer cycles, same single-chunk job count.
printf 'RMSNorm 512 1024\n'   > "$WORK/v10a.txt"
printf 'LayerNorm 512 1024\n' > "$WORK/v10b.txt"
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v10a.txt" -o "$WORK/v10a_s.txt" > "$WORK/v10a.log" 2>&1
rc=$?
"$BIN" -c 1 -sa_sz 64 -vu_sz 64 -f 1 -i "$WORK/v10b.txt" -o "$WORK/v10b_s.txt" > "$WORK/v10b.log" 2>&1
cr=$(cycles_of "$WORK/v10a_s.txt"); cl=$(cycles_of "$WORK/v10b_s.txt")
if [ "$rc" -eq 0 ] && [ -n "$cr" ] && [ -n "$cl" ] && [ "$cr" -lt "$cl" ]; then
  ok "V10 RMSNorm runs and is cheaper than LayerNorm ($cr < $cl)"
else
  bad "V10 rc=$rc rms=$cr ln=$cl"
fi
```

- [ ] **Step 2: Run to verify V10 fails** — `Unknown layer type: RMSNorm`.

- [ ] **Step 3: Implement** in `src/frontends/standard/StandardLayers.cc`:

```cpp
static const std::vector<std::pair<VectorUnit::VPUPhase, int>> rmsnorm_phases =
    {{VectorUnit::VPUPhase::REDUCE, 2}, {VectorUnit::VPUPhase::BROADCAST, 1}};

// RMSNorm over par_dim rows of length lin_dim, chunked to the buffer the
// same way LayerNorm is.
static JobList makeRMSNormJobs(int lin_dim, int par_dim) {
  JobList jl;
  int par_acc = par_dim;
  int dec_amt = buffer_size_bytes / data_type_width / lin_dim;
  while (par_acc > 0) {
    jl.push_back(new VectorUnit::VecUnitJob(lin_dim, std::min(dec_amt, par_acc), false, rmsnorm_phases));
    par_acc -= dec_amt;
  }
  return jl;
}

JobPair RMSNorm(const ArchConfig &a_config, const LayerConfig &l_config) {
  int lin_dim, par_dim = 1;
  if (l_config.dimensions.size() == 1) {
    lin_dim = l_config.dimensions[0];
  } else if (l_config.dimensions.size() == 2) {
    par_dim = l_config.dimensions[0];
    lin_dim = l_config.dimensions[1];
  } else {
    std::cerr << "RMSNorm expects 1 or 2 dimensions, got " << l_config.dimensions.size() << std::endl;
    throw std::exception();
  }
  JobList jl = makeRMSNormJobs(lin_dim, par_dim);
  return {jl, jl};
}
```
Register `RMSNorm` in `getLayerLambda`.

Extract the softmax chunker (verbatim logic move from `Softmax`, generalized names):
```cpp
// n_rows independent softmax rows of length row_len, chunked so each job's
// working set fits the buffer and no job exceeds 1024 rows.
static JobList makeSoftmaxJobs(int row_len, int n_rows) {
  int spl = 1;
  int Mp = n_rows;
  if (row_len * n_rows * data_type_width * batch_size > buffer_size_bytes || Mp > 1024) {
    spl = std::max(div_ru(row_len * n_rows * data_type_width * batch_size, buffer_size_bytes),
                   div_ru(Mp, 1024));
    if (spl > Mp) {
      std::cerr << "Can't split this enough to fit inside buffer." << std::endl;
      throw std::exception();
    }
    Mp /= spl;
  }
  int n_jobs = div_ru(n_rows, Mp);
  JobList softmax_layer;
  for (int i = 0; i < n_jobs; ++i)
    softmax_layer.push_back(new VectorUnit::VecUnitJob(row_len, Mp, false, softmax_phases));
  return softmax_layer;
}
```
and shrink `Softmax` to parse dims then `JobList softmax_layer = makeSoftmaxJobs(M, M * heads); return {softmax_layer, softmax_layer};` — for the old code `row_len = M` and total rows `= M * heads` with the buffer test `heads*M*M*dtw <= buf`: `row_len*n_rows*dtw = M*(M*heads)*dtw` — identical. `Mp` init `M*heads` — identical. T2 must still pass.

- [ ] **Step 4: Build; run both suites** — regression 5/5 (T2 proves Softmax refactor is behavior-preserving), tpuv6e 10/10.

- [ ] **Step 5: Commit**

```bash
git add src/frontends/standard/StandardLayers.cc tests/tpuv6e.sh
git commit -m "Add RMSNorm layer; extract reusable softmax row-chunker"
```

---

### Task 9: `Transformer` composite — prefill

Spec §3.4. One text line expands to the full decoder-stack DAG. Line grammar (integers only — the file parser sscanf's `%d`, so mode is numeric; this is the documented encoding of the spec's `prefill|decode`):

`Transformer n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch`  (mode: 0=prefill, 1=decode)

Per layer: RMSNorm → {Q, K, V} projections → RoPE → per-head scores → softmax → per-head AV → O projection → residual-add(1) → RMSNorm → {gate, up} → SiLU-multiply (binary) → down → residual-add(2). Residual adds get BOTH true parents. This task implements prefill (`M = seq_len`); decode (`M = batch`, GQA sizing already in place) is verified in Task 10.

**Files:**
- Modify: `src/frontends/standard/StandardLayers.cc`
- Test: `tests/tpuv6e.sh` (V11)

**Interfaces:**
- Consumes: `createSAJobs(M, K, N, sa_sz, n_cores)` (Task 4), `makeRMSNormJobs`, `makeSoftmaxJobs` (Task 8), `VecUnitJob.n_read_operands` (Task 7), `Matmul(...)` JobPair, `connectJobLists`.
- Produces: layer keyword `Transformer`; file-local `JobPair Transformer(const ArchConfig &, const LayerConfig &)`. Task 11 adds the `-fuse_epilogue` branch inside it.

- [ ] **Step 1: Append failing test V11**

```bash
# V11: tiny prefill Transformer job count + DAG shape.
#   Transformer 1 8 2 2 16 8 0 1  with -sa_sz 4 -vu_sz 4 -c 1, OS mode.
# Derivation (head_dim=8/2=4, M=seq=8, S=8):
#   norm1: 1 (VPU)          qkv: 3 matmuls M8 K8 N8 -> ceil(8/4)=2 jobs each = 6 (SA)
#   rope: 1 (VPU)           scores: n_heads=2 SA jobs (M8 K4 N8)
#   softmax: rows=M*nh=16 len=8 -> 1 job (VPU)
#   av: 2 (SA)              o_proj: M8 K8 N8 -> 2 (SA)
#   res1: 1 (VPU)           norm2: 1 (VPU)
#   gate,up: M8 K8 N16 -> 2 each = 4 (SA)   silu_mul: 1 (VPU)
#   down: M8 K16 N8 -> 2 (SA)               res2: 1 (VPU)
# total = 1+6+1+2+1+2+2+1+1+4+1+2+1 = 25
printf 'Transformer 1 8 2 2 16 8 0 1\n' > "$WORK/v11.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v11.txt" -o "$WORK/v11_s.txt" > "$WORK/v11.log" 2>&1
rc=$?
total=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v11.log" | tail -1 | sed 's|.*/||')
fin=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v11.log" | tail -1 | grep -o '[0-9]*')
# DAG check via jobs.dot: the silu-multiply node is the only "8 x 16" VPU
# node; it must have in-degree 4 (2 gate jobs + 2 up jobs), proving the
# expansion wires real fan-in, not a linear chain.
silu=$(awk -F'[" ]' '/label="8 x 16"/{print $1}' jobs.dot | tr -d ' ')
indeg=$(grep -c -- "-> ${silu};" jobs.dot)
if [ "$rc" -eq 0 ] && [ "${total:-0}" = "25" ] && [ "$fin" = "$total" ] && [ "${indeg:-0}" -eq 4 ]; then
  ok "V11 Transformer prefill: 25 jobs, all finish, silu fan-in=4"
else
  bad "V11 rc=$rc total=${total:-?} fin=${fin:-?} silu_indeg=${indeg:-?}"
fi
```
Note `jobs.dot` is written to the CWD (`build/`) by `main.cc`. The awk extracts the node name from a line like `  job17 [label="8 x 16"];` — field 1 after splitting on quote/space is the indented name; verify the extraction interactively if it misfires (the label is `parallel x linearized` = `8 x 16` for `VecUnitJob(16, 8, ...)`).

- [ ] **Step 2: Run to verify V11 fails** — `Unknown layer type: Transformer`.

- [ ] **Step 3: Implement** in `src/frontends/standard/StandardLayers.cc` (after `RMSNorm`):

```cpp
// Llama/Gemma-style decoder stack (spec 3.4).
// Line: Transformer n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch
//   mode 0 = prefill: every GEMM has M = seq_len rows, attention context = seq_len
//   mode 1 = decode:  every GEMM has M = batch rows, KV context = seq_len
// Residual adds depend on BOTH true parents so the simulator is allowed the
// same SA/VPU overlap the hardware has. KV-cache traffic rides the score/AV
// jobs' weight-side reads. For layer 0 the residual's block-input parent is
// approximated by norm1 (the true parent is the external predecessor line).
JobPair Transformer(const ArchConfig &a_config, const LayerConfig &l_config) {
  if (l_config.dimensions.size() != 8) {
    std::cerr << "Transformer expects 8 dims: n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch" << std::endl;
    throw std::exception();
  }
  const auto &d = l_config.dimensions;
  int n_layers = d[0], d_model = d[1], nh = d[2], nkv = d[3], d_ff = d[4], seq_len = d[5], mode = d[6], batch = d[7];
  if (n_layers < 1 || d_model < 1 || nh < 1 || nkv < 1 || d_ff < 1 || seq_len < 1 || batch < 1 ||
      (mode != 0 && mode != 1) || d_model % nh != 0 || nh % nkv != 0) {
    std::cerr << "Transformer: invalid dims (need positive sizes, mode 0|1, nh | d_model, nkv | nh)" << std::endl;
    throw std::exception();
  }
  int head_dim = d_model / nh;
  int M = (mode == 0) ? seq_len : batch;// rows through every GEMM
  int S = seq_len;                      // attention context length

  auto mk_binary_ew = [&](int rows, int cols) -> JobList {
    auto *jb = new VectorUnit::VecUnitJob(cols, rows, false, {{VectorUnit::VPUPhase::BROADCAST, 1}});
    jb->n_read_operands = 2;
    return {jb};
  };

  JobList model_head, prev_tail;
  for (int l = 0; l < n_layers; ++l) {
    JobList norm1 = makeRMSNormJobs(d_model, M);
    if (l == 0) model_head = norm1;
    else connectJobLists(prev_tail, norm1);

    auto q = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_model}));
    auto k = Matmul(a_config, LayerConfig("Matmul", {M, d_model, nkv * head_dim}));
    auto v = Matmul(a_config, LayerConfig("Matmul", {M, d_model, nkv * head_dim}));
    connectJobLists(norm1, q.first);
    connectJobLists(norm1, k.first);
    connectJobLists(norm1, v.first);

    JobList rope = {new VectorUnit::VecUnitJob(1, M * (d_model + nkv * head_dim), false,
                                               {{VectorUnit::VPUPhase::BROADCAST, 1}})};
    connectJobLists(q.second, rope);
    connectJobLists(k.second, rope);

    JobList scores, av;
    for (int h = 0; h < nh; ++h) scores.push_back(new SystolicArray::SysArrayJob(M, head_dim, S));
    for (int h = 0; h < nh; ++h) av.push_back(new SystolicArray::SysArrayJob(M, S, head_dim));
    connectJobLists(rope, scores);

    JobList sm = makeSoftmaxJobs(S, M * nh);
    connectJobLists(scores, sm);
    connectJobLists(sm, av);
    connectJobLists(v.second, av);

    auto o = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_model}));
    connectJobLists(av, o.first);

    JobList block_in = (l == 0) ? norm1 : prev_tail;
    JobList res1 = mk_binary_ew(M, d_model);
    connectJobLists(o.second, res1);
    connectJobLists(block_in, res1);

    JobList norm2 = makeRMSNormJobs(d_model, M);
    connectJobLists(res1, norm2);

    auto gate = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_ff}));
    auto up = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_ff}));
    connectJobLists(norm2, gate.first);
    connectJobLists(norm2, up.first);

    JobList silu_mul = mk_binary_ew(M, d_ff);
    connectJobLists(gate.second, silu_mul);
    connectJobLists(up.second, silu_mul);

    auto down = Matmul(a_config, LayerConfig("Matmul", {M, d_ff, d_model}));
    connectJobLists(silu_mul, down.first);

    JobList res2 = mk_binary_ew(M, d_model);
    connectJobLists(down.second, res2);
    connectJobLists(res1, res2);

    prev_tail = res2;
  }
  return {model_head, prev_tail};
}
```
Register `Transformer` in `getLayerLambda`.

- [ ] **Step 4: Build; run both suites** — regression 5/5, tpuv6e 11/11. If the V11 count differs, dump `jobs.dot` node labels and re-derive per the Global Constraints rule.

- [ ] **Step 5: Commit**

```bash
git add src/frontends/standard/StandardLayers.cc tests/tpuv6e.sh
git commit -m "Add Transformer composite layer with DAG-wired residuals (prefill)"
```

---

### Task 10: `Transformer` decode-mode verification (GQA + KV context)

The decode path (`mode=1`) is already coded in Task 9 (`M = batch`, context `S = seq_len`, K/V projections sized `nkv * head_dim`). This task pins its semantics with tests so a regression in the mode logic can't slip through.

**Files:**
- Test: `tests/tpuv6e.sh` (V12)

**Interfaces:**
- Consumes: `Transformer` keyword (Task 9); `jobs.dot` labels.
- Produces: verified decode contract for the calibration plan's workload generation.

- [ ] **Step 1: Append failing-or-passing test V12 (write it, expect it to PASS if Task 9 was faithful; a FAIL here is a real decode bug, not a missing feature)**

```bash
# V12: decode semantics. Transformer 1 8 2 1 16 32 1 4 -sa_sz 4:
#   GQA: K/V projections have N = nkv*head_dim = 1*4 = 4 -> SA job "4 x 8 x 4"
#   scores read the 32-token KV context: SA job "4 x 4 x 32"
#   AV contracts over the context:       SA job "4 x 32 x 4"
#   M = batch = 4 everywhere (no seq_len-sized GEMM rows: no "32 x 8 x" node)
printf 'Transformer 1 8 2 1 16 32 1 4\n' > "$WORK/v12.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -i "$WORK/v12.txt" -o "$WORK/v12_s.txt" > "$WORK/v12.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v12.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4)?1:0}')
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ] \
   && grep -q 'label="4 x 8 x 4"' jobs.dot \
   && grep -q 'label="4 x 4 x 32"' jobs.dot \
   && grep -q 'label="4 x 32 x 4"' jobs.dot \
   && ! grep -q 'label="32 x 8 x' jobs.dot; then
  ok "V12 decode: GQA K/V, KV-context scores/AV, batch-sized rows"
else
  bad "V12 rc=$rc fin_eq=${fin_eq:-?} (check jobs.dot labels)"
fi
```

- [ ] **Step 2: Run** — if V12 fails, apply superpowers:systematic-debugging to the decode branch of `Transformer` (most likely suspects: M/S swap, or K/V matmul N using `d_model` instead of `nkv * head_dim`). Fix, re-run both suites.

- [ ] **Step 3: Commit**

```bash
git add tests/tpuv6e.sh
git commit -m "Pin Transformer decode semantics (GQA, KV context, batch rows)"
```

---

### Task 11: `-fuse_epilogue` flag

Spec §3.4: when XProf shows residual adds fused into GEMM epilogues, the model must be able to absorb them. `-fuse_epilogue 1` suppresses the two residual-add jobs per layer (their parents wire straight through); default 0 keeps them.

**Files:**
- Modify: `include/global.h`, `src/global.cc`, `src/frontends/standard/StandardParser.cc` (flag + validation), `src/frontends/standard/StandardLayers.cc` (`Transformer`)
- Test: `tests/tpuv6e.sh` (V13)

**Interfaces:**
- Consumes: `Transformer` expansion (Task 9), flag machinery (Task 1).
- Produces: global `int fuse_epilogue = 0;`, flag `-fuse_epilogue <0|1>`. With fusing on, `res1 := o.second ∪ block_in` and `res2 := down.second ∪ res1` (dependency union, no VPU job).

- [ ] **Step 1: Append failing test V13**

```bash
# V13: -fuse_epilogue 1 removes exactly 2 VPU jobs per layer (res1, res2).
# Tiny prefill config from V11 has 25 jobs -> 23 fused. All still finish.
printf 'Transformer 1 8 2 2 16 8 0 1\n' > "$WORK/v13.txt"
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_epilogue 1 -i "$WORK/v13.txt" -o "$WORK/v13_s.txt" > "$WORK/v13.log" 2>&1
rc=$?
total=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v13.log" | tail -1 | sed 's|.*/||')
fin=$(grep -o 'Jobs finished: [0-9]*/' "$WORK/v13.log" | tail -1 | grep -o '[0-9]*')
"$BIN" -c 1 -sa_sz 4 -vu_sz 4 -f 1 -fuse_epilogue 2 -i "$WORK/v13.txt" -o "$WORK/v13b_s.txt" > "$WORK/v13b.log" 2>&1
rcb=$?
if [ "$rc" -eq 0 ] && [ "${total:-0}" = "23" ] && [ "$fin" = "$total" ] && [ "$rcb" -eq 1 ]; then
  ok "V13 -fuse_epilogue 1 -> 23 jobs; invalid value rejected"
else
  bad "V13 rc=$rc total=${total:-?} fin=${fin:-?} badval_rc=$rcb"
fi
```

- [ ] **Step 2: Run to verify V13 fails** — `Failed to parse passed flag: '-fuse_epilogue'`.

- [ ] **Step 3: Implement**

`include/global.h`: `extern int fuse_epilogue;` — `src/global.cc`: `int fuse_epilogue = 0;`

`StandardParser.cc`: add `{"-fuse_epilogue", &fuse_epilogue}` to `parse_args` (help line: `"-fuse_epilogue residual adds fused into GEMM epilogue: 0 off (default), 1 on"`), and validation:
```cpp
  if (fuse_epilogue != 0 && fuse_epilogue != 1) {
    std::cerr << "Error: -fuse_epilogue must be 0 or 1, got " << fuse_epilogue << std::endl;
    exit(1);
  }
```

`Transformer` in `StandardLayers.cc` — replace the res1 block:
```cpp
    JobList block_in = (l == 0) ? norm1 : prev_tail;
    JobList res1;
    if (fuse_epilogue) {
      // Residual absorbed into the projection epilogue: downstream work
      // depends on both contributors directly, no VPU job.
      res1 = o.second;
      res1.insert(res1.end(), block_in.begin(), block_in.end());
    } else {
      res1 = mk_binary_ew(M, d_model);
      connectJobLists(o.second, res1);
      connectJobLists(block_in, res1);
    }
```
and the res2 block:
```cpp
    JobList res2;
    if (fuse_epilogue) {
      res2 = down.second;
      res2.insert(res2.end(), res1.begin(), res1.end());
    } else {
      res2 = mk_binary_ew(M, d_model);
      connectJobLists(down.second, res2);
      connectJobLists(res1, res2);
    }
```
(Note: with fusing on, `res1` may contain jobs already in `prev_tail`'s lineage; `connectJobLists(prev_tail, norm1)` in the next iteration handles duplicates fine — `add_child` just increments `rem_deps` per edge, and duplicate edges are not created here because each list element is distinct.)

- [ ] **Step 4: Build; run both suites** — regression 5/5, tpuv6e 13/13 (V11/V12 confirm default-off still yields 25 jobs).

- [ ] **Step 5: Commit**

```bash
git add include/global.h src/global.cc src/frontends/standard/StandardParser.cc src/frontends/standard/StandardLayers.cc tests/tpuv6e.sh
git commit -m "Add -fuse_epilogue switch for residual-add fusion"
```

---

### Task 12: Pinned v6e config artifact + final sweep

Spec §3.6. One committed script holds the canonical v6e flag set (geometry hypothesis + priors; fitted values get frozen here after calibration).

**Files:**
- Create: `configs/tpuv6e.sh`
- Test: `tests/tpuv6e.sh` (V14)

**Interfaces:**
- Consumes: every flag from Tasks 1-3 and 11.
- Produces: `configs/tpuv6e.sh <layer_file> <stats_out> [extra flags...]` — the measurement/calibration plans invoke the simulator only through this script.

- [ ] **Step 1: Append failing test V14**

```bash
# V14: the pinned config runs a decode workload end-to-end on the v6e model.
printf 'Transformer 2 512 8 4 1024 128 1 8\n' > "$WORK/v14.txt"
"$REPO/configs/tpuv6e.sh" "$WORK/v14.txt" "$WORK/v14_s.txt" > "$WORK/v14.log" 2>&1
rc=$?
fin_eq=$(grep -o 'Jobs finished: [0-9]*/[0-9]*' "$WORK/v14.log" | tail -1 | awk -F'[:/ ]+' '{print ($3==$4)?1:0}')
n_acct=$(grep -c '^ACCT' "$WORK/v14_s.txt")
if [ "$rc" -eq 0 ] && [ "${fin_eq:-0}" -eq 1 ] && [ "${n_acct:-0}" -eq 8 ]; then
  ok "V14 tpuv6e.sh runs decode workload (8 ACCT units)"
else
  bad "V14 rc=$rc fin_eq=${fin_eq:-?} acct_units=${n_acct:-?}"
fi
```

- [ ] **Step 2: Run to verify V14 fails** — `configs/tpuv6e.sh` does not exist.

- [ ] **Step 3: Create `configs/tpuv6e.sh`**

```bash
#!/usr/bin/env bash
# Canonical TPU v6e model configuration (spec 3.1/3.6).
# Geometry hypothesis: 4 MXUs of 256x256 at 1.75 GHz (peak-TFLOPS
# decomposition, v5e lineage) - Phase C probes confirm or falsify.
# PRIORS, to be frozen after calibration (spec 5): -f, -dram_enq,
# -job_overhead, -vu_sz. -buf_mb 128 is the nominal VMEM hypothesis.
# Usage: configs/tpuv6e.sh <layer_file> <stats_out> [extra flags...]
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAYERS="$1"; OUT="$2"; shift 2
cd "$REPO/build"
exec ./perf_model \
  -c 4 \
  -sa_sz 256 \
  -vu_sz 256 \
  -f 1.75 \
  -ws 0 \
  -buf_mb 128 \
  -dram_ini ../dramsim3/configs/HBM2e_v6e.ini \
  -dram_enq 32 \
  -job_overhead 0 \
  -fuse_epilogue 0 \
  -i "$LAYERS" -o "$OUT" "$@"
```
Run: `chmod +x configs/tpuv6e.sh`

- [ ] **Step 4: Full final sweep**

Run: `tests/regression.sh && tests/tpuv6e.sh` and additionally the three shipped examples through `build/perf_model` with defaults (exit 0, all jobs finish) — the same check Task 0 relied on.
Expected: 5/5, 14/14, examples clean.

- [ ] **Step 5: Commit**

```bash
git add configs/tpuv6e.sh tests/tpuv6e.sh
git commit -m "Add pinned TPUv6e model configuration script"
```

---

## Out of scope for this plan (tracked in the spec)

- Measurement harness `benchmarks/tpuv6e/` (spec §4) — Plan 2, separate environment (GCP/JAX).
- Fit driver and calibration (spec §5) — Plan 3, needs Plan 2's data.
- Known pre-existing defects deliberately untouched: `createSAJobs` static counter first-call sizing, multi-period `state_updates.at(-1)` crash, unchecked `fopen(ofile)` (three open task chips).
