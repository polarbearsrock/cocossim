# Compute-only SA / VPU ratio sweep

This standalone COCOSSim experiment reuses the checkout's **unmodified**
`SysArray.cc` and `VectorUnit.cc`. `compute_state.cc` replaces only the
memory-facing `State` implementation with immediate memory completion.
Internal SA/VPU compute and stage-transition timing is retained.

`runner.cc` supplies a Qwen 2.5 7B decoder-layer DAG, resource-independent
tiling, and FIFO event scheduling. In the absence of memory coupling,
native tile service times can be measured once and cached for exact replay
under this experiment's scheduling policy. This is an extension of the
compute model, not a run of the stock `Transformer` frontend or a validated
MTIA hardware model. It does not evaluate numerical tensors.

The primary sweep holds 144 32×32 arrays fixed and varies 128-lane VPUs.
The ratio is peak **MACs/cycle : simple vector operations/cycle**;
72 VPUs therefore gives 16:1. Area and power are not normalized.

## Run

Use Python 3 with matplotlib to generate plots (CSV/report generation also
works without matplotlib), CMake, and a C++17 compiler. `--source` must name
the extended checkout containing `mxu_macs_per_pe` and the current unit API.
The source checkout is read-only to this runner. Builds and matplotlib
cache files go under `TMPDIR`.

```bash
python run.py --source ../.. \
  --batch 64 --seq 1024 --sa 144 \
  --vpus 288,144,72,36,18,9,6,4,3,2,1 \
  --nonlinear-costs 1,4 \
  --out "$TMPDIR/qwen25_prefill_ratio"
```

For this machine, `/data2/s2chitni/venvs/tpu-analysis/bin/python` has the
plotting dependencies installed. The current report expects the reference
72-VPU point and nonlinear cost 1 to be included in a custom sweep.

The checked-in `results/` contains the corrected sweep and its methods,
assumptions, work census, timings, figures, source hashes, and validation
evidence. See `results/report.md` for the model boundaries and interpretation.

## Two SAs and one VPU: attention only

```bash
python run_attention.py --source ../.. \
  --batch 64 --seq 1024 --nonlinear-costs 1,4 \
  --out "$TMPDIR/qwen25_attention_2sa_1vpu"
```

This runs two 32×32 SAs and one 128-lane VPU (16:1 MAC/vector-op capacity).
The scope is input RMSNorm, QKV projections, bias, RoPE, dense masked causal
attention, output projection, and the residual add. The MLP is excluded.
`attention_2sa_1vpu/` contains results, a per-stage trace, a timeline, and the
operation-by-operation explanation. The native runner accepts `--scope
attention`, `--trace-stages 1`, and `--bin-width N`; full-layer scope remains
the default. The separate attention harness does not require a 72-VPU point.

## Timing correction, 2026-09-11

The original oracle reset `gcycles` after native unit initialization, which
captures an SA job's dispatch timestamp. A shorter following measurement
could therefore incur an artificial completion stall. The reset now occurs
before initialization, with regression checks for measurements following
longer jobs. Native K=128 and K=1024 tiles take 161 and 1057 cycles instead
of 385 and 2049. The full sweep has been rerun; prior results and source
snapshots are archived in `superseded_20260911/` and should not be used for
conclusions. See the corrected report for the updated tradeoffs.

## Modeling details

The layer includes QKV biases, two RMSNorms, cached-table RoPE, dense masked
causal attention with stable softmax, both residual adds, and the full
gated MLP. Attention operates independently per sequence/query head and
128-row group. It uses GQA's four K/V projection heads and all 28 query
heads. Upper-triangular attention tiles still execute, as in dense masked
attention; this is not a FlashAttention schedule.

MAC output tiles are at most 32×32. Elementwise jobs have at most 16,384
elements; reductions batch at most 128 independent rows. These sizes do
not change with SA/VPU counts. Exact tails avoid the stock N/core
integer-division truncation. All work waits for its producers; the custom
frontend does not use the stock traffic-free vector sidecar shortcut.

The nonlinear-cost sensitivity assigns exp, reciprocal, and rsqrt either
one or four lane-cycle equivalents. It is a throughput-cost assumption,
not an operation's measured pipeline latency. Ordinary operations cost
one. COCOSSim's reduction timing is retained (one row per lane, serial
along the row); changing reduction hardware would require a separate study.

## Validation

`compute_ratio --selftest` checks native unit timings, a hand-computed
dependency schedule, exact tail work, and event/cycle-step agreement on
50 deterministic random DAGs. Every layer construction asserts independent
analytical MAC/vector censuses. Every run asserts work/service conservation
and capacity. The Python harness also checks invariant work and job counts
across ratios.
