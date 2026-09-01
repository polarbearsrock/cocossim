# TPU v6e measurement harness (spec §4, Plan 2)

Measures a real `v6e-1` to calibrate and validate the COCOSSim TPUv6e model
(`configs/tpuv6e.sh`). Session 0 (2026-08-31) validated the full pipeline:
provision → vLLM offline serve (Qwen3-8B) → JAX profiler trace → kernel
census → teardown, ~$3.

## Layout

- `env.sh` — shared config (project/zone/VM name), `tpu_ssh`/`tpu_scp` helpers
- `provision.sh` / `setup_vm.sh` / `teardown.sh` — VM lifecycle (idempotent)
- `probes/` — raw-JAX microbenchmarks (session 2): C1 array rows, C3
  saturation TFLOPS, C4 stream bandwidth, C5 VMEM cliff, B1 elementwise
  roofline + overhead intercept. All support `--dry-run` locally and are
  resumable (points already in the CSV are skipped).
- `holdout/dh_offline.py` — fixed-shape Qwen3-8B points via vLLM offline
  mode; maps 1:1 onto simulator `Transformer` runs. `--trace-dir` captures an
  xplane per point.
- `analysis/kernel_census.py` — xplane trace → phase-class breakdown
  (gemm/attention/norm/elementwise/data/idle/other), comparable to the
  simulator's ACCT lines. Local venv: `/data2/s2chitni/venvs/tpu-analysis`
  (`pip install xprof`).

## Session runbook

```
source benchmarks/tpuv6e/env.sh
benchmarks/tpuv6e/provision.sh --if-needed
# push probes and run (example: session 2)
tpu_scp benchmarks/tpuv6e/probes "$TPU_NAME":~/probes
tpu_ssh "source ~/venv/bin/activate && cd ~/probes && python c1_msweep.py"
...
tpu_scp "$TPU_NAME":~/probes/'*.csv' "$RESULTS_DIR"/
benchmarks/tpuv6e/teardown.sh        # ALWAYS - on-demand bills ~$3/hr
```

## Hard-won facts (do not rediscover)

- Home dir is quota-full: gcloud config + SSH key live under
  `/data2/s2chitni/.gcloud` (`env.sh` sets both). ssh known-hosts writes fail
  harmlessly.
- Quota map (project `rta-tpu-research`): on-demand v6e in **us-east5** only;
  us-east1-d is spot-only (spot pool preempted 2x/40min on 2026-08-31);
  europe-west4 blocked by org policy. User ruling: on-demand for all sessions.
- The TPU VM's system Python 3.10 cannot build `vllm-tpu`; use `uv` + py3.12
  venv (`setup_vm.sh`).
- vLLM v1 engine owns the TPU in a subprocess: profiling requires
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in `dh_offline.py`). A shutdown
  `AttributeError` after `stop_trace` is cosmetic.
- Launching detached over ssh needs `< /dev/null` + `disown` or the session
  hangs.

## Measurement discipline (spec §4)

XProf device time for per-op numbers; wall clock only end-to-end. Compile
iteration discarded; medians over ≥20 reps with p10/p90 recorded. HLO/kernel
names captured via the trace (the census keys on them). Raw traces + CSVs
land in `$RESULTS_DIR`, never in git; the extracted CSVs that feed the fit
get committed under `benchmarks/tpuv6e/results/` once a session is accepted.
