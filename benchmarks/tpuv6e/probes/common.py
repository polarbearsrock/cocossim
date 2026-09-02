"""Shared measurement discipline for all raw-JAX probes (spec 4).

Every probe: compile once and DISCARD, then >=20 timed reps with
block_until_ready, report the median (plus p10/p90 so jitter is visible in
the CSV rather than silently averaged away). CSV rows are append-only and
self-describing; a re-run continues a sweep rather than clobbering it.
"""
import csv
import os
import statistics
import time

REPS = int(os.environ.get("PROBE_REPS", "20"))


def time_op(fn, reps=None):
    """Median seconds per call of fn() after one discarded compile call."""
    import jax
    reps = reps or REPS
    jax.block_until_ready(fn())  # compile + warm, discarded
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {
        "median_s": statistics.median(ts),
        "p10_s": ts[max(0, int(0.1 * len(ts)) - 1)],
        "p90_s": ts[int(0.9 * len(ts)) - 1],
        "reps": reps,
    }


def time_chain_slope(make_fn, chain, min_call_s=1e-3, max_chain=4096, reps=None):
    """Device-side per-step time of a chained kernel by the SLOPE method.

    A chained call is one executable, so its launch + completion cost (the
    ~113 us host floor, spec 2) lands once per call and leaks floor/CHAIN
    into a naive per-step division (session H1: 7.1 us/step at chain 16 on
    tiny elementwise ops = 113/16). Timing the chain at C and 2C and taking
    per_step = (t_2C - t_C) / C cancels every per-call constant; the
    intercept 2*t_C - t_2C is that constant, measured rather than assumed.

    make_fn(C) must return a zero-arg callable that runs a C-step chain
    (compiled per C; C is a static scan length). C is grown until a call
    lasts at least min_call_s so the difference is well above jitter.
    Returns per_step_s, intercept_s, chain, both raw timings, and a
    conservative per-step band from the two runs' p10/p90.
    """
    C = chain
    t1 = time_op(make_fn(C), reps=reps)
    while t1["median_s"] < min_call_s and C < max_chain:
        C = min(max_chain, C * 2)
        t1 = time_op(make_fn(C), reps=reps)
    t2 = time_op(make_fn(2 * C), reps=reps)
    per_step = (t2["median_s"] - t1["median_s"]) / C
    return {
        "chain": C,
        "per_step_s": per_step,
        "per_step_p10_s": (t2["p10_s"] - t1["p90_s"]) / C,
        "per_step_p90_s": (t2["p90_s"] - t1["p10_s"]) / C,
        "intercept_s": 2 * t1["median_s"] - t2["median_s"],
        "t_c_s": t1["median_s"], "t_2c_s": t2["median_s"],
        "reps": t1["reps"],
    }


def csv_append(path, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def already_done(path, key_fields: dict) -> bool:
    """Resumability: skip a point whose key fields already appear in the CSV."""
    if not os.path.exists(path):
        return False
    with open(path) as f:
        for r in csv.DictReader(f):
            if all(str(r.get(k)) == str(v) for k, v in key_fields.items()):
                return True
    return False
