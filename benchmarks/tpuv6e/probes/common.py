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
