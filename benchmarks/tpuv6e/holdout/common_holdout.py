"""Minimal CSV append helper for holdout scripts (no jax dependency)."""
import csv
import os


def csv_append(path, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
