#!/usr/bin/env python3
"""Generate and run the default Kimi K3 / TPU7x proxy sweep."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence

import generate_kimi_k3_traces as generator
import run_kimi_k3_tpu7x as runner


@dataclass(frozen=True)
class SweepPoint:
    mode: str
    batch: int
    query_tokens: int
    context_tokens: int
    include_lm_head: bool

    @property
    def label(self) -> str:
        return "{}_b{}_q{}_ctx{}".format(
            self.mode, self.batch, self.query_tokens, self.context_tokens
        )


DEFAULT_POINTS = (
    SweepPoint("prefill", 1, 128, 128, False),
    SweepPoint("prefill", 1, 1024, 1024, False),
    SweepPoint("prefill", 1, 8192, 8192, False),
    SweepPoint("decode", 1, 1, 1024, True),
    SweepPoint("decode", 1, 1, 8192, True),
    SweepPoint("decode", 1, 1, 131072, True),
    SweepPoint("decode", 32, 1, 8192, True),
    SweepPoint("decode", 256, 1, 8192, True),
)

QUICK_POINTS = (
    SweepPoint("prefill", 1, 128, 128, False),
    SweepPoint("decode", 1, 1, 1024, True),
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate traces and run the standard Kimi K3 TPU7x proxy sweep."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument("--simulator", type=Path, default=runner.DEFAULT_SIMULATOR)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="keep generated trace manifests here; defaults to an auto-removed TMPDIR directory",
    )
    parser.add_argument("--quick", action="store_true", help="run one prefill and one decode point")
    parser.add_argument("--analytical-only", action="store_true")
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="reuse matching simulator stats already present in --output-dir",
    )
    parser.add_argument(
        "--expert-layout",
        choices=("aggregate", "per-expert"),
        default="aggregate",
    )
    parser.add_argument(
        "--max-sim-query-tokens",
        type=int,
        default=1024,
        help="keep larger prefill points analytical-only (default: 1024)",
    )
    return parser


def _generate_manifests(
    work_dir: Path, points: Sequence[SweepPoint], expert_layout: str
) -> List[Path]:
    manifests = []
    for point in points:
        request = generator.GenerationRequest(
            mode=point.mode,
            tokens=point.query_tokens,
            context_tokens=point.context_tokens,
            batch=point.batch,
            routing=generator.RoutingSpec(
                scenario="balanced",
                expert_token_multiplier=Decimal("1"),
                expert_layout=expert_layout,
            ),
            include_lm_head=point.include_lm_head,
        )
        manifests.append(
            generator.generate(work_dir / point.label, request, generator.WORKLOADS)
        )
    return manifests


def _run(
    args: argparse.Namespace, work_dir: Path, points: Sequence[SweepPoint]
) -> int:
    manifests = _generate_manifests(work_dir, points, args.expert_layout)
    runner_args: List[str] = [
        "--config",
        str(args.config),
        "--simulator",
        str(args.simulator),
        "--output-dir",
        str(args.output_dir),
        "--max-sim-query-tokens",
        str(args.max_sim_query_tokens),
    ]
    if args.analytical_only:
        runner_args.append("--analytical-only")
    if args.reuse_stats:
        runner_args.append("--reuse-stats")
    for manifest in manifests:
        runner_args.extend(("--manifest", str(manifest)))
    return runner.main(runner_args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    points = QUICK_POINTS if args.quick else DEFAULT_POINTS
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        return _run(args, args.work_dir, points)

    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir:
        raise runner.RunnerError("TMPDIR must be set for generated sweep traces")
    with tempfile.TemporaryDirectory(prefix="kimi-k3-tpu7x-", dir=tmpdir) as directory:
        return _run(args, Path(directory), points)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.RunnerError as error:
        raise SystemExit("error: {}".format(error))
