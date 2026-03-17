#!/usr/bin/env python3
"""
Benchmark script for COCOSSim matmul configurations.
Runs simulations and extracts key metrics into a table.
"""

import subprocess
import re
import sys
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MatmulConfig:
    """Configuration for a matmul benchmark."""
    name: str           # Descriptive name
    M: int
    K: int
    N: int
    ws: int = 1         # Weight stationary (1) or output stationary (0)
    sw: int = 1
    cores: int = 1
    sa_sz: int = 32
    act_in_buf: bool = False
    no_wb: bool = False

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    matmul_dims: str
    dram_reads: int
    dram_writes: int
    sram_reads: int
    sram_writes: int
    sram_max_cap_mb: float
    latency_us: float
    cycles: int
    compute_energy_mj: float
    sram_energy_mj: float
    hbm_energy_mj: float
    total_energy_mj: float
    avg_power_w: float

# ============================================================
# EDIT YOUR CONFIGURATIONS HERE
# ============================================================
split = 1
CONFIGS: List[MatmulConfig] = [
    # MatmulConfig("ActInBuf", M=4096, K=64, N=64, act_in_buf=True),
    MatmulConfig("L1a", M=1048576//split, K=9, N=16),
    MatmulConfig("L1b", M=1048576//split, K=144, N=16),
    MatmulConfig("L2a", M=262144//split, K=256, N=32),
    MatmulConfig("L2b", M=262144//split, K=288, N=32),
    MatmulConfig("L3a", M=65536//split, K=512, N=64),
    MatmulConfig("L3b", M=65536//split, K=576, N=64),
    MatmulConfig("L4a", M=16384//split, K=1024, N=128),
    MatmulConfig("L4b", M=16384//split, K=1152, N=128),
    MatmulConfig("L5", M=4096//split, K=2048, N=128),
    MatmulConfig("L6a", M=16384//split, K=2048, N=128),
    MatmulConfig("L6b", M=1638//split, K=2304, N=128),
    MatmulConfig("L7a", M=65536//split, K=2048, N=64),
    MatmulConfig("L7b", M=65536//split, K=1152, N=64),
    MatmulConfig("L8a", M=262144//split, K=1024, N=32),
    MatmulConfig("L8b", M=262144//split, K=576, N=32),
    MatmulConfig("L9a", M=1048576//split, K=512, N=16),
    MatmulConfig("L9b", M=1048576//split, K=288, N=16),
]
# CONFIGS: List[MatmulConfig] = [
#     # MatmulConfig("ActInBuf", M=4096, K=64, N=64, act_in_buf=True),
#     MatmulConfig("L1a", M=1048576//split, K=9, N=16, no_wb=True),
#     MatmulConfig("L1b", M=1048576//split, K=144, N=16, act_in_buf=True, no_wb=True),
#     MatmulConfig("L2a", M=262144//split, K=256, N=32, act_in_buf=True, no_wb=True),
#     MatmulConfig("L2b", M=262144//split, K=288, N=32, act_in_buf=True, no_wb=True),
#     MatmulConfig("L3a", M=65536//split, K=512, N=64, act_in_buf=True, no_wb=True),
#     MatmulConfig("L3b", M=65536//split, K=576, N=64, act_in_buf=True, no_wb=True),
#     MatmulConfig("L4a", M=16384//split, K=1024, N=128, act_in_buf=True, no_wb=True),
#     MatmulConfig("L4b", M=16384//split, K=1152, N=128, act_in_buf=True, no_wb=True),
#     MatmulConfig("L5", M=4096//split, K=2048, N=128, act_in_buf=True, no_wb=True),
#     MatmulConfig("L6a", M=16384//split, K=2048, N=128, act_in_buf=True, no_wb=True),
#     MatmulConfig("L6b", M=1638//split, K=2304, N=128, act_in_buf=True, no_wb=True),
#     MatmulConfig("L7a", M=65536//split, K=2048, N=64, act_in_buf=True, no_wb=True),
#     MatmulConfig("L7b", M=65536//split, K=1152, N=64, act_in_buf=True, no_wb=True),
#     MatmulConfig("L8a", M=262144//split, K=1024, N=32, act_in_buf=True, no_wb=True),
#     MatmulConfig("L8b", M=262144//split, K=576, N=32, act_in_buf=True, no_wb=True),
#     MatmulConfig("L9a", M=1048576//split, K=512, N=16, act_in_buf=True, no_wb=True),
#     MatmulConfig("L9b", M=1048576//split, K=288, N=16, act_in_buf=True, no_wb=True),
# ]
# ============================================================

def run_benchmark(config: MatmulConfig, build_dir: str) -> Optional[BenchmarkResult]:
    """Run a single benchmark and parse the output."""

    # Create temp input file
    matmul_line = f"Matmul {config.M} {config.K} {config.N}"

    # Build command
    cmd = [
        f"{build_dir}/perf_model",
        "-i", "/dev/stdin",
        "-o", "/dev/null",
        "-c", str(config.cores),
        "-sa_sz", str(config.sa_sz),
        "-vu_sz", str(config.sa_sz),
        "-ws", str(config.ws),
        "-sw", str(config.sw),
    ]

    if config.act_in_buf:
        cmd.append("-act_in_buf")
    if config.no_wb:
        cmd.append("-no_wb")

    try:
        result = subprocess.run(
            cmd,
            input=matmul_line,
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {config.name}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR: {config.name}: {e}", file=sys.stderr)
        return None

    # Parse output
    return parse_output(config, output)

def parse_output(config: MatmulConfig, output: str) -> Optional[BenchmarkResult]:
    """Parse simulator output and extract metrics."""

    try:
        # DRAM commands - find the LAST occurrence (final stats)
        dram_matches = re.findall(r'DRAM CMDs: \d+ \(R: (\d+), W: (\d+)\)', output)
        if dram_matches:
            dram_reads = int(dram_matches[-1][0])
            dram_writes = int(dram_matches[-1][1])
        else:
            dram_reads = 0
            dram_writes = 0

        # SRAM reads/writes from buffer stats
        sram_reads_match = re.search(r'OnChipSRAM ===\n\s+Reads:\s+\d+ \((\d+) bytes\)', output)
        sram_reads = int(sram_reads_match.group(1)) if sram_reads_match else 0

        # Find writes after the OnChipSRAM section
        sram_section = re.search(r'OnChipSRAM ===\n(.*?)(?:===|$)', output, re.DOTALL)
        if sram_section:
            writes_match = re.search(r'Writes:\s+\d+ \((\d+) bytes\)', sram_section.group(1))
            sram_writes = int(writes_match.group(1)) if writes_match else 0
        else:
            sram_writes = 0

        # Peak occupancy
        peak_match = re.search(r'Peak Occupancy: ([\d.]+) MB', output)
        sram_max_cap = float(peak_match.group(1)) if peak_match else 0.0

        # Cycles - look for the performance section
        cycles_match = re.search(r'Cycles:\s+(\d+)\n', output)
        cycles = int(cycles_match.group(1)) if cycles_match else 0

        latency_match = re.search(r'Latency:\s+([\d.]+) ms', output)
        latency_ms = float(latency_match.group(1)) if latency_match else 0.0
        latency_us = latency_ms * 1000

        # Energy breakdown
        compute_match = re.search(r'Compute \(MACs\):\s+([\d.]+) mJ', output)
        compute_energy = float(compute_match.group(1)) if compute_match else 0.0

        sram_total_match = re.search(r'SRAM Total:\s+([\d.]+) mJ', output)
        sram_energy = float(sram_total_match.group(1)) if sram_total_match else 0.0

        hbm_match = re.search(r'DRAM \(HBM\):\s+([\d.]+) mJ', output)
        hbm_energy = float(hbm_match.group(1)) if hbm_match else 0.0

        total_match = re.search(r'TOTAL:\s+([\d.]+) mJ', output)
        total_energy = float(total_match.group(1)) if total_match else 0.0

        power_match = re.search(r'Average Power:\s+([\d.]+) W', output)
        avg_power = float(power_match.group(1)) if power_match else 0.0

        return BenchmarkResult(
            config_name=config.name,
            matmul_dims=f"{config.M}x{config.K}x{config.N}",
            dram_reads=dram_reads,
            dram_writes=dram_writes,
            sram_reads=sram_reads,
            sram_writes=sram_writes,
            sram_max_cap_mb=sram_max_cap,
            latency_us=latency_us,
            cycles=cycles,
            compute_energy_mj=compute_energy,
            sram_energy_mj=sram_energy,
            hbm_energy_mj=hbm_energy,
            total_energy_mj=total_energy,
            avg_power_w=avg_power
        )
    except Exception as e:
        print(f"  PARSE ERROR: {config.name}: {e}", file=sys.stderr)
        return None

def print_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""

    # Header
    headers = [
        "Config", "Matmul", "DRAM R", "DRAM W", "SRAM R (B)", "SRAM W (B)",
        "SRAM Cap (MB)", "Latency (µs)", "Cycles", "Compute (mJ)",
        "SRAM (mJ)", "HBM (mJ)", "Total (mJ)", "Power (W)"
    ]

    # Format rows
    rows = []
    for r in results:
        rows.append([
            r.config_name,
            r.matmul_dims,
            str(r.dram_reads),
            str(r.dram_writes),
            str(r.sram_reads),
            str(r.sram_writes),
            f"{r.sram_max_cap_mb:.2f}",
            f"{r.latency_us:.2f}",
            str(r.cycles),
            f"{r.compute_energy_mj:.4f}",
            f"{r.sram_energy_mj:.4f}",
            f"{r.hbm_energy_mj:.4f}",
            f"{r.total_energy_mj:.4f}",
            f"{r.avg_power_w:.3f}"
        ])

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)

    print(header_line)
    print(separator)

    # Print rows
    for row in rows:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

def print_csv(results: List[BenchmarkResult]):
    """Print results as CSV."""
    headers = [
        "Config", "Matmul", "DRAM_Reads", "DRAM_Writes", "SRAM_Reads_Bytes",
        "SRAM_Writes_Bytes", "SRAM_Cap_MB", "Latency_us", "Cycles",
        "Compute_mJ", "SRAM_mJ", "HBM_mJ", "Total_mJ", "Power_W"
    ]
    print(",".join(headers))

    for r in results:
        row = [
            r.config_name,
            r.matmul_dims,
            str(r.dram_reads),
            str(r.dram_writes),
            str(r.sram_reads),
            str(r.sram_writes),
            f"{r.sram_max_cap_mb:.2f}",
            f"{r.latency_us:.2f}",
            str(r.cycles),
            f"{r.compute_energy_mj:.4f}",
            f"{r.sram_energy_mj:.4f}",
            f"{r.hbm_energy_mj:.4f}",
            f"{r.total_energy_mj:.4f}",
            f"{r.avg_power_w:.3f}"
        ]
        print(",".join(row))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark COCOSSim matmul configurations")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--build-dir", default=".", help="Build directory containing perf_model")
    args = parser.parse_args()

    # Find build directory
    build_dir = args.build_dir
    if not os.path.exists(f"{build_dir}/perf_model"):
        # Try common locations
        for path in [".", "./build", "../build"]:
            if os.path.exists(f"{path}/perf_model"):
                build_dir = path
                break
        else:
            print("ERROR: Cannot find perf_model executable", file=sys.stderr)
            sys.exit(1)

    print(f"Running benchmarks from: {build_dir}/perf_model", file=sys.stderr)
    print(f"Total configs: {len(CONFIGS)}", file=sys.stderr)

    results = []
    for i, config in enumerate(CONFIGS):
        print(f"  [{i+1}/{len(CONFIGS)}] {config.name}...", file=sys.stderr, end=" ", flush=True)
        result = run_benchmark(config, build_dir)
        if result:
            results.append(result)
            print("OK", file=sys.stderr)
        else:
            print("FAILED", file=sys.stderr)

    print(file=sys.stderr)

    if args.csv:
        print_csv(results)
    else:
        print_table(results)

if __name__ == "__main__":
    main()
