#!/usr/bin/env python3
"""Run Kimi K3 COCOSSim traces and derive transparent TPU7x proxy bounds.

This runner deliberately keeps three quantities separate:

* raw, compute-only COCOSSim cycles;
* a COCOSSim-derived shape/scheduling efficiency; and
* analytical TPU7x compute, HBM, and routed-MoE ICI lower bounds.

It does not turn COCOSSim's normalized 1 GHz clock into a claimed TPU clock.
The generated latency remains a proxy until it is calibrated with TPU7x
microbenchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "tpu7x_ironwood_16chip_proxy.json"
DEFAULT_SIMULATOR = REPOSITORY_ROOT / "build" / "perf_model"

REPRESENTATIVE_LAYER_COUNTS = {
    "dense-kda": 1,
    "kda-moe": 68,
    "mla-moe": 24,
}

# M and K match a large K3 projection.  N is chosen so that N / 32 proxy
# TensorCores is 512, giving an aligned, saturated two-tile local output.
DEFAULT_CALIBRATION_GEMM = (8192, 7168, 16384)


class RunnerError(RuntimeError):
    """A user-facing runner or input validation error."""


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("failed to read JSON from {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise RunnerError("{} must contain a JSON object".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunnerError("failed to hash {}: {}".format(path, exc)) from exc
    return digest.hexdigest()


def _run_fingerprint(
    simulator: Path, trace: Path, config: Mapping[str, object]
) -> Mapping[str, object]:
    payload = {
        "schema_version": 1,
        "simulator_sha256": _sha256(simulator),
        "trace_sha256": _sha256(trace),
        "cocossim_proxy": config["cocossim_proxy"],
        "runtime_batch_size": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **payload,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _stats_metadata_path(stats: Path) -> Path:
    return stats.with_name(stats.name + ".meta.json")


def _positive_int(mapping: Mapping[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RunnerError("{}.{} must be a positive integer".format(context, key))
    return value


def _positive_number(mapping: Mapping[str, object], key: str, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RunnerError("{}.{} must be a positive number".format(context, key))
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RunnerError("{}.{} must be a positive number".format(context, key))
    return result


def validate_config(config: Mapping[str, object]) -> None:
    hardware = config.get("published_hardware")
    proxy = config.get("cocossim_proxy")
    precision = config.get("roofline_precision")
    capacity = config.get("model_capacity")
    if (
        not isinstance(hardware, dict)
        or not isinstance(proxy, dict)
        or not isinstance(precision, dict)
        or not isinstance(capacity, dict)
    ):
        raise RunnerError(
            "config requires published_hardware, model_capacity, cocossim_proxy, and roofline_precision objects"
        )
    for key in (
        "chips",
        "tensor_cores_per_chip",
        "mxu_rows",
        "mxu_columns",
    ):
        _positive_int(hardware, key, "published_hardware")
    for key in (
        "bf16_tflops_per_chip",
        "fp8_tflops_per_chip",
        "hbm_gib_per_chip",
        "hbm_bandwidth_gbps_per_chip",
        "ici_bidirectional_bandwidth_gbps_per_chip",
    ):
        _positive_number(hardware, key, "published_hardware")
    for key in ("cores", "systolic_array_size", "vector_unit_width", "buffer_mib_per_core", "data_type_bits"):
        _positive_int(proxy, key, "cocossim_proxy")
    _positive_number(proxy, "normalized_frequency_ghz", "cocossim_proxy")
    for key in ("weight_stationary", "compute_only"):
        if not isinstance(proxy.get(key), bool):
            raise RunnerError("cocossim_proxy.{} must be a boolean".format(key))
    for key in (
        "activation_bits",
        "routed_expert_weight_bits",
        "unquantized_weight_bits",
        "cache_and_state_bits",
    ):
        _positive_number(precision, key, "roofline_precision")
    _positive_int(precision, "routed_expert_group_size", "roofline_precision")
    scale_bits = precision.get("routed_expert_scale_bits_per_group")
    if (
        not isinstance(scale_bits, (int, float))
        or isinstance(scale_bits, bool)
        or not math.isfinite(float(scale_bits))
        or float(scale_bits) < 0
    ):
        raise RunnerError(
            "roofline_precision.routed_expert_scale_bits_per_group must be non-negative"
        )
    for key in ("mxfp4_weight_classes", "fp8_compute_categories"):
        values = precision.get(key)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise RunnerError("roofline_precision.{} must be a non-empty string array".format(key))
    _positive_number(capacity, "checkpoint_size_tb_decimal", "model_capacity")


def _scenario_id(manifest: Mapping[str, object]) -> str:
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise RunnerError("manifest.request must be an object")
    routing = request.get("routing")
    if not isinstance(routing, dict):
        raise RunnerError("manifest.request.routing must be an object")
    mode = str(request.get("mode_label"))
    batch = _positive_int(request, "batch", "manifest.request")
    query = _positive_int(request, "query_tokens_per_sequence", "manifest.request")
    context = _positive_int(request, "context_tokens_per_sequence", "manifest.request")
    route = str(routing.get("scenario", "unknown"))
    layout = str(routing.get("expert_layout", "unknown")).replace("-", "_")
    multiplier = routing.get("expert_token_multiplier_decimal", "1")
    if route == "multiplier":
        route = "multiplier_{}".format(str(multiplier).replace(".", "p"))
    label = "{}_b{}_q{}_ctx{}_{}_{}".format(
        mode, batch, query, context, route, layout
    )
    repeat = int(request.get("repeat", 1))
    if repeat != 1:
        label += "_x{}".format(repeat)
    if request.get("include_lm_head"):
        label += "_lmhead"
    return label


def _workload_map(manifest: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list):
        raise RunnerError("manifest.workloads must be an array")
    result: Dict[str, Mapping[str, object]] = {}
    for entry in workloads:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise RunnerError("each manifest workload must be an object with a name")
        result[str(entry["name"])] = entry
    return result


def _ceil_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _split_widths(width: int, cores: int) -> Iterable[int]:
    base, remainder = divmod(width, cores)
    for core in range(cores):
        local = base + (1 if core < remainder else 0)
        if local:
            yield local


def trace_shape_accounting(trace_path: Path, cores: int, array_size: int) -> Mapping[str, object]:
    """Return useful and padded GEMM work for COCOSSim's N-sharding policy."""

    useful_macs = 0
    padded_macs = 0
    matmuls = 0
    operation_count = 0
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunnerError("failed to read trace {}: {}".format(trace_path, exc)) from exc
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        operation_count += 1
        if fields[0] != "Matmul":
            continue
        if len(fields) != 4:
            raise RunnerError(
                "{}:{} generated Matmul must have M K N".format(trace_path, line_number)
            )
        try:
            m_dim, k_dim, n_dim = (int(value) for value in fields[1:])
        except ValueError as exc:
            raise RunnerError(
                "{}:{} contains a non-integer Matmul dimension".format(trace_path, line_number)
            ) from exc
        if min(m_dim, k_dim, n_dim) <= 0:
            raise RunnerError(
                "{}:{} contains a non-positive Matmul dimension".format(trace_path, line_number)
            )
        useful_macs += m_dim * k_dim * n_dim
        padded_local_n = sum(_ceil_to(width, array_size) for width in _split_widths(n_dim, cores))
        padded_macs += (
            _ceil_to(m_dim, array_size)
            * _ceil_to(k_dim, array_size)
            * padded_local_n
        )
        matmuls += 1
    return {
        "matmul_count": matmuls,
        "operation_count": operation_count,
        "trace_sha256": _sha256(trace_path),
        "useful_macs": useful_macs,
        "padded_macs_after_n_sharding": padded_macs,
        "tile_fill_ratio": (useful_macs / padded_macs) if padded_macs else 1.0,
    }


def parse_stats(path: Path) -> Mapping[str, object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunnerError("failed to read simulator stats {}: {}".format(path, exc)) from exc
    cycles: Optional[int] = None
    active: MutableMapping[str, List[float]] = {}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "Cycles" and len(fields) == 2:
            cycles = int(fields[1])
        elif len(fields) == 2:
            active.setdefault(fields[0], []).append(float(fields[1]))
    if cycles is None or cycles <= 0:
        raise RunnerError("{} contains no positive Cycles result".format(path))
    grouped = {
        name: {
            "instances": len(values),
            "mean_scheduled_active_pct": sum(values) / len(values),
            "min_scheduled_active_pct": min(values),
            "max_scheduled_active_pct": max(values),
        }
        for name, values in sorted(active.items())
    }
    return {"cycles": cycles, "unit_activity": grouped}


def _simulator_command(
    simulator: Path,
    trace: Path,
    stats: Path,
    config: Mapping[str, object],
) -> List[str]:
    proxy = config["cocossim_proxy"]
    assert isinstance(proxy, dict)
    return [
        str(simulator),
        "-c",
        str(proxy["cores"]),
        "-sa_sz",
        str(proxy["systolic_array_size"]),
        "-vu_sz",
        str(proxy["vector_unit_width"]),
        "-ws",
        "1" if proxy.get("weight_stationary") else "0",
        "-f",
        str(proxy["normalized_frequency_ghz"]),
        "--batch-size",
        "1",
        "--data-bits",
        str(proxy["data_type_bits"]),
        "--buffer-bytes",
        str(int(proxy["buffer_mib_per_core"]) * 1024 * 1024),
        "--compute-only",
        "1" if proxy.get("compute_only") else "0",
        "-i",
        str(trace.resolve()),
        "-o",
        str(stats.resolve()),
    ]


def run_simulator(
    simulator: Path,
    trace: Path,
    stats: Path,
    log: Path,
    config: Mapping[str, object],
    work_dir: Path,
) -> Mapping[str, object]:
    command = _simulator_command(simulator, trace, stats, config)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=work_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise RunnerError(
            "simulator failed for {} with exit code {}:\n{}".format(
                trace, completed.returncode, tail
            )
        )
    parsed = dict(parse_stats(stats))
    parsed["stats_file"] = stats.name
    parsed["log_file"] = log.name
    _stats_metadata_path(stats).write_text(
        json.dumps(_run_fingerprint(simulator, trace, config), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    for sidecar_name in ("jobs.dot", "dramsim3.json", "dramsim3.txt", "dramsim3epoch.json"):
        (work_dir / sidecar_name).unlink(missing_ok=True)
    return parsed


def reuse_simulator_stats(
    simulator: Path,
    trace: Path,
    stats: Path,
    log: Path,
    config: Mapping[str, object],
) -> Optional[Mapping[str, object]]:
    metadata_path = _stats_metadata_path(stats)
    if not stats.is_file() or not metadata_path.is_file():
        return None
    metadata = _read_json(metadata_path)
    expected = _run_fingerprint(simulator, trace, config)
    if metadata.get("fingerprint_sha256") != expected["fingerprint_sha256"]:
        return None
    parsed = dict(parse_stats(stats))
    parsed["stats_file"] = stats.name
    parsed["log_file"] = log.name
    return parsed


def _weighted_activity(
    runs: Mapping[str, Mapping[str, object]], counts: Mapping[str, int], total_cycles: int
) -> Mapping[str, float]:
    active_cycles: Dict[str, float] = {}
    instances: Dict[str, int] = {}
    for name, count in counts.items():
        run = runs[name]
        cycles = int(run["cycles"])
        activity = run.get("unit_activity", {})
        if not isinstance(activity, dict):
            continue
        for unit, summary in activity.items():
            if not isinstance(summary, dict):
                continue
            pct = float(summary["mean_scheduled_active_pct"])
            active_cycles[unit] = active_cycles.get(unit, 0.0) + count * cycles * pct / 100.0
            instances[unit] = int(summary["instances"])
    return {
        unit: 100.0 * cycles / total_cycles
        for unit, cycles in sorted(active_cycles.items())
    }


def _sum_full_shape(
    shapes: Mapping[str, Mapping[str, object]], counts: Mapping[str, int]
) -> Mapping[str, object]:
    useful = sum(int(shapes[name]["useful_macs"]) * count for name, count in counts.items())
    padded = sum(
        int(shapes[name]["padded_macs_after_n_sharding"]) * count
        for name, count in counts.items()
    )
    return {
        "useful_macs": useful,
        "padded_macs_after_n_sharding": padded,
        "tile_fill_ratio": useful / padded if padded else 1.0,
    }


def _class_weight_bytes(
    class_elements: Mapping[str, object], precision: Mapping[str, object]
) -> Tuple[float, Mapping[str, float]]:
    routed_bits = float(precision.get("routed_expert_weight_bits", 4))
    other_bits = float(precision.get("unquantized_weight_bits", 16))
    group_size = int(precision.get("routed_expert_group_size", 0) or 0)
    scale_bits = float(precision.get("routed_expert_scale_bits_per_group", 0) or 0)
    compressed_classes = set(
        str(value)
        for value in precision.get("mxfp4_weight_classes", ["routed_expert"])
    )
    by_class: Dict[str, float] = {}
    for name, raw_elements in class_elements.items():
        elements = int(raw_elements)
        compressed = name in compressed_classes
        bits = elements * (routed_bits if compressed else other_bits)
        if compressed and group_size and scale_bits:
            bits += math.ceil(elements / group_size) * scale_bits
        by_class[name] = bits / 8.0
    return sum(by_class.values()), by_class


def _dynamic_rhs_bytes(full: Mapping[str, object], precision: Mapping[str, object]) -> float:
    categories = full.get("by_category")
    if not isinstance(categories, dict):
        elements = int(full.get("matmul_dynamic_rhs_elements", 0))
        return elements * float(precision.get("activation_bits", 8)) / 8.0
    cache_bits = float(precision.get("cache_and_state_bits", 16))
    activation_bits = float(precision.get("activation_bits", 8))
    total = 0.0
    for name, bucket in categories.items():
        if not isinstance(bucket, dict):
            continue
        elements = int(bucket.get("matmul_dynamic_rhs_elements", 0))
        bits = cache_bits if name in {"mla_attention", "kda_recurrence"} else activation_bits
        total += elements * bits / 8.0
    return total


def _mixed_compute_seconds(
    full: Mapping[str, object], config: Mapping[str, object], efficiency: float
) -> Mapping[str, float]:
    hardware = config["published_hardware"]
    assert isinstance(hardware, dict)
    chips = int(hardware["chips"])
    total_flops = 2.0 * int(full["matmul_macs"])
    categories = full.get("by_category")
    fp8_flops = 0.0
    fp8_categories = set(
        str(value)
        for value in config["roofline_precision"].get(
            "fp8_compute_categories", ["routed_expert"]
        )
    )
    if isinstance(categories, dict):
        for name in fp8_categories:
            bucket = categories.get(name)
            if isinstance(bucket, dict):
                fp8_flops += 2.0 * int(bucket.get("matmul_macs", 0))
    bf16_flops = total_flops - fp8_flops
    fp8_peak = chips * float(hardware["fp8_tflops_per_chip"]) * 1.0e12
    bf16_peak = chips * float(hardware["bf16_tflops_per_chip"]) * 1.0e12
    ideal = fp8_flops / fp8_peak + bf16_flops / bf16_peak
    return {
        "total_flops": total_flops,
        "fp8_eligible_flops": fp8_flops,
        "bf16_flops": bf16_flops,
        "ideal_seconds": ideal,
        "shape_adjusted_seconds": ideal / max(efficiency, 1.0e-12),
    }


def analyze_scenario(
    manifest_path: Path,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
    runs: Optional[Mapping[str, Mapping[str, object]]],
    calibration: Optional[Mapping[str, object]],
) -> Mapping[str, object]:
    request = manifest["request"]
    assert isinstance(request, dict)
    workloads = _workload_map(manifest)
    missing = [name for name in REPRESENTATIVE_LAYER_COUNTS if name not in workloads]
    if missing:
        raise RunnerError("{} lacks representative workloads: {}".format(manifest_path, ", ".join(missing)))
    if "full-model" not in workloads:
        raise RunnerError("{} lacks the full-model analytical workload".format(manifest_path))

    proxy = config["cocossim_proxy"]
    hardware = config["published_hardware"]
    precision = config["roofline_precision"]
    assert isinstance(proxy, dict) and isinstance(hardware, dict) and isinstance(precision, dict)
    cores = int(proxy["cores"])
    array_size = int(proxy["systolic_array_size"])
    manifest_dir = manifest_path.parent

    def validated_shape(name: str) -> Mapping[str, object]:
        entry = workloads[name]
        trace = manifest_dir / str(entry["trace_file"])
        shape = trace_shape_accounting(trace, cores, array_size)
        expected_macs = int(entry.get("matmul_macs", -1))
        expected_operations = int(entry.get("trace_line_count", -1))
        if int(shape["useful_macs"]) != expected_macs:
            raise RunnerError(
                "trace/manifest MAC mismatch for {}: trace {} versus manifest {}".format(
                    name, shape["useful_macs"], expected_macs
                )
            )
        if int(shape["operation_count"]) != expected_operations:
            raise RunnerError(
                "trace/manifest operation-count mismatch for {}: trace {} versus manifest {}".format(
                    name, shape["operation_count"], expected_operations
                )
            )
        return shape

    shapes: Dict[str, Mapping[str, object]] = {}
    for name in REPRESENTATIVE_LAYER_COUNTS:
        shapes[name] = validated_shape(name)
    full_trace_shape = validated_shape("full-model")
    extrapolated_shape = _sum_full_shape(shapes, REPRESENTATIVE_LAYER_COUNTS)

    simulation: Optional[Mapping[str, object]] = None
    eta_ccs = float(extrapolated_shape["tile_fill_ratio"])
    if runs is not None and calibration is not None:
        full_cycles = sum(
            int(runs[name]["cycles"]) * count
            for name, count in REPRESENTATIVE_LAYER_COUNTS.items()
        )
        calibration_rate = float(calibration["flops_per_cycle"])
        full_rate = 2.0 * int(extrapolated_shape["useful_macs"]) / full_cycles
        rate_ratio = full_rate / calibration_rate
        eta_ccs = min(1.0, float(extrapolated_shape["tile_fill_ratio"]), rate_ratio)
        simulation = {
            "representative_runs": runs,
            "layer_counts": dict(REPRESENTATIVE_LAYER_COUNTS),
            "extrapolated_full_model_cycles": full_cycles,
            "extrapolated_mean_scheduled_active_pct": _weighted_activity(
                runs, REPRESENTATIVE_LAYER_COUNTS, full_cycles
            ),
            "flops_per_cycle": full_rate,
            "rate_ratio_to_calibration": rate_ratio,
            "tile_fill_ratio_after_n_sharding": extrapolated_shape["tile_fill_ratio"],
            "eta_ccs": eta_ccs,
            "cycle_interpretation": (
                "Normalized compute-only COCOSSim cycles; not TPU clock cycles. "
                "The full count is 1*dense-KDA + 68*KDA-MoE + 24*MLA-MoE."
            ),
        }

    full = workloads["full-model"]
    class_elements = full.get("weight_elements_by_class")
    if not isinstance(class_elements, dict):
        raise RunnerError("full-model workload lacks weight_elements_by_class")
    weight_bytes, weight_bytes_by_class = _class_weight_bytes(class_elements, precision)
    unique = full.get("unique_across_repeat")
    unique_class_elements = (
        unique.get("weight_elements_by_class") if isinstance(unique, dict) else None
    )
    if not isinstance(unique_class_elements, dict):
        raise RunnerError("full-model workload lacks unique_across_repeat weight classes")
    resident_weight_bytes, resident_weight_bytes_by_class = _class_weight_bytes(
        unique_class_elements, precision
    )
    activation_bits = float(precision.get("activation_bits", 8))
    batch = int(request["batch"])
    query = int(request["query_tokens_per_sequence"])
    context = int(request["context_tokens_per_sequence"])
    token_rows = int(request["flattened_query_token_rows"])
    repeat = int(request.get("repeat", 1))
    cache_state = {
        "compressed_mla_cache_bytes": 24 * batch * context * (512 + 64) * 2,
        "expanded_reference_mla_cache_bytes": 24 * batch * context * 96 * (192 + 128) * 2,
        "bf16_kda_recurrent_state_bytes": 69 * batch * 96 * 128 * 128 * 2,
        "bf16_attnres_transient_bytes": 8 * token_rows * 7168 * 2,
    }
    # The lower bound assumes a latent MLA kernel reads the compressed 512+64
    # cache and reads+writes the fixed KDA recurrent state once.  The emitted
    # naive MLA GEMMs carry expanded K/V tensors; retain that as a sensitivity,
    # not as compulsory traffic.
    compressed_cache_state_traffic_bytes = (
        cache_state["compressed_mla_cache_bytes"]
        + 2 * cache_state["bf16_kda_recurrent_state_bytes"]
    ) * repeat
    expanded_cache_state_traffic_bytes = (
        cache_state["expanded_reference_mla_cache_bytes"]
        + 2 * cache_state["bf16_kda_recurrent_state_bytes"]
    ) * repeat
    manifest_dynamic_rhs_bytes = _dynamic_rhs_bytes(full, precision)
    vector_bytes_unfused = 2.0 * int(full.get("vector_elements", 0)) * activation_bits / 8.0
    unfused_hbm_bytes = (
        weight_bytes
        + manifest_dynamic_rhs_bytes
        + (
            int(full.get("matmul_lhs_elements", 0))
            + int(full.get("matmul_output_elements", 0))
        )
        * activation_bits
        / 8.0
        + vector_bytes_unfused
    )
    compulsory_hbm_bytes = weight_bytes + compressed_cache_state_traffic_bytes
    expanded_reference_hbm_bytes = weight_bytes + expanded_cache_state_traffic_bytes

    compute = _mixed_compute_seconds(full, config, eta_ccs)
    chips = int(hardware["chips"])
    hbm_bandwidth = chips * float(hardware["hbm_bandwidth_gbps_per_chip"]) * 1.0e9
    hbm_seconds = compulsory_hbm_bytes / hbm_bandwidth
    expanded_reference_hbm_seconds = expanded_reference_hbm_bytes / hbm_bandwidth

    model = manifest.get("model")
    official = model.get("official_config") if isinstance(model, dict) else None
    if not isinstance(official, dict):
        raise RunnerError("manifest.model.official_config must be an object")
    top_k = int(official["num_experts_per_token"])
    latent = int(official["routed_expert_hidden_size"])
    executed_mix = full.get("executed_layer_mix")
    if not isinstance(executed_mix, dict):
        raise RunnerError("full-model workload lacks executed_layer_mix")
    moe_layers = int(executed_mix.get("kda_moe", 0)) + int(
        executed_mix.get("mla_moe", 0)
    )
    remote_fraction = (chips - 1) / chips
    routed_ici_bytes = (
        2.0
        * moe_layers
        * token_rows
        * top_k
        * latent
        * activation_bits
        / 8.0
        * remote_fraction
    )
    ici_bidirectional_bandwidth = (
        chips * float(hardware["ici_bidirectional_bandwidth_gbps_per_chip"]) * 1.0e9
    )
    ici_one_way_injection_bandwidth = ici_bidirectional_bandwidth / 2.0
    routed_ici_seconds_bidirectional_optimistic = (
        routed_ici_bytes / ici_bidirectional_bandwidth
    )
    routed_ici_seconds = routed_ici_bytes / ici_one_way_injection_bandwidth

    proxy_seconds = max(compute["shape_adjusted_seconds"], hbm_seconds, routed_ici_seconds)
    ideal_roofline_seconds = max(compute["ideal_seconds"], hbm_seconds, routed_ici_seconds)
    if proxy_seconds == compute["shape_adjusted_seconds"]:
        bottleneck = "shape-adjusted compute"
    elif proxy_seconds == hbm_seconds:
        bottleneck = "HBM"
    else:
        bottleneck = "routed-MoE ICI payload"

    exact = full.get("by_quality", {}).get("exact", {}) if isinstance(full.get("by_quality"), dict) else {}
    proxy_quality = full.get("by_quality", {}).get("proxy", {}) if isinstance(full.get("by_quality"), dict) else {}
    return {
        "scenario_id": _scenario_id(manifest),
        "manifest_sha256": _sha256(manifest_path),
        "request": request,
        "full_model_accounting": {
            "layer_mix": full["executed_layer_mix"],
            "matmul_macs": int(full["matmul_macs"]),
            "matmul_flops": int(full["matmul_flops_convention_2_per_mac"]),
            "exact_matmul_flops": 2 * int(exact.get("matmul_macs", 0)),
            "proxy_matmul_flops": 2 * int(proxy_quality.get("matmul_macs", 0)),
            "full_trace_shape": full_trace_shape,
            "representative_extrapolated_shape": extrapolated_shape,
            "lm_head_included": bool(request.get("include_lm_head")),
        },
        "simulation": simulation,
        "analytical_proxy": {
            "estimate_level": "COCOSSim-shape-adjusted analytical proxy; not hardware-calibrated",
            "eta_ccs": eta_ccs,
            "mixed_precision": {
                "policy": (
                    "MXFP4-targeted routed-expert and latent-MoE projection GEMMs use FP8 peak; all remaining modeled GEMMs use BF16 peak"
                ),
                **compute,
            },
            "hbm": {
                "learned_weight_traffic_bytes": weight_bytes,
                "learned_weight_traffic_bytes_by_class": weight_bytes_by_class,
                "resident_learned_weight_capacity_bytes": resident_weight_bytes,
                "resident_learned_weight_capacity_bytes_by_class": resident_weight_bytes_by_class,
                "compressed_cache_state_traffic_bytes": compressed_cache_state_traffic_bytes,
                "expanded_reference_cache_state_traffic_bytes": expanded_cache_state_traffic_bytes,
                "manifest_naive_dynamic_rhs_bytes": manifest_dynamic_rhs_bytes,
                "compulsory_proxy_bytes": compulsory_hbm_bytes,
                "expanded_reference_proxy_bytes": expanded_reference_hbm_bytes,
                "unfused_traffic_sensitivity_bytes": unfused_hbm_bytes,
                "compulsory_proxy_seconds": hbm_seconds,
                "expanded_reference_proxy_seconds": expanded_reference_hbm_seconds,
            },
            "ici": {
                "routed_payload_bytes": routed_ici_bytes,
                "routed_payload_one_way_injection_seconds": routed_ici_seconds,
                "routed_payload_full_bidirectional_optimistic_seconds": routed_ici_seconds_bidirectional_optimistic,
                "remote_fraction_assumption": remote_fraction,
                "excluded": "tensor-parallel collectives, topology hops, contention, and expert imbalance",
            },
            "ideal_roofline_seconds": ideal_roofline_seconds,
            "shape_adjusted_no_contention_seconds": proxy_seconds,
            "bottleneck": bottleneck,
            "aggregate_tokens_per_second": token_rows * repeat / proxy_seconds,
            "cache_and_state_capacity": cache_state,
        },
    }


def _format_count(value: float) -> str:
    absolute = abs(value)
    for scale, suffix in ((1.0e15, "P"), (1.0e12, "T"), (1.0e9, "G"), (1.0e6, "M")):
        if absolute >= scale:
            return "{:.3g}{}".format(value / scale, suffix)
    return "{:.3g}".format(value)


def _format_ms(seconds: float) -> str:
    return "{:.3f}".format(seconds * 1.0e3)


def render_markdown(
    config: Mapping[str, object],
    calibration: Optional[Mapping[str, object]],
    scenarios: Sequence[Mapping[str, object]],
) -> str:
    hardware = config["published_hardware"]
    capacity = config["model_capacity"]
    assert isinstance(hardware, dict) and isinstance(capacity, dict)
    chips = int(hardware["chips"])
    hbm_gib_per_chip = float(hardware["hbm_gib_per_chip"])
    checkpoint_gib = float(capacity["checkpoint_size_tb_decimal"]) * 1.0e12 / (1024.0**3)
    raw_fit_chips = math.ceil(checkpoint_gib / hbm_gib_per_chip)
    slice_hbm_gib = chips * hbm_gib_per_chip
    lines = [
        "# Kimi K3 on TPU7x: COCOSSim proxy results",
        "",
        "> These are **not measurements from Google TPU hardware**. Raw cycles come from a compute-only COCOSSim proxy; latency and throughput are uncalibrated analytical proxy estimates assembled from published TPU7x lower-bound terms. ICI collectives, XLA fusion/layout, and hardware calibration remain outstanding.",
        "",
        "## Configuration",
        "",
        "- Scope: Kimi K3 text decoder (vision tower omitted).",
        "- Layer mix: 1 dense KDA + 68 KDA-MoE + 24 MLA-MoE.",
        "- TPU7x slice: {} chips; {} TensorCores/chip; {}x{} MXUs.".format(
            chips,
            hardware["tensor_cores_per_chip"],
            hardware["mxu_rows"],
            hardware["mxu_columns"],
        ),
        "- Published peaks per chip: {} BF16 TFLOP/s, {} FP8 TFLOP/s, {} GB/s HBM, {} GB/s bidirectional ICI.".format(
            hardware["bf16_tflops_per_chip"],
            hardware["fp8_tflops_per_chip"],
            hardware["hbm_bandwidth_gbps_per_chip"],
            hardware["ici_bidirectional_bandwidth_gbps_per_chip"],
        ),
        "- K3 precision proxy: routed-expert and latent-MoE projection weights use MXFP4 plus group scales and their GEMMs use FP8 peak; ignored/unquantized GEMMs and cache/state traffic use BF16.",
        "",
    ]
    if calibration is not None:
        dims = calibration["gemm"]
        lines.extend(
            [
                "The aligned calibration GEMM {}x{}x{} completed in {:,} COCOSSim cycles ({:.3g} FLOP/proxy-cycle). This normalizes shape/scheduling efficiency only; it does not infer a TPU clock.".format(
                    dims[0], dims[1], dims[2], int(calibration["cycles"]), calibration["flops_per_cycle"]
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Primary results",
            "",
            "| Phase | B | Query | Context | Modeled FLOPs | Raw full-stack cycles* | eta_CCS | Shape compute (ms) | Compressed-cache HBM (ms) | Expanded-K/V HBM sensitivity (ms) | No-contention proxy (ms) | Bottleneck | Aggregate tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for scenario in scenarios:
        request = scenario["request"]
        analytical = scenario["analytical_proxy"]
        mixed = analytical["mixed_precision"]
        simulation = scenario["simulation"]
        cycles = "analytic only" if simulation is None else "{:,}".format(
            int(simulation["extrapolated_full_model_cycles"])
        )
        lines.append(
            "| {mode} | {batch} | {query} | {context} | {flops} | {cycles} | {eta:.4f} | {compute} | {hbm} | {expanded_hbm} | {proxy} | {bottleneck} | {tps:,.0f} |".format(
                mode=request["mode_label"],
                batch=request["batch"],
                query=request["query_tokens_per_sequence"],
                context=request["context_tokens_per_sequence"],
                flops=_format_count(float(mixed["total_flops"])),
                cycles=cycles,
                eta=float(analytical["eta_ccs"]),
                compute=_format_ms(float(mixed["shape_adjusted_seconds"])),
                hbm=_format_ms(float(analytical["hbm"]["compulsory_proxy_seconds"])),
                expanded_hbm=_format_ms(float(analytical["hbm"]["expanded_reference_proxy_seconds"])),
                proxy=_format_ms(float(analytical["shape_adjusted_no_contention_seconds"])),
                bottleneck=analytical["bottleneck"],
                tps=float(analytical["aggregate_tokens_per_second"]),
            )
        )
    lines.extend(
        [
            "",
            "\\* Raw full-stack cycles are extrapolated from three simulated representative blocks using the exact 1/68/24 layer mix. They exclude the final norm and optional LM head, and they are normalized COCOSSim cycles—not TPU7x cycles.",
            "",
            "The primary HBM column counts learned weights plus a compressed 512+64 MLA cache and BF16 KDA-state read/write traffic. The expanded-K/V column is a sensitivity for the current reference-style 96-head K/V representation; it is not treated as compulsory traffic. Both assume perfect sharding and exclude contention. The no-contention proxy is `max(shape-adjusted compute, compressed-cache HBM, ideal routed-MoE one-way-injection ICI)`; tensor-parallel collectives and topology effects are not yet included.",
            "",
            "## Representative COCOSSim runs",
            "",
            "| Scenario | Dense-KDA cycles | KDA-MoE cycles | MLA-MoE cycles | Mean SA scheduled-active | Mean VU scheduled-active |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenarios:
        simulation = scenario["simulation"]
        if simulation is None:
            continue
        runs = simulation["representative_runs"]
        activity = simulation["extrapolated_mean_scheduled_active_pct"]
        lines.append(
            "| {scenario} | {dense:,} | {kda:,} | {mla:,} | {sa:.2f}% | {vu:.2f}% |".format(
                scenario=scenario["scenario_id"],
                dense=int(runs["dense-kda"]["cycles"]),
                kda=int(runs["kda-moe"]["cycles"]),
                mla=int(runs["mla-moe"]["cycles"]),
                sa=float(activity.get("SYSTOLIC_ARRAY", 0.0)),
                vu=float(activity.get("VECTOR_UNIT", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "`scheduled-active` is the fraction of COCOSSim cycles in a non-idle unit state. It is not MXU lane utilization or TPU profiler utilization.",
            "",
            "## Capacity and fidelity notes",
            "",
            "- The configured K3 checkpoint is {:.2f} TB decimal ({:.1f} GiB). This {}-chip slice supplies {:.0f} GiB HBM ({:.1f} GiB raw headroom). The raw-fit minimum is {} chips and does not reserve runtime/cache workspace.".format(
                float(capacity["checkpoint_size_tb_decimal"]),
                checkpoint_gib,
                chips,
                slice_hbm_gib,
                slice_hbm_gib - checkpoint_gib,
                raw_fit_chips,
            ),
            "- The manifest labels exact learned GEMM shapes separately from proxies. KDA recurrence/short convolution, RMSNorm and gating, MLA attention, and grouped MoE execution are approximations; AttnRes scoring and the vision tower are omitted.",
            "- Aggregate expert traces model ideal packed expert-token rows. They do not preserve per-expert small-M utilization or routing skew; use `--expert-layout per-expert` for that sensitivity.",
            "- COCOSSim's legacy DRAMSim3 HBM2 model is intentionally bypassed. TPU7x HBM and ICI appear only in the analytical bounds.",
            "- A hardware-calibrated result requires exact-shape KDA, MLA, and grouped-MoE microbenchmarks on TPU7x plus XProf measurements.",
            "",
            "## Sources",
            "",
        ]
    )
    sources = config.get("sources", {})
    if isinstance(sources, dict):
        source_labels = {
            "tpu7x": "Google TPU7x documentation",
            "tpu_architecture": "Google TPU architecture",
            "ironwood_performance": "Google Ironwood performance guidance",
            "kimi_k3": "Kimi K3 repository",
            "kimi_k3_config": "Kimi K3 official configuration",
        }
        for name, url in sources.items():
            lines.append(
                "- [{}]({})".format(
                    source_labels.get(name, name.replace("_", " ").title()), url
                )
            )
    lines.append("")
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Kimi K3 traces and derive TPU7x COCOSSim/roofline proxy results."
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--analytical-only",
        action="store_true",
        help="skip COCOSSim execution and use tile fill as the shape-efficiency proxy",
    )
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="reuse matching stats files already present in --output-dir",
    )
    parser.add_argument(
        "--max-sim-query-tokens",
        type=int,
        help="make larger-query manifests analytical-only while still simulating smaller points",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    config = _read_json(args.config)
    validate_config(config)
    if args.max_sim_query_tokens is not None and args.max_sim_query_tokens <= 0:
        raise RunnerError("--max-sim-query-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration: Optional[Mapping[str, object]] = None
    if not args.analytical_only:
        simulator = args.simulator.resolve()
        if not simulator.is_file():
            raise RunnerError(
                "simulator binary not found at {}; build COCOSSim first".format(simulator)
            )
        calibration_trace = args.output_dir / "calibration_matmul.txt"
        calibration_trace.write_text(
            "Matmul {} {} {}\n".format(*DEFAULT_CALIBRATION_GEMM), encoding="utf-8"
        )
        calibration_stats = args.output_dir / "calibration_stats.txt"
        calibration_log = args.output_dir / "calibration.log"
        run = None
        if args.reuse_stats:
            run = reuse_simulator_stats(
                simulator,
                calibration_trace,
                calibration_stats,
                calibration_log,
                config,
            )
        if run is None:
            run = run_simulator(
                simulator,
                calibration_trace,
                calibration_stats,
                calibration_log,
                config,
                args.output_dir,
            )
        calibration_flops = 2 * math.prod(DEFAULT_CALIBRATION_GEMM)
        calibration = {
            "gemm": list(DEFAULT_CALIBRATION_GEMM),
            **run,
            "flops": calibration_flops,
            "flops_per_cycle": calibration_flops / int(run["cycles"]),
        }

    scenarios = []
    for manifest_path in args.manifest:
        manifest_path = manifest_path.resolve()
        manifest = _read_json(manifest_path)
        scenario_id = _scenario_id(manifest)
        runs: Optional[Dict[str, Mapping[str, object]]] = None
        request = manifest.get("request")
        if not isinstance(request, dict):
            raise RunnerError("manifest.request must be an object")
        query_tokens = int(request.get("query_tokens_per_sequence", 0))
        simulate_scenario = not args.analytical_only and (
            args.max_sim_query_tokens is None
            or query_tokens <= args.max_sim_query_tokens
        )
        if simulate_scenario:
            runs = {}
            scenario_dir = args.output_dir / scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)
            workload_entries = _workload_map(manifest)
            for name in REPRESENTATIVE_LAYER_COUNTS:
                entry = workload_entries.get(name)
                if entry is None:
                    raise RunnerError("{} lacks workload {}".format(manifest_path, name))
                trace = manifest_path.parent / str(entry["trace_file"])
                stats = scenario_dir / "{}_stats.txt".format(name.replace("-", "_"))
                log = scenario_dir / "{}.log".format(name.replace("-", "_"))
                reused = None
                if args.reuse_stats:
                    reused = reuse_simulator_stats(
                        simulator, trace, stats, log, config
                    )
                if reused is not None:
                    runs[name] = reused
                else:
                    runs[name] = run_simulator(
                        simulator,
                        trace,
                        stats,
                        log,
                        config,
                        scenario_dir,
                    )
        scenarios.append(
            analyze_scenario(manifest_path, manifest, config, runs, calibration)
        )

    result = {
        "schema_version": 1,
        "estimate_level": "proxy",
        "config": config,
        "calibration": calibration,
        "scenarios": scenarios,
        "global_limitations": [
            "No TPU7x hardware was used; latency and throughput are uncalibrated proxy bounds.",
            "COCOSSim cycles are compute-only normalized cycles and are not converted with a guessed TPU clock.",
            "Representative blocks are extrapolated with the exact 1/68/24 text-decoder layer mix.",
            "Tensor-parallel collectives, topology contention, XLA fusion/layout, host overhead, and sampling are excluded.",
            "The vision tower is outside the modeled scope.",
        ],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        render_markdown(config, calibration, scenarios), encoding="utf-8"
    )
    print(args.output_dir / "results.json")
    print(args.output_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
