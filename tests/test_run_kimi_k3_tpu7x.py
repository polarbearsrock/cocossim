"""Tests for the Kimi K3 TPU7x proxy runner."""

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_kimi_k3_traces as generator  # noqa: E402
import run_kimi_k3_tpu7x as runner  # noqa: E402


class KimiK3TPU7xRunnerTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])

    def test_trace_shape_accounting_includes_core_and_tile_tails(self):
        with self.temporary_directory() as directory:
            trace = Path(directory) / "tail.txt"
            trace.write_text("Matmul 1 257 33\n", encoding="utf-8")
            result = runner.trace_shape_accounting(trace, cores=2, array_size=256)
        self.assertEqual(result["useful_macs"], 1 * 257 * 33)
        self.assertEqual(
            result["padded_macs_after_n_sharding"], 256 * 512 * (256 + 256)
        )
        self.assertLess(result["tile_fill_ratio"], 1.0)

    def test_parse_stats_groups_repeated_unit_names(self):
        with self.temporary_directory() as directory:
            stats = Path(directory) / "stats.txt"
            stats.write_text(
                "Cycles 100\n"
                "SystolicArray 50\n"
                "SystolicArray 100\n"
                "VectorUnit 25\n",
                encoding="utf-8",
            )
            result = runner.parse_stats(stats)
        self.assertEqual(result["cycles"], 100)
        self.assertEqual(result["unit_activity"]["SystolicArray"]["instances"], 2)
        self.assertEqual(
            result["unit_activity"]["SystolicArray"]["mean_scheduled_active_pct"],
            75.0,
        )

    def test_analytical_scenario_uses_full_model_accounting(self):
        with self.temporary_directory() as directory:
            directory_path = Path(directory)
            request = generator.GenerationRequest(
                mode="decode",
                tokens=1,
                context_tokens=1024,
                batch=1,
                routing=generator.RoutingSpec(
                    scenario="balanced",
                    expert_token_multiplier=Decimal("1"),
                    expert_layout="aggregate",
                ),
            )
            manifest_path = generator.generate(
                directory_path,
                request,
                generator.WORKLOADS,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
            result = runner.analyze_scenario(
                manifest_path, manifest, config, runs=None, calibration=None
            )
        self.assertIsNone(result["simulation"])
        self.assertEqual(
            result["full_model_accounting"]["layer_mix"],
            {"dense_kda": 1, "kda_moe": 68, "mla_moe": 24},
        )
        self.assertGreater(result["full_model_accounting"]["matmul_macs"], 100_000_000_000)
        self.assertGreater(
            result["analytical_proxy"]["hbm"]["resident_learned_weight_capacity_bytes"],
            50_000_000_000,
        )
        self.assertGreater(
            result["analytical_proxy"]["hbm"]["expanded_reference_proxy_seconds"],
            result["analytical_proxy"]["hbm"]["compulsory_proxy_seconds"],
        )
        self.assertEqual(result["analytical_proxy"]["bottleneck"], "HBM")

    def test_mxfp4_group_scale_bytes_are_exact(self):
        config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        total, by_class = runner._class_weight_bytes(
            {"routed_expert": 32, "moe_projection": 32, "attention": 1},
            config["roofline_precision"],
        )
        # Each compressed 32-element group is 16 data bytes + one scale byte.
        self.assertEqual(by_class["routed_expert"], 17.0)
        self.assertEqual(by_class["moe_projection"], 17.0)
        self.assertEqual(by_class["attention"], 2.0)
        self.assertEqual(total, 36.0)

    def test_trace_manifest_mismatch_is_rejected(self):
        with self.temporary_directory() as directory:
            directory_path = Path(directory)
            request = generator.GenerationRequest(
                mode="decode",
                tokens=1,
                context_tokens=16,
                batch=1,
                routing=generator.RoutingSpec(expert_layout="aggregate"),
            )
            manifest_path = generator.generate(
                directory_path / "repeat2", request, generator.WORKLOADS
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dense = next(
                entry for entry in manifest["workloads"] if entry["name"] == "dense-kda"
            )
            trace_path = manifest_path.parent / dense["trace_file"]
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write("Activation 1\n")
            config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(runner.RunnerError, "operation-count mismatch"):
                runner.analyze_scenario(
                    manifest_path, manifest, config, runs=None, calibration=None
                )

    def test_repeat_scales_traffic_and_scenario_identity(self):
        with self.temporary_directory() as directory:
            directory_path = Path(directory)
            request = generator.GenerationRequest(
                mode="decode",
                tokens=1,
                context_tokens=1024,
                batch=1,
                routing=generator.RoutingSpec(expert_layout="aggregate"),
                repeat=2,
                include_lm_head=True,
            )
            manifest_path = generator.generate(
                directory_path, request, generator.WORKLOADS
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
            result = runner.analyze_scenario(
                manifest_path, manifest, config, runs=None, calibration=None
            )
            once_request = generator.GenerationRequest(
                mode="decode",
                tokens=1,
                context_tokens=1024,
                batch=1,
                routing=generator.RoutingSpec(expert_layout="aggregate"),
                repeat=1,
                include_lm_head=True,
            )
            once_manifest_path = generator.generate(
                directory_path / "repeat1", once_request, generator.WORKLOADS
            )
            once_manifest = json.loads(
                once_manifest_path.read_text(encoding="utf-8")
            )
            once = runner.analyze_scenario(
                once_manifest_path,
                once_manifest,
                config,
                runs=None,
                calibration=None,
            )
        self.assertTrue(result["scenario_id"].endswith("_x2_lmhead"))
        cache = result["analytical_proxy"]["cache_and_state_capacity"]
        expected_traffic = 2 * (
            cache["compressed_mla_cache_bytes"]
            + 2 * cache["bf16_kda_recurrent_state_bytes"]
        )
        self.assertEqual(
            result["analytical_proxy"]["hbm"]["compressed_cache_state_traffic_bytes"],
            expected_traffic,
        )
        self.assertAlmostEqual(
            result["analytical_proxy"]["shape_adjusted_no_contention_seconds"],
            2 * once["analytical_proxy"]["shape_adjusted_no_contention_seconds"],
        )
        self.assertAlmostEqual(
            result["analytical_proxy"]["aggregate_tokens_per_second"],
            once["analytical_proxy"]["aggregate_tokens_per_second"],
        )

    def test_config_is_well_formed(self):
        config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        runner.validate_config(config)

        config["roofline_precision"]["activation_bits"] = -8
        with self.assertRaisesRegex(runner.RunnerError, "activation_bits"):
            runner.validate_config(config)


if __name__ == "__main__":
    unittest.main()
