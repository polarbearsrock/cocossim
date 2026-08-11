"""Tests for the weight-free Kimi K3 COCOSSim trace generator."""

import json
import os
import re
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import generate_kimi_k3_traces as generator  # noqa: E402


class KimiK3TraceGeneratorTests(unittest.TestCase):
    def make_request(
        self,
        *,
        mode="prefill",
        tokens=2,
        context_tokens=2,
        batch=1,
        scenario="balanced",
        multiplier="1",
        layout="aggregate",
        repeat=1,
    ):
        return generator.GenerationRequest(
            mode=mode,
            tokens=tokens,
            context_tokens=context_tokens,
            batch=batch,
            routing=generator.RoutingSpec(
                scenario=scenario,
                expert_token_multiplier=Decimal(multiplier),
                expert_layout=layout,
            ),
            repeat=repeat,
        )

    def test_official_dimensions_and_full_model_mix(self):
        self.assertEqual(generator.KIMI_K3_CONFIG["hidden_size"], 7168)
        self.assertEqual(generator.KIMI_K3_CONFIG["num_experts"], 896)
        self.assertEqual(generator.KIMI_K3_CONFIG["num_experts_per_token"], 16)
        self.assertEqual(generator.KDA_PROJECTION, 12288)
        self.assertEqual(generator.MLA_Q_PROJECTION, 18432)
        self.assertEqual(generator.MLA_KV_PROJECTION, 24576)

        workload = generator.build_workload("full-model", self.make_request())
        self.assertEqual(
            workload.layer_mix_per_repeat,
            {"dense_kda": 1, "kda_moe": 68, "mla_moe": 24},
        )

    def test_mla_uses_separate_query_and_context_lengths(self):
        request = self.make_request(
            mode="decode", tokens=1, context_tokens=1024, batch=2
        )
        workload = generator.build_workload("mla-moe", request)
        attention_ops = [
            op for op in workload.base_ops if op.category == "mla_attention"
        ]
        self.assertEqual(len(attention_ops), 2)
        self.assertEqual(attention_ops[0].dims, (192, 192, 1024))
        self.assertEqual(attention_ops[1].dims, (192, 1024, 128))

    def test_per_expert_decode_preserves_small_m_geometry(self):
        request = self.make_request(
            mode="decode", tokens=1, context_tokens=128, layout="per-expert"
        )
        workload = generator.build_workload("kda-moe", request)
        routed_matmuls = [
            op
            for op in workload.base_ops
            if op.category == "routed_expert" and op.op == "Matmul"
        ]
        self.assertEqual(workload.routing_accounting["unique_routed_experts_per_moe_layer"], 16)
        self.assertEqual(len(routed_matmuls), 16 * 3)
        self.assertTrue(all(op.dims[0] == 1 for op in routed_matmuls))
        self.assertTrue(all(op.quality == "exact" for op in routed_matmuls))

    def test_multiplier_scales_work_but_not_unique_experts(self):
        request = self.make_request(
            tokens=2,
            context_tokens=2,
            scenario="multiplier",
            multiplier="1.5",
        )
        workload = generator.build_workload("kda-moe", request)
        routing = workload.routing_accounting
        self.assertEqual(routing["base_expert_token_assignments_per_moe_layer"], 32)
        self.assertEqual(routing["effective_expert_token_rows_per_moe_layer"], 48)
        self.assertEqual(routing["unique_routed_experts_per_moe_layer"], 32)

        routed_matmuls = [
            op
            for op in workload.base_ops
            if op.category == "routed_expert" and op.op == "Matmul"
        ]
        self.assertEqual(len(routed_matmuls), 3)
        self.assertTrue(all(op.dims[0] == 48 for op in routed_matmuls))
        self.assertTrue(all(op.weight_repetitions == 32 for op in routed_matmuls))

    def test_manifest_reports_weights_macs_and_repeat_uniqueness(self):
        request = self.make_request(repeat=3)
        workload = generator.build_workload("dense-kda", request)
        summary = generator.summarize_ops(workload.ops)
        base_summary = generator.summarize_ops(workload.base_ops)
        self.assertEqual(summary["matmul_macs"], 3 * base_summary["matmul_macs"])
        self.assertEqual(
            summary["weight_elements_by_class"]["dense"],
            3 * base_summary["weight_elements_by_class"]["dense"],
        )
        self.assertGreater(summary["matmul_activation_elements"], 0)
        self.assertEqual(
            summary["matmul_activation_elements"],
            summary["matmul_lhs_elements"]
            + summary["matmul_output_elements"]
            + summary["matmul_dynamic_rhs_elements"],
        )
        self.assertGreater(summary["matmul_dynamic_rhs_elements"], 0)
        self.assertGreater(summary["by_quality"]["proxy"]["operation_count"], 0)
        self.assertGreater(summary["by_quality"]["exact"]["matmul_macs"], 0)

    def test_cli_writes_integer_only_traces_and_machine_readable_manifest(self):
        tmpdir = os.environ["TMPDIR"]
        with tempfile.TemporaryDirectory(dir=tmpdir) as directory:
            output_dir = Path(directory) / "traces"
            result = generator.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "decode",
                    "--tokens",
                    "1",
                    "--context-tokens",
                    "128",
                    "--workload",
                    "mla-moe",
                ]
            )
            self.assertEqual(result, 0)

            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertFalse(manifest["generator"]["requires_model_weights"])
            self.assertFalse(manifest["generator"]["requires_network"])
            self.assertEqual(len(manifest["workloads"]), 1)

            entry = manifest["workloads"][0]
            trace_path = output_dir / entry["trace_file"]
            self.assertEqual(entry["trace_line_count"], entry["operation_count"])
            self.assertIn("attention", entry["weight_elements_by_class"])
            self.assertIn("routed_expert", entry["weight_elements_by_class"])
            self.assertIn("shared_expert", entry["weight_elements_by_class"])
            self.assertIn("router", entry["weight_elements_by_class"])

            lines = trace_path.read_text(encoding="utf-8").splitlines()
            grammar = re.compile(r"^(Matmul|Activation|LayerNorm)( [1-9][0-9]*)+$")
            self.assertTrue(lines)
            self.assertTrue(all(grammar.fullmatch(line) for line in lines))

    def test_balanced_routing_rejects_non_unit_multiplier(self):
        with self.assertRaisesRegex(ValueError, "balanced routing"):
            generator.RoutingSpec(
                scenario="balanced",
                expert_token_multiplier=Decimal("1.1"),
                expert_layout="aggregate",
            )


if __name__ == "__main__":
    unittest.main()
