#!/usr/bin/env python3
"""Generate weight-free Kimi K3 workload traces for COCOSSim.

The emitted ``.txt`` files contain only operation names followed by positive
integer dimensions, which is the input grammar consumed by COCOSSim's
standard frontend.  A JSON manifest records which shapes correspond to real
Kimi K3 linear layers and which operations are performance proxies.

Example:

    python3 scripts/generate_kimi_k3_traces.py \
        --output-dir generated/kimi_k3_prefill_128 \
        --mode prefill --tokens 128 --context-tokens 128

No model weights, third-party Python packages, or network access are needed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


GENERATOR_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = 1
INT32_MAX = 2_147_483_647

# Values are from the official Kimi K3 configuration and model card.  Derived
# values are kept separate below so that the source fields remain recognizable.
KIMI_K3_CONFIG: Mapping[str, object] = {
    "hidden_size": 7168,
    "dense_intermediate_size": 33792,
    "num_hidden_layers": 93,
    "num_dense_layers": 1,
    "num_kda_layers": 69,
    "num_gated_mla_layers": 24,
    "num_attention_heads": 96,
    "head_dim": 128,
    "kda_short_conv_kernel_size": 4,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "routed_expert_hidden_size": 3584,
    "moe_intermediate_size": 3072,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_shared_experts": 2,
    "attn_res_block_size": 12,
    "vocab_size": 163840,
    "max_position_embeddings": 1048576,
    "weight_format": "MXFP4",
    "activation_format": "MXFP8",
}

MODEL_SOURCES = [
    "https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json",
    "https://github.com/MoonshotAI/Kimi-K3",
]

WORKLOADS = ("dense-kda", "kda-moe", "mla-moe", "full-model")
WEIGHT_CLASSES = (
    "attention",
    "dense",
    "router",
    "routed_expert",
    "shared_expert",
    "moe_projection",
    "lm_head",
)

CATEGORY_DEFINITIONS: Mapping[str, str] = {
    "decoder_norm": "COCOSSim LayerNorm proxy for a decoder RMSNorm.",
    "kda_projection": "Shape-exact learned KDA linear projection.",
    "kda_short_conv": (
        "Vector-pass proxy for one four-tap depthwise short convolution or "
        "its SiLU; it does not reproduce depthwise-convolution scheduling."
    ),
    "kda_qk_norm": "LayerNorm proxy for the per-head Q/K L2 normalization.",
    "kda_recurrence": (
        "Two per-head dense GEMM proxies for recurrent-state update and read; "
        "the KDA kernel, temporal dependencies, and state traffic are absent."
    ),
    "kda_output_norm_gate": (
        "LayerNorm and vector-pass proxies for fused gated RMS normalization."
    ),
    "mla_projection": "Shape-exact learned gated-MLA linear projection.",
    "mla_norm": "LayerNorm proxy for an MLA RMSNorm.",
    "mla_rope": "Single vector-pass proxy for rotary-position processing.",
    "mla_attention": (
        "Dense GEMM proxy with naive attention MAC count; it is not a TPU "
        "FlashAttention, paged-attention, or latent-cache implementation."
    ),
    "mla_softmax": (
        "Three generic vector passes stand in for max, exponentiation/reduction, "
        "and normalization over a rectangular Q-by-context score tensor."
    ),
    "mla_output_gate": "Generic vector-pass proxies for sigmoid and gating.",
    "dense_ffn": "Shape-exact learned linear layer in the dense SiTU-GLU FFN.",
    "dense_activation": "Generic vector-pass proxy for SiTU-GLU.",
    "moe_router": "Shape-exact learned 7168-by-896 router projection.",
    "moe_routing": (
        "Generic vector passes for router sigmoid and top-k selection; token "
        "dispatch and inter-chip communication are not represented."
    ),
    "moe_projection": "Shape-exact learned Stable LatentMoE down/up projection.",
    "routed_expert": "Routed-expert SiTU-GLU linear layer or activation proxy.",
    "moe_combine": "Generic vector-pass proxy for weighting and combining routes.",
    "moe_latent_norm": "LayerNorm proxy for routed-expert latent RMSNorm.",
    "shared_expert": "Learned shared-expert FFN linear layer or activation proxy.",
    "residual_add": "Generic vector-pass proxy for an elementwise residual add.",
    "final_norm": "COCOSSim LayerNorm proxy for the final RMSNorm.",
    "lm_head": "Shape-exact optional learned vocabulary projection.",
}

GLOBAL_LIMITATIONS = [
    "These are performance traces, not numerically executable Kimi K3 graphs.",
    "The vision tower, embedding lookup, sampling, and tokenizer are omitted.",
    "AttnRes block scoring/softmax, residual-state storage, and its blockwise dataflow are omitted.",
    "MXFP4 weights and MXFP8 activations are metadata only; COCOSSim's compiled datatype controls simulation.",
    "KDA recurrence, four-tap depthwise convolution, cache traffic, and fusion use coarse proxy operations.",
    "MLA score/value GEMMs use naive dense MAC counts and do not model XLA fusion or TPU attention kernels.",
    "MoE dispatch, combine collectives, expert parallel all-to-all, capacity limits, and routing skew are not modeled directly.",
    "Trace lines execute sequentially, so routed experts and the shared-expert branch do not overlap.",
    "COCOSSim uses 32-bit integer dimensions and has additional internal product limits for very long contexts.",
    "A trace alone cannot predict Google TPU latency until architecture, memory, precision, and interconnect behavior are calibrated.",
]


def _config_int(name: str) -> int:
    value = KIMI_K3_CONFIG[name]
    assert isinstance(value, int)
    return value


HIDDEN = _config_int("hidden_size")
DENSE_INTERMEDIATE = _config_int("dense_intermediate_size")
HEADS = _config_int("num_attention_heads")
HEAD_DIM = _config_int("head_dim")
KDA_PROJECTION = HEADS * HEAD_DIM
KDA_CONV_KERNEL = _config_int("kda_short_conv_kernel_size")
Q_LORA_RANK = _config_int("q_lora_rank")
KV_LORA_RANK = _config_int("kv_lora_rank")
QK_NOPE_DIM = _config_int("qk_nope_head_dim")
QK_ROPE_DIM = _config_int("qk_rope_head_dim")
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM
V_HEAD_DIM = _config_int("v_head_dim")
MLA_Q_PROJECTION = HEADS * Q_HEAD_DIM
MLA_KV_PROJECTION = HEADS * (QK_NOPE_DIM + V_HEAD_DIM)
MLA_OUTPUT = HEADS * V_HEAD_DIM
LATENT_MOE = _config_int("routed_expert_hidden_size")
MOE_INTERMEDIATE = _config_int("moe_intermediate_size")
EXPERTS = _config_int("num_experts")
TOP_K = _config_int("num_experts_per_token")
SHARED_EXPERTS = _config_int("num_shared_experts")
VOCAB_SIZE = _config_int("vocab_size")


@dataclass(frozen=True)
class TraceOp:
    """One COCOSSim trace operation plus manifest-only accounting metadata."""

    op: str
    dims: Tuple[int, ...]
    category: str
    quality: str
    weight_class: Optional[str] = None
    weight_repetitions: int = 1
    dynamic_rhs_repetitions: int = 1

    def __post_init__(self) -> None:
        if self.op not in {"Matmul", "Activation", "LayerNorm"}:
            raise ValueError("unsupported trace operation: {}".format(self.op))
        if not self.dims:
            raise ValueError("trace operations need at least one dimension")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.dims):
            raise ValueError("trace dimensions must be positive integers: {}".format(self.dims))
        if any(dim > INT32_MAX for dim in self.dims):
            raise ValueError("trace dimension exceeds COCOSSim's signed 32-bit parser: {}".format(self.dims))
        if self.op == "Matmul" and len(self.dims) != 3:
            raise ValueError("generated Matmul operations must have M, K, and N")
        if self.quality not in {"exact", "proxy"}:
            raise ValueError("quality must be exact or proxy")
        if self.weight_class is not None and self.weight_class not in WEIGHT_CLASSES:
            raise ValueError("unknown weight class: {}".format(self.weight_class))
        if self.weight_class is not None and self.op != "Matmul":
            raise ValueError("only Matmul operations can carry learned weights")
        if self.weight_repetitions <= 0 or self.dynamic_rhs_repetitions <= 0:
            raise ValueError("tensor repetition counts must be positive")

    def line(self) -> str:
        return "{} {}".format(self.op, " ".join(str(dim) for dim in self.dims))

    @property
    def elements(self) -> int:
        return math.prod(self.dims)

    @property
    def matmul_macs(self) -> int:
        return self.elements if self.op == "Matmul" else 0

    @property
    def static_weight_elements(self) -> int:
        if self.op != "Matmul" or self.weight_class is None:
            return 0
        _, k_dim, n_dim = self.dims
        return k_dim * n_dim * self.weight_repetitions

    @property
    def matmul_activation_elements(self) -> int:
        return (
            self.matmul_lhs_elements
            + self.matmul_output_elements
            + self.matmul_dynamic_rhs_elements
        )

    @property
    def matmul_lhs_elements(self) -> int:
        if self.op != "Matmul":
            return 0
        m_dim, k_dim, _ = self.dims
        return m_dim * k_dim

    @property
    def matmul_output_elements(self) -> int:
        if self.op != "Matmul":
            return 0
        m_dim, _, n_dim = self.dims
        return m_dim * n_dim

    @property
    def matmul_dynamic_rhs_elements(self) -> int:
        if self.op != "Matmul" or self.weight_class is not None:
            return 0
        _, k_dim, n_dim = self.dims
        return k_dim * n_dim * self.dynamic_rhs_repetitions


class TraceBuilder:
    def __init__(self) -> None:
        self.ops: List[TraceOp] = []

    def add(
        self,
        op: str,
        dims: Iterable[int],
        category: str,
        quality: str,
        weight_class: Optional[str] = None,
        weight_repetitions: int = 1,
        dynamic_rhs_repetitions: int = 1,
    ) -> None:
        self.ops.append(
            TraceOp(
                op=op,
                dims=tuple(dims),
                category=category,
                quality=quality,
                weight_class=weight_class,
                weight_repetitions=weight_repetitions,
                dynamic_rhs_repetitions=dynamic_rhs_repetitions,
            )
        )


@dataclass(frozen=True)
class RoutingSpec:
    scenario: str = "balanced"
    expert_token_multiplier: Decimal = Decimal("1")
    expert_layout: str = "aggregate"

    def __post_init__(self) -> None:
        if self.scenario not in {"balanced", "multiplier"}:
            raise ValueError("routing scenario must be balanced or multiplier")
        if self.expert_layout not in {"aggregate", "per-expert"}:
            raise ValueError("expert layout must be aggregate or per-expert")
        if not self.expert_token_multiplier.is_finite():
            raise ValueError("expert-token multiplier must be finite")
        if self.expert_token_multiplier < Decimal("1"):
            raise ValueError("expert-token multiplier must be at least 1")
        if self.scenario == "balanced" and self.expert_token_multiplier != Decimal("1"):
            raise ValueError("balanced routing requires an expert-token multiplier of 1")

    def accounting(self, token_rows: int) -> Mapping[str, object]:
        base_assignments = token_rows * TOP_K
        scaled = Decimal(base_assignments) * self.expert_token_multiplier
        effective_assignments = int(scaled.to_integral_value(rounding=ROUND_CEILING))
        active_experts = min(base_assignments, EXPERTS)
        floor_rows, experts_with_extra = divmod(effective_assignments, active_experts)
        return {
            "scenario": self.scenario,
            "expert_layout": self.expert_layout,
            "expert_layout_interpretation": (
                "One flattened expert GEMM with exact aggregate MACs but optimistic M geometry."
                if self.expert_layout == "aggregate"
                else "One GEMM set per active expert using an even processed-row distribution."
            ),
            "expert_token_multiplier": float(self.expert_token_multiplier),
            "expert_token_multiplier_decimal": _decimal_text(self.expert_token_multiplier),
            "expert_token_multiplier_interpretation": (
                "Scales processed expert-token rows to represent padding or imbalance overhead; "
                "it does not change top-k routes or the number of unique experts."
            ),
            "selected_experts_per_token": TOP_K,
            "total_routed_experts": EXPERTS,
            "base_expert_token_assignments_per_moe_layer": base_assignments,
            "effective_expert_token_rows_per_moe_layer": effective_assignments,
            "unique_routed_experts_per_moe_layer": active_experts,
            "unique_routed_experts_assumption": (
                "min(batch * query_tokens * selected_experts_per_token, total_routed_experts)"
            ),
            "processed_rows_per_active_expert_floor": floor_rows,
            "active_experts_with_one_extra_processed_row": experts_with_extra,
        }


@dataclass(frozen=True)
class GenerationRequest:
    mode: str
    tokens: int
    context_tokens: int
    batch: int
    routing: RoutingSpec
    repeat: int = 1
    include_lm_head: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError("mode must be prefill or decode")
        for name, value in (
            ("tokens", self.tokens),
            ("context_tokens", self.context_tokens),
            ("batch", self.batch),
            ("repeat", self.repeat),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.context_tokens < self.tokens:
            raise ValueError("context-tokens must be greater than or equal to query tokens")
        if self.tokens > _config_int("max_position_embeddings"):
            raise ValueError("tokens exceeds Kimi K3's configured context limit")
        if self.context_tokens > _config_int("max_position_embeddings"):
            raise ValueError("context-tokens exceeds Kimi K3's configured context limit")

    @property
    def token_rows(self) -> int:
        return self.batch * self.tokens


@dataclass(frozen=True)
class BuiltWorkload:
    name: str
    base_ops: Tuple[TraceOp, ...]
    ops: Tuple[TraceOp, ...]
    layer_mix_per_repeat: Mapping[str, int]
    routing_accounting: Optional[Mapping[str, object]]
    description: str


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _routing_label(routing: RoutingSpec) -> str:
    if routing.scenario == "balanced":
        return "balanced"
    return "multiplier_{}".format(
        _decimal_text(routing.expert_token_multiplier).replace(".", "p")
    )


def _add_decoder_input_norm(builder: TraceBuilder, rows: int) -> None:
    builder.add("LayerNorm", (rows, HIDDEN), "decoder_norm", "proxy")


def _add_kda(builder: TraceBuilder, request: GenerationRequest) -> None:
    rows = request.token_rows
    _add_decoder_input_norm(builder, rows)

    for _ in ("q", "k", "v"):
        builder.add(
            "Matmul",
            (rows, HIDDEN, KDA_PROJECTION),
            "kda_projection",
            "exact",
            weight_class="attention",
        )

    # One vector pass per convolution tap plus one pass for SiLU, for Q/K/V.
    # The manifest deliberately labels these as proxies rather than claiming a
    # depthwise convolution implementation in the standard frontend.
    for _stream in ("q", "k", "v"):
        for _pass in range(KDA_CONV_KERNEL + 1):
            builder.add(
                "Activation",
                (rows, KDA_PROJECTION),
                "kda_short_conv",
                "proxy",
            )

    for _ in ("q", "k"):
        builder.add(
            "LayerNorm",
            (rows * HEADS, HEAD_DIM),
            "kda_qk_norm",
            "proxy",
        )

    builder.add(
        "Matmul",
        (rows, HIDDEN, HEAD_DIM),
        "kda_projection",
        "exact",
        weight_class="attention",
    )
    builder.add(
        "Matmul",
        (rows, HEAD_DIM, KDA_PROJECTION),
        "kda_projection",
        "exact",
        weight_class="attention",
    )
    builder.add(
        "Matmul",
        (rows, HIDDEN, HEADS),
        "kda_projection",
        "exact",
        weight_class="attention",
    )

    # A pair of D-by-D operations per token and head preserves a simple linear
    # KDA MAC proxy while keeping the recurrent-state implementation explicit.
    for _ in ("state_update", "state_read"):
        builder.add(
            "Matmul",
            (rows * HEADS, HEAD_DIM, HEAD_DIM),
            "kda_recurrence",
            "proxy",
            dynamic_rhs_repetitions=request.batch * HEADS,
        )

    builder.add(
        "Matmul",
        (rows, HIDDEN, KDA_PROJECTION),
        "kda_projection",
        "exact",
        weight_class="attention",
    )
    builder.add(
        "LayerNorm",
        (rows * HEADS, HEAD_DIM),
        "kda_output_norm_gate",
        "proxy",
    )
    builder.add(
        "Activation",
        (rows, KDA_PROJECTION),
        "kda_output_norm_gate",
        "proxy",
    )
    builder.add(
        "Matmul",
        (rows, KDA_PROJECTION, HIDDEN),
        "kda_projection",
        "exact",
        weight_class="attention",
    )


def _add_mla(builder: TraceBuilder, request: GenerationRequest) -> None:
    rows = request.token_rows
    _add_decoder_input_norm(builder, rows)

    builder.add(
        "Matmul",
        (rows, HIDDEN, Q_LORA_RANK),
        "mla_projection",
        "exact",
        weight_class="attention",
    )
    builder.add("LayerNorm", (rows, Q_LORA_RANK), "mla_norm", "proxy")
    builder.add(
        "Matmul",
        (rows, Q_LORA_RANK, MLA_Q_PROJECTION),
        "mla_projection",
        "exact",
        weight_class="attention",
    )
    builder.add(
        "Matmul",
        (rows, HIDDEN, KV_LORA_RANK + QK_ROPE_DIM),
        "mla_projection",
        "exact",
        weight_class="attention",
    )
    builder.add("LayerNorm", (rows, KV_LORA_RANK), "mla_norm", "proxy")
    builder.add(
        "Matmul",
        (rows, KV_LORA_RANK, MLA_KV_PROJECTION),
        "mla_projection",
        "exact",
        weight_class="attention",
    )
    builder.add(
        "Activation",
        (rows, HEADS, QK_ROPE_DIM),
        "mla_rope",
        "proxy",
    )

    attention_rows = request.batch * HEADS * request.tokens
    rhs_repetitions = request.batch * HEADS
    builder.add(
        "Matmul",
        (attention_rows, Q_HEAD_DIM, request.context_tokens),
        "mla_attention",
        "proxy",
        dynamic_rhs_repetitions=rhs_repetitions,
    )
    for _ in range(3):
        builder.add(
            "Activation",
            (request.batch, HEADS, request.tokens, request.context_tokens),
            "mla_softmax",
            "proxy",
        )
    builder.add(
        "Matmul",
        (attention_rows, request.context_tokens, V_HEAD_DIM),
        "mla_attention",
        "proxy",
        dynamic_rhs_repetitions=rhs_repetitions,
    )

    builder.add(
        "Matmul",
        (rows, HIDDEN, MLA_OUTPUT),
        "mla_projection",
        "exact",
        weight_class="attention",
    )
    for _ in range(2):
        builder.add(
            "Activation", (rows, MLA_OUTPUT), "mla_output_gate", "proxy"
        )
    builder.add(
        "Matmul",
        (rows, MLA_OUTPUT, HIDDEN),
        "mla_projection",
        "exact",
        weight_class="attention",
    )


def _add_situ_glu(
    builder: TraceBuilder,
    rows: int,
    input_size: int,
    intermediate_size: int,
    output_size: int,
    category: str,
    activation_category: str,
    weight_class: str,
    quality: str = "exact",
    weight_repetitions: int = 1,
) -> None:
    for _ in ("gate", "up"):
        builder.add(
            "Matmul",
            (rows, input_size, intermediate_size),
            category,
            quality,
            weight_class=weight_class,
            weight_repetitions=weight_repetitions,
        )
    builder.add(
        "Activation",
        (rows, 2 * intermediate_size),
        activation_category,
        "proxy",
    )
    builder.add(
        "Activation",
        (rows, intermediate_size),
        activation_category,
        "proxy",
    )
    builder.add(
        "Matmul",
        (rows, intermediate_size, output_size),
        category,
        quality,
        weight_class=weight_class,
        weight_repetitions=weight_repetitions,
    )


def _add_dense_ffn(builder: TraceBuilder, request: GenerationRequest) -> None:
    rows = request.token_rows
    builder.add("LayerNorm", (rows, HIDDEN), "decoder_norm", "proxy")
    _add_situ_glu(
        builder,
        rows=rows,
        input_size=HIDDEN,
        intermediate_size=DENSE_INTERMEDIATE,
        output_size=HIDDEN,
        category="dense_ffn",
        activation_category="dense_activation",
        weight_class="dense",
    )
    builder.add("Activation", (rows, HIDDEN), "residual_add", "proxy")


def _expert_row_distribution(accounting: Mapping[str, object]) -> List[int]:
    active = int(accounting["unique_routed_experts_per_moe_layer"])
    effective = int(accounting["effective_expert_token_rows_per_moe_layer"])
    floor_rows, extra = divmod(effective, active)
    return [floor_rows + (1 if expert < extra else 0) for expert in range(active)]


def _add_moe(builder: TraceBuilder, request: GenerationRequest) -> Mapping[str, object]:
    rows = request.token_rows
    accounting = request.routing.accounting(rows)
    effective_rows = int(accounting["effective_expert_token_rows_per_moe_layer"])
    active_experts = int(accounting["unique_routed_experts_per_moe_layer"])

    builder.add("LayerNorm", (rows, HIDDEN), "decoder_norm", "proxy")
    builder.add(
        "Matmul",
        (rows, HIDDEN, EXPERTS),
        "moe_router",
        "exact",
        weight_class="router",
    )
    for _ in ("sigmoid", "topk"):
        builder.add("Activation", (rows, EXPERTS), "moe_routing", "proxy")

    builder.add(
        "Matmul",
        (rows, HIDDEN, LATENT_MOE),
        "moe_projection",
        "exact",
        weight_class="moe_projection",
    )

    if request.routing.expert_layout == "aggregate":
        _add_situ_glu(
            builder,
            rows=effective_rows,
            input_size=LATENT_MOE,
            intermediate_size=MOE_INTERMEDIATE,
            output_size=LATENT_MOE,
            category="routed_expert",
            activation_category="routed_expert",
            weight_class="routed_expert",
            quality="proxy",
            weight_repetitions=active_experts,
        )
    else:
        for expert_rows in _expert_row_distribution(accounting):
            _add_situ_glu(
                builder,
                rows=expert_rows,
                input_size=LATENT_MOE,
                intermediate_size=MOE_INTERMEDIATE,
                output_size=LATENT_MOE,
                category="routed_expert",
                activation_category="routed_expert",
                weight_class="routed_expert",
            )

    builder.add(
        "Activation",
        (rows, TOP_K, LATENT_MOE),
        "moe_combine",
        "proxy",
    )
    builder.add("LayerNorm", (rows, LATENT_MOE), "moe_latent_norm", "proxy")
    builder.add(
        "Matmul",
        (rows, LATENT_MOE, HIDDEN),
        "moe_projection",
        "exact",
        weight_class="moe_projection",
    )

    _add_situ_glu(
        builder,
        rows=rows,
        input_size=HIDDEN,
        intermediate_size=SHARED_EXPERTS * MOE_INTERMEDIATE,
        output_size=HIDDEN,
        category="shared_expert",
        activation_category="shared_expert",
        weight_class="shared_expert",
    )
    builder.add("Activation", (rows, HIDDEN), "residual_add", "proxy")
    return accounting


def _build_dense_kda(request: GenerationRequest) -> Tuple[Tuple[TraceOp, ...], Mapping[str, int], None]:
    builder = TraceBuilder()
    _add_kda(builder, request)
    _add_dense_ffn(builder, request)
    return tuple(builder.ops), {"dense_kda": 1, "kda_moe": 0, "mla_moe": 0}, None


def _build_kda_moe(
    request: GenerationRequest,
) -> Tuple[Tuple[TraceOp, ...], Mapping[str, int], Mapping[str, object]]:
    builder = TraceBuilder()
    _add_kda(builder, request)
    accounting = _add_moe(builder, request)
    return tuple(builder.ops), {"dense_kda": 0, "kda_moe": 1, "mla_moe": 0}, accounting


def _build_mla_moe(
    request: GenerationRequest,
) -> Tuple[Tuple[TraceOp, ...], Mapping[str, int], Mapping[str, object]]:
    builder = TraceBuilder()
    _add_mla(builder, request)
    accounting = _add_moe(builder, request)
    return tuple(builder.ops), {"dense_kda": 0, "kda_moe": 0, "mla_moe": 1}, accounting


def _build_full_model(
    request: GenerationRequest,
) -> Tuple[Tuple[TraceOp, ...], Mapping[str, int], Mapping[str, object]]:
    builder = TraceBuilder()
    accounting: Optional[Mapping[str, object]] = None
    mix = {"dense_kda": 0, "kda_moe": 0, "mla_moe": 0}

    # Official layer numbering is one-based in the published KDA/MLA lists:
    # Gated MLA appears at 4, 8, ..., 92, and 93.  Layer 1 is the only dense
    # FFN layer and uses KDA.
    for layer_number in range(1, _config_int("num_hidden_layers") + 1):
        if layer_number == 1:
            _add_kda(builder, request)
            _add_dense_ffn(builder, request)
            mix["dense_kda"] += 1
        elif layer_number % 4 == 0 or layer_number == 93:
            _add_mla(builder, request)
            accounting = _add_moe(builder, request)
            mix["mla_moe"] += 1
        else:
            _add_kda(builder, request)
            accounting = _add_moe(builder, request)
            mix["kda_moe"] += 1

    builder.add(
        "LayerNorm", (request.token_rows, HIDDEN), "final_norm", "proxy"
    )
    if request.include_lm_head:
        builder.add(
            "Matmul",
            (request.token_rows, HIDDEN, VOCAB_SIZE),
            "lm_head",
            "exact",
            weight_class="lm_head",
        )
    assert accounting is not None
    return tuple(builder.ops), mix, accounting


def build_workload(name: str, request: GenerationRequest) -> BuiltWorkload:
    if name == "dense-kda":
        base_ops, mix, accounting = _build_dense_kda(request)
        description = "One KDA decoder layer with Kimi K3's sole dense SiTU-GLU FFN."
    elif name == "kda-moe":
        base_ops, mix, accounting = _build_kda_moe(request)
        description = "One KDA decoder layer with Stable LatentMoE."
    elif name == "mla-moe":
        base_ops, mix, accounting = _build_mla_moe(request)
        description = "One gated-MLA decoder layer with Stable LatentMoE."
    elif name == "full-model":
        base_ops, mix, accounting = _build_full_model(request)
        description = (
            "The 93-layer Kimi K3 text decoder stack: one dense KDA layer, "
            "68 KDA+MoE layers, and 24 gated-MLA+MoE layers."
        )
    else:
        raise ValueError("unknown workload: {}".format(name))

    return BuiltWorkload(
        name=name,
        base_ops=base_ops,
        ops=base_ops * request.repeat,
        layer_mix_per_repeat=mix,
        routing_accounting=accounting,
        description=description,
    )


def _empty_summary_bucket() -> Dict[str, object]:
    return {
        "operation_count": 0,
        "matmul_macs": 0,
        "matmul_weight_elements": 0,
        "matmul_lhs_elements": 0,
        "matmul_output_elements": 0,
        "matmul_dynamic_rhs_elements": 0,
        "matmul_activation_elements": 0,
        "vector_elements": 0,
    }


def summarize_ops(ops: Sequence[TraceOp]) -> Mapping[str, object]:
    operation_counts: Counter[str] = Counter()
    by_quality: Dict[str, Dict[str, object]] = {
        "exact": _empty_summary_bucket(),
        "proxy": _empty_summary_bucket(),
    }
    by_category: Dict[str, Dict[str, object]] = defaultdict(_empty_summary_bucket)
    weight_elements_by_class: Dict[str, int] = {name: 0 for name in WEIGHT_CLASSES}

    total_macs = 0
    total_weight_elements = 0
    total_lhs_elements = 0
    total_output_elements = 0
    total_dynamic_rhs_elements = 0
    total_activation_elements = 0
    total_vector_elements = 0

    for op in ops:
        operation_counts[op.op] += 1
        macs = op.matmul_macs
        weights = op.static_weight_elements
        lhs_elements = op.matmul_lhs_elements
        output_elements = op.matmul_output_elements
        dynamic_rhs_elements = op.matmul_dynamic_rhs_elements
        activations = op.matmul_activation_elements
        vectors = op.elements if op.op != "Matmul" else 0

        total_macs += macs
        total_weight_elements += weights
        total_lhs_elements += lhs_elements
        total_output_elements += output_elements
        total_dynamic_rhs_elements += dynamic_rhs_elements
        total_activation_elements += activations
        total_vector_elements += vectors
        if op.weight_class is not None:
            weight_elements_by_class[op.weight_class] += weights

        for bucket in (by_quality[op.quality], by_category[op.category]):
            bucket["operation_count"] = int(bucket["operation_count"]) + 1
            bucket["matmul_macs"] = int(bucket["matmul_macs"]) + macs
            bucket["matmul_weight_elements"] = int(bucket["matmul_weight_elements"]) + weights
            bucket["matmul_lhs_elements"] = int(bucket["matmul_lhs_elements"]) + lhs_elements
            bucket["matmul_output_elements"] = int(bucket["matmul_output_elements"]) + output_elements
            bucket["matmul_dynamic_rhs_elements"] = int(bucket["matmul_dynamic_rhs_elements"]) + dynamic_rhs_elements
            bucket["matmul_activation_elements"] = int(bucket["matmul_activation_elements"]) + activations
            bucket["vector_elements"] = int(bucket["vector_elements"]) + vectors

    return {
        "operation_count": len(ops),
        "operation_counts": dict(sorted(operation_counts.items())),
        "matmul_macs": total_macs,
        "matmul_flops_convention_2_per_mac": 2 * total_macs,
        "matmul_weight_elements": total_weight_elements,
        "matmul_lhs_elements": total_lhs_elements,
        "matmul_output_elements": total_output_elements,
        "matmul_dynamic_rhs_elements": total_dynamic_rhs_elements,
        "matmul_activation_elements": total_activation_elements,
        "matmul_total_tensor_elements": total_weight_elements + total_activation_elements,
        "vector_elements": total_vector_elements,
        "weight_elements_by_class": weight_elements_by_class,
        "by_quality": by_quality,
        "by_category": dict(sorted(by_category.items())),
    }


def _trace_filename(workload: BuiltWorkload, request: GenerationRequest) -> str:
    repeat_suffix = "" if request.repeat == 1 else "_x{}".format(request.repeat)
    workload_label = workload.name.replace("-", "_")
    layout_label = request.routing.expert_layout.replace("-", "_")
    return (
        "kimi_k3_{mode}_b{batch}_q{tokens}_ctx{context}_{routing}_{layout}_{workload}{repeat}.txt"
    ).format(
        mode=request.mode,
        batch=request.batch,
        tokens=request.tokens,
        context=request.context_tokens,
        routing=_routing_label(request.routing),
        layout=layout_label,
        workload=workload_label,
        repeat=repeat_suffix,
    )


def _model_manifest() -> Mapping[str, object]:
    derived = {
        "kda_projection_size": KDA_PROJECTION,
        "mla_q_head_dim": Q_HEAD_DIM,
        "mla_q_projection_size": MLA_Q_PROJECTION,
        "mla_kv_a_projection_size": KV_LORA_RANK + QK_ROPE_DIM,
        "mla_kv_b_projection_size": MLA_KV_PROJECTION,
        "mla_output_projection_size": MLA_OUTPUT,
        "gated_mla_one_based_layers": list(range(4, 93, 4)) + [93],
    }
    return {
        "name": "Kimi K3",
        "scope": "text decoder",
        "official_config": dict(KIMI_K3_CONFIG),
        "derived_dimensions": derived,
        "sources": MODEL_SOURCES,
    }


def _request_manifest(request: GenerationRequest) -> Mapping[str, object]:
    return {
        "mode_label": request.mode,
        "query_tokens_per_sequence": request.tokens,
        "context_tokens_per_sequence": request.context_tokens,
        "batch": request.batch,
        "flattened_query_token_rows": request.token_rows,
        "repeat": request.repeat,
        "include_lm_head": request.include_lm_head,
        "routing": dict(request.routing.accounting(request.token_rows)),
    }


def _workload_manifest_entry(
    workload: BuiltWorkload,
    request: GenerationRequest,
    trace_filename: str,
) -> Mapping[str, object]:
    executed_summary = dict(summarize_ops(workload.ops))
    unique_summary = summarize_ops(workload.base_ops)
    layer_mix = {
        key: value * request.repeat
        for key, value in workload.layer_mix_per_repeat.items()
    }

    entry: Dict[str, object] = {
        "name": workload.name,
        "description": workload.description,
        "trace_file": trace_filename,
        "repeat": request.repeat,
        "layer_mix_per_repeat": dict(workload.layer_mix_per_repeat),
        "executed_layer_mix": layer_mix,
        "trace_line_count": len(workload.ops),
        **executed_summary,
        "unique_across_repeat": {
            "definition": (
                "Counts one base workload. Repetition expands execution/tensor "
                "touches but is assumed to reuse the same learned weights."
            ),
            "matmul_weight_elements": unique_summary["matmul_weight_elements"],
            "weight_elements_by_class": unique_summary["weight_elements_by_class"],
        },
    }
    if workload.routing_accounting is not None:
        entry["routing"] = dict(workload.routing_accounting)
    else:
        entry["routing"] = None
    return entry


def generate(
    output_dir: Path,
    request: GenerationRequest,
    workload_names: Sequence[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = [build_workload(name, request) for name in workload_names]
    entries = []

    for workload in workloads:
        filename = _trace_filename(workload, request)
        trace_path = output_dir / filename
        trace_text = "\n".join(op.line() for op in workload.ops) + "\n"
        trace_path.write_text(trace_text, encoding="utf-8")
        entries.append(_workload_manifest_entry(workload, request, filename))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator": {
            "name": "generate_kimi_k3_traces.py",
            "version": GENERATOR_VERSION,
            "dependencies": "Python standard library only",
            "requires_model_weights": False,
            "requires_network": False,
        },
        "trace_format": {
            "grammar": "Operation positive_integer [positive_integer ...]",
            "supported_operations_emitted": ["Matmul", "Activation", "LayerNorm"],
            "comments_or_labels_in_trace": False,
        },
        "model": _model_manifest(),
        "request": _request_manifest(request),
        "accounting_definitions": {
            "matmul_macs": "Sum of M*K*N over emitted Matmul operations.",
            "matmul_weight_elements": (
                "Learned K*N elements touched by emitted linear layers. Aggregate "
                "expert traces multiply each expert matrix by the number of unique "
                "active routed experts. Dynamic attention/state RHS matrices are excluded."
            ),
            "matmul_activation_elements": (
                "Sum of matmul_lhs_elements, matmul_output_elements, and "
                "matmul_dynamic_rhs_elements."
            ),
            "matmul_lhs_elements": "M*K dynamic left-hand-side elements over Matmul operations.",
            "matmul_output_elements": "M*N output elements over Matmul operations.",
            "matmul_dynamic_rhs_elements": (
                "K*N dynamic right-hand-side elements for attention/state proxies, "
                "including batch/head replication; learned weights are excluded."
            ),
            "vector_elements": (
                "Product of dimensions over emitted Activation and LayerNorm lines; "
                "this is an element count, not an exact operation or byte count."
            ),
            "quality_exact": (
                "The Kimi K3 learned linear-layer shape and MAC count are represented exactly."
            ),
            "quality_proxy": (
                "The trace operation preserves a stated work dimension or coarse MAC count "
                "but not the original kernel's implementation or scheduling."
            ),
        },
        "category_definitions": dict(CATEGORY_DEFINITIONS),
        "limitations": GLOBAL_LIMITATIONS,
        "workloads": entries,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parse_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("expected a decimal value") from exc
    if not value.is_finite():
        raise argparse.ArgumentTypeError("expected a finite decimal value")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate weight-free Kimi K3 COCOSSim traces and a JSON manifest."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument(
        "--tokens",
        type=_positive_int,
        default=128,
        help="query/input tokens per sequence (use 1 for ordinary decode)",
    )
    parser.add_argument(
        "--context-tokens",
        type=_positive_int,
        help="KV context per sequence; defaults to --tokens",
    )
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument(
        "--workload",
        action="append",
        choices=WORKLOADS + ("all",),
        help="repeat to select multiple workloads; defaults to all",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="repeat each selected workload in its trace",
    )
    parser.add_argument(
        "--routing",
        choices=("balanced", "multiplier"),
        default="balanced",
        help="balanced top-k work or a padded/imbalanced expert-token multiplier",
    )
    parser.add_argument(
        "--expert-token-multiplier",
        type=_parse_decimal,
        default=Decimal("1"),
        help="processed expert-token rows relative to balanced top-16 routing; must be >= 1",
    )
    parser.add_argument(
        "--expert-layout",
        choices=("aggregate", "per-expert"),
        default="aggregate",
        help=(
            "aggregate preserves expert MACs in three compact GEMMs; per-expert "
            "emits one small GEMM set per active expert"
        ),
    )
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="append the 7168-by-163840 vocabulary projection to full-model",
    )
    return parser


def _selected_workloads(values: Optional[Sequence[str]]) -> List[str]:
    if not values or "all" in values:
        return list(WORKLOADS)
    # Preserve command-line order while avoiding duplicate output paths.
    return list(dict.fromkeys(values))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    context_tokens = args.context_tokens if args.context_tokens is not None else args.tokens
    try:
        routing = RoutingSpec(
            scenario=args.routing,
            expert_token_multiplier=args.expert_token_multiplier,
            expert_layout=args.expert_layout,
        )
        request = GenerationRequest(
            mode=args.mode,
            tokens=args.tokens,
            context_tokens=context_tokens,
            batch=args.batch,
            routing=routing,
            repeat=args.repeat,
            include_lm_head=args.include_lm_head,
        )
        manifest_path = generate(
            output_dir=args.output_dir,
            request=request,
            workload_names=_selected_workloads(args.workload),
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
