/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/TensorPartition.h"
#include "chiplets/ChipletTopology.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace chiplets {

// ============================================================================
// ModelPartitionPlan Implementation
// ============================================================================

ModelPartitionPlan::ModelPartitionPlan(const std::string& model_name,
                                       ParallelismType primary_parallelism,
                                       int num_chiplets)
    : model_name_(model_name)
    , primary_parallelism_(primary_parallelism)
    , num_chiplets_(num_chiplets) {
}

void ModelPartitionPlan::add_layer_strategy(const LayerPartitionStrategy& strategy) {
    layer_id_to_index_[strategy.layer_id] = layer_strategies_.size();
    layer_strategies_.push_back(strategy);
}

const LayerPartitionStrategy& ModelPartitionPlan::get_layer_strategy(int layer_id) const {
    auto it = layer_id_to_index_.find(layer_id);
    if (it == layer_id_to_index_.end()) {
        throw std::runtime_error("Layer ID not found in partition plan: " +
                                 std::to_string(layer_id));
    }
    return layer_strategies_[it->second];
}

std::vector<int> ModelPartitionPlan::get_chiplets_for_layer(int layer_id) const {
    const auto& strategy = get_layer_strategy(layer_id);

    std::vector<int> chiplets;
    if (strategy.parallelism == ParallelismType::PIPELINE_PARALLEL) {
        chiplets = strategy.stage_chiplets;
    } else if (strategy.parallelism == ParallelismType::DATA_PARALLEL) {
        chiplets = strategy.dp_group;
    } else {
        // For other types, return all chiplets (simplified)
        for (int i = 0; i < num_chiplets_; i++) {
            chiplets.push_back(i);
        }
    }

    return chiplets;
}

std::vector<CollectiveOperation> ModelPartitionPlan::get_required_collectives() const {
    return collective_operations_;
}

void ModelPartitionPlan::add_collective_operation(const CollectiveOperation& op) {
    collective_operations_.push_back(op);
}

uint64_t ModelPartitionPlan::get_total_communication_bytes() const {
    uint64_t total = 0;
    for (const auto& op : collective_operations_) {
        total += op.data_size_bytes;
    }
    return total;
}

int ModelPartitionPlan::get_max_pipeline_depth() const {
    int max_depth = 0;
    for (const auto& strategy : layer_strategies_) {
        if (strategy.pipeline_stage > max_depth) {
            max_depth = strategy.pipeline_stage;
        }
    }
    return max_depth + 1;  // Stages are 0-indexed
}

double ModelPartitionPlan::estimate_parallelism_efficiency() const {
    /**
     * Estimate parallelism efficiency based on strategy
     *
     * Reference: [1] Megatron-LM paper Section 5
     *
     * Efficiency factors:
     * - Data Parallel: ~95% (communication overhead minimal)
     * - Tensor Parallel: ~80-90% (depends on TP degree)
     * - Pipeline Parallel: ~70-85% (bubble overhead)
     * - Hybrid: Product of individual efficiencies
     */

    double efficiency = 1.0;

    switch (primary_parallelism_) {
        case ParallelismType::DATA_PARALLEL:
            // DP efficiency: high for small model, lower for large
            efficiency = 0.95;
            break;

        case ParallelismType::TENSOR_PARALLEL:
            // TP efficiency decreases with degree (more communication)
            if (num_chiplets_ <= 2) {
                efficiency = 0.90;
            } else if (num_chiplets_ <= 4) {
                efficiency = 0.85;
            } else {
                efficiency = 0.75;
            }
            break;

        case ParallelismType::PIPELINE_PARALLEL: {
            // PP efficiency depends on number of stages and microbatches
            // Assume 8 microbatches (typical)
            int num_stages = get_max_pipeline_depth();
            double bubble_fraction = static_cast<double>(num_stages - 1) /
                                    (8.0 + num_stages - 1);
            efficiency = 1.0 - bubble_fraction;
            break;
        }

        case ParallelismType::HYBRID_2D:
            // Combine TP and DP inefficiencies
            efficiency = 0.85 * 0.95;  // TP * DP
            break;

        case ParallelismType::HYBRID_3D:
            // Combine all three
            efficiency = 0.80 * 0.90 * 0.80;  // TP * DP * PP
            break;

        default:
            efficiency = 0.8;
    }

    return efficiency;
}

void ModelPartitionPlan::print_summary(std::ostream& os) const {
    os << "\n=== Model Partition Plan: " << model_name_ << " ===\n";
    os << "  Primary Parallelism: ";
    switch (primary_parallelism_) {
        case ParallelismType::DATA_PARALLEL:     os << "Data Parallel"; break;
        case ParallelismType::TENSOR_PARALLEL:   os << "Tensor Parallel"; break;
        case ParallelismType::PIPELINE_PARALLEL: os << "Pipeline Parallel"; break;
        case ParallelismType::HYBRID_2D:         os << "Hybrid 2D (TP+DP)"; break;
        case ParallelismType::HYBRID_3D:         os << "Hybrid 3D (TP+DP+PP)"; break;
        case ParallelismType::EXPERT_PARALLEL:   os << "Expert Parallel"; break;
        default:                                 os << "Custom"; break;
    }
    os << "\n";
    os << "  Num Chiplets: " << num_chiplets_ << "\n";
    os << "  Num Layers: " << layer_strategies_.size() << "\n";

    if (primary_parallelism_ == ParallelismType::PIPELINE_PARALLEL ||
        primary_parallelism_ == ParallelismType::HYBRID_3D) {
        os << "  Pipeline Depth: " << get_max_pipeline_depth() << " stages\n";
    }

    os << "\n  Communication Requirements:\n";
    os << "    Total Collective Ops: " << collective_operations_.size() << "\n";
    os << "    Total Data Transfer: "
       << (get_total_communication_bytes() / 1024.0 / 1024.0) << " MB\n";

    os << "\n  Estimated Efficiency: " << std::fixed << std::setprecision(1)
       << (estimate_parallelism_efficiency() * 100.0) << "%\n";
}

// ============================================================================
// PartitionStrategyBuilder Implementation
// ============================================================================

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_data_parallel(
    const std::string& model_name,
    int num_layers,
    int num_chiplets,
    int batch_size,
    uint64_t model_size_bytes) {
    /**
     * Data Parallel Strategy
     *
     * Reference: [1] Classic data parallelism
     *
     * Each chiplet has full copy of model.
     * Batch is split: batch_per_chiplet = batch_size / num_chiplets
     *
     * Forward pass: Independent computation on each chiplet
     * Backward pass: AllReduce gradients (sum across chiplets)
     *
     * Communication: One AllReduce per layer with gradient size
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::DATA_PARALLEL, num_chiplets);

    // Per-chiplet batch size
    int local_batch = batch_size / num_chiplets;

    // Create DP group (all chiplets)
    std::vector<int> dp_group;
    for (int i = 0; i < num_chiplets; i++) {
        dp_group.push_back(i);
    }

    // Add strategy for each layer
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        LayerPartitionStrategy strategy = create_data_parallel_layer(
            layer_id, num_chiplets, local_batch);
        strategy.dp_group = dp_group;
        plan->add_layer_strategy(strategy);

        // Add AllReduce for gradient synchronization
        CollectiveOperation allreduce;
        allreduce.type = CollectiveOperation::Type::ALLREDUCE;
        allreduce.participating_chiplets = dp_group;
        // Assume gradient size = model_size / num_layers
        allreduce.data_size_bytes = model_size_bytes / num_layers;
        plan->add_collective_operation(allreduce);
    }

    return plan;
}

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_tensor_parallel(
    const std::string& model_name,
    int num_layers,
    int num_chiplets,
    uint64_t model_size_bytes) {
    /**
     * Tensor Parallel Strategy (Megatron-LM)
     *
     * Reference: [1] Megatron-LM Section 3
     *
     * Split weight matrices across chiplets:
     * - Column Parallel: Split columns (e.g., A = [A1 A2 ... An])
     *   Y = X @ A = [X@A1 X@A2 ... X@An] (concat outputs)
     *
     * - Row Parallel: Split rows (e.g., A = [A1; A2; ...; An])
     *   Y = X @ A = sum(X@Ai) (AllReduce outputs)
     *
     * Transformer MLP pattern:
     * FC1: Column parallel (no communication)
     * GeLU: Local
     * FC2: Row parallel (AllReduce)
     *
     * Attention pattern:
     * QKV projection: Column parallel
     * Attention: Local
     * Output projection: Row parallel (AllReduce)
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::TENSOR_PARALLEL, num_chiplets);

    // All chiplets participate in TP
    std::vector<int> tp_group;
    for (int i = 0; i < num_chiplets; i++) {
        tp_group.push_back(i);
    }

    // Alternate between column and row parallel
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        // Assume transformer pattern: QKV (col) -> Attn -> Out (row)
        // FC1 (col) -> GeLU -> FC2 (row)

        // Column parallel (QKV, FC1)
        LayerPartitionStrategy col_strategy = create_tensor_parallel_layer(
            layer_id * 2, num_chiplets, TensorParallelMode::COLUMN_PARALLEL);
        col_strategy.tp_degree = num_chiplets;
        plan->add_layer_strategy(col_strategy);

        // Row parallel (Output, FC2) - needs AllReduce
        LayerPartitionStrategy row_strategy = create_tensor_parallel_layer(
            layer_id * 2 + 1, num_chiplets, TensorParallelMode::ROW_PARALLEL);
        row_strategy.tp_degree = num_chiplets;
        plan->add_layer_strategy(row_strategy);

        // Add AllReduce for row parallel operation
        CollectiveOperation allreduce;
        allreduce.type = CollectiveOperation::Type::ALLREDUCE;
        allreduce.participating_chiplets = tp_group;
        // Activation size (assume hidden_dim = 4096, batch = 32, seq = 512)
        allreduce.data_size_bytes = 4096 * 32 * 512 * sizeof(float);
        plan->add_collective_operation(allreduce);
    }

    return plan;
}

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_pipeline_parallel(
    const std::string& model_name,
    int num_layers,
    int num_chiplets,
    int batch_size,
    uint64_t model_size_bytes) {
    /**
     * Pipeline Parallel Strategy
     *
     * Reference: [2] GPipe, PipeDream
     *
     * Split layers into stages:
     * - Stage 0: Layers 0 to L/N-1 on chiplet 0
     * - Stage 1: Layers L/N to 2L/N-1 on chiplet 1
     * - ...
     *
     * Microbatching: Split batch into microbatches to fill pipeline
     * Communication: Point-to-point between consecutive stages
     *
     * Pipeline bubble: (num_stages - 1) / num_microbatches
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::PIPELINE_PARALLEL, num_chiplets);

    int layers_per_stage = (num_layers + num_chiplets - 1) / num_chiplets;

    for (int stage = 0; stage < num_chiplets; stage++) {
        int start_layer = stage * layers_per_stage;
        int end_layer = std::min(start_layer + layers_per_stage, num_layers);

        std::vector<int> stage_chiplets = {stage};

        for (int layer_id = start_layer; layer_id < end_layer; layer_id++) {
            LayerPartitionStrategy strategy = create_pipeline_parallel_layer(
                layer_id, stage, stage_chiplets);
            plan->add_layer_strategy(strategy);
        }

        // Add point-to-point communication to next stage
        if (stage < num_chiplets - 1) {
            CollectiveOperation p2p;
            p2p.type = CollectiveOperation::Type::POINT_TO_POINT;
            p2p.src_chiplet = stage;
            p2p.dst_chiplet = stage + 1;
            // Activation size
            p2p.data_size_bytes = 4096 * (batch_size / 8) * 512 * sizeof(float);
            plan->add_collective_operation(p2p);
        }
    }

    return plan;
}

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_hybrid_2d(
    const std::string& model_name,
    int num_layers,
    int num_chiplets,
    int tp_degree,
    int batch_size,
    uint64_t model_size_bytes) {
    /**
     * Hybrid 2D: Tensor Parallel + Data Parallel
     *
     * Reference: [1] Megatron-LM Section 4
     *
     * Example: 8 chiplets, TP=2, DP=4
     * - TP group 0: [0, 1] - DP rank 0
     * - TP group 1: [2, 3] - DP rank 1
     * - TP group 2: [4, 5] - DP rank 2
     * - TP group 3: [6, 7] - DP rank 3
     *
     * Communication:
     * 1. TP: AllReduce within TP group (row parallel ops)
     * 2. DP: AllReduce across DP groups (gradients)
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::HYBRID_2D, num_chiplets);

    int dp_degree = num_chiplets / tp_degree;
    int local_batch = batch_size / dp_degree;

    // Create TP and DP groups
    std::vector<std::vector<int>> tp_groups(dp_degree);
    for (int dp_rank = 0; dp_rank < dp_degree; dp_rank++) {
        for (int tp_rank = 0; tp_rank < tp_degree; tp_rank++) {
            int chiplet_id = dp_rank * tp_degree + tp_rank;
            tp_groups[dp_rank].push_back(chiplet_id);
        }
    }

    // Add strategies for each layer
    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        // Column parallel
        LayerPartitionStrategy col_strategy = create_tensor_parallel_layer(
            layer_id * 2, num_chiplets, TensorParallelMode::COLUMN_PARALLEL);
        col_strategy.tp_degree = tp_degree;
        col_strategy.dp_degree = dp_degree;
        plan->add_layer_strategy(col_strategy);

        // Row parallel with TP AllReduce
        LayerPartitionStrategy row_strategy = create_tensor_parallel_layer(
            layer_id * 2 + 1, num_chiplets, TensorParallelMode::ROW_PARALLEL);
        row_strategy.tp_degree = tp_degree;
        row_strategy.dp_degree = dp_degree;
        plan->add_layer_strategy(row_strategy);

        // TP AllReduce (within each TP group)
        for (const auto& tp_group : tp_groups) {
            CollectiveOperation tp_allreduce;
            tp_allreduce.type = CollectiveOperation::Type::ALLREDUCE;
            tp_allreduce.participating_chiplets = tp_group;
            tp_allreduce.data_size_bytes = 4096 * local_batch * 512 * sizeof(float);
            plan->add_collective_operation(tp_allreduce);
        }

        // DP AllReduce (across TP groups - for gradients)
        for (int tp_rank = 0; tp_rank < tp_degree; tp_rank++) {
            std::vector<int> dp_group;
            for (int dp_rank = 0; dp_rank < dp_degree; dp_rank++) {
                dp_group.push_back(dp_rank * tp_degree + tp_rank);
            }

            CollectiveOperation dp_allreduce;
            dp_allreduce.type = CollectiveOperation::Type::ALLREDUCE;
            dp_allreduce.participating_chiplets = dp_group;
            dp_allreduce.data_size_bytes = model_size_bytes / (num_layers * tp_degree);
            plan->add_collective_operation(dp_allreduce);
        }
    }

    return plan;
}

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_hybrid_3d(
    const std::string& model_name,
    int num_layers,
    int num_chiplets,
    int tp_degree,
    int pp_degree,
    int batch_size,
    uint64_t model_size_bytes) {
    /**
     * Hybrid 3D: Tensor + Data + Pipeline Parallel
     *
     * Reference: [2] Megatron-LM Section 5, GPT-3 training
     *
     * Example: 64 chiplets, TP=2, DP=4, PP=8
     * - 8 pipeline stages
     * - Each stage has 8 chiplets (2 TP × 4 DP)
     *
     * Communication:
     * 1. TP: AllReduce within TP group
     * 2. DP: AllReduce across DP groups
     * 3. PP: Point-to-point between stages
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::HYBRID_3D, num_chiplets);

    int dp_degree = num_chiplets / (tp_degree * pp_degree);
    int layers_per_stage = num_layers / pp_degree;

    // Build pipeline stages
    for (int stage = 0; stage < pp_degree; stage++) {
        int start_layer = stage * layers_per_stage;
        int end_layer = std::min(start_layer + layers_per_stage, num_layers);

        // Chiplets in this stage
        std::vector<int> stage_chiplets;
        int base_chiplet = stage * (tp_degree * dp_degree);
        for (int i = 0; i < tp_degree * dp_degree; i++) {
            stage_chiplets.push_back(base_chiplet + i);
        }

        for (int layer_id = start_layer; layer_id < end_layer; layer_id++) {
            LayerPartitionStrategy strategy;
            strategy.layer_id = layer_id;
            strategy.parallelism = ParallelismType::HYBRID_3D;
            strategy.pipeline_stage = stage;
            strategy.stage_chiplets = stage_chiplets;
            strategy.tp_degree = tp_degree;
            strategy.dp_degree = dp_degree;
            plan->add_layer_strategy(strategy);
        }

        // PP: Point-to-point to next stage
        if (stage < pp_degree - 1) {
            CollectiveOperation p2p;
            p2p.type = CollectiveOperation::Type::POINT_TO_POINT;
            p2p.src_chiplet = stage_chiplets[0];
            p2p.dst_chiplet = stage_chiplets[0] + tp_degree * dp_degree;
            p2p.data_size_bytes = 4096 * (batch_size / 8 / dp_degree) * 512 * sizeof(float);
            plan->add_collective_operation(p2p);
        }
    }

    return plan;
}

std::unique_ptr<ModelPartitionPlan> PartitionStrategyBuilder::build_expert_parallel(
    const std::string& model_name,
    int num_layers,
    int num_experts,
    int num_chiplets,
    int batch_size,
    uint64_t model_size_bytes) {
    /**
     * Expert Parallel (Mixture of Experts)
     *
     * Reference: [3] Switch Transformer, GShard
     *
     * Each chiplet has subset of experts.
     * Tokens are routed to different experts dynamically.
     *
     * Communication: AllToAll to route tokens to experts
     */

    auto plan = std::make_unique<ModelPartitionPlan>(
        model_name, ParallelismType::EXPERT_PARALLEL, num_chiplets);

    int experts_per_chiplet = num_experts / num_chiplets;

    for (int layer_id = 0; layer_id < num_layers; layer_id++) {
        LayerPartitionStrategy strategy;
        strategy.layer_id = layer_id;
        strategy.parallelism = ParallelismType::EXPERT_PARALLEL;

        // Each chiplet gets experts_per_chiplet experts
        for (int chiplet = 0; chiplet < num_chiplets; chiplet++) {
            strategy.stage_chiplets.push_back(chiplet);
        }

        plan->add_layer_strategy(strategy);

        // AllToAll for token routing
        CollectiveOperation alltoall;
        alltoall.type = CollectiveOperation::Type::ALLGATHER;  // Simplified
        for (int i = 0; i < num_chiplets; i++) {
            alltoall.participating_chiplets.push_back(i);
        }
        // Token routing data
        alltoall.data_size_bytes = 4096 * batch_size * 512 * sizeof(float) / num_chiplets;
        plan->add_collective_operation(alltoall);
    }

    return plan;
}

// ============================================================================
// Helper Functions
// ============================================================================

LayerPartitionStrategy PartitionStrategyBuilder::create_data_parallel_layer(
    int layer_id, int num_chiplets, int batch_size) {
    LayerPartitionStrategy strategy;
    strategy.layer_id = layer_id;
    strategy.parallelism = ParallelismType::DATA_PARALLEL;
    strategy.dp_degree = num_chiplets;

    // Input partitioning: split batch
    TensorPartitionDescriptor input_partition;
    input_partition.tensor_name = "input";
    input_partition.dimension = PartitionDimension::BATCH;
    input_partition.num_partitions = num_chiplets;
    input_partition.requires_allreduce = true;  // Gradients
    strategy.input_partitions.push_back(input_partition);

    return strategy;
}

LayerPartitionStrategy PartitionStrategyBuilder::create_tensor_parallel_layer(
    int layer_id, int num_chiplets, TensorParallelMode mode) {
    LayerPartitionStrategy strategy;
    strategy.layer_id = layer_id;
    strategy.parallelism = ParallelismType::TENSOR_PARALLEL;
    strategy.tp_mode = mode;
    strategy.tp_degree = num_chiplets;

    // Weight partitioning
    TensorPartitionDescriptor weight_partition;
    weight_partition.tensor_name = "weight";
    if (mode == TensorParallelMode::COLUMN_PARALLEL) {
        weight_partition.dimension = PartitionDimension::CHANNEL;
    } else {
        weight_partition.dimension = PartitionDimension::HIDDEN;
        weight_partition.requires_allreduce = true;  // Row parallel needs AllReduce
    }
    weight_partition.num_partitions = num_chiplets;
    strategy.weight_partitions.push_back(weight_partition);

    return strategy;
}

LayerPartitionStrategy PartitionStrategyBuilder::create_pipeline_parallel_layer(
    int layer_id, int stage, const std::vector<int>& stage_chiplets) {
    LayerPartitionStrategy strategy;
    strategy.layer_id = layer_id;
    strategy.parallelism = ParallelismType::PIPELINE_PARALLEL;
    strategy.pipeline_stage = stage;
    strategy.stage_chiplets = stage_chiplets;

    // Pipeline requires point-to-point communication
    TensorPartitionDescriptor activation_partition;
    activation_partition.tensor_name = "activation";
    activation_partition.dimension = PartitionDimension::LAYER;
    activation_partition.requires_point_to_point = true;
    strategy.input_partitions.push_back(activation_partition);

    return strategy;
}

// ============================================================================
// PartitionValidator Implementation
// ============================================================================

PartitionValidator::PartitionValidator(const ModelPartitionPlan& plan,
                                      const ChipletTopology& topology)
    : plan_(plan)
    , topology_(topology) {
}

bool PartitionValidator::validate_chiplet_assignment() const {
    // Check that all chiplet IDs are valid
    for (const auto& strategy : plan_.get_layer_strategies()) {
        for (int chiplet : strategy.stage_chiplets) {
            if (chiplet < 0 || chiplet >= topology_.get_num_chiplets()) {
                return false;
            }
        }
        for (int chiplet : strategy.dp_group) {
            if (chiplet < 0 || chiplet >= topology_.get_num_chiplets()) {
                return false;
            }
        }
    }
    return true;
}

bool PartitionValidator::validate_communication_feasibility() const {
    // Check that all required communication paths exist in topology
    for (const auto& op : plan_.get_required_collectives()) {
        if (op.type == CollectiveOperation::Type::POINT_TO_POINT) {
            // Check if path exists
            auto route = topology_.get_route(op.src_chiplet, op.dst_chiplet);
            if (route.empty()) {
                return false;
            }
        }
    }
    return true;
}

bool PartitionValidator::check_memory_constraints(
    uint64_t per_chiplet_memory_bytes) const {
    if (per_chiplet_memory_bytes == 0) return false;

    // Pass 1: accumulate per-chiplet weight memory from explicit partition sizes.
    int n = topology_.get_num_chiplets();
    std::vector<uint64_t> chiplet_mem(n, 0);
    bool has_explicit_sizes = false;

    for (const auto& strategy : plan_.get_layer_strategies()) {
        auto account = [&](const std::vector<TensorPartitionDescriptor>& descs) {
            for (const auto& td : descs) {
                if (td.partition_size_bytes == 0) continue;
                has_explicit_sizes = true;
                for (int cid : td.chiplet_ids) {
                    if (cid >= 0 && cid < n)
                        chiplet_mem[cid] += td.partition_size_bytes;
                }
            }
        };
        account(strategy.weight_partitions);
        account(strategy.output_partitions);
    }

    if (has_explicit_sizes) {
        for (int i = 0; i < n; i++) {
            if (chiplet_mem[i] > per_chiplet_memory_bytes) return false;
        }
        return true;
    }

    // Pass 2 (fallback): builders don't always populate partition_size_bytes.
    // Estimate from total_communication_bytes and primary parallelism type.
    //
    // Data Parallel:  AllReduce size ≈ model size; each chiplet holds full model.
    // Tensor Parallel: weight shards ≪ AllReduce activations — can't estimate reliably.
    // Pipeline Parallel: P2P activations ≪ model size — can't estimate reliably.
    //
    // For TP and PP we conservatively return true (no info to reject).
    uint64_t total_comm = plan_.get_total_communication_bytes();

    switch (plan_.get_primary_parallelism()) {
        case ParallelismType::DATA_PARALLEL:
            // Each layer's AllReduce size ≈ that layer's gradient = its weight size.
            // Summed over all layers → total_comm ≈ total model size.
            // Each DP chiplet holds the full model.
            return total_comm <= per_chiplet_memory_bytes;

        default:
            // Cannot reliably estimate per-chiplet footprint without explicit sizes.
            return true;
    }
}

PartitionValidator::PerformanceEstimate PartitionValidator::estimate_performance(
    uint64_t compute_cycles_per_layer,
    int num_microbatches) const {
    /**
     * Estimate Performance
     *
     * Compute: compute_cycles_per_layer × num_layers
     * Communication: collective_latency × num_collectives
     *
     * Efficiency = compute / (compute + communication)
     */

    PerformanceEstimate estimate;

    int num_layers = plan_.get_layer_strategies().size();
    estimate.compute_cycles = compute_cycles_per_layer * num_layers;

    // Estimate communication cycles
    estimate.communication_cycles = 0;
    for (const auto& op : plan_.get_required_collectives()) {
        // Simplified: assume 100 cycles per collective
        // In reality, would calculate based on topology and data size
        estimate.communication_cycles += 100;
    }

    estimate.total_cycles = estimate.compute_cycles + estimate.communication_cycles;
    estimate.efficiency = static_cast<double>(estimate.compute_cycles) /
                         estimate.total_cycles;

    return estimate;
}

void PartitionValidator::print_validation_report(std::ostream& os) const {
    os << "\n=== Partition Validation Report ===\n";

    bool valid_assignment = validate_chiplet_assignment();
    os << "  Chiplet Assignment: " << (valid_assignment ? "VALID" : "INVALID") << "\n";

    bool valid_communication = validate_communication_feasibility();
    os << "  Communication Paths: " << (valid_communication ? "VALID" : "INVALID") << "\n";

    auto perf = estimate_performance(10000);
    perf.print(os);
}

void PartitionValidator::PerformanceEstimate::print(std::ostream& os) const {
    os << "\n  Performance Estimate:\n";
    os << "    Compute Cycles: " << compute_cycles << "\n";
    os << "    Communication Cycles: " << communication_cycles << "\n";
    os << "    Total Cycles: " << total_cycles << "\n";
    os << "    Efficiency: " << std::fixed << std::setprecision(1)
       << (efficiency * 100.0) << "%\n";
}

} // namespace chiplets
