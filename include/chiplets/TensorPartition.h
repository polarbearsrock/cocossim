/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

/**
 * TensorPartition.h
 *
 * Tensor Partitioning Strategies for Multi-Chiplet AI Accelerators
 *
 * This module implements various parallelism strategies for distributing
 * neural network computation across multiple chiplets:
 *
 * 1. Data Parallel (DP): Split batch dimension
 * 2. Tensor Parallel (TP): Split weight matrices (Megatron-LM style)
 * 3. Pipeline Parallel (PP): Split layers across chiplets
 * 4. Hybrid 2D (TP + DP): Megatron-LM style
 * 5. Hybrid 3D (TP + DP + PP): Full parallelism (GPT-3 style)
 *
 * References:
 * [1] Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
 *     Models Using Model Parallelism", arXiv:1909.08053, 2019
 * [2] Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
 *     Clusters Using Megatron-LM", SC'21
 * [3] Brown et al., "Language Models are Few-Shot Learners" (GPT-3), NeurIPS 2020
 * [4] Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
 *     Parameter Models", SC'20
 */

#ifndef COCOSSIM_CHIPLETS_TENSOR_PARTITION_H
#define COCOSSIM_CHIPLETS_TENSOR_PARTITION_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace chiplets {

// Forward declarations
class ChipletTopology;

/**
 * Parallelism Strategy Types
 *
 * Reference: [1] Megatron-LM paper Section 3
 */
enum class ParallelismType {
    DATA_PARALLEL,      // Split batch across chiplets (all-reduce gradients)
    TENSOR_PARALLEL,    // Split weight matrices (column/row parallel)
    PIPELINE_PARALLEL,  // Split layers across chiplets (pipeline stages)
    HYBRID_2D,          // TP + DP (Megatron-LM)
    HYBRID_3D,          // TP + DP + PP (GPT-3 style)
    EXPERT_PARALLEL,    // MoE: Different experts on different chiplets
    CUSTOM              // User-defined partitioning
};

/**
 * Tensor Parallelism Mode
 *
 * Reference: [1] Megatron-LM Section 3.2
 */
enum class TensorParallelMode {
    COLUMN_PARALLEL,    // Split columns of weight matrix (e.g., FC1, QKV projection)
    ROW_PARALLEL,       // Split rows of weight matrix (e.g., FC2, output projection)
    BOTH                // Use both modes (typical for transformer layers)
};

/**
 * Dimension to partition
 */
enum class PartitionDimension {
    BATCH,              // Batch dimension (N)
    CHANNEL,            // Channel dimension (C)
    HEIGHT,             // Height dimension (H)
    WIDTH,              // Width dimension (W)
    SEQUENCE,           // Sequence length (T for transformers)
    HIDDEN,             // Hidden dimension (D for transformers)
    LAYER,              // Layer dimension (for pipeline parallel)
    EXPERT              // Expert dimension (for MoE)
};

/**
 * Tensor Partitioning Descriptor
 *
 * Describes how a single tensor is partitioned across chiplets
 */
struct TensorPartitionDescriptor {
    std::string tensor_name;        // Name of the tensor (e.g., "query_weight")
    PartitionDimension dimension;   // Which dimension to partition
    int num_partitions;             // How many partitions
    std::vector<int> chiplet_ids;   // Which chiplets hold each partition

    // Size information
    uint64_t original_size_bytes;
    uint64_t partition_size_bytes;

    // Requires communication?
    bool requires_allreduce;        // After backward pass (DP)
    bool requires_allgather;        // To reconstruct full tensor (TP)
    bool requires_reduce_scatter;   // Optimize allreduce (DP)
    bool requires_point_to_point;   // Pipeline parallel boundaries

    TensorPartitionDescriptor()
        : dimension(PartitionDimension::BATCH)
        , num_partitions(1)
        , original_size_bytes(0)
        , partition_size_bytes(0)
        , requires_allreduce(false)
        , requires_allgather(false)
        , requires_reduce_scatter(false)
        , requires_point_to_point(false) {}
};

/**
 * Layer Partitioning Strategy
 *
 * Describes how a single layer is partitioned across chiplets
 */
struct LayerPartitionStrategy {
    int layer_id;
    std::string layer_type;         // "conv", "matmul", "attention", etc.
    ParallelismType parallelism;

    // Tensor-level partitioning
    std::vector<TensorPartitionDescriptor> input_partitions;
    std::vector<TensorPartitionDescriptor> weight_partitions;
    std::vector<TensorPartitionDescriptor> output_partitions;

    // Pipeline parallel info
    int pipeline_stage;             // Which stage in pipeline (-1 if not PP)
    std::vector<int> stage_chiplets; // Chiplets assigned to this stage

    // Tensor parallel info
    TensorParallelMode tp_mode;
    int tp_degree;                  // Number of chiplets for TP

    // Data parallel info
    int dp_degree;                  // Number of chiplets for DP
    std::vector<int> dp_group;      // Chiplets in DP group

    LayerPartitionStrategy()
        : layer_id(-1)
        , parallelism(ParallelismType::DATA_PARALLEL)
        , pipeline_stage(-1)
        , tp_mode(TensorParallelMode::COLUMN_PARALLEL)
        , tp_degree(1)
        , dp_degree(1) {}
};

/**
 * Communication Operation for Collective
 *
 * Represents a collective communication operation required by partitioning
 */
struct CollectiveOperation {
    enum class Type {
        ALLREDUCE,          // Sum/Max across all chiplets (gradient sync)
        ALLGATHER,          // Gather data from all chiplets
        REDUCE_SCATTER,     // Reduce and scatter results
        BROADCAST,          // Send from one to all
        REDUCE,             // Reduce from all to one
        SCATTER,            // Scatter from one to all
        GATHER,             // Gather from all to one
        POINT_TO_POINT      // Pipeline parallel send/recv
    };

    Type type;
    std::vector<int> participating_chiplets;
    uint64_t data_size_bytes;
    int root_chiplet;               // For broadcast/reduce/scatter/gather
    int src_chiplet;                // For point-to-point
    int dst_chiplet;                // For point-to-point

    // Timing
    uint64_t scheduled_cycle;
    uint64_t completion_cycle;
    uint64_t latency_cycles;

    CollectiveOperation()
        : type(Type::ALLREDUCE)
        , data_size_bytes(0)
        , root_chiplet(-1)
        , src_chiplet(-1)
        , dst_chiplet(-1)
        , scheduled_cycle(0)
        , completion_cycle(0)
        , latency_cycles(0) {}
};

/**
 * Full Model Partitioning Plan
 *
 * Complete description of how an entire neural network model is partitioned
 */
class ModelPartitionPlan {
public:
    ModelPartitionPlan(const std::string& model_name,
                      ParallelismType primary_parallelism,
                      int num_chiplets);

    // Add layer partitioning strategies
    void add_layer_strategy(const LayerPartitionStrategy& strategy);

    // Query strategies
    const LayerPartitionStrategy& get_layer_strategy(int layer_id) const;
    std::vector<int> get_chiplets_for_layer(int layer_id) const;

    // Communication requirements
    std::vector<CollectiveOperation> get_required_collectives() const;
    void add_collective_operation(const CollectiveOperation& op);

    // Statistics
    uint64_t get_total_communication_bytes() const;
    int get_max_pipeline_depth() const;
    double estimate_parallelism_efficiency() const;

    // Getters
    const std::string& get_model_name() const { return model_name_; }
    ParallelismType get_primary_parallelism() const { return primary_parallelism_; }
    int get_num_chiplets() const { return num_chiplets_; }
    const std::vector<LayerPartitionStrategy>& get_layer_strategies() const {
        return layer_strategies_;
    }

    // Print summary
    void print_summary(std::ostream& os) const;

private:
    std::string model_name_;
    ParallelismType primary_parallelism_;
    int num_chiplets_;

    std::vector<LayerPartitionStrategy> layer_strategies_;
    std::vector<CollectiveOperation> collective_operations_;

    std::unordered_map<int, int> layer_id_to_index_;
};

/**
 * Partition Strategy Builder
 *
 * Helper class to build common partitioning strategies
 */
class PartitionStrategyBuilder {
public:
    /**
     * Data Parallel Strategy
     *
     * Split batch dimension across all chiplets.
     * Each chiplet has full model, different data.
     *
     * Communication: AllReduce gradients after backward pass
     *
     * Reference: [1] Classic data parallelism
     */
    static std::unique_ptr<ModelPartitionPlan> build_data_parallel(
        const std::string& model_name,
        int num_layers,
        int num_chiplets,
        int batch_size,
        uint64_t model_size_bytes);

    /**
     * Tensor Parallel Strategy (Megatron-LM style)
     *
     * Split weight matrices across chiplets.
     * Uses column-parallel and row-parallel patterns.
     *
     * Communication: AllReduce after row-parallel operations
     *
     * Reference: [1] Megatron-LM Section 3
     */
    static std::unique_ptr<ModelPartitionPlan> build_tensor_parallel(
        const std::string& model_name,
        int num_layers,
        int num_chiplets,
        uint64_t model_size_bytes);

    /**
     * Pipeline Parallel Strategy
     *
     * Split layers across chiplets (stages).
     * Each chiplet has subset of layers.
     *
     * Communication: Point-to-point between stages
     *
     * Reference: [2] GPipe, PipeDream
     */
    static std::unique_ptr<ModelPartitionPlan> build_pipeline_parallel(
        const std::string& model_name,
        int num_layers,
        int num_chiplets,
        int batch_size,
        uint64_t model_size_bytes);

    /**
     * Hybrid 2D Strategy (TP + DP)
     *
     * Combine tensor parallel and data parallel.
     * Example: 8 chiplets = 2 TP groups × 4 DP groups
     *
     * Communication:
     * - TP: AllReduce within TP group
     * - DP: AllReduce gradients across DP groups
     *
     * Reference: [1] Megatron-LM Section 4
     */
    static std::unique_ptr<ModelPartitionPlan> build_hybrid_2d(
        const std::string& model_name,
        int num_layers,
        int num_chiplets,
        int tp_degree,  // Tensor parallel degree
        int batch_size,
        uint64_t model_size_bytes);

    /**
     * Hybrid 3D Strategy (TP + DP + PP)
     *
     * Combine all three parallelism types.
     * Example: 64 chiplets = 2 TP × 4 DP × 8 PP
     *
     * Communication:
     * - TP: AllReduce within TP group
     * - DP: AllReduce gradients across DP groups
     * - PP: Point-to-point between pipeline stages
     *
     * Reference: [2] Megatron-LM Section 5, GPT-3 training
     */
    static std::unique_ptr<ModelPartitionPlan> build_hybrid_3d(
        const std::string& model_name,
        int num_layers,
        int num_chiplets,
        int tp_degree,
        int pp_degree,
        int batch_size,
        uint64_t model_size_bytes);

    /**
     * Expert Parallel Strategy (MoE)
     *
     * For Mixture of Experts models.
     * Different experts on different chiplets.
     *
     * Communication: Dynamic routing, AllToAll
     *
     * Reference: [3] Switch Transformer, GShard
     */
    static std::unique_ptr<ModelPartitionPlan> build_expert_parallel(
        const std::string& model_name,
        int num_layers,
        int num_experts,
        int num_chiplets,
        int batch_size,
        uint64_t model_size_bytes);

private:
    // Helper methods for building strategies
    static LayerPartitionStrategy create_data_parallel_layer(
        int layer_id, int num_chiplets, int batch_size);

    static LayerPartitionStrategy create_tensor_parallel_layer(
        int layer_id, int num_chiplets, TensorParallelMode mode);

    static LayerPartitionStrategy create_pipeline_parallel_layer(
        int layer_id, int stage, const std::vector<int>& stage_chiplets);
};

/**
 * Partition Validator
 *
 * Validates that a partition plan is feasible for a given topology
 */
class PartitionValidator {
public:
    PartitionValidator(const ModelPartitionPlan& plan,
                      const ChipletTopology& topology);

    // Validation checks
    bool validate_chiplet_assignment() const;
    bool validate_communication_feasibility() const;
    bool check_memory_constraints(uint64_t per_chiplet_memory_bytes) const;

    // Estimate performance
    struct PerformanceEstimate {
        uint64_t compute_cycles;
        uint64_t communication_cycles;
        double efficiency;  // compute / (compute + communication)
        uint64_t total_cycles;

        void print(std::ostream& os) const;
    };

    PerformanceEstimate estimate_performance(
        uint64_t compute_cycles_per_layer,
        int num_microbatches = 1) const;

    // Print warnings
    void print_validation_report(std::ostream& os) const;

private:
    const ModelPartitionPlan& plan_;
    const ChipletTopology& topology_;
};

} // namespace chiplets

#endif // COCOSSIM_CHIPLETS_TENSOR_PARTITION_H
