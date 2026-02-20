/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

/**
 * ChipletArch.h
 *
 * Multi-Chiplet Architecture Manager
 *
 * This module integrates all chiplet components to simulate a complete
 * multi-chiplet AI accelerator system:
 *
 * - Physical topology (UCIe links, routing)
 * - Tensor partitioning strategies
 * - Collective operations
 * - Job scheduling and execution
 *
 * References:
 * [1] Shoeybi et al., "Megatron-LM", arXiv:1909.08053, 2019
 * [2] AMD, "MI300A Architecture Whitepaper", 2023
 * [3] Intel, "Ponte Vecchio Architecture", 2022
 */

#ifndef COCOSSIM_CHIPLETS_CHIPLET_ARCH_H
#define COCOSSIM_CHIPLETS_CHIPLET_ARCH_H

#include "chiplets/ChipletTopology.h"
#include "chiplets/TensorPartition.h"
#include "chiplets/UCIeLink.h"
#include "chiplets/UCIePacket.h"
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>

namespace chiplets {

/**
 * Individual Chiplet in the Architecture
 *
 * Represents a single compute chiplet with its own resources
 */
struct Chiplet {
    int chiplet_id;
    std::string name;

    // Compute resources (simplified - will integrate with COCOSSim units later)
    uint64_t compute_capacity_ops;      // Operations per cycle
    uint64_t memory_capacity_bytes;     // On-chiplet memory (HBM)
    uint64_t memory_bandwidth_gbps;     // Memory bandwidth

    // Current state
    uint64_t memory_used_bytes;
    bool is_active;
    uint64_t total_compute_cycles;
    uint64_t idle_cycles;

    // Statistics
    uint64_t packets_sent;
    uint64_t packets_received;
    uint64_t bytes_sent;
    uint64_t bytes_received;

    Chiplet()
        : chiplet_id(-1)
        , compute_capacity_ops(0)
        , memory_capacity_bytes(0)
        , memory_bandwidth_gbps(0)
        , memory_used_bytes(0)
        , is_active(false)
        , total_compute_cycles(0)
        , idle_cycles(0)
        , packets_sent(0)
        , packets_received(0)
        , bytes_sent(0)
        , bytes_received(0) {}
};

/**
 * Multi-Chiplet Job
 *
 * Represents a distributed computation job across chiplets
 */
struct ChipletJob {
    int job_id;
    std::string operation_type;  // "matmul", "conv", "attention", "allreduce", etc.

    // Partitioning info
    ParallelismType parallelism;
    std::vector<int> participating_chiplets;

    // Data transfer requirements
    std::vector<CollectiveOperation> collective_ops;
    std::vector<UCIePacket*> pending_packets;

    // Timing
    uint64_t creation_cycle;
    uint64_t start_cycle;
    uint64_t completion_cycle;
    uint64_t compute_cycles;
    uint64_t communication_cycles;

    // Status
    enum class Status {
        CREATED,
        SCHEDULED,
        COMPUTING,
        COMMUNICATING,
        COMPLETED,
        FAILED
    };
    Status status;

    ChipletJob()
        : job_id(-1)
        , parallelism(ParallelismType::DATA_PARALLEL)
        , creation_cycle(0)
        , start_cycle(0)
        , completion_cycle(0)
        , compute_cycles(0)
        , communication_cycles(0)
        , status(Status::CREATED) {}
};

/**
 * Multi-Chiplet Architecture
 *
 * Top-level class managing the entire multi-chiplet system
 */
class ChipletArch {
public:
    /**
     * Constructor
     *
     * @param num_chiplets Number of chiplets in the system
     * @param topology_type Type of interconnect topology
     * @param phy_config UCIe physical layer configuration
     */
    ChipletArch(int num_chiplets,
                TopologyType topology_type,
                const UCIePhyConfig& phy_config);

    // ========================================================================
    // Configuration
    // ========================================================================

    /**
     * Configure chiplet resources
     */
    void configure_chiplet(int chiplet_id,
                          uint64_t compute_capacity_ops,
                          uint64_t memory_capacity_bytes,
                          uint64_t memory_bandwidth_gbps);

    /**
     * Set partition plan for a model
     */
    void set_partition_plan(std::unique_ptr<ModelPartitionPlan> plan);

    /**
     * Get topology for custom configuration
     */
    ChipletTopology* get_topology() { return topology_.get(); }

    // ========================================================================
    // Job Submission
    // ========================================================================

    /**
     * Submit a compute job
     *
     * @param operation_type Type of operation (e.g., "matmul", "conv")
     * @param layer_id Layer ID for partitioning strategy lookup
     * @param data_size_bytes Size of input data
     * @param compute_cycles Expected compute cycles per chiplet
     * @return job_id
     */
    int submit_job(const std::string& operation_type,
                   int layer_id,
                   uint64_t data_size_bytes,
                   uint64_t compute_cycles);

    /**
     * Submit a collective communication operation
     *
     * @param collective Collective operation descriptor
     * @return job_id
     */
    int submit_collective(const CollectiveOperation& collective);

    // ========================================================================
    // Simulation
    // ========================================================================

    /**
     * Advance simulation by one cycle
     */
    void tick(uint64_t current_cycle);

    /**
     * Run until all jobs complete or timeout
     */
    void run_until_idle(uint64_t max_cycles = 1000000);

    // ========================================================================
    // Query
    // ========================================================================

    /**
     * Check if a job is complete
     */
    bool is_job_complete(int job_id) const;

    /**
     * Get job status
     */
    ChipletJob::Status get_job_status(int job_id) const;

    /**
     * Get total latency for a job
     */
    uint64_t get_job_latency(int job_id) const;

    /**
     * Check if system is idle (no pending jobs)
     */
    bool is_idle() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    struct ArchStats {
        // Timing
        uint64_t total_cycles;
        uint64_t total_compute_cycles;
        uint64_t total_communication_cycles;
        uint64_t total_idle_cycles;

        // Jobs
        int jobs_submitted;
        int jobs_completed;
        int jobs_failed;

        // Data movement
        uint64_t total_bytes_transferred;
        uint64_t total_packets_sent;

        // Utilization
        double avg_chiplet_utilization;
        double avg_link_utilization;
        double communication_overhead;  // comm / (compute + comm)

        // Energy (from UCIe links)
        double total_energy_mJ;
        double avg_power_W;

        void reset();
        void print_summary(std::ostream& os) const;
    };

    const ArchStats& get_stats() const { return stats_; }
    void reset_stats();

    /**
     * Print status of all chiplets and links
     */
    void print_status(std::ostream& os) const;

    /**
     * Get detailed per-chiplet statistics
     */
    const Chiplet& get_chiplet(int chiplet_id) const;

private:
    // ========================================================================
    // Internal State
    // ========================================================================

    int num_chiplets_;
    uint64_t current_cycle_;

    // Components
    std::unique_ptr<ChipletTopology> topology_;
    std::unique_ptr<ModelPartitionPlan> partition_plan_;
    std::vector<Chiplet> chiplets_;

    // Job management
    std::vector<ChipletJob> jobs_;
    std::queue<int> pending_job_queue_;
    std::vector<int> active_jobs_;
    int next_job_id_;

    // Statistics
    ArchStats stats_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /**
     * Schedule jobs from pending queue
     */
    void schedule_jobs();

    /**
     * Execute compute phase of active jobs
     */
    void execute_compute();

    /**
     * Execute communication phase of active jobs
     */
    void execute_communication();

    /**
     * Check for job completions
     */
    void check_completions();

    /**
     * Update statistics
     */
    void update_stats();

    /**
     * Create packets for a collective operation
     *
     * Generates the specific UCIe packets needed to implement
     * a collective operation (AllReduce, AllGather, etc.)
     */
    void create_collective_packets(ChipletJob& job,
                                   const CollectiveOperation& collective);

    /**
     * Route and send packets through topology
     */
    void route_and_send_packet(UCIePacket* packet);

    /**
     * Get chiplet mutable reference
     */
    Chiplet& get_chiplet_mut(int chiplet_id);
};

/**
 * Common Chiplet Architectures
 *
 * Pre-configured architectures matching real systems
 */
namespace architectures {

/**
 * AMD MI300A Style
 *
 * 4 compute chiplets in 2x2 mesh with 1 I/O die
 * Standard 16GT/s x16 UCIe links
 *
 * Reference: [2] AMD MI300A Whitepaper
 */
std::unique_ptr<ChipletArch> create_mi300a_style(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb);

/**
 * Intel Ponte Vecchio Style
 *
 * 2 compute tiles with high-bandwidth links
 * High BW 32GT/s x32 UCIe links
 *
 * Reference: [3] Ponte Vecchio Architecture
 */
std::unique_ptr<ChipletArch> create_ponte_vecchio_style(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb);

/**
 * Ring Topology (8 chiplets)
 *
 * Good for Ring AllReduce collective
 * Balanced 24GT/s x16 UCIe links
 */
std::unique_ptr<ChipletArch> create_8chiplet_ring(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb);

/**
 * 4x4 Mesh Topology (16 chiplets)
 *
 * Scalable mesh for large models
 * Standard 16GT/s x16 UCIe links
 */
std::unique_ptr<ChipletArch> create_16chiplet_mesh(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb);

} // namespace architectures

} // namespace chiplets

#endif // COCOSSIM_CHIPLETS_CHIPLET_ARCH_H
