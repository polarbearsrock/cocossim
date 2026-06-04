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
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"
#include "Job.h"
#include <memory>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>

// Forward-declare MemorySystem so the header doesn't pull in dramsim3 internals.
// The full definition is only needed in ChipletArch.cc.
namespace dramsim3 { class MemorySystem; }

// Forward-declare State (used in drain_compute_dram_queue signature).
struct State;

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
    uint64_t dram_stall_cycles;  // cycles compute waited for DRAM

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
        , bytes_received(0)
        , dram_stall_cycles(0) {}
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

    // Input data size (bytes, per job — each chiplet fetches its shard from HBM)
    uint64_t data_size_bytes;

    // Cycle-accurate compute job (SA or VPU). When non-null, the FSM is ticked
    // each cycle instead of using the countdown timer below.
    Job* compute_job_ptr = nullptr;

    // Timing
    uint64_t creation_cycle;
    uint64_t start_cycle;
    uint64_t compute_start_cycle;  // cycle when COMPUTING phase actually began
    uint64_t completion_cycle;
    uint64_t compute_cycles;       // used only when compute_job_ptr == nullptr
    uint64_t communication_cycles;

    // DRAM tracking
    bool dram_issued;              // have DRAM reads been enqueued for this job?

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
        , data_size_bytes(0)
        , creation_cycle(0)
        , start_cycle(0)
        , compute_start_cycle(0)
        , completion_cycle(0)
        , compute_cycles(0)
        , communication_cycles(0)
        , dram_issued(false)
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

    /**
     * Initialize per-chiplet DRAMSim3 instances.
     * Must be called before run_until_idle().
     *
     * @param dram_config_path  Path to DRAMSim3 .ini config (e.g. "../dramsim3/configs/HBM2_8Gb_x128.ini")
     * @param sa_freq_ghz       Systolic array clock frequency in GHz (default 1.0)
     */
    void init_dram(const std::string& dram_config_path, double sa_freq_ghz = 1.0);

    // ========================================================================
    // Job Submission
    // ========================================================================

    /**
     * Submit a compute job (countdown timer — backward compat)
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
     * Submit a compute job (cycle-accurate SA/VPU simulation)
     *
     * @param operation_type Type of operation (e.g., "Matmul")
     * @param layer_id Layer ID for partitioning strategy lookup
     * @param data_size_bytes Size of input data (for DRAMSim3)
     * @param compute_job Heap-allocated SA or VPU job (ownership transferred)
     * @return job_id
     */
    int submit_job(const std::string& operation_type,
                   int layer_id,
                   uint64_t data_size_bytes,
                   Job* compute_job);

    /**
     * Submit a compute job pinned to a specific chiplet subset.
     * Used for data-parallel replicas where each replica owns a disjoint
     * set of chiplets. Bypasses partition_plan_.
     */
    int submit_job(const std::string& operation_type,
                   int layer_id,
                   uint64_t data_size_bytes,
                   Job* compute_job,
                   const std::vector<int>& chiplet_subset);

    /**
     * Set the systolic array size (must match the SysArrayState used).
     * Called once before simulation begins.
     */
    void set_sa_sz(int sa_sz);

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
    void run_until_idle(uint64_t max_cycles = 100000000ULL);

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
        uint64_t total_dram_stall_cycles;  // cycles compute FSM was done but DRAM outstanding

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

    // ── Cycle-accurate compute units ────────────────────────────────────────
    // One representative SA and VPU state shared across all chiplets.
    // All chiplets execute the same-shaped workload shard, so a single
    // state machine correctly models when compute finishes.
    int sa_sz_ = 64;
    SystolicArray::SysArrayState* sa_state_ = nullptr;
    VectorUnit::VecUnitState*     vpu_state_ = nullptr;

    // Job management
    std::vector<ChipletJob> jobs_;
    std::queue<int> pending_job_queue_;
    std::vector<int> active_jobs_;
    int next_job_id_;

    // Statistics
    ArchStats stats_;

    // ========================================================================
    // Per-Chiplet DRAMSim3 State
    // ========================================================================

    struct ChipletDRAMState {
        dramsim3::MemorySystem* mem_sys = nullptr;
        double tck_ns       = 1.0;   // DRAM cycle time in ns
        double clock_accum  = 0.0;   // fractional DRAM ticks accumulated
        uint64_t bytes_per_tx = 64;  // bytes per DRAM transaction
        uint64_t outstanding_reads = 0;  // decremented by read callback
        uint64_t next_addr_offset = 0;   // next free address offset (within this chiplet's space)
        uint64_t dram_reqs_remaining = 0; // reads not yet issued to DRAMSim3 for current job
    };

    // Stride between chiplet address spaces: 1 TB per chiplet, avoids collisions
    static constexpr uint64_t CHIPLET_ADDR_STRIDE = 1ULL << 40;

    std::vector<ChipletDRAMState> dram_states_;
    double sa_freq_ghz_ = 1.0;
    bool dram_enabled_ = false;

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

    /**
     * Drain the global to_enqueue list for a specific state machine, instantly
     * completing all pending DRAM transactions it issued.
     * This decouples the SA/VPU arithmetic timing from the DRAM model:
     * the FSM sees "instant DRAM" while DRAMSim3 independently models real timing.
     */
    void drain_compute_dram_queue(State* state);
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
