/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/ChipletArch.h"
#include "chiplets/UCIeConfig.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace chiplets {

// ============================================================================
// ArchStats Implementation
// ============================================================================

void ChipletArch::ArchStats::reset() {
    total_cycles = 0;
    total_compute_cycles = 0;
    total_communication_cycles = 0;
    total_idle_cycles = 0;
    jobs_submitted = 0;
    jobs_completed = 0;
    jobs_failed = 0;
    total_bytes_transferred = 0;
    total_packets_sent = 0;
    avg_chiplet_utilization = 0.0;
    avg_link_utilization = 0.0;
    communication_overhead = 0.0;
    total_energy_mJ = 0.0;
    avg_power_W = 0.0;
}

void ChipletArch::ArchStats::print_summary(std::ostream& os) const {
    os << "\n=== Multi-Chiplet Architecture Statistics ===\n";

    // Timing
    os << "\n  Timing:\n";
    os << "    Total Cycles: " << total_cycles << "\n";
    os << "    Compute Cycles: " << total_compute_cycles
       << " (" << (total_compute_cycles * 100.0 / std::max(total_cycles, 1ULL)) << "%)\n";
    os << "    Communication Cycles: " << total_communication_cycles
       << " (" << (total_communication_cycles * 100.0 / std::max(total_cycles, 1ULL)) << "%)\n";
    os << "    Idle Cycles: " << total_idle_cycles
       << " (" << (total_idle_cycles * 100.0 / std::max(total_cycles, 1ULL)) << "%)\n";

    // Jobs
    os << "\n  Jobs:\n";
    os << "    Submitted: " << jobs_submitted << "\n";
    os << "    Completed: " << jobs_completed << "\n";
    os << "    Failed: " << jobs_failed << "\n";

    // Data Movement
    os << "\n  Data Movement:\n";
    os << "    Bytes Transferred: " << (total_bytes_transferred / 1024.0 / 1024.0) << " MB\n";
    os << "    Packets Sent: " << total_packets_sent << "\n";

    // Utilization
    os << "\n  Utilization:\n";
    os << "    Avg Chiplet Utilization: " << std::fixed << std::setprecision(1)
       << (avg_chiplet_utilization * 100.0) << "%\n";
    os << "    Avg Link Utilization: " << (avg_link_utilization * 100.0) << "%\n";
    os << "    Communication Overhead: " << (communication_overhead * 100.0) << "%\n";

    // Energy
    os << "\n  Energy:\n";
    os << "    Total Energy: " << std::setprecision(2) << total_energy_mJ << " mJ\n";
    os << "    Avg Power: " << std::setprecision(1) << avg_power_W << " W\n";
}

// ============================================================================
// ChipletArch Implementation
// ============================================================================

ChipletArch::ChipletArch(int num_chiplets,
                        TopologyType topology_type,
                        const UCIePhyConfig& phy_config)
    : num_chiplets_(num_chiplets)
    , current_cycle_(0)
    , next_job_id_(0) {

    // Create topology
    topology_ = std::make_unique<ChipletTopology>(topology_type, num_chiplets);
    topology_->build_topology(phy_config);

    // Initialize chiplets
    chiplets_.resize(num_chiplets);
    for (int i = 0; i < num_chiplets; i++) {
        chiplets_[i].chiplet_id = i;
        chiplets_[i].name = "Chiplet_" + std::to_string(i);
    }

    stats_.reset();
}

// ============================================================================
// Configuration
// ============================================================================

void ChipletArch::configure_chiplet(int chiplet_id,
                                   uint64_t compute_capacity_ops,
                                   uint64_t memory_capacity_bytes,
                                   uint64_t memory_bandwidth_gbps) {
    if (chiplet_id < 0 || chiplet_id >= num_chiplets_) {
        std::cerr << "ERROR: Invalid chiplet ID: " << chiplet_id << "\n";
        return;
    }

    Chiplet& chiplet = chiplets_[chiplet_id];
    chiplet.compute_capacity_ops = compute_capacity_ops;
    chiplet.memory_capacity_bytes = memory_capacity_bytes;
    chiplet.memory_bandwidth_gbps = memory_bandwidth_gbps;
}

void ChipletArch::set_partition_plan(std::unique_ptr<ModelPartitionPlan> plan) {
    partition_plan_ = std::move(plan);
}

// ============================================================================
// Job Submission
// ============================================================================

int ChipletArch::submit_job(const std::string& operation_type,
                           int layer_id,
                           uint64_t data_size_bytes,
                           uint64_t compute_cycles) {
    ChipletJob job;
    job.job_id = next_job_id_++;
    job.operation_type = operation_type;
    job.creation_cycle = current_cycle_;
    job.compute_cycles = compute_cycles;
    job.status = ChipletJob::Status::CREATED;

    // Get partitioning strategy from plan
    if (partition_plan_ && layer_id >= 0) {
        const auto& strategy = partition_plan_->get_layer_strategy(layer_id);
        job.parallelism = strategy.parallelism;
        job.participating_chiplets = partition_plan_->get_chiplets_for_layer(layer_id);
    } else {
        // Default: all chiplets participate
        job.parallelism = ParallelismType::DATA_PARALLEL;
        for (int i = 0; i < num_chiplets_; i++) {
            job.participating_chiplets.push_back(i);
        }
    }

    // Add to job list and pending queue
    jobs_.push_back(job);
    pending_job_queue_.push(job.job_id);

    stats_.jobs_submitted++;

    return job.job_id;
}

int ChipletArch::submit_collective(const CollectiveOperation& collective) {
    ChipletJob job;
    job.job_id = next_job_id_++;
    job.operation_type = "collective";
    job.creation_cycle = current_cycle_;
    job.compute_cycles = 0;  // Pure communication
    job.status = ChipletJob::Status::CREATED;
    job.participating_chiplets = collective.participating_chiplets;

    // Create packets for collective
    job.collective_ops.push_back(collective);
    create_collective_packets(job, collective);

    // Add to job list and pending queue
    jobs_.push_back(job);
    pending_job_queue_.push(job.job_id);

    stats_.jobs_submitted++;

    return job.job_id;
}

// ============================================================================
// Simulation
// ============================================================================

void ChipletArch::tick(uint64_t current_cycle) {
    current_cycle_ = current_cycle;

    // 1. Schedule new jobs
    schedule_jobs();

    // 2. Execute compute phase
    execute_compute();

    // 3. Execute communication phase
    execute_communication();

    // 4. Tick the topology (UCIe links)
    topology_->tick(current_cycle);

    // 5. Check for completions
    check_completions();

    // 6. Update statistics
    update_stats();
}

void ChipletArch::run_until_idle(uint64_t max_cycles) {
    uint64_t start_cycle = current_cycle_;

    while (!is_idle() && (current_cycle_ - start_cycle) < max_cycles) {
        tick(current_cycle_);
        current_cycle_++;
    }

    if (!is_idle()) {
        std::cerr << "WARNING: Simulation timeout after " << max_cycles << " cycles\n";
    }
}

// ============================================================================
// Query
// ============================================================================

bool ChipletArch::is_job_complete(int job_id) const {
    if (job_id < 0 || job_id >= static_cast<int>(jobs_.size())) {
        return false;
    }
    return jobs_[job_id].status == ChipletJob::Status::COMPLETED;
}

ChipletJob::Status ChipletArch::get_job_status(int job_id) const {
    if (job_id < 0 || job_id >= static_cast<int>(jobs_.size())) {
        return ChipletJob::Status::FAILED;
    }
    return jobs_[job_id].status;
}

uint64_t ChipletArch::get_job_latency(int job_id) const {
    if (job_id < 0 || job_id >= static_cast<int>(jobs_.size())) {
        return 0;
    }
    const auto& job = jobs_[job_id];
    if (job.status != ChipletJob::Status::COMPLETED) {
        return 0;
    }
    return job.completion_cycle - job.creation_cycle;
}

bool ChipletArch::is_idle() const {
    return pending_job_queue_.empty() && active_jobs_.empty();
}

// ============================================================================
// Statistics
// ============================================================================

void ChipletArch::reset_stats() {
    stats_.reset();
}

void ChipletArch::print_status(std::ostream& os) const {
    os << "\n=== Multi-Chiplet Architecture Status ===\n";
    os << "  Num Chiplets: " << num_chiplets_ << "\n";
    os << "  Current Cycle: " << current_cycle_ << "\n";
    os << "  Pending Jobs: " << pending_job_queue_.size() << "\n";
    os << "  Active Jobs: " << active_jobs_.size() << "\n";

    // Print chiplet states
    os << "\n  Chiplet States:\n";
    for (const auto& chiplet : chiplets_) {
        os << "    " << chiplet.name << ": "
           << (chiplet.is_active ? "ACTIVE" : "IDLE")
           << " | Memory: " << (chiplet.memory_used_bytes / 1024.0 / 1024.0) << "/"
           << (chiplet.memory_capacity_bytes / 1024.0 / 1024.0) << " MB\n";
    }

    // Print topology status
    // TODO: Implement ChipletTopology::print_status()
    // topology_->print_status(os);
}

const Chiplet& ChipletArch::get_chiplet(int chiplet_id) const {
    if (chiplet_id < 0 || chiplet_id >= num_chiplets_) {
        throw std::out_of_range("Invalid chiplet ID: " + std::to_string(chiplet_id));
    }
    return chiplets_[chiplet_id];
}

Chiplet& ChipletArch::get_chiplet_mut(int chiplet_id) {
    if (chiplet_id < 0 || chiplet_id >= num_chiplets_) {
        throw std::out_of_range("Invalid chiplet ID: " + std::to_string(chiplet_id));
    }
    return chiplets_[chiplet_id];
}

// ============================================================================
// Internal Methods
// ============================================================================

void ChipletArch::schedule_jobs() {
    // Simple FIFO scheduling
    while (!pending_job_queue_.empty()) {
        int job_id = pending_job_queue_.front();
        ChipletJob& job = jobs_[job_id];

        // Check if all participating chiplets are available
        bool all_available = true;
        for (int chiplet_id : job.participating_chiplets) {
            if (chiplets_[chiplet_id].is_active) {
                all_available = false;
                break;
            }
        }

        if (all_available) {
            // Schedule the job
            pending_job_queue_.pop();
            job.status = ChipletJob::Status::SCHEDULED;
            job.start_cycle = current_cycle_;
            active_jobs_.push_back(job_id);

            // Mark chiplets as active
            for (int chiplet_id : job.participating_chiplets) {
                chiplets_[chiplet_id].is_active = true;
            }
        } else {
            // Can't schedule yet, stop trying
            break;
        }
    }
}

void ChipletArch::execute_compute() {
    for (int job_id : active_jobs_) {
        ChipletJob& job = jobs_[job_id];

        if (job.status == ChipletJob::Status::SCHEDULED) {
            // Start compute phase
            job.status = ChipletJob::Status::COMPUTING;
        }

        if (job.status == ChipletJob::Status::COMPUTING) {
            // Check if compute phase is done
            uint64_t cycles_elapsed = current_cycle_ - job.start_cycle;
            if (cycles_elapsed >= job.compute_cycles) {
                // Move to communication phase
                if (!job.collective_ops.empty() || !job.pending_packets.empty()) {
                    job.status = ChipletJob::Status::COMMUNICATING;
                } else {
                    // No communication needed, complete
                    job.status = ChipletJob::Status::COMPLETED;
                    job.completion_cycle = current_cycle_;
                }
            }
        }
    }
}

void ChipletArch::execute_communication() {
    for (int job_id : active_jobs_) {
        ChipletJob& job = jobs_[job_id];

        if (job.status == ChipletJob::Status::COMMUNICATING) {
            // Try to send pending packets
            for (auto* packet : job.pending_packets) {
                if (packet->status == PacketStatus::CREATED) {
                    route_and_send_packet(packet);
                }
            }

            // Check if all packets completed
            bool all_done = true;
            for (const auto* packet : job.pending_packets) {
                if (packet->status != PacketStatus::COMPLETED) {
                    all_done = false;
                    break;
                }
            }

            if (all_done) {
                job.status = ChipletJob::Status::COMPLETED;
                job.completion_cycle = current_cycle_;
            }
        }
    }
}

void ChipletArch::check_completions() {
    // Remove completed jobs from active list
    active_jobs_.erase(
        std::remove_if(active_jobs_.begin(), active_jobs_.end(),
            [this](int job_id) {
                ChipletJob& job = jobs_[job_id];
                if (job.status == ChipletJob::Status::COMPLETED ||
                    job.status == ChipletJob::Status::FAILED) {

                    // Free chiplets
                    for (int chiplet_id : job.participating_chiplets) {
                        chiplets_[chiplet_id].is_active = false;
                    }

                    // Update stats
                    if (job.status == ChipletJob::Status::COMPLETED) {
                        stats_.jobs_completed++;
                    } else {
                        stats_.jobs_failed++;
                    }

                    return true;  // Remove from active list
                }
                return false;
            }),
        active_jobs_.end());
}

void ChipletArch::update_stats() {
    stats_.total_cycles = current_cycle_;

    // Calculate chiplet utilization
    int active_chiplets = 0;
    for (const auto& chiplet : chiplets_) {
        if (chiplet.is_active) {
            active_chiplets++;
        }
    }
    stats_.avg_chiplet_utilization = static_cast<double>(active_chiplets) / num_chiplets_;

    // Get link statistics from topology
    double total_link_utilization = 0.0;
    int num_links = 0;
    for (int i = 0; i < topology_->get_num_links(); i++) {
        const auto& link = topology_->get_link(i);
        const auto& link_stats = link.get_stats();
        total_link_utilization += link_stats.utilization;
        stats_.total_energy_mJ += link_stats.total_energy_mJ;
        num_links++;
    }
    if (num_links > 0) {
        stats_.avg_link_utilization = total_link_utilization / num_links;
    }

    // Calculate average power
    if (current_cycle_ > 0) {
        stats_.avg_power_W = (stats_.total_energy_mJ / current_cycle_) * 1e-3;  // mJ/cycle to W
    }
}

void ChipletArch::create_collective_packets(ChipletJob& job,
                                           const CollectiveOperation& collective) {
    /**
     * Create UCIe packets to implement a collective operation
     *
     * This is simplified - a full implementation would generate
     * the exact packet sequence for each collective algorithm
     */

    switch (collective.type) {
        case CollectiveOperation::Type::ALLREDUCE:
            // Ring AllReduce: N-1 reduce-scatter + N-1 allgather
            // Simplified: create packets between adjacent chiplets
            for (size_t i = 0; i < collective.participating_chiplets.size(); i++) {
                int src = collective.participating_chiplets[i];
                int dst = collective.participating_chiplets[(i + 1) %
                         collective.participating_chiplets.size()];

                auto* packet = new UCIePacket();
                packet->packet_id = job.pending_packets.size();
                packet->type = PacketType::WRITE_REQUEST;
                packet->src_chiplet = src;
                packet->dst_chiplet = dst;
                packet->size_bytes = collective.data_size_bytes /
                                    collective.participating_chiplets.size();
                packet->creation_cycle = current_cycle_;
                packet->job_id = job.job_id;
                packet->operation_type = "allreduce";

                job.pending_packets.push_back(packet);
            }
            break;

        case CollectiveOperation::Type::BROADCAST:
            // Broadcast from root to all others
            for (int dst : collective.participating_chiplets) {
                if (dst == collective.root_chiplet) continue;

                auto* packet = new UCIePacket();
                packet->packet_id = job.pending_packets.size();
                packet->type = PacketType::WRITE_REQUEST;
                packet->src_chiplet = collective.root_chiplet;
                packet->dst_chiplet = dst;
                packet->size_bytes = collective.data_size_bytes;
                packet->creation_cycle = current_cycle_;
                packet->job_id = job.job_id;
                packet->operation_type = "broadcast";

                job.pending_packets.push_back(packet);
            }
            break;

        case CollectiveOperation::Type::POINT_TO_POINT:
            // Simple point-to-point transfer
            {
                auto* packet = new UCIePacket();
                packet->packet_id = 0;
                packet->type = PacketType::WRITE_REQUEST;
                packet->src_chiplet = collective.src_chiplet;
                packet->dst_chiplet = collective.dst_chiplet;
                packet->size_bytes = collective.data_size_bytes;
                packet->creation_cycle = current_cycle_;
                packet->job_id = job.job_id;
                packet->operation_type = "p2p";

                job.pending_packets.push_back(packet);
            }
            break;

        default:
            std::cerr << "WARNING: Collective type not yet implemented\n";
            break;
    }
}

void ChipletArch::route_and_send_packet(UCIePacket* packet) {
    // Get route through topology
    auto route = topology_->get_route(packet->src_chiplet, packet->dst_chiplet);
    if (route.empty()) {
        std::cerr << "ERROR: No route from " << packet->src_chiplet
                  << " to " << packet->dst_chiplet << "\n";
        packet->status = PacketStatus::DROPPED;
        return;
    }

    // Get first link on route and try to enqueue
    if (route.size() >= 2) {
        int link_id = topology_->get_link_id(route[0], route[1]);
        if (link_id >= 0) {
            UCIeLink& link = topology_->get_link_mut(link_id);
            if (link.enqueue_packet(packet)) {
                stats_.total_packets_sent++;
                stats_.total_bytes_transferred += packet->size_bytes;

                // Update source chiplet stats
                chiplets_[packet->src_chiplet].packets_sent++;
                chiplets_[packet->src_chiplet].bytes_sent += packet->size_bytes;
            }
        }
    }
}

// ============================================================================
// Pre-configured Architectures
// ============================================================================

namespace architectures {

std::unique_ptr<ChipletArch> create_mi300a_style(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb) {
    /**
     * AMD MI300A: 4 compute chiplets in 2x2 mesh
     *
     * Reference: AMD MI300A Architecture Whitepaper
     */
    auto arch = std::make_unique<ChipletArch>(
        4,
        TopologyType::MESH_2D,
        ucie_configs::standard_16gt_x16());

    // Configure 2x2 mesh
    arch->get_topology()->set_mesh_dimensions(2, 2);
    arch->get_topology()->build_topology(ucie_configs::standard_16gt_x16());

    // Configure each chiplet
    uint64_t memory_bytes = memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;
    for (int i = 0; i < 4; i++) {
        arch->configure_chiplet(i, compute_capacity_per_chiplet_ops,
                               memory_bytes, 2048);  // 2 TB/s HBM3
    }

    return arch;
}

std::unique_ptr<ChipletArch> create_ponte_vecchio_style(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb) {
    /**
     * Intel Ponte Vecchio: 2 compute tiles
     */
    auto arch = std::make_unique<ChipletArch>(
        2,
        TopologyType::LINEAR,
        ucie_configs::high_bw_32gt_x32());

    uint64_t memory_bytes = memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;
    for (int i = 0; i < 2; i++) {
        arch->configure_chiplet(i, compute_capacity_per_chiplet_ops,
                               memory_bytes, 1600);  // HBM2e
    }

    return arch;
}

std::unique_ptr<ChipletArch> create_8chiplet_ring(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb) {
    auto arch = std::make_unique<ChipletArch>(
        8,
        TopologyType::RING,
        ucie_configs::balanced_24gt_x16());

    uint64_t memory_bytes = memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;
    for (int i = 0; i < 8; i++) {
        arch->configure_chiplet(i, compute_capacity_per_chiplet_ops,
                               memory_bytes, 2048);
    }

    return arch;
}

std::unique_ptr<ChipletArch> create_16chiplet_mesh(
    uint64_t compute_capacity_per_chiplet_ops,
    uint64_t memory_per_chiplet_gb) {
    auto arch = std::make_unique<ChipletArch>(
        16,
        TopologyType::MESH_2D,
        ucie_configs::standard_16gt_x16());

    arch->get_topology()->set_mesh_dimensions(4, 4);
    arch->get_topology()->build_topology(ucie_configs::standard_16gt_x16());

    uint64_t memory_bytes = memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;
    for (int i = 0; i < 16; i++) {
        arch->configure_chiplet(i, compute_capacity_per_chiplet_ops,
                               memory_bytes, 2048);
    }

    return arch;
}

} // namespace architectures

} // namespace chiplets
