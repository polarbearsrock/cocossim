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
#include "memory_system.h"  // dramsim3::MemorySystem (full def needed here only)
#include "State.h"
#include "global.h"
#include "frontends/standard/StandardArch.h"
#include "frontends/standard/StandardUnits.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace chiplets {

// ============================================================================
// ArchStats Implementation
// ============================================================================

void ChipletArch::ArchStats::reset() {
    total_cycles = 0;
    total_compute_cycles = 0;
    total_communication_cycles = 0;
    total_idle_cycles = 0;
    total_dram_stall_cycles = 0;
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
    os << "    DRAM Stall Cycles: " << total_dram_stall_cycles
       << " (" << (total_dram_stall_cycles * 100.0 / std::max(total_cycles, 1ULL)) << "%)\n";

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

    // SA/VPU states are created in set_sa_sz() after sa_sz is known.
    // Ensure the global DRAM transaction size is set to a sensible default;
    // init_dram() will update it from the actual DRAMSim3 config.
    if (bytes_per_tx == 0) bytes_per_tx = 64;

    // Chiplet mode: outputs stay in the on-chip register file / L1;
    // DRAMSim3 models actual weight/activation traffic separately.
    skip_dram_writeback = true;

    stats_.reset();
}

// ============================================================================
// Compute-unit initialisation
// ============================================================================

void ChipletArch::set_sa_sz(int sa_sz) {
    sa_sz_ = sa_sz;

    // (Re-)create representative compute-unit states.
    delete sa_state_;
    delete vpu_state_;
    sa_state_ = new SystolicArray::SysArrayState(sa_sz, /*ws=*/false);
    vpu_state_ = new VectorUnit::VecUnitState(sa_sz);

    // Configure the global arch_config that SysArray.cc reads.
    // n_cores=1 (one SA per chiplet), OS dataflow, no shared-weight skipping.
    frontend::standard::arch_config =
        frontend::standard::ArchConfig(1, sa_sz, sa_sz, /*ws=*/false, /*shared=*/false);
}

// ============================================================================
// DRAMSim3 Integration
// ============================================================================

void ChipletArch::init_dram(const std::string& dram_config_path, double sa_freq_ghz) {
    sa_freq_ghz_ = sa_freq_ghz;
    dram_states_.resize(num_chiplets_);

    for (int i = 0; i < num_chiplets_; i++) {
        // Callback: decrement outstanding_reads for this chiplet.
        // Each chiplet has at most one active compute job (enforced by schedule_jobs),
        // so we can unconditionally decrement the chiplet's counter.
        auto read_cb = [this, i](uint64_t /*addr*/) {
            if (dram_states_[i].outstanding_reads > 0)
                dram_states_[i].outstanding_reads--;
        };
        auto write_cb = [](uint64_t /*addr*/) {};  // writes fire-and-forget

        dram_states_[i].mem_sys = new dramsim3::MemorySystem(
            dram_config_path, "./", read_cb, write_cb);

        dram_states_[i].tck_ns = dram_states_[i].mem_sys->GetTCK();

        // bytes_per_tx = bus width in bytes * burst length
        uint64_t bus_bytes = static_cast<uint64_t>(
            dram_states_[i].mem_sys->GetBusBits()) / 8;
        uint64_t burst = static_cast<uint64_t>(
            dram_states_[i].mem_sys->GetBurstLength());
        dram_states_[i].bytes_per_tx = bus_bytes * burst;
        if (dram_states_[i].bytes_per_tx == 0)
            dram_states_[i].bytes_per_tx = 64;  // safe fallback
    }

    dram_enabled_ = true;
    std::cout << "ChipletArch: DRAMSim3 initialized for " << num_chiplets_
              << " chiplets (tCK=" << dram_states_[0].tck_ns
              << "ns, bytes_per_tx=" << dram_states_[0].bytes_per_tx << ")\n";
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
    job.data_size_bytes = data_size_bytes;
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

int ChipletArch::submit_job(const std::string& operation_type,
                           int layer_id,
                           uint64_t data_size_bytes,
                           Job* compute_job) {
    // Build the ChipletJob the same way as the countdown overload, but store
    // the SA/VPU job pointer so execute_compute() can tick the FSM.
    ChipletJob job;
    job.job_id = next_job_id_++;
    job.operation_type = operation_type;
    job.creation_cycle = current_cycle_;
    job.compute_cycles = 0;           // unused when compute_job_ptr is set
    job.compute_job_ptr = compute_job;
    job.data_size_bytes = data_size_bytes;
    job.status = ChipletJob::Status::CREATED;

    if (partition_plan_ && layer_id >= 0) {
        const auto& strategy = partition_plan_->get_layer_strategy(layer_id);
        job.parallelism = strategy.parallelism;
        job.participating_chiplets = partition_plan_->get_chiplets_for_layer(layer_id);
    } else {
        job.parallelism = ParallelismType::DATA_PARALLEL;
        for (int i = 0; i < num_chiplets_; i++)
            job.participating_chiplets.push_back(i);
    }

    jobs_.push_back(job);
    pending_job_queue_.push(job.job_id);
    stats_.jobs_submitted++;
    return job.job_id;
}

int ChipletArch::submit_job(const std::string& operation_type,
                           int layer_id,
                           uint64_t data_size_bytes,
                           Job* compute_job,
                           const std::vector<int>& chiplet_subset) {
    ChipletJob job;
    job.job_id = next_job_id_++;
    job.operation_type = operation_type;
    job.creation_cycle = current_cycle_;
    job.compute_cycles = 0;
    job.compute_job_ptr = compute_job;
    job.data_size_bytes = data_size_bytes;
    job.status = ChipletJob::Status::CREATED;
    job.parallelism = ParallelismType::DATA_PARALLEL;
    job.participating_chiplets = chiplet_subset;

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

    // 2. Tick each chiplet's DRAMSim3 instance.
    //    Uses the same differential-accumulator approach as Arch.cc so that
    //    DRAM and SA clock domains stay correctly phased even if freq_sa != 1.0.
    if (dram_enabled_) {
        for (auto& ds : dram_states_) {
            // differential = tCK_ns * sa_freq_ghz  (DRAM ticks per SA cycle)
            ds.clock_accum += ds.tck_ns * sa_freq_ghz_;
            while (ds.clock_accum >= 1.0) {
                ds.mem_sys->ClockTick();
                ds.clock_accum -= 1.0;
            }
        }
    }

    // 3. Execute compute phase (issues DRAM reads, stalls on completion)
    execute_compute();

    // 4. Execute communication phase
    execute_communication();

    // 5. Tick the topology (UCIe links)
    topology_->tick(current_cycle);

    // 6. Check for completions
    check_completions();

    // 7. Update statistics
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
           << (chiplet.memory_capacity_bytes / 1024.0 / 1024.0) << " MB"
           << " | DRAM stall cycles: " << chiplet.dram_stall_cycles << "\n";
    }

    // Print topology status
    topology_->print_status(os, current_cycle_);
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

void ChipletArch::drain_compute_dram_queue(State* state) {
    // Iterate the global to_enqueue list and instantly complete every read/write
    // issued by this state machine so mem_read_left / mem_write_left reach zero
    // without real DRAM latency. DRAMSim3 provides the actual timing independently.
    auto it = to_enqueue.begin();
    while (it != to_enqueue.end()) {
        State* target = std::get<3>(*it);
        if (target == state) {
            bool is_write = std::get<1>(*it);
            if (is_write) {
                if (state->mem_write_left > 0) state->mem_write_left--;
            } else {
                if (state->mem_read_left  > 0) state->mem_read_left--;
            }
            if (state->mem_queued > 0) state->mem_queued--;
            it = to_enqueue.erase(it);
        } else {
            ++it;
        }
    }
}

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
            job.status = ChipletJob::Status::COMPUTING;
            job.compute_start_cycle = current_cycle_;
            job.dram_issued = false;
        }

        if (job.status == ChipletJob::Status::COMPUTING) {

            // On the first cycle of COMPUTING, compute how many DRAM reads are
            // needed per chiplet. Issuance is spread across cycles (streaming)
            // to avoid overwhelming the DRAM queue in a single tick.
            if (!job.dram_issued && dram_enabled_) {
                job.dram_issued = true;

                uint64_t n_chiplets = std::max<uint64_t>(
                    1, job.participating_chiplets.size());
                uint64_t bytes_per_chiplet = job.data_size_bytes / n_chiplets;

                for (int cid : job.participating_chiplets) {
                    auto& ds = dram_states_[cid];
                    ds.dram_reqs_remaining = (bytes_per_chiplet + ds.bytes_per_tx - 1)
                                             / ds.bytes_per_tx;
                }
            }

            // Each COMPUTING cycle, stream up to MAX_DRAM_ISSUE_PER_CYCLE reads
            // per chiplet into DRAMSim3 (bandwidth-limited streaming model).
            static constexpr uint64_t MAX_DRAM_ISSUE_PER_CYCLE = 16;
            if (dram_enabled_) {
                for (int cid : job.participating_chiplets) {
                    auto& ds = dram_states_[cid];
                    uint64_t issued = 0;
                    while (ds.dram_reqs_remaining > 0 && issued < MAX_DRAM_ISSUE_PER_CYCLE) {
                        uint64_t addr = (static_cast<uint64_t>(cid) * CHIPLET_ADDR_STRIDE)
                                        + ds.next_addr_offset;
                        if (!ds.mem_sys->WillAcceptTransaction(addr, false))
                            break;  // queue full — retry next cycle
                        if (ds.mem_sys->AddTransaction(addr, false)) {
                            ds.outstanding_reads++;
                            ds.next_addr_offset += ds.bytes_per_tx;
                            ds.dram_reqs_remaining--;
                            issued++;
                        } else {
                            break;
                        }
                    }
                }
            }

            // ── Compute completion check ──────────────────────────────────
            // If a cycle-accurate SA/VPU job was provided, tick its FSM and
            // drain the to_enqueue list so the FSM sees "instant DRAM" (the
            // actual DRAM timing is captured by DRAMSim3 above).
            // Fall back to the countdown timer for backward compatibility.
            bool compute_done;
            if (job.compute_job_ptr != nullptr) {
                Job* cj = job.compute_job_ptr;
                if (!cj->is_done) {
                    // Assign to the representative state machine on first tick.
                    State* active_state = nullptr;
                    if (sa_state_ != nullptr && cj->get_type() == SYSTOLIC_ARRAY_IDX) {
                        active_state = sa_state_;
                    } else if (vpu_state_ != nullptr && cj->get_type() == VECTOR_UNIT_IDX) {
                        active_state = vpu_state_;
                    }
                    if (active_state != nullptr) {
                        if (active_state->j == nullptr) {
                            active_state->j = cj;
                            active_state->init();
                        }
                        int dummy_idle = 0;
                        int dummy_n[4] = {0};
                        active_state->increment(
                            [](Job*) {},   // no-op enqueue callback
                            dummy_idle, dummy_n);
                        // Drain the to_enqueue entries for this state: the FSM
                        // sees instant DRAM so arithmetic cycles are the bottleneck.
                        drain_compute_dram_queue(active_state);
                    }
                }
                compute_done = cj->is_done;
            } else {
                uint64_t cycles_elapsed = current_cycle_ - job.compute_start_cycle;
                compute_done = (cycles_elapsed >= job.compute_cycles);
            }

            bool dram_done = true;
            if (dram_enabled_) {
                for (int cid : job.participating_chiplets) {
                    if (dram_states_[cid].dram_reqs_remaining > 0 ||
                        dram_states_[cid].outstanding_reads > 0) {
                        dram_done = false;
                        break;
                    }
                }
            }

            // Count cycles where compute is finished but DRAM is still outstanding
            if (compute_done && !dram_done) {
                for (int cid : job.participating_chiplets)
                    chiplets_[cid].dram_stall_cycles++;
            }

            if (compute_done && dram_done) {
                if (!job.collective_ops.empty() || !job.pending_packets.empty()) {
                    job.status = ChipletJob::Status::COMMUNICATING;
                } else {
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
            // Forward packets that finished a hop but haven't reached their final
            // destination yet (multi-hop routing: current_chiplet != dst_chiplet).
            for (auto* packet : job.pending_packets) {
                if (packet->status == PacketStatus::COMPLETED &&
                    packet->current_chiplet != packet->dst_chiplet) {
                    // Advance the logical source to the intermediate chiplet so
                    // route_and_send_packet() computes the remaining sub-route.
                    packet->src_chiplet = packet->current_chiplet;
                    packet->status = PacketStatus::CREATED;
                }
            }

            // Try to send CREATED packets (includes freshly forwarded ones above)
            for (auto* packet : job.pending_packets) {
                if (packet->status == PacketStatus::CREATED) {
                    route_and_send_packet(packet);
                }
            }

            // Check if all packets have reached their final destination.
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

    // Track per-cycle compute/communication cycles
    for (int job_id : active_jobs_) {
        const ChipletJob& job = jobs_[job_id];
        if (job.status == ChipletJob::Status::COMPUTING)
            stats_.total_compute_cycles++;
        else if (job.status == ChipletJob::Status::COMMUNICATING)
            stats_.total_communication_cycles++;
    }

    // Snapshot DRAM stall cycles (max across chiplets — they stall in lockstep for TP)
    uint64_t max_dram_stall = 0;
    for (const auto& chiplet : chiplets_)
        max_dram_stall = std::max(max_dram_stall, chiplet.dram_stall_cycles);
    stats_.total_dram_stall_cycles = max_dram_stall;

    // Calculate chiplet utilization
    int active_chiplets = 0;
    for (const auto& chiplet : chiplets_) {
        if (chiplet.is_active) active_chiplets++;
    }
    stats_.avg_chiplet_utilization = static_cast<double>(active_chiplets) / num_chiplets_;

    // Snapshot link statistics (not accumulated — links track their own running totals)
    double total_link_utilization = 0.0;
    double total_energy_mJ = 0.0;
    int num_links = topology_->get_num_links();
    for (int i = 0; i < num_links; i++) {
        UCIeLink& link = topology_->get_link_mut(i);
        auto ls = link.get_stats();
        ls.calculate_derived_stats(current_cycle_);
        total_link_utilization += ls.utilization;
        total_energy_mJ += ls.total_energy_mJ;
    }
    stats_.total_energy_mJ = total_energy_mJ;
    if (num_links > 0) {
        stats_.avg_link_utilization = total_link_utilization / num_links;
    }

    // Average power: total_energy_mJ / time_ms (time_ms = cycles / freq_MHz / 1000)
    // Here we don't know freq, so store energy only; power printed in main.
    stats_.avg_power_W = 0.0;
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
                packet->current_chiplet = packet->src_chiplet;

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
                packet->current_chiplet = packet->src_chiplet;

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
                packet->current_chiplet = packet->src_chiplet;

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
    // An empty route means the topology is disconnected — this is a bug, not a
    // valid simulation outcome. Abort with a clear message rather than silently
    // producing wrong results or spinning forever.
    if (route.empty()) {
        std::cerr << "FATAL: no route from chiplet " << packet->src_chiplet
                  << " to chiplet " << packet->dst_chiplet
                  << " — topology is disconnected (bug in topology construction)\n";
        std::abort();
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
