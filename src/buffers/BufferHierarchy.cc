/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "buffers/BufferHierarchy.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace buffers {

// ============================================================================
// BufferLevel Implementation
// ============================================================================

BufferLevel::BufferLevel(const BufferLevelConfig& config)
    : config_(config)
    , current_occupancy_bytes_(0)
    , current_cycle_(0)
{
    // Initialize bank availability tracking
    bank_available_at_.resize(config_.num_banks, 0);
}

void BufferLevel::reset_stats() {
    stats_.reset();
}

bool BufferLevel::can_allocate(uint64_t size_bytes) const {
    return (current_occupancy_bytes_ + size_bytes) <= config_.size_bytes;
}

bool BufferLevel::allocate(uint64_t addr, uint64_t size_bytes) {
    if (!can_allocate(size_bytes)) {
        return false;
    }

    // Check if already allocated
    if (allocated_blocks_.find(addr) != allocated_blocks_.end()) {
        std::cerr << "Warning: Address " << std::hex << addr
                  << " already allocated in " << config_.name << std::endl;
        return false;
    }

    allocated_blocks_[addr] = size_bytes;
    current_occupancy_bytes_ += size_bytes;
    return true;
}

void BufferLevel::deallocate(uint64_t addr) {
    auto it = allocated_blocks_.find(addr);
    if (it != allocated_blocks_.end()) {
        current_occupancy_bytes_ -= it->second;
        allocated_blocks_.erase(it);
    }
}

uint64_t BufferLevel::get_free_bytes() const {
    return config_.size_bytes - current_occupancy_bytes_;
}

double BufferLevel::get_utilization() const {
    return static_cast<double>(current_occupancy_bytes_) / static_cast<double>(config_.size_bytes);
}

bool BufferLevel::can_read(uint64_t addr, uint64_t size_bytes) const {
    // Check if data is allocated
    auto it = allocated_blocks_.find(addr);
    if (it == allocated_blocks_.end()) {
        return false;  // Not allocated
    }

    // For now, assume reads can always proceed if allocated
    // In future, can add bandwidth checks
    return true;
}

bool BufferLevel::can_write(uint64_t addr, uint64_t size_bytes) const {
    // Check capacity
    if (!can_allocate(size_bytes)) {
        return false;
    }

    // Check if read-only buffer
    if (config_.read_only) {
        return false;
    }

    return true;
}

void BufferLevel::record_read(uint64_t addr, uint64_t size_bytes, const std::string& op_type) {
    stats_.record_read(size_bytes, op_type);

    // Check for bank conflicts
    if (config_.num_banks > 1) {
        int bank_idx = get_bank_index(addr);
        if (!is_bank_available(bank_idx, current_cycle_)) {
            stats_.bank_conflicts++;
        }
        mark_bank_busy(bank_idx, current_cycle_, config_.read_latency_cycles);
    }
}

void BufferLevel::record_write(uint64_t addr, uint64_t size_bytes, const std::string& op_type) {
    stats_.record_write(size_bytes, op_type);

    // Auto-allocate if not already allocated
    if (allocated_blocks_.find(addr) == allocated_blocks_.end()) {
        allocate(addr, size_bytes);
    }

    // Check for bank conflicts
    if (config_.num_banks > 1) {
        int bank_idx = get_bank_index(addr);
        if (!is_bank_available(bank_idx, current_cycle_)) {
            stats_.bank_conflicts++;
        }
        mark_bank_busy(bank_idx, current_cycle_, config_.write_latency_cycles);
    }
}

int BufferLevel::get_bank_index(uint64_t addr) const {
    if (config_.num_banks <= 1) return 0;

    // Simple bank interleaving: use bank_size as granularity
    uint64_t bank_offset = addr / config_.bank_size_bytes;
    return static_cast<int>(bank_offset % config_.num_banks);
}

bool BufferLevel::is_bank_available(int bank_idx, uint64_t cycle) const {
    if (bank_idx < 0 || bank_idx >= static_cast<int>(bank_available_at_.size())) {
        return true;
    }
    return cycle >= bank_available_at_[bank_idx];
}

void BufferLevel::mark_bank_busy(int bank_idx, uint64_t cycle, int duration) {
    if (bank_idx >= 0 && bank_idx < static_cast<int>(bank_available_at_.size())) {
        bank_available_at_[bank_idx] = cycle + duration;
    }
}

void BufferLevel::tick(uint64_t cycle) {
    current_cycle_ = cycle;
    stats_.record_occupancy(current_occupancy_bytes_);
}

// ============================================================================
// BufferHierarchy Implementation
// ============================================================================

BufferHierarchy::BufferHierarchy(const BufferHierarchyConfig& config)
    : config_(config)
{
    if (!config_.validate()) {
        throw std::runtime_error("Invalid BufferHierarchyConfig");
    }

    initialize_levels();

    if (config_.partitioned_per_core) {
        initialize_partitions();
    }
}

void BufferHierarchy::initialize_levels() {
    for (const auto& level_config : config_.levels) {
        levels_.push_back(std::make_unique<BufferLevel>(level_config));
    }
}

void BufferHierarchy::initialize_partitions() {
    for (int core_id = 0; core_id < config_.num_partitions; ++core_id) {
        std::vector<std::unique_ptr<BufferLevel>> core_levels;

        for (const auto& level_config : config_.levels) {
            // Create partitioned version with reduced capacity
            BufferLevelConfig partitioned_config = level_config;
            partitioned_config.size_bytes /= config_.num_partitions;
            partitioned_config.bank_size_bytes /= config_.num_partitions;
            partitioned_config.name = level_config.name + "_Core" + std::to_string(core_id);

            core_levels.push_back(std::make_unique<BufferLevel>(partitioned_config));
        }

        core_partitions_[core_id] = std::move(core_levels);
    }
}

BufferLevel* BufferHierarchy::get_level(int level_idx) {
    if (level_idx < 0 || level_idx >= static_cast<int>(levels_.size())) {
        return nullptr;
    }
    return levels_[level_idx].get();
}

const BufferLevel* BufferHierarchy::get_level(int level_idx) const {
    if (level_idx < 0 || level_idx >= static_cast<int>(levels_.size())) {
        return nullptr;
    }
    return levels_[level_idx].get();
}

BufferLevel* BufferHierarchy::get_level_by_name(const std::string& name) {
    for (auto& level : levels_) {
        if (level->get_config().name == name) {
            return level.get();
        }
    }
    return nullptr;
}

uint64_t BufferHierarchy::get_unified_buffer_size() const {
    // For backward compatibility: return size of first level
    if (levels_.empty()) return 0;
    return levels_[0]->get_config().size_bytes;
}

bool BufferHierarchy::fits_in_level(int level_idx, uint64_t size_bytes) const {
    const BufferLevel* level = get_level(level_idx);
    if (!level) return false;
    return level->can_allocate(size_bytes);
}

uint64_t BufferHierarchy::get_available_capacity(int level_idx) const {
    const BufferLevel* level = get_level(level_idx);
    if (!level) return 0;
    return level->get_free_bytes();
}

BufferLevel* BufferHierarchy::get_partition(int core_id, int level_idx) {
    if (!config_.partitioned_per_core) {
        return get_level(level_idx);
    }

    auto it = core_partitions_.find(core_id);
    if (it == core_partitions_.end()) {
        return nullptr;
    }

    if (level_idx < 0 || level_idx >= static_cast<int>(it->second.size())) {
        return nullptr;
    }

    return it->second[level_idx].get();
}

BufferAccessStats BufferHierarchy::get_total_stats() const {
    BufferAccessStats total;

    for (const auto& level : levels_) {
        total.merge(level->get_stats());
    }

    // Also merge partition stats if applicable
    if (config_.partitioned_per_core) {
        for (const auto& [core_id, core_levels] : core_partitions_) {
            for (const auto& level : core_levels) {
                total.merge(level->get_stats());
            }
        }
    }

    return total;
}

BufferAccessStats BufferHierarchy::get_level_stats(int level_idx) const {
    const BufferLevel* level = get_level(level_idx);
    if (!level) {
        return BufferAccessStats();
    }
    return level->get_stats();
}

void BufferHierarchy::reset_all_stats() {
    for (auto& level : levels_) {
        level->reset_stats();
    }

    if (config_.partitioned_per_core) {
        for (auto& [core_id, core_levels] : core_partitions_) {
            for (auto& level : core_levels) {
                level->reset_stats();
            }
        }
    }
}

void BufferHierarchy::tick(uint64_t cycle) {
    for (auto& level : levels_) {
        level->tick(cycle);
    }

    if (config_.partitioned_per_core) {
        for (auto& [core_id, core_levels] : core_partitions_) {
            for (auto& level : core_levels) {
                level->tick(cycle);
            }
        }
    }
}

void BufferHierarchy::print_hierarchy(std::ostream& os) const {
    os << "\n=== Buffer Hierarchy Configuration ===\n";
    os << "Style: " << (config_.use_unified_buffer ? "Unified" : "Separate") << "\n";
    os << "Partitioned: " << (config_.partitioned_per_core ? "Yes" : "No");
    if (config_.partitioned_per_core) {
        os << " (" << config_.num_partitions << " partitions)";
    }
    os << "\n\n";

    for (size_t i = 0; i < levels_.size(); ++i) {
        const auto& level = levels_[i];
        const auto& cfg = level->get_config();

        os << "Level " << i << ": " << cfg.name << "\n";
        os << "  Capacity: " << (cfg.size_bytes / 1024.0 / 1024.0) << " MB\n";
        os << "  Banks: " << cfg.num_banks << "\n";
        os << "  Bandwidth: " << cfg.bandwidth_bytes_per_cycle << " bytes/cycle\n";
        os << "  Latency: R=" << cfg.read_latency_cycles
           << " W=" << cfg.write_latency_cycles << " cycles\n";

        if (cfg.use_dramsim3) {
            os << "  DRAMSim3: " << cfg.dramsim3_config_path << "\n";
        }

        os << "\n";
    }
}

void BufferHierarchy::print_utilization(std::ostream& os) const {
    os << "\n=== Buffer Utilization ===\n";

    for (size_t i = 0; i < levels_.size(); ++i) {
        const auto& level = levels_[i];
        os << level->get_config().name << ": "
           << std::fixed << std::setprecision(1)
           << (level->get_utilization() * 100.0) << "% ("
           << (level->get_used_bytes() / 1024.0 / 1024.0) << " / "
           << (level->get_config().size_bytes / 1024.0 / 1024.0) << " MB)\n";
    }
}

} // namespace buffers
