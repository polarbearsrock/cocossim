/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_BUFFER_HIERARCHY_H
#define COCOSSIM_BUFFER_HIERARCHY_H

#include "BufferConfig.h"
#include "BufferStats.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace buffers {

/**
 * Represents a single buffer level with access tracking
 */
class BufferLevel {
public:
    explicit BufferLevel(const BufferLevelConfig& config);

    // Configuration
    const BufferLevelConfig& get_config() const { return config_; }

    // Statistics
    BufferAccessStats& get_stats() { return stats_; }
    const BufferAccessStats& get_stats() const { return stats_; }
    void reset_stats();

    // Capacity management
    bool can_allocate(uint64_t size_bytes) const;
    bool allocate(uint64_t addr, uint64_t size_bytes);
    void deallocate(uint64_t addr);
    uint64_t get_free_bytes() const;
    uint64_t get_used_bytes() const { return current_occupancy_bytes_; }
    double get_utilization() const;

    // Access methods with stats tracking
    bool can_read(uint64_t addr, uint64_t size_bytes) const;
    bool can_write(uint64_t addr, uint64_t size_bytes) const;
    void record_read(uint64_t addr, uint64_t size_bytes, const std::string& op_type = "");
    void record_write(uint64_t addr, uint64_t size_bytes, const std::string& op_type = "");

    // Banking / conflict tracking
    int get_bank_index(uint64_t addr) const;
    bool is_bank_available(int bank_idx, uint64_t cycle) const;
    void mark_bank_busy(int bank_idx, uint64_t cycle, int duration);

    // Per-cycle update (for occupancy tracking)
    void tick(uint64_t cycle);

private:
    BufferLevelConfig config_;
    BufferAccessStats stats_;

    // Address space management
    std::unordered_map<uint64_t, uint64_t> allocated_blocks_;  // addr -> size
    uint64_t current_occupancy_bytes_;

    // Bank availability tracking (bank_idx -> cycle when available)
    std::vector<uint64_t> bank_available_at_;

    uint64_t current_cycle_;
};

/**
 * Manages a hierarchy of buffer levels
 */
class BufferHierarchy {
public:
    explicit BufferHierarchy(const BufferHierarchyConfig& config);

    // Configuration
    const BufferHierarchyConfig& get_config() const { return config_; }

    // Access specific levels
    BufferLevel* get_level(int level_idx);
    const BufferLevel* get_level(int level_idx) const;
    BufferLevel* get_level_by_name(const std::string& name);
    int get_num_levels() const { return static_cast<int>(levels_.size()); }

    // Capacity queries (backward compatibility helpers)
    uint64_t get_unified_buffer_size() const;
    bool fits_in_level(int level_idx, uint64_t size_bytes) const;
    uint64_t get_available_capacity(int level_idx) const;

    // Multi-core partitioning support
    BufferLevel* get_partition(int core_id, int level_idx);

    // Statistics aggregation
    BufferAccessStats get_total_stats() const;
    BufferAccessStats get_level_stats(int level_idx) const;
    void reset_all_stats();

    // Per-cycle update
    void tick(uint64_t cycle);

    // Debug / reporting
    void print_hierarchy(std::ostream& os) const;
    void print_utilization(std::ostream& os) const;

private:
    BufferHierarchyConfig config_;
    std::vector<std::unique_ptr<BufferLevel>> levels_;

    // For multi-core partitioning: core_id -> level_idx -> BufferLevel
    std::map<int, std::vector<std::unique_ptr<BufferLevel>>> core_partitions_;

    void initialize_levels();
    void initialize_partitions();
};

} // namespace buffers

#endif // COCOSSIM_BUFFER_HIERARCHY_H
