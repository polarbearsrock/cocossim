/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_BUFFER_CONFIG_H
#define COCOSSIM_BUFFER_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

namespace buffers {

/**
 * Configuration for a single level in the buffer hierarchy
 */
struct BufferLevelConfig {
    std::string name;                  // Descriptive name (e.g., "UnifiedBuffer", "WeightSRAM")
    int level;                         // Hierarchy level (0=closest to compute, higher=further)

    // Capacity
    uint64_t size_bytes;               // Total capacity in bytes

    // Performance parameters
    int bandwidth_bytes_per_cycle;     // Maximum bytes transferred per cycle
    int read_latency_cycles;           // Cycles to complete a read
    int write_latency_cycles;          // Cycles to complete a write

    // Banking for parallel access
    int num_banks;                     // Number of independent banks
    uint64_t bank_size_bytes;          // Size of each bank

    // Special flags
    bool read_only;                    // True for read-only buffers (e.g., weight buffers)
    bool write_through;                // Write-through to next level
    bool use_dramsim3;                 // Use DRAMSim3 for this level (typically DRAM)
    std::string dramsim3_config_path;  // Path to DRAMSim3 config if used

    BufferLevelConfig()
        : name("Unnamed")
        , level(0)
        , size_bytes(0)
        , bandwidth_bytes_per_cycle(0)
        , read_latency_cycles(0)
        , write_latency_cycles(0)
        , num_banks(1)
        , bank_size_bytes(0)
        , read_only(false)
        , write_through(false)
        , use_dramsim3(false) {}
};

/**
 * Configuration for the entire buffer hierarchy
 */
struct BufferHierarchyConfig {
    // Buffer levels ordered from closest to compute (level 0) to furthest (DRAM)
    std::vector<BufferLevelConfig> levels;

    // Architecture style
    bool use_unified_buffer;           // True: single unified buffer, False: separate weight/activation

    // Multi-core partitioning
    bool partitioned_per_core;         // True: each core has its own buffer partition
    int num_partitions;                // Number of partitions (typically = num_cores)

    // Default level for allocations
    int default_level;                 // Which level to use by default (typically 0 or 1)

    BufferHierarchyConfig()
        : use_unified_buffer(true)
        , partitioned_per_core(false)
        , num_partitions(1)
        , default_level(0) {}

    // Helper: Get total capacity at a specific level
    uint64_t get_level_capacity(int level) const;

    // Helper: Find level by name
    const BufferLevelConfig* find_level(const std::string& name) const;

    // Validation
    bool validate() const;
};

/**
 * Create default configurations for common accelerator architectures
 */
namespace configs {
    // TPU-style unified buffer (64MB)
    BufferHierarchyConfig create_tpu_v3_style();

    // Eyeriss-style hierarchical scratchpads
    BufferHierarchyConfig create_eyeriss_style();

    // Simple two-level: SRAM + DRAM
    BufferHierarchyConfig create_simple_two_level(
        uint64_t sram_size_bytes,
        const std::string& dram_config_path
    );
}

} // namespace buffers

#endif // COCOSSIM_BUFFER_CONFIG_H
