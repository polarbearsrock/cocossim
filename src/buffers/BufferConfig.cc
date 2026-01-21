/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "buffers/BufferConfig.h"
#include <iostream>
#include <algorithm>

namespace buffers {

uint64_t BufferHierarchyConfig::get_level_capacity(int level) const {
    for (const auto& lvl : levels) {
        if (lvl.level == level) {
            return lvl.size_bytes;
        }
    }
    return 0;
}

const BufferLevelConfig* BufferHierarchyConfig::find_level(const std::string& name) const {
    for (const auto& lvl : levels) {
        if (lvl.name == name) {
            return &lvl;
        }
    }
    return nullptr;
}

bool BufferHierarchyConfig::validate() const {
    if (levels.empty()) {
        std::cerr << "BufferHierarchyConfig: No levels defined" << std::endl;
        return false;
    }

    // Check for duplicate level numbers
    std::vector<int> level_nums;
    for (const auto& lvl : levels) {
        if (std::find(level_nums.begin(), level_nums.end(), lvl.level) != level_nums.end()) {
            std::cerr << "BufferHierarchyConfig: Duplicate level number " << lvl.level << std::endl;
            return false;
        }
        level_nums.push_back(lvl.level);
    }

    // Validate each level
    for (const auto& lvl : levels) {
        if (lvl.size_bytes == 0) {
            std::cerr << "BufferHierarchyConfig: Level " << lvl.name << " has zero capacity" << std::endl;
            return false;
        }
        if (lvl.num_banks > 0 && lvl.bank_size_bytes > 0) {
            if (lvl.num_banks * lvl.bank_size_bytes != lvl.size_bytes) {
                std::cerr << "BufferHierarchyConfig: Level " << lvl.name
                          << " bank configuration inconsistent" << std::endl;
                return false;
            }
        }
    }

    return true;
}

namespace configs {

BufferHierarchyConfig create_tpu_style() {
    BufferHierarchyConfig config;
    config.use_unified_buffer = true;
    config.partitioned_per_core = false;
    config.num_partitions = 1;
    config.default_level = 0;

    // Unified Buffer (64MB on-chip SRAM)
    BufferLevelConfig unified;
    unified.name = "UnifiedBuffer";
    unified.level = 0;
    unified.size_bytes = 64ULL * 1024 * 1024;  // 64 MB
    unified.bandwidth_bytes_per_cycle = 256;
    unified.read_latency_cycles = 2;
    unified.write_latency_cycles = 2;
    unified.num_banks = 32;
    unified.bank_size_bytes = unified.size_bytes / unified.num_banks;
    unified.read_only = false;
    unified.write_through = false;
    unified.use_dramsim3 = false;

    config.levels.push_back(unified);

    // DRAM (HBM2)
    BufferLevelConfig dram;
    dram.name = "DRAM";
    dram.level = 1;
    dram.size_bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    dram.bandwidth_bytes_per_cycle = 128;
    dram.read_latency_cycles = 100;
    dram.write_latency_cycles = 100;
    dram.num_banks = 1;
    dram.bank_size_bytes = dram.size_bytes;
    dram.use_dramsim3 = true;
    dram.dramsim3_config_path = "../dramsim3/configs/HBM2_8Gb_x128.ini";

    config.levels.push_back(dram);

    return config;
}

BufferHierarchyConfig create_simple_two_level(
    uint64_t sram_size_bytes,
    const std::string& dram_config_path
) {
    BufferHierarchyConfig config;
    config.use_unified_buffer = true;
    config.partitioned_per_core = false;
    config.num_partitions = 1;
    config.default_level = 0;

    // On-chip SRAM
    BufferLevelConfig sram;
    sram.name = "OnChipSRAM";
    sram.level = 0;
    sram.size_bytes = sram_size_bytes;
    sram.bandwidth_bytes_per_cycle = 128;
    sram.read_latency_cycles = 2;
    sram.write_latency_cycles = 2;
    sram.num_banks = 16;
    sram.bank_size_bytes = sram_size_bytes / sram.num_banks;

    config.levels.push_back(sram);

    // Off-chip DRAM
    BufferLevelConfig dram;
    dram.name = "DRAM";
    dram.level = 1;
    dram.size_bytes = 8ULL * 1024 * 1024 * 1024;
    dram.use_dramsim3 = true;
    dram.dramsim3_config_path = dram_config_path;

    config.levels.push_back(dram);

    return config;
}

} // namespace configs

} // namespace buffers
