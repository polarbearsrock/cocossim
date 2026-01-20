/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "buffers/PowerModel.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace buffers {

EnergyBreakdown PowerModel::calculate_energy(
    const BufferHierarchy& hierarchy,
    uint64_t total_cycles,
    double frequency_GHz
) {
    EnergyBreakdown breakdown;

    // Calculate energy for each level
    for (int i = 0; i < hierarchy.get_num_levels(); ++i) {
        const BufferLevel* level = hierarchy.get_level(i);
        if (!level) continue;

        const auto& config = level->get_config();
        const auto& stats = level->get_stats();

        double level_energy = calculate_level_energy(stats, config, total_cycles, frequency_GHz);

        breakdown.energy_per_level[config.name] = level_energy;
        breakdown.total_energy_mJ += level_energy;

        // Accumulate read/write/static components
        breakdown.read_energy_mJ += calculate_read_energy_mJ(
            stats.bytes_read,
            config.power.read_energy_pJ_per_byte
        );

        breakdown.write_energy_mJ += calculate_write_energy_mJ(
            stats.bytes_written,
            config.power.write_energy_pJ_per_byte
        );

        breakdown.static_energy_mJ += calculate_static_energy_mJ(
            config.power.static_power_mW,
            total_cycles,
            frequency_GHz
        );
    }

    // Calculate per-operation energy breakdown
    // Aggregate across all levels
    BufferAccessStats total_stats = hierarchy.get_total_stats();

    for (const auto& [op_type, bytes] : total_stats.bytes_read_per_op_type) {
        // Use average energy across all levels (simplified)
        double avg_read_energy_pJ = 2.5;  // Default approximation
        double op_read_energy = pJ_to_mJ(bytes * avg_read_energy_pJ);
        breakdown.energy_per_op_type[op_type] += op_read_energy;
    }

    for (const auto& [op_type, bytes] : total_stats.bytes_written_per_op_type) {
        double avg_write_energy_pJ = 3.0;
        double op_write_energy = pJ_to_mJ(bytes * avg_write_energy_pJ);
        breakdown.energy_per_op_type[op_type] += op_write_energy;
    }

    return breakdown;
}

double PowerModel::calculate_level_energy(
    const BufferAccessStats& stats,
    const BufferLevelConfig& config,
    uint64_t total_cycles,
    double frequency_GHz
) {
    double read_energy = calculate_read_energy_mJ(
        stats.bytes_read,
        config.power.read_energy_pJ_per_byte
    );

    double write_energy = calculate_write_energy_mJ(
        stats.bytes_written,
        config.power.write_energy_pJ_per_byte
    );

    double static_energy = calculate_static_energy_mJ(
        config.power.static_power_mW,
        total_cycles,
        frequency_GHz
    );

    return read_energy + write_energy + static_energy;
}

double PowerModel::calculate_read_energy_mJ(
    uint64_t bytes_read,
    double read_energy_pJ_per_byte
) {
    double energy_pJ = static_cast<double>(bytes_read) * read_energy_pJ_per_byte;
    return pJ_to_mJ(energy_pJ);
}

double PowerModel::calculate_write_energy_mJ(
    uint64_t bytes_written,
    double write_energy_pJ_per_byte
) {
    double energy_pJ = static_cast<double>(bytes_written) * write_energy_pJ_per_byte;
    return pJ_to_mJ(energy_pJ);
}

double PowerModel::calculate_static_energy_mJ(
    double static_power_mW,
    uint64_t total_cycles,
    double frequency_GHz
) {
    // Convert cycles to seconds
    double time_seconds = cycles_to_seconds(total_cycles, frequency_GHz);

    // Energy = Power * Time
    // Power is in mW, time in seconds, so energy is in mJ
    return static_power_mW * time_seconds * 1000.0;  // Convert to mJ
}

void PowerModel::apply_technology_scaling(
    BufferLevelConfig& config,
    int technology_nm
) {
    // Simplified scaling rules based on technology node
    // More accurate would be to use CACTI or similar tools

    // Baseline: assume config is for 7nm
    const int baseline_nm = 7;

    if (technology_nm == baseline_nm) {
        return;  // No scaling needed
    }

    // Scaling factor for energy (approximately quadratic with feature size)
    // Smaller nodes are more energy efficient
    double energy_scale = std::pow(static_cast<double>(technology_nm) / baseline_nm, 2.0);

    // Scaling factor for static power (increases with smaller nodes due to leakage)
    // But also improves with better transistor designs - simplified model
    double static_scale = std::pow(static_cast<double>(baseline_nm) / technology_nm, 1.5);

    config.power.read_energy_pJ_per_byte *= energy_scale;
    config.power.write_energy_pJ_per_byte *= energy_scale;
    config.power.static_power_mW *= static_scale;

    std::cout << "Technology scaling applied: " << baseline_nm << "nm -> "
              << technology_nm << "nm\n";
    std::cout << "  Energy scale: " << energy_scale << "\n";
    std::cout << "  Static power scale: " << static_scale << "\n";
}

void PowerModel::print_power_report(
    std::ostream& os,
    const EnergyBreakdown& energy,
    const BufferHierarchy& hierarchy,
    uint64_t total_cycles,
    double frequency_GHz
) {
    energy.print_report(os, total_cycles, frequency_GHz);

    // Print per-level access statistics
    os << "\n=== Buffer Access Statistics ===\n";
    for (int i = 0; i < hierarchy.get_num_levels(); ++i) {
        const BufferLevel* level = hierarchy.get_level(i);
        if (!level) continue;

        const auto& stats = level->get_stats();
        stats.print_summary(os, level->get_config().name);
    }

    // Print utilization
    hierarchy.print_utilization(os);

    os << "\n=== End of Power Report ===\n";
}

} // namespace buffers
