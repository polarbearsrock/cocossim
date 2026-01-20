/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_POWER_MODEL_H
#define COCOSSIM_POWER_MODEL_H

#include "BufferConfig.h"
#include "BufferStats.h"
#include "BufferHierarchy.h"

namespace buffers {

/**
 * Power modeling utilities for buffer hierarchy
 */
class PowerModel {
public:
    /**
     * Calculate energy consumption from buffer hierarchy stats
     *
     * @param hierarchy Buffer hierarchy with collected statistics
     * @param total_cycles Total simulation cycles
     * @param frequency_GHz Operating frequency in GHz
     * @return Detailed energy breakdown
     */
    static EnergyBreakdown calculate_energy(
        const BufferHierarchy& hierarchy,
        uint64_t total_cycles,
        double frequency_GHz
    );

    /**
     * Calculate energy for a single buffer level
     *
     * @param stats Access statistics for the level
     * @param config Configuration including power parameters
     * @param total_cycles Total simulation cycles
     * @param frequency_GHz Operating frequency in GHz
     * @return Energy in millijoules
     */
    static double calculate_level_energy(
        const BufferAccessStats& stats,
        const BufferLevelConfig& config,
        uint64_t total_cycles,
        double frequency_GHz
    );

    /**
     * Calculate read energy for a buffer level
     */
    static double calculate_read_energy_mJ(
        uint64_t bytes_read,
        double read_energy_pJ_per_byte
    );

    /**
     * Calculate write energy for a buffer level
     */
    static double calculate_write_energy_mJ(
        uint64_t bytes_written,
        double write_energy_pJ_per_byte
    );

    /**
     * Calculate static/leakage energy
     */
    static double calculate_static_energy_mJ(
        double static_power_mW,
        uint64_t total_cycles,
        double frequency_GHz
    );

    /**
     * Apply technology node scaling to power parameters
     * Uses simplified scaling rules (actual CACTI integration would be better)
     *
     * @param config Buffer configuration to modify
     * @param technology_nm Technology node in nanometers (e.g., 7, 5, 3)
     */
    static void apply_technology_scaling(
        BufferLevelConfig& config,
        int technology_nm
    );

    /**
     * Print detailed power report
     */
    static void print_power_report(
        std::ostream& os,
        const EnergyBreakdown& energy,
        const BufferHierarchy& hierarchy,
        uint64_t total_cycles,
        double frequency_GHz
    );

private:
    // Helper: Convert cycles to time (seconds)
    static double cycles_to_seconds(uint64_t cycles, double frequency_GHz) {
        return static_cast<double>(cycles) / (frequency_GHz * 1e9);
    }

    // Helper: Convert pJ to mJ
    static double pJ_to_mJ(double pJ) {
        return pJ / 1e9;
    }
};

} // namespace buffers

#endif // COCOSSIM_POWER_MODEL_H
