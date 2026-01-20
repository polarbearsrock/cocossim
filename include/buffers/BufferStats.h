/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_BUFFER_STATS_H
#define COCOSSIM_BUFFER_STATS_H

#include <cstdint>
#include <map>
#include <string>
#include <iostream>

namespace buffers {

/**
 * Statistics for buffer accesses - tracks reads, writes, and energy
 */
struct BufferAccessStats {
    // Access counts
    uint64_t num_reads;                // Number of read operations
    uint64_t num_writes;               // Number of write operations
    uint64_t bytes_read;               // Total bytes read
    uint64_t bytes_written;            // Total bytes written

    // Cache-like metrics (optional, for future extensions)
    uint64_t hits;                     // Number of hits (if cache behavior is modeled)
    uint64_t misses;                   // Number of misses

    // Banking conflicts
    uint64_t bank_conflicts;           // Number of cycles lost to bank conflicts
    uint64_t stall_cycles;             // Total cycles stalled on buffer unavailability

    // Occupancy tracking
    uint64_t peak_occupancy_bytes;     // Maximum occupancy observed
    uint64_t total_occupancy_cycles;   // Sum of (occupancy * cycles) for average calculation

    // Per-operation type breakdown (e.g., "matmul", "conv", "softmax")
    std::map<std::string, uint64_t> reads_per_op_type;
    std::map<std::string, uint64_t> writes_per_op_type;
    std::map<std::string, uint64_t> bytes_read_per_op_type;
    std::map<std::string, uint64_t> bytes_written_per_op_type;

    BufferAccessStats()
        : num_reads(0)
        , num_writes(0)
        , bytes_read(0)
        , bytes_written(0)
        , hits(0)
        , misses(0)
        , bank_conflicts(0)
        , stall_cycles(0)
        , peak_occupancy_bytes(0)
        , total_occupancy_cycles(0) {}

    // Reset all statistics
    void reset();

    // Merge stats from another instance (for multi-core aggregation)
    void merge(const BufferAccessStats& other);

    // Record a read access
    void record_read(uint64_t bytes, const std::string& op_type = "");

    // Record a write access
    void record_write(uint64_t bytes, const std::string& op_type = "");

    // Record occupancy for this cycle
    void record_occupancy(uint64_t current_bytes);

    // Get average occupancy
    double get_average_occupancy(uint64_t total_cycles) const;

    // Print summary
    void print_summary(std::ostream& os, const std::string& buffer_name) const;
};

/**
 * Energy breakdown calculated from buffer access stats
 */
struct EnergyBreakdown {
    double total_energy_mJ;            // Total energy in millijoules
    double read_energy_mJ;             // Energy for reads
    double write_energy_mJ;            // Energy for writes
    double static_energy_mJ;           // Static/leakage energy

    // Per-level breakdown (level name -> energy)
    std::map<std::string, double> energy_per_level;

    // Per-operation breakdown (operation type -> energy)
    std::map<std::string, double> energy_per_op_type;

    EnergyBreakdown()
        : total_energy_mJ(0.0)
        , read_energy_mJ(0.0)
        , write_energy_mJ(0.0)
        , static_energy_mJ(0.0) {}

    // Print detailed energy report
    void print_report(std::ostream& os, uint64_t total_cycles, double frequency_GHz) const;
};

} // namespace buffers

#endif // COCOSSIM_BUFFER_STATS_H
