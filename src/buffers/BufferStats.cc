/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "buffers/BufferStats.h"
#include <iomanip>

namespace buffers {

void BufferAccessStats::reset() {
    num_reads = 0;
    num_writes = 0;
    bytes_read = 0;
    bytes_written = 0;
    hits = 0;
    misses = 0;
    bank_conflicts = 0;
    stall_cycles = 0;
    peak_occupancy_bytes = 0;
    total_occupancy_cycles = 0;
    reads_per_op_type.clear();
    writes_per_op_type.clear();
    bytes_read_per_op_type.clear();
    bytes_written_per_op_type.clear();
}

void BufferAccessStats::merge(const BufferAccessStats& other) {
    num_reads += other.num_reads;
    num_writes += other.num_writes;
    bytes_read += other.bytes_read;
    bytes_written += other.bytes_written;
    hits += other.hits;
    misses += other.misses;
    bank_conflicts += other.bank_conflicts;
    stall_cycles += other.stall_cycles;
    peak_occupancy_bytes = std::max(peak_occupancy_bytes, other.peak_occupancy_bytes);
    total_occupancy_cycles += other.total_occupancy_cycles;

    // Merge per-operation stats
    for (const auto& [op, count] : other.reads_per_op_type) {
        reads_per_op_type[op] += count;
    }
    for (const auto& [op, count] : other.writes_per_op_type) {
        writes_per_op_type[op] += count;
    }
    for (const auto& [op, bytes] : other.bytes_read_per_op_type) {
        bytes_read_per_op_type[op] += bytes;
    }
    for (const auto& [op, bytes] : other.bytes_written_per_op_type) {
        bytes_written_per_op_type[op] += bytes;
    }
}

void BufferAccessStats::record_read(uint64_t bytes, const std::string& op_type) {
    num_reads++;
    bytes_read += bytes;

    if (!op_type.empty()) {
        reads_per_op_type[op_type]++;
        bytes_read_per_op_type[op_type] += bytes;
    }
}

void BufferAccessStats::record_write(uint64_t bytes, const std::string& op_type) {
    num_writes++;
    bytes_written += bytes;

    if (!op_type.empty()) {
        writes_per_op_type[op_type]++;
        bytes_written_per_op_type[op_type] += bytes;
    }
}

void BufferAccessStats::record_occupancy(uint64_t current_bytes) {
    total_occupancy_cycles += current_bytes;
    if (current_bytes > peak_occupancy_bytes) {
        peak_occupancy_bytes = current_bytes;
    }
}

double BufferAccessStats::get_average_occupancy(uint64_t total_cycles) const {
    if (total_cycles == 0) return 0.0;
    return static_cast<double>(total_occupancy_cycles) / static_cast<double>(total_cycles);
}

void BufferAccessStats::print_summary(std::ostream& os, const std::string& buffer_name) const {
    os << "\n=== Buffer Access Stats: " << buffer_name << " ===\n";
    os << "  Reads:        " << num_reads << " (" << bytes_read << " bytes)\n";
    os << "  Writes:       " << num_writes << " (" << bytes_written << " bytes)\n";
    os << "  Total Bytes:  " << (bytes_read + bytes_written) << "\n";

    if (hits > 0 || misses > 0) {
        double hit_rate = static_cast<double>(hits) / (hits + misses) * 100.0;
        os << "  Hit Rate:     " << std::fixed << std::setprecision(2) << hit_rate << "%\n";
    }

    if (bank_conflicts > 0) {
        os << "  Bank Conflicts: " << bank_conflicts << "\n";
    }

    if (stall_cycles > 0) {
        os << "  Stall Cycles: " << stall_cycles << "\n";
    }

    if (peak_occupancy_bytes > 0) {
        os << "  Peak Occupancy: " << (peak_occupancy_bytes / 1024.0 / 1024.0)
           << " MB\n";
    }

    // Per-operation breakdown
    if (!reads_per_op_type.empty()) {
        os << "\n  Reads by Operation:\n";
        for (const auto& [op, count] : reads_per_op_type) {
            uint64_t op_bytes = bytes_read_per_op_type.at(op);
            os << "    " << op << ": " << count << " reads ("
               << (op_bytes / 1024.0 / 1024.0) << " MB)\n";
        }
    }

    if (!writes_per_op_type.empty()) {
        os << "\n  Writes by Operation:\n";
        for (const auto& [op, count] : writes_per_op_type) {
            uint64_t op_bytes = bytes_written_per_op_type.at(op);
            os << "    " << op << ": " << count << " writes ("
               << (op_bytes / 1024.0 / 1024.0) << " MB)\n";
        }
    }
}

} // namespace buffers
