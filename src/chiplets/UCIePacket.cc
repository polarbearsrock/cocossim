/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/UCIePacket.h"
#include <sstream>
#include <iomanip>

namespace chiplets {

std::string UCIePacket::to_string() const {
    std::ostringstream oss;

    oss << "UCIePacket[" << packet_id << "]: ";
    oss << get_type_string() << " ";
    oss << "(" << src_chiplet << " → " << dst_chiplet << ") ";
    oss << "| Size: " << size_bytes << " B ";
    oss << "| Addr: 0x" << std::hex << std::setw(12) << std::setfill('0') << address << std::dec;

    if (total_latency_cycles > 0) {
        oss << " | Latency: " << total_latency_cycles << " cycles";
        oss << " (ser: " << serialization_cycles;
        oss << ", tx: " << transmission_cycles;
        oss << ", q: " << queueing_cycles << ")";
    }

    if (!operation_type.empty()) {
        oss << " | Op: " << operation_type;
    }

    if (has_error) {
        oss << " | ERROR: " << error_msg;
    }

    return oss.str();
}

} // namespace chiplets
