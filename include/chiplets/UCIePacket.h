/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_UCIEPACKET_H
#define COCOSSIM_UCIEPACKET_H

#include <cstdint>
#include <string>
#include <memory>

/**
 * UCIe Packet Model
 *
 * Implements packet structure for UCIe transfers based on the specification.
 *
 * REFERENCES:
 * [1] UCIe Spec 1.0, Section 3.3 - Packet Format
 * [2] UCIe Spec 1.0, Section 3.4 - Flow Control
 * [3] CXL 3.0 Spec - CXL.io over UCIe (for CXL packet types)
 */

namespace chiplets {

/**
 * UCIe Packet Type
 * Reference: [1] Section 3.3.1 - Packet Types
 *
 * UCIe supports various packet types depending on the protocol layer:
 * - Memory requests/responses (read/write)
 * - Control packets (flow control, link management)
 * - Protocol-specific packets (CXL, PCIe)
 */
enum class PacketType {
    // Memory Operations
    READ_REQUEST,       // Memory read request
    WRITE_REQUEST,      // Memory write request (data included)
    READ_RESPONSE,      // Memory read response (data included)
    WRITE_RESPONSE,     // Memory write completion acknowledgment

    // Flow Control
    CREDIT_RETURN,      // Credit return packet (no data)

    // Link Management
    LINK_INIT,          // Link initialization
    LINK_TRAINING,      // Link training sequence
    LINK_ERROR,         // Error notification

    // Protocol-Specific
    CXL_SNOOP,          // CXL coherency snoop
    CXL_MEMORY,         // CXL memory operation
    PCIE_TLP,           // PCIe Transaction Layer Packet

    // Custom
    CUSTOM              // User-defined packet type
};

/**
 * Packet Status
 * Tracks the lifecycle of a packet through the system
 */
enum class PacketStatus {
    CREATED,            // Packet created, not yet transmitted
    IN_TRANSIT,         // Currently being transmitted
    COMPLETED,          // Successfully received at destination
};

/**
 * UCIe Packet Structure
 *
 * Represents a single packet transmitted over a UCIe link.
 *
 * Packet Size Calculation (Reference: [1] Section 3.3.2):
 * - Header: 128 bits (16 bytes) - includes address, type, length
 * - Data: Variable (0-256 bytes typical)
 * - CRC: 32 bits (4 bytes) - error detection
 *
 * Maximum packet size:
 * - UCIe 1.0: 256 bytes data + 20 bytes overhead = 276 bytes
 * - UCIe 1.1: 512 bytes data + 20 bytes overhead = 532 bytes
 */
struct UCIePacket {
    // Packet Identification
    uint64_t packet_id;             // Unique packet ID for tracking
    PacketType type;                // Packet type
    PacketStatus status;            // Current status

    // Source and Destination
    int src_chiplet;                // Source chiplet ID (original sender)
    int dst_chiplet;                // Destination chiplet ID (final receiver)
    int current_chiplet;            // Chiplet where the packet currently resides
                                    // (updated per hop for multi-hop routing)

    // Address and Data
    uint64_t address;               // Physical address (48-bit typical)
    uint64_t size_bytes;            // Payload size in bytes
    void* data;                     // Pointer to payload data (optional)

    // Protocol Information
    std::string protocol;           // "STREAMING", "CXL", "PCIE"

    // Timing Information
    uint64_t creation_cycle;        // Cycle when packet was created
    uint64_t tx_start_cycle;        // Cycle when transmission started
    uint64_t tx_end_cycle;          // Cycle when transmission completed
    uint64_t rx_cycle;              // Cycle when received at destination

    // Latency Tracking
    uint64_t serialization_cycles;  // Time to serialize packet to flits
    uint64_t transmission_cycles;   // Time on wire
    uint64_t queueing_cycles;       // Time spent waiting in queues
    uint64_t total_latency_cycles;  // Total end-to-end latency

    // Flow Control
    int credits_required;           // Number of credits needed to send
    bool credit_granted;            // Whether credits are available

    // Associated Job (for statistics)
    int job_id;                     // Job ID this packet belongs to (-1 if none)
    std::string operation_type;     // Operation type: "matmul", "allreduce", etc.

    // Error Tracking
    bool has_error;                 // Error flag
    std::string error_msg;          // Error message if any

    // Constructor
    UCIePacket()
        : packet_id(0)
        , type(PacketType::READ_REQUEST)
        , status(PacketStatus::CREATED)
        , src_chiplet(-1)
        , dst_chiplet(-1)
        , current_chiplet(-1)
        , address(0)
        , size_bytes(0)
        , data(nullptr)
        , protocol("STREAMING")
        , creation_cycle(0)
        , tx_start_cycle(0)
        , tx_end_cycle(0)
        , rx_cycle(0)
        , serialization_cycles(0)
        , transmission_cycles(0)
        , queueing_cycles(0)
        , total_latency_cycles(0)
        , credits_required(1)
        , credit_granted(false)
        , job_id(-1)
        , operation_type("")
        , has_error(false)
        , error_msg("")
    {}

    // Calculate total packet size including overhead
    // Reference: [1] Section 3.3.2 - Packet Format
    uint64_t get_total_size_bytes() const {
        const uint64_t HEADER_SIZE = 16;  // 128 bits
        const uint64_t CRC_SIZE = 4;      // 32 bits
        return size_bytes + HEADER_SIZE + CRC_SIZE;
    }

    // Calculate serialization latency based on link speed
    // Reference: [1] Section 4.3.1 - Serialization Latency
    int calculate_serialization_cycles(double link_bandwidth_gbps) const {
        // Serialization time = packet_size / bandwidth
        // Convert to cycles assuming 1 GHz reference clock
        double time_ns = (get_total_size_bytes() / link_bandwidth_gbps);
        return static_cast<int>(std::ceil(time_ns));
    }

    // Get packet type as string
    std::string get_type_string() const {
        switch (type) {
            case PacketType::READ_REQUEST:    return "READ_REQ";
            case PacketType::WRITE_REQUEST:   return "WRITE_REQ";
            case PacketType::READ_RESPONSE:   return "READ_RESP";
            case PacketType::WRITE_RESPONSE:  return "WRITE_RESP";
            case PacketType::CREDIT_RETURN:   return "CREDIT";
            case PacketType::LINK_INIT:       return "LINK_INIT";
            case PacketType::LINK_TRAINING:   return "LINK_TRAIN";
            case PacketType::LINK_ERROR:      return "LINK_ERROR";
            case PacketType::CXL_SNOOP:       return "CXL_SNOOP";
            case PacketType::CXL_MEMORY:      return "CXL_MEM";
            case PacketType::PCIE_TLP:        return "PCIE_TLP";
            case PacketType::CUSTOM:          return "CUSTOM";
            default:                          return "UNKNOWN";
        }
    }

    // Print packet details for debugging
    std::string to_string() const;
};

/**
 * UCIe Flit (Flow Control Unit)
 *
 * Reference: [1] Section 3.4.2 - Flit-based Flow Control
 *
 * UCIe uses flit-based flow control where credits are granted in units
 * of flits rather than packets. A large packet may span multiple flits.
 *
 * Typical flit size: 64-128 bytes
 */
struct UCIeFlit {
    uint64_t flit_id;               // Unique flit ID
    uint64_t packet_id;             // Parent packet ID

    bool is_header;                 // First flit of packet (contains header)
    bool is_tail;                   // Last flit of packet

    uint64_t size_bytes;            // Flit size (fixed per link)

    // Timing
    uint64_t tx_start_cycle;        // Start of transmission
    uint64_t tx_end_cycle;          // End of transmission

    UCIeFlit()
        : flit_id(0)
        , packet_id(0)
        , is_header(false)
        , is_tail(false)
        , size_bytes(64)
        , tx_start_cycle(0)
        , tx_end_cycle(0)
    {}
};

} // namespace chiplets

#endif // COCOSSIM_UCIEPACKET_H
