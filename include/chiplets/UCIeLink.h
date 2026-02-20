/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_UCIELINK_H
#define COCOSSIM_UCIELINK_H

#include "chiplets/UCIeConfig.h"
#include "chiplets/UCIePacket.h"
#include <deque>
#include <vector>
#include <memory>

/**
 * UCIe Link Model
 *
 * Implements cycle-accurate UCIe link with:
 * - Credit-based flow control
 * - Flit-level transmission
 * - Buffering and queueing delays
 * - Power and bandwidth tracking
 *
 * REFERENCES:
 * [1] UCIe Spec 1.0, Section 3.4 - Flow Control
 * [2] UCIe Spec 1.0, Section 4.3 - Latency Analysis
 * [3] "Modeling UCIe Interconnects for AI Accelerators", ISCA 2024
 */

namespace chiplets {

/**
 * Link State
 * Reference: [1] Section 5.1 - Link Power States
 */
enum class LinkState {
    L0_ACTIVE,          // Fully active, can transmit/receive
    L1_STANDBY,         // Low power standby, quick wake
    L2_OFF,             // Powered off, requires reinitialization
    INIT,               // Link initialization in progress
    TRAINING,           // Link training/calibration
    ERROR               // Link error state
};

/**
 * Link Direction
 * UCIe links are typically bidirectional, but we model each direction separately
 */
enum class LinkDirection {
    FORWARD,            // src_chiplet → dst_chiplet
    REVERSE             // dst_chiplet → src_chiplet
};

/**
 * UCIe Link Statistics
 */
struct UCIeLinkStats {
    // Traffic Statistics
    uint64_t packets_transmitted;
    uint64_t packets_received;
    uint64_t bytes_transmitted;
    uint64_t bytes_received;

    // Latency Statistics
    uint64_t total_serialization_cycles;
    uint64_t total_transmission_cycles;
    uint64_t total_queueing_cycles;
    double avg_packet_latency;

    // Flow Control Statistics
    uint64_t credit_stalls;             // Cycles stalled due to lack of credits
    uint64_t buffer_full_stalls;        // Cycles stalled due to full buffer
    uint64_t total_credits_granted;
    uint64_t total_credits_returned;

    // Utilization
    uint64_t active_cycles;             // Cycles with data transmission
    uint64_t idle_cycles;               // Cycles idle
    double utilization;                 // active / (active + idle)

    // Power
    double total_energy_mJ;             // Total energy consumed (millijoules)
    double avg_power_mW;                // Average power (milliwatts)

    // Errors
    uint64_t dropped_packets;
    uint64_t crc_errors;

    UCIeLinkStats()
        : packets_transmitted(0)
        , packets_received(0)
        , bytes_transmitted(0)
        , bytes_received(0)
        , total_serialization_cycles(0)
        , total_transmission_cycles(0)
        , total_queueing_cycles(0)
        , avg_packet_latency(0.0)
        , credit_stalls(0)
        , buffer_full_stalls(0)
        , total_credits_granted(0)
        , total_credits_returned(0)
        , active_cycles(0)
        , idle_cycles(0)
        , utilization(0.0)
        , total_energy_mJ(0.0)
        , avg_power_mW(0.0)
        , dropped_packets(0)
        , crc_errors(0)
    {}

    void reset() {
        *this = UCIeLinkStats();
    }

    void calculate_derived_stats(uint64_t total_cycles);
    void print_summary(std::ostream& os, const std::string& link_name) const;
};

/**
 * UCIe Link
 *
 * Models a single unidirectional UCIe link between two chiplets.
 * For bidirectional communication, two UCIeLink instances are used
 * (one for each direction).
 */
class UCIeLink {
public:
    UCIeLink(int link_id, int src_chiplet, int dst_chiplet, const UCIePhyConfig& phy_config);
    ~UCIeLink() = default;

    // Configuration
    int get_link_id() const { return link_id_; }
    int get_src_chiplet() const { return src_chiplet_; }
    int get_dst_chiplet() const { return dst_chiplet_; }
    const UCIePhyConfig& get_phy_config() const { return phy_config_; }

    // State Management
    LinkState get_state() const { return state_; }
    void set_state(LinkState state) { state_ = state; }
    bool is_active() const { return state_ == LinkState::L0_ACTIVE; }

    // Flow Control - Credit-based (Reference: [1] Section 3.4)
    /**
     * Credit-Based Flow Control
     *
     * UCIe uses credit-based flow control to prevent buffer overflow:
     * 1. Receiver advertises available buffer space as "credits"
     * 2. Sender consumes credits when transmitting packets
     * 3. Receiver returns credits after processing packets
     *
     * Credit granularity: One credit = one flit (typically 64 bytes)
     */
    bool has_credits(int required_credits) const;
    void consume_credits(int num_credits);
    void return_credits(int num_credits);
    int get_available_credits() const { return tx_credits_; }
    int get_max_credits() const { return max_credits_; }

    // Buffer Management
    bool can_enqueue_packet(const UCIePacket* packet) const;
    bool enqueue_packet(UCIePacket* packet);
    UCIePacket* dequeue_packet();
    int get_buffer_occupancy() const { return tx_buffer_.size(); }
    int get_buffer_capacity() const { return max_buffer_size_; }

    // Transmission
    /**
     * Transmit a packet on this link
     *
     * Steps:
     * 1. Check if credits available
     * 2. Calculate serialization latency
     * 3. Break packet into flits
     * 4. Track transmission cycles
     * 5. Update statistics
     *
     * Returns: true if transmission started, false if blocked
     */
    bool transmit(UCIePacket* packet);

    // Simulation
    void tick(uint64_t current_cycle);
    void reset_stats();

    // Statistics
    const UCIeLinkStats& get_stats() const { return stats_; }
    UCIeLinkStats& get_stats_mutable() { return stats_; }

    // Diagnostics
    void print_status(std::ostream& os) const;

private:
    // Link Identification
    int link_id_;
    int src_chiplet_;
    int dst_chiplet_;

    // Physical Configuration
    UCIePhyConfig phy_config_;

    // Link State
    LinkState state_;
    uint64_t current_cycle_;

    // Flow Control (Credit-based)
    int tx_credits_;                    // Available transmit credits
    int rx_credits_;                    // Credits from receiver (for tracking)
    int max_credits_;                   // Maximum credits (= buffer capacity)
    std::deque<int> pending_credit_returns_;  // Credits to be returned with delay

    // Buffering
    // Reference: [1] Section 3.4.3 - Buffer Requirements
    // Typical buffer: 16-32 flits (1-2 KB)
    std::deque<UCIePacket*> tx_buffer_;     // Transmit buffer
    std::deque<UCIePacket*> rx_buffer_;     // Receive buffer (destination side)
    int max_buffer_size_;                   // Max packets in buffer

    // In-flight Transmission
    struct InFlightPacket {
        UCIePacket* packet;
        uint64_t completion_cycle;          // When transmission completes
    };
    std::vector<InFlightPacket> in_flight_packets_;

    // Statistics
    UCIeLinkStats stats_;

    // Helper Functions
    int calculate_flit_count(uint64_t packet_size_bytes) const;
    int calculate_transmission_cycles(const UCIePacket* packet) const;
    void process_in_flight_packets();
    void process_credit_returns();
    void update_power_stats();
};

} // namespace chiplets

#endif // COCOSSIM_UCIELINK_H
