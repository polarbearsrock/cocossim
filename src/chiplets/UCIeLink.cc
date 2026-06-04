/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/UCIeLink.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace chiplets {

// ============================================================================
// UCIeLinkStats Implementation
// ============================================================================

void UCIeLinkStats::calculate_derived_stats(uint64_t total_cycles) {
    // Calculate average latency
    if (packets_transmitted > 0) {
        uint64_t total_latency = total_serialization_cycles +
                                 total_transmission_cycles +
                                 total_queueing_cycles;
        avg_packet_latency = static_cast<double>(total_latency) / packets_transmitted;
    }

    // Calculate utilization
    if (total_cycles > 0) {
        utilization = static_cast<double>(active_cycles) / total_cycles;
    }

    // Calculate average power.
    // energy_per_cycle_mJ = power_mW × 1e-9  (1 mW × 1 ns = 1 pJ = 1e-9 mJ)
    // → avg_power_mW = total_energy_mJ / total_cycles × 1e9
    if (total_cycles > 0) {
        avg_power_mW = (total_energy_mJ / total_cycles) * 1e9;
    }
}

void UCIeLinkStats::print_summary(std::ostream& os, const std::string& link_name) const {
    os << "\n=== UCIe Link Stats: " << link_name << " ===\n";

    // Traffic
    os << "  Packets TX: " << packets_transmitted
       << " (" << (bytes_transmitted / 1024.0 / 1024.0) << " MB)\n";
    os << "  Packets RX: " << packets_received
       << " (" << (bytes_received / 1024.0 / 1024.0) << " MB)\n";

    // Latency
    os << "  Avg Latency: " << std::fixed << std::setprecision(1)
       << avg_packet_latency << " cycles\n";
    os << "    Serialization: " << total_serialization_cycles << " cycles\n";
    os << "    Transmission: " << total_transmission_cycles << " cycles\n";
    os << "    Queueing: " << total_queueing_cycles << " cycles\n";

    // Flow Control
    os << "  Credit Stalls: " << credit_stalls << " cycles\n";
    os << "  Buffer Stalls: " << buffer_full_stalls << " cycles\n";

    // Utilization
    os << "  Utilization: " << std::setprecision(1) << (utilization * 100.0) << "%\n";
    os << "  Active Cycles: " << active_cycles << " / " << (active_cycles + idle_cycles) << "\n";

    // Power
    os << "  Total Energy: " << std::setprecision(2) << total_energy_mJ << " mJ\n";
    os << "  Avg Power: " << std::setprecision(1) << avg_power_mW << " mW\n";

    // Errors
    if (dropped_packets > 0 || crc_errors > 0) {
        os << "  Dropped Packets: " << dropped_packets << "\n";
        os << "  CRC Errors: " << crc_errors << "\n";
    }
}

// ============================================================================
// UCIeLink Implementation
// ============================================================================

UCIeLink::UCIeLink(int link_id, int src_chiplet, int dst_chiplet, const UCIePhyConfig& phy_config)
    : link_id_(link_id)
    , src_chiplet_(src_chiplet)
    , dst_chiplet_(dst_chiplet)
    , phy_config_(phy_config)
    , state_(LinkState::L0_ACTIVE)
    , current_cycle_(0)
    , tx_credits_(0)
    , rx_credits_(0)
    , max_credits_(4096)  // 4096 flits of 64 bytes = 256 KB; sized to never block realistic packets
    , max_buffer_size_(256)  // 256 packets max
    , consecutive_idle_cycles_(0)
    , wakeup_remaining_(0)
{
    // Initialize credits to max (buffer is empty)
    tx_credits_ = max_credits_;
    rx_credits_ = max_credits_;
}

// ============================================================================
// Flow Control
// ============================================================================

bool UCIeLink::has_credits(int required_credits) const {
    return tx_credits_ >= required_credits;
}

void UCIeLink::consume_credits(int num_credits) {
    tx_credits_ -= num_credits;
    if (tx_credits_ < 0) {
        std::cerr << "ERROR: Link " << link_id_ << " consumed more credits than available!\n";
        tx_credits_ = 0;
    }
    stats_.total_credits_granted += num_credits;
}

void UCIeLink::return_credits(int num_credits) {
    /**
     * Credit Return with Latency
     *
     * Reference: [1] UCIe Spec Section 3.4.4 - Credit Return Latency
     *
     * Credits are not returned immediately. They experience round-trip latency:
     * 1. Packet arrives at destination
     * 2. Receiver processes packet and frees buffer space
     * 3. Credit return message sent back to sender
     * 4. Credit return message experiences link latency
     *
     * Total delay: ~20-50 cycles depending on link configuration
     */
    uint64_t return_cycle = current_cycle_ + phy_config_.credit_return_latency_cycles;

    // Schedule credit return — credits become available at return_cycle.
    pending_credit_returns_.push_back({return_cycle, num_credits});

    stats_.total_credits_returned += num_credits;
}

// ============================================================================
// Buffer Management
// ============================================================================

bool UCIeLink::can_enqueue_packet(const UCIePacket* packet) const {
    // Packets may be buffered in L1/L2 — enqueueing triggers a wakeup.
    // Only refuse if the link is in a non-operational state (error/init/training).
    if (state_ == LinkState::ERROR ||
        state_ == LinkState::INIT  ||
        state_ == LinkState::TRAINING) {
        return false;
    }

    // Credits gate wire transmission (in transmit()), not sender-side buffering.
    // Only check that the tx buffer has space.
    return tx_buffer_.size() < static_cast<size_t>(max_buffer_size_);
}

bool UCIeLink::enqueue_packet(UCIePacket* packet) {
    if (!can_enqueue_packet(packet)) {
        return false;
    }

    tx_buffer_.push_back(packet);
    // Mark IN_TRANSIT immediately so the caller does not re-enqueue next cycle.
    // The packet is still in tx_buffer_ awaiting credit-gated wire transmission.
    packet->status = PacketStatus::IN_TRANSIT;
    packet->tx_start_cycle = current_cycle_;

    return true;
}

UCIePacket* UCIeLink::dequeue_packet() {
    if (tx_buffer_.empty()) {
        return nullptr;
    }

    UCIePacket* packet = tx_buffer_.front();
    tx_buffer_.pop_front();
    return packet;
}

// ============================================================================
// Transmission
// ============================================================================

bool UCIeLink::transmit(UCIePacket* packet) {
    if (!is_active()) {
        return false;
    }

    // Credits model back-pressure at the flit level; our simulation works at
    // packet granularity with a serialization-delay model, so we skip the
    // credit-blocking check and just consume/return credits for statistics.
    int required_credits = calculate_flit_count(packet->get_total_size_bytes());
    if (has_credits(required_credits)) {
        consume_credits(required_credits);
    }
    // (If credits are insufficient, proceed anyway — timing is still correct
    //  via serialization + link latency; credits replenish when packet lands.)

    // Calculate transmission time
    int tx_cycles = calculate_transmission_cycles(packet);

    // Calculate and set serialization cycles for this packet
    int ser_cycles = packet->calculate_serialization_cycles(phy_config_.effective_bandwidth_gbps);
    packet->serialization_cycles = ser_cycles;

    // Mark packet as in transit
    packet->status = PacketStatus::IN_TRANSIT;
    packet->tx_start_cycle = current_cycle_;
    packet->tx_end_cycle = current_cycle_ + tx_cycles;
    packet->transmission_cycles = tx_cycles;

    // Add to in-flight tracking
    InFlightPacket in_flight;
    in_flight.packet = packet;
    in_flight.completion_cycle = packet->tx_end_cycle;
    in_flight_packets_.push_back(in_flight);

    // Update statistics
    stats_.packets_transmitted++;
    stats_.bytes_transmitted += packet->size_bytes;
    stats_.total_serialization_cycles += packet->serialization_cycles;
    stats_.total_transmission_cycles += packet->transmission_cycles;

    return true;
}

// ============================================================================
// Simulation
// ============================================================================

void UCIeLink::tick(uint64_t current_cycle) {
    current_cycle_ = current_cycle;

    // Process in-flight packets (check for completions)
    process_in_flight_packets();

    // Process credit returns
    process_credit_returns();

    // Advance the L0/L1/L2 power state FSM.
    // Must run after process_in_flight_packets() so link_busy reflects current activity.
    update_power_state();

    // Try to transmit packets from buffer.
    // Block while waking from L1/L2 so the wakeup latency is correctly modeled.
    if (wakeup_remaining_ > 0) {
        wakeup_remaining_--;
    } else {
        while (!tx_buffer_.empty()) {
            UCIePacket* packet = tx_buffer_.front();

            if (transmit(packet)) {
                tx_buffer_.pop_front();
            } else {
                // Blocked - stop trying for this cycle
                if (!has_credits(calculate_flit_count(packet->get_total_size_bytes()))) {
                    stats_.credit_stalls++;
                } else {
                    stats_.buffer_full_stalls++;
                }
                break;
            }
        }
    }

    // Update power consumption
    update_power_stats();

    // Track utilization
    if (!in_flight_packets_.empty()) {
        stats_.active_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void UCIeLink::reset_stats() {
    stats_.reset();
}

// ============================================================================
// Helper Functions
// ============================================================================

int UCIeLink::calculate_flit_count(uint64_t packet_size_bytes) const {
    /**
     * Flit Count Calculation
     *
     * Reference: [1] UCIe Spec Section 3.4.2 - Flit-based Flow Control
     *
     * UCIe uses fixed-size flits (typically 64 or 128 bytes).
     * Credits are granted per flit, not per packet.
     *
     * Example:
     * - Flit size: 64 bytes
     * - Packet size: 256 bytes
     * - Flit count: ceil(256 / 64) = 4 flits
     */
    const int FLIT_SIZE_BYTES = 64;  // Standard flit size
    return static_cast<int>(std::ceil(static_cast<double>(packet_size_bytes) / FLIT_SIZE_BYTES));
}

int UCIeLink::calculate_transmission_cycles(const UCIePacket* packet) const {
    /**
     * Transmission Latency Calculation
     *
     * Reference: [2] UCIe Spec Section 4.3 - Latency Analysis
     *
     * Total latency = Serialization + Link Latency + Adapter Latency
     *
     * 1. Serialization: Time to convert packet to flits and transmit on wire
     *    = packet_size / bandwidth
     *
     * 2. Link Latency: Physical layer latency (PHY TX + RX)
     *    = phy_tx_latency + phy_rx_latency
     *
     * 3. Adapter Latency: Die adapter processing
     *    = adapter_tx_latency + adapter_rx_latency
     */

    // Serialization latency
    int ser_cycles = packet->calculate_serialization_cycles(phy_config_.effective_bandwidth_gbps);

    // Link layer and adapter latency
    int link_cycles = phy_config_.get_total_latency_cycles();

    return ser_cycles + link_cycles;
}

void UCIeLink::process_in_flight_packets() {
    /**
     * Process packets that have completed transmission
     */
    auto it = in_flight_packets_.begin();
    while (it != in_flight_packets_.end()) {
        if (current_cycle_ >= it->completion_cycle) {
            // Packet transmission complete
            UCIePacket* packet = it->packet;
            packet->status = PacketStatus::COMPLETED;
            packet->rx_cycle = current_cycle_;
            packet->total_latency_cycles = current_cycle_ - packet->creation_cycle;
            // Record where the packet is now (needed for multi-hop forwarding).
            packet->current_chiplet = dst_chiplet_;

            // Return credits (with latency)
            int credits = calculate_flit_count(packet->get_total_size_bytes());
            return_credits(credits);

            // Update stats
            stats_.packets_received++;
            stats_.bytes_received += packet->size_bytes;

            // Remove from in-flight
            it = in_flight_packets_.erase(it);
        } else {
            ++it;
        }
    }
}

void UCIeLink::process_credit_returns() {
    // Only release credits whose return_cycle has been reached.
    while (!pending_credit_returns_.empty()) {
        auto [return_cycle, credits] = pending_credit_returns_.front();
        if (current_cycle_ < return_cycle)
            break;
        pending_credit_returns_.pop_front();
        tx_credits_ = std::min(tx_credits_ + credits, max_credits_);
    }
}

void UCIeLink::update_power_stats() {
    /**
     * Update Power Consumption
     *
     * Reference: [1] UCIe Spec Section 5.2 - Power Model
     *
     * Power = Static Power + Dynamic Power
     *
     * Dynamic Power = (lanes × power_per_lane) when active
     * Static Power = constant
     *
     * Energy = Power × time
     */
    double power_mW = 0.0;

    if (state_ == LinkState::L0_ACTIVE) {
        // Active state: static + dynamic
        power_mW = phy_config_.get_total_power_mW();

        // Scale dynamic power by utilization (if idle, only static)
        if (in_flight_packets_.empty()) {
            // Idle: only static power
            power_mW = phy_config_.static_power_mW;
        }
    } else if (state_ == LinkState::L1_STANDBY) {
        // Standby: reduced power (~10% of active)
        power_mW = phy_config_.static_power_mW * 0.1;
    }
    // L2_OFF: 0 power

    // Energy = power × time
    // Assuming 1 cycle = 1 ns @ 1 GHz
    double energy_per_cycle_mJ = power_mW * 1e-9;  // mW × ns = pJ, convert to mJ
    stats_.total_energy_mJ += energy_per_cycle_mJ;
}

void UCIeLink::update_power_state() {
    /**
     * Advance the L0/L1/L2 power state FSM.
     *
     * Reference: UCIe Spec 1.0 Section 5.1 - Link Power States
     *
     * Entry policy (from L0):
     *   Idle for L1_ENTRY_THRESHOLD cycles  → enter L1 (standby, 10% static power)
     *   Idle for L2_ENTRY_THRESHOLD cycles total → enter L2 (off, 0 power)
     *
     * Exit policy:
     *   Non-empty tx_buffer detected in L1/L2 → transition to L0 but stall
     *   transmission for L1_WAKEUP_CYCLES / L2_WAKEUP_CYCLES via wakeup_remaining_.
     */
    bool link_busy = !tx_buffer_.empty() || !in_flight_packets_.empty();

    switch (state_) {
        case LinkState::L0_ACTIVE:
            if (link_busy) {
                consecutive_idle_cycles_ = 0;
            } else {
                consecutive_idle_cycles_++;
                if (consecutive_idle_cycles_ >= L2_ENTRY_THRESHOLD) {
                    state_ = LinkState::L2_OFF;
                    consecutive_idle_cycles_ = 0;
                } else if (consecutive_idle_cycles_ >= L1_ENTRY_THRESHOLD) {
                    state_ = LinkState::L1_STANDBY;
                    consecutive_idle_cycles_ = 0;
                }
            }
            break;

        case LinkState::L1_STANDBY:
            if (link_busy) {
                // Packet arrived while in standby — wake up with L1 latency.
                state_ = LinkState::L0_ACTIVE;
                wakeup_remaining_ = L1_WAKEUP_CYCLES;
                consecutive_idle_cycles_ = 0;
            } else {
                consecutive_idle_cycles_++;
                if (consecutive_idle_cycles_ >= L2_ENTRY_THRESHOLD) {
                    state_ = LinkState::L2_OFF;
                    consecutive_idle_cycles_ = 0;
                }
            }
            break;

        case LinkState::L2_OFF:
            if (link_busy) {
                // Packet arrived while powered off — reinitialization required.
                state_ = LinkState::L0_ACTIVE;
                wakeup_remaining_ = L2_WAKEUP_CYCLES;
                consecutive_idle_cycles_ = 0;
            }
            break;

        default:
            break;
    }
}

void UCIeLink::print_status(std::ostream& os) const {
    os << "UCIeLink[" << link_id_ << "]: "
       << src_chiplet_ << " → " << dst_chiplet_ << "\n";
    os << "  Config: " << phy_config_.to_string() << "\n";
    os << "  State: ";
    switch (state_) {
        case LinkState::L0_ACTIVE:  os << "L0_ACTIVE"; break;
        case LinkState::L1_STANDBY: os << "L1_STANDBY"; break;
        case LinkState::L2_OFF:     os << "L2_OFF"; break;
        case LinkState::INIT:       os << "INIT"; break;
        case LinkState::TRAINING:   os << "TRAINING"; break;
        case LinkState::ERROR:      os << "ERROR"; break;
    }
    os << "\n";
    os << "  Credits: " << tx_credits_ << " / " << max_credits_ << "\n";
    os << "  TX Buffer: " << tx_buffer_.size() << " / " << max_buffer_size_ << "\n";
    os << "  In Flight: " << in_flight_packets_.size() << " packets\n";
}

} // namespace chiplets
