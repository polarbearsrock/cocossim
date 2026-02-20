/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_UCIECONFIG_H
#define COCOSSIM_UCIECONFIG_H

#include <string>
#include <cstdint>

/**
 * UCIe (Universal Chiplet Interconnect Express) Configuration
 *
 * This file implements the UCIe 1.0 and 1.1 specifications for die-to-die
 * interconnect modeling in multi-chiplet systems.
 *
 * REFERENCES:
 * [1] UCIe Consortium, "Universal Chiplet Interconnect Express (UCIe)
 *     Specification Rev 1.0", March 2022
 *     https://www.uciexpress.org/specification
 *
 * [2] UCIe Consortium, "Universal Chiplet Interconnect Express (UCIe)
 *     Specification Rev 1.1", June 2023
 *
 * [3] Intel, "UCIe: An Open Chiplet Interconnect Standard", 2022
 *     https://www.intel.com/content/www/us/en/newsroom/opinion/updates-ucie-universal-chiplet-interconnect-express.html
 *
 * [4] Saptarshi Das et al., "Design and Analysis of a UCIe Based Multi-Chiplet
 *     AI Accelerator", IEEE ICCAD 2023
 */

namespace chiplets {

/**
 * UCIe Specification Version
 * Reference: [1] Section 1.3, [2]
 */
enum class UCIeVersion {
    UCIE_1_0,    // Initial specification (March 2022)
    UCIE_1_1     // Enhanced specification with improved signaling (June 2023)
};

/**
 * UCIe Protocol Layer
 * Reference: [1] Section 3.2 - Protocol Layer Options
 *
 * UCIe supports multiple protocol layers on top of the physical layer:
 * - CXL: Compute Express Link for cache-coherent memory access
 * - PCIe: PCI Express for standard device communication
 * - Streaming: Simple streaming protocol for maximum bandwidth
 */
enum class UCIeProtocol {
    STREAMING,   // Streaming protocol (lowest overhead, highest BW)
    CXL,         // Compute Express Link (cache coherency)
    PCIE,        // PCIe over UCIe (device communication)
    CUSTOM       // Custom protocol (user-defined)
};

/**
 * UCIe Physical Layer Link Width
 * Reference: [1] Section 2.3.1 - Lane Configuration
 *
 * UCIe supports scalable link widths from 4 to 32 lanes.
 * Each lane is a differential pair carrying data.
 */
enum class UCIeLinkWidth {
    X4,    // 4 data lanes (minimum configuration)
    X8,    // 8 data lanes
    X16,   // 16 data lanes (standard configuration)
    X32    // 32 data lanes (maximum bandwidth)
};

/**
 * UCIe Physical Layer Link Speed
 * Reference: [1] Section 2.2.2 - Signaling Rates
 *            [2] Section 2.2.3 - Enhanced Signaling (UCIe 1.1)
 *
 * UCIe supports both SDR (Single Data Rate) and DDR (Double Data Rate):
 * - SDR: Data transmitted on one edge of clock
 * - DDR: Data transmitted on both edges of clock (2× effective rate)
 *
 * Speed Progression:
 * UCIe 1.0: Up to 16 GT/s DDR
 * UCIe 1.1: Up to 32 GT/s DDR (improved signaling)
 */
enum class UCIeLinkSpeed {
    SDR_4GT,     // 4 GT/s SDR (gigatransfers per second)
    SDR_8GT,     // 8 GT/s SDR
    SDR_12GT,    // 12 GT/s SDR
    SDR_16GT,    // 16 GT/s SDR
    DDR_16GT,    // 16 GT/s DDR (8 GHz clock, both edges)
    DDR_24GT,    // 24 GT/s DDR (12 GHz clock, both edges)
    DDR_32GT     // 32 GT/s DDR (16 GHz clock, both edges) - UCIe 1.1+
};

/**
 * UCIe Physical Layer Configuration
 *
 * This structure contains all physical layer parameters for a UCIe link,
 * derived from the UCIe specification and academic studies.
 */
struct UCIePhyConfig {
    UCIeVersion version;
    UCIeLinkWidth width;
    UCIeLinkSpeed speed;

    // Bandwidth Calculations
    // Reference: [1] Section 2.2 - Bandwidth Calculation
    // Raw BW = (GT/s) × (width) × (bytes per transfer)
    // Effective BW accounts for protocol overhead, CRC, framing
    double raw_bandwidth_gbps;          // Raw physical bandwidth (GB/s)
    double effective_bandwidth_gbps;    // After overhead (typically 85-90% of raw)

    // Latency Components
    // Reference: [1] Section 4.3 - Latency Budget
    //            [4] Table 2 - UCIe Latency Analysis
    //
    // Total latency = PHY + Link Layer + Adapter
    int phy_tx_latency_cycles;          // Transmit PHY serialization (5-15 cycles)
    int phy_rx_latency_cycles;          // Receive PHY deserialization (5-15 cycles)
    int link_layer_latency_cycles;      // Link layer protocol overhead (2-8 cycles)
    int adapter_tx_latency_cycles;      // Die adapter transmit (10-25 cycles)
    int adapter_rx_latency_cycles;      // Die adapter receive (10-25 cycles)

    // Flow Control
    // Reference: [1] Section 3.4 - Flow Control
    // UCIe uses credit-based flow control
    int credit_return_latency_cycles;   // Round-trip credit latency (20-50 cycles)

    // Power Parameters
    // Reference: [1] Section 5.2 - Power Specifications
    //            [3] Intel UCIe Power Analysis
    //
    // Power scales with speed and width:
    // - Higher speed → higher SerDes power
    // - More lanes → more static power
    double power_per_lane_mW;           // Dynamic power per lane at speed (mW)
    double static_power_mW;             // Static power for link (mW)

    // Calculate total one-way latency
    int get_total_latency_cycles() const {
        return phy_tx_latency_cycles +
               phy_rx_latency_cycles +
               link_layer_latency_cycles +
               adapter_tx_latency_cycles +
               adapter_rx_latency_cycles;
    }

    // Calculate total power consumption
    double get_total_power_mW() const {
        int num_lanes = get_num_lanes();
        return (power_per_lane_mW * num_lanes) + static_power_mW;
    }

    // Get number of lanes from width
    int get_num_lanes() const {
        switch (width) {
            case UCIeLinkWidth::X4:  return 4;
            case UCIeLinkWidth::X8:  return 8;
            case UCIeLinkWidth::X16: return 16;
            case UCIeLinkWidth::X32: return 32;
            default: return 16;
        }
    }

    // Get speed in GT/s
    double get_speed_gtps() const {
        switch (speed) {
            case UCIeLinkSpeed::SDR_4GT:  return 4.0;
            case UCIeLinkSpeed::SDR_8GT:  return 8.0;
            case UCIeLinkSpeed::SDR_12GT: return 12.0;
            case UCIeLinkSpeed::SDR_16GT: return 16.0;
            case UCIeLinkSpeed::DDR_16GT: return 16.0;
            case UCIeLinkSpeed::DDR_24GT: return 24.0;
            case UCIeLinkSpeed::DDR_32GT: return 32.0;
            default: return 16.0;
        }
    }

    std::string to_string() const;
};

/**
 * Standard UCIe Configurations
 *
 * These presets represent common UCIe configurations used in industry
 * and academic studies.
 */
namespace ucie_configs {
    /**
     * Standard 16 GT/s, x16 Configuration
     * Reference: [1] Section 2.4 - Typical Configuration
     *
     * This is the most common UCIe configuration, providing:
     * - Raw bandwidth: 32 GB/s
     * - Effective bandwidth: ~28 GB/s (87.5% efficiency)
     * - Latency: ~60 cycles typical
     * - Power: ~250 mW
     *
     * Used in: AMD MI300A, Intel Meteor Lake
     */
    UCIePhyConfig standard_16gt_x16();

    /**
     * High Bandwidth 32 GT/s, x32 Configuration
     * Reference: [2] Section 2.2 - UCIe 1.1 High Speed
     *
     * High-performance configuration for data-intensive workloads:
     * - Raw bandwidth: 128 GB/s
     * - Effective bandwidth: ~112 GB/s
     * - Latency: ~80 cycles (higher due to SerDes complexity)
     * - Power: ~1.2 W
     *
     * Target: Next-gen AI accelerators, HPC chiplets
     */
    UCIePhyConfig high_bw_32gt_x32();

    /**
     * Power-Efficient 8 GT/s, x8 Configuration
     * Reference: [1] Section 5.3 - Low Power Mode
     *
     * Low-power configuration for edge/mobile:
     * - Raw bandwidth: 8 GB/s
     * - Effective bandwidth: ~7 GB/s
     * - Latency: ~50 cycles (simpler SerDes)
     * - Power: ~60 mW
     *
     * Target: Mobile SoCs, edge AI accelerators
     */
    UCIePhyConfig low_power_8gt_x8();

    /**
     * Balanced 24 GT/s, x16 Configuration
     * Reference: [2] UCIe 1.1 Standard
     *
     * Balanced performance/power for mainstream accelerators:
     * - Raw bandwidth: 48 GB/s
     * - Effective bandwidth: ~42 GB/s
     * - Latency: ~65 cycles
     * - Power: ~400 mW
     *
     * Target: Mainstream AI accelerators, server chiplets
     */
    UCIePhyConfig balanced_24gt_x16();
}

} // namespace chiplets

#endif // COCOSSIM_UCIECONFIG_H
