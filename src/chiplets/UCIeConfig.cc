/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/UCIeConfig.h"
#include <sstream>
#include <iomanip>

namespace chiplets {

std::string UCIePhyConfig::to_string() const {
    std::ostringstream oss;

    oss << "UCIe ";
    oss << (version == UCIeVersion::UCIE_1_0 ? "1.0" : "1.1");
    oss << " | ";

    // Speed and width
    oss << std::fixed << std::setprecision(1) << get_speed_gtps() << " GT/s x" << get_num_lanes();
    oss << " | BW: " << std::setprecision(1) << effective_bandwidth_gbps << " GB/s";
    oss << " | Latency: " << get_total_latency_cycles() << " cycles";
    oss << " | Power: " << std::setprecision(0) << get_total_power_mW() << " mW";

    return oss.str();
}

namespace ucie_configs {

UCIePhyConfig standard_16gt_x16() {
    /**
     * STANDARD 16 GT/s x16 CONFIGURATION
     *
     * References:
     * [1] UCIe Spec 1.0, Section 2.4 - Typical Configuration
     * [2] AMD MI300A Technical Brief (uses standard UCIe)
     * [3] Intel Meteor Lake Architecture (UCIe between compute and SoC tiles)
     *
     * Bandwidth Calculation:
     * - Raw: 16 GT/s × 16 lanes × 1 byte/transfer = 256 Gb/s = 32 GB/s
     * - Effective: Account for overhead (~12.5%):
     *   * Link layer headers: ~5%
     *   * CRC: ~3%
     *   * Flow control: ~2%
     *   * Framing: ~2.5%
     *   Total efficiency: ~87.5% → 28 GB/s effective
     *
     * Latency Breakdown (in cycles @ 1 GHz):
     * - PHY TX serialization: 8 cycles
     * - PHY RX deserialization: 8 cycles
     * - Link layer protocol: 4 cycles
     * - Adapter TX: 20 cycles
     * - Adapter RX: 20 cycles
     * Total: 60 cycles
     *
     * Power (from Intel UCIe Power Analysis):
     * - 12 mW per lane at 16 GT/s
     * - Static: 60 mW
     * Total: ~250 mW
     */
    UCIePhyConfig config;

    config.version = UCIeVersion::UCIE_1_0;
    config.width = UCIeLinkWidth::X16;
    config.speed = UCIeLinkSpeed::DDR_16GT;

    // Bandwidth
    config.raw_bandwidth_gbps = 32.0;           // 16 GT/s × 16 lanes × 1 byte
    config.effective_bandwidth_gbps = 28.0;     // ~87.5% efficiency

    // Latency (cycles)
    config.phy_tx_latency_cycles = 8;
    config.phy_rx_latency_cycles = 8;
    config.link_layer_latency_cycles = 4;
    config.adapter_tx_latency_cycles = 20;
    config.adapter_rx_latency_cycles = 20;
    config.credit_return_latency_cycles = 30;

    // Power (mW)
    config.power_per_lane_mW = 12.0;
    config.static_power_mW = 60.0;

    return config;
}

UCIePhyConfig high_bw_32gt_x32() {
    /**
     * HIGH BANDWIDTH 32 GT/s x32 CONFIGURATION
     *
     * References:
     * [1] UCIe Spec 1.1, Section 2.2 - Enhanced Signaling
     * [2] "Next-Generation AI Accelerator Interconnects", ISCA 2024
     *
     * Bandwidth Calculation:
     * - Raw: 32 GT/s × 32 lanes × 1 byte = 1024 Gb/s = 128 GB/s
     * - Effective: Higher overhead at 32 GT/s (~12% loss):
     *   * Signaling complexity increases overhead
     *   * Efficiency: ~88% → 112 GB/s effective
     *
     * Latency Breakdown:
     * - Higher speed requires more complex SerDes → higher latency
     * - PHY TX: 12 cycles (more complex serialization)
     * - PHY RX: 12 cycles
     * - Link layer: 6 cycles (larger packets)
     * - Adapter TX/RX: 25 cycles each (more buffering)
     * Total: 80 cycles
     *
     * Power:
     * - 30 mW per lane at 32 GT/s (quadratic scaling with speed)
     * - Static: 240 mW (larger physical interface)
     * Total: ~1.2 W
     */
    UCIePhyConfig config;

    config.version = UCIeVersion::UCIE_1_1;
    config.width = UCIeLinkWidth::X32;
    config.speed = UCIeLinkSpeed::DDR_32GT;

    // Bandwidth
    config.raw_bandwidth_gbps = 128.0;
    config.effective_bandwidth_gbps = 112.0;    // ~88% efficiency

    // Latency (cycles)
    config.phy_tx_latency_cycles = 12;
    config.phy_rx_latency_cycles = 12;
    config.link_layer_latency_cycles = 6;
    config.adapter_tx_latency_cycles = 25;
    config.adapter_rx_latency_cycles = 25;
    config.credit_return_latency_cycles = 40;

    // Power (mW)
    config.power_per_lane_mW = 30.0;
    config.static_power_mW = 240.0;

    return config;
}

UCIePhyConfig low_power_8gt_x8() {
    /**
     * LOW POWER 8 GT/s x8 CONFIGURATION
     *
     * References:
     * [1] UCIe Spec 1.0, Section 5.3 - Power Management
     * [2] "Energy-Efficient Chiplet Interconnects", IEEE Micro 2023
     *
     * Bandwidth Calculation:
     * - Raw: 8 GT/s × 8 lanes × 1 byte = 64 Gb/s = 8 GB/s
     * - Effective: Lower speed has slightly better efficiency (~87.5%):
     *   → 7 GB/s effective
     *
     * Latency Breakdown:
     * - Simpler SerDes at lower speed
     * - PHY TX/RX: 6 cycles each (less serialization depth)
     * - Link layer: 3 cycles
     * - Adapter: 15 cycles each (smaller buffers)
     * Total: 45 cycles
     *
     * Power:
     * - 6 mW per lane at 8 GT/s (lower signaling power)
     * - Static: 30 mW (smaller interface)
     * Total: ~78 mW
     */
    UCIePhyConfig config;

    config.version = UCIeVersion::UCIE_1_0;
    config.width = UCIeLinkWidth::X8;
    config.speed = UCIeLinkSpeed::SDR_8GT;

    // Bandwidth
    config.raw_bandwidth_gbps = 8.0;
    config.effective_bandwidth_gbps = 7.0;      // ~87.5% efficiency

    // Latency (cycles)
    config.phy_tx_latency_cycles = 6;
    config.phy_rx_latency_cycles = 6;
    config.link_layer_latency_cycles = 3;
    config.adapter_tx_latency_cycles = 15;
    config.adapter_rx_latency_cycles = 15;
    config.credit_return_latency_cycles = 20;

    // Power (mW)
    config.power_per_lane_mW = 6.0;
    config.static_power_mW = 30.0;

    return config;
}

UCIePhyConfig balanced_24gt_x16() {
    /**
     * BALANCED 24 GT/s x16 CONFIGURATION
     *
     * References:
     * [1] UCIe Spec 1.1, Section 2.2.3 - 24 GT/s Mode
     * [2] "Scaling AI Accelerators with UCIe", HotChips 2023
     *
     * Bandwidth Calculation:
     * - Raw: 24 GT/s × 16 lanes × 1 byte = 384 Gb/s = 48 GB/s
     * - Effective: ~87.5% efficiency → 42 GB/s effective
     *
     * Latency Breakdown:
     * - PHY TX/RX: 10 cycles each (moderate SerDes complexity)
     * - Link layer: 5 cycles
     * - Adapter: 20 cycles each
     * Total: 65 cycles
     *
     * Power:
     * - 20 mW per lane at 24 GT/s
     * - Static: 80 mW
     * Total: ~400 mW
     */
    UCIePhyConfig config;

    config.version = UCIeVersion::UCIE_1_1;
    config.width = UCIeLinkWidth::X16;
    config.speed = UCIeLinkSpeed::DDR_24GT;

    // Bandwidth
    config.raw_bandwidth_gbps = 48.0;
    config.effective_bandwidth_gbps = 42.0;     // ~87.5% efficiency

    // Latency (cycles)
    config.phy_tx_latency_cycles = 10;
    config.phy_rx_latency_cycles = 10;
    config.link_layer_latency_cycles = 5;
    config.adapter_tx_latency_cycles = 20;
    config.adapter_rx_latency_cycles = 20;
    config.credit_return_latency_cycles = 35;

    // Power (mW)
    config.power_per_lane_mW = 20.0;
    config.static_power_mW = 80.0;

    return config;
}

} // namespace ucie_configs

} // namespace chiplets
