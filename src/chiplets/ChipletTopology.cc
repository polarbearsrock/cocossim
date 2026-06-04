/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/ChipletTopology.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>
#include <cmath>
#include <limits>

namespace chiplets {

// ============================================================================
// ChipletTopology Implementation
// ============================================================================

ChipletTopology::ChipletTopology(TopologyType type, int num_chiplets)
    : type_(type)
    , num_chiplets_(num_chiplets)
    , routing_algo_(RoutingAlgorithm::SHORTEST_PATH)
    , mesh_width_(0)
    , mesh_height_(0)
{
    initialize_matrices();
}

void ChipletTopology::initialize_matrices() {
    // Initialize connectivity matrix (direct links)
    connectivity_matrix_.resize(num_chiplets_);
    for (int i = 0; i < num_chiplets_; i++) {
        connectivity_matrix_[i].resize(num_chiplets_, -1);  // -1 = no link
        connectivity_matrix_[i][i] = 0;  // 0 hops to self
    }

    // Initialize hop distance matrix
    hop_distances_.resize(num_chiplets_);
    for (int i = 0; i < num_chiplets_; i++) {
        hop_distances_[i].resize(num_chiplets_, std::numeric_limits<int>::max());
        hop_distances_[i][i] = 0;
    }
}

void ChipletTopology::set_mesh_dimensions(int width, int height) {
    mesh_width_ = width;
    mesh_height_ = height;
}

// ============================================================================
// Topology Building
// ============================================================================

void ChipletTopology::build_topology(const UCIePhyConfig& default_phy_config) {
    switch (type_) {
        case TopologyType::LINEAR:
            build_linear_topology(default_phy_config);
            break;
        case TopologyType::RING:
            build_ring_topology(default_phy_config);
            break;
        case TopologyType::MESH_2D:
            build_mesh_2d_topology(default_phy_config);
            break;
        case TopologyType::TORUS_2D:
            build_torus_2d_topology(default_phy_config);
            break;
        case TopologyType::STAR:
            build_star_topology(default_phy_config);
            break;
        case TopologyType::TREE:
            build_tree_topology(default_phy_config);
            break;
        case TopologyType::FULLY_CONNECTED:
            build_fully_connected_topology(default_phy_config);
            break;
        case TopologyType::CUSTOM:
            std::cerr << "Warning: Use build_custom_topology() for CUSTOM type\n";
            break;
    }

    compute_hop_distances();
}

void ChipletTopology::build_linear_topology(const UCIePhyConfig& phy_config) {
    /**
     * Linear Topology: 0 -- 1 -- 2 -- 3
     *
     * Properties:
     * - Simple, low cost
     * - High diameter (N-1)
     * - Used for small systems (2-4 chiplets)
     *
     * Example: AMD MI300A (compute + I/O die)
     */
    for (int i = 0; i < num_chiplets_ - 1; i++) {
        add_bidirectional_link(i, i + 1, phy_config);
        chiplet_positions_[i] = ChipletPosition(i, 0);
    }
    chiplet_positions_[num_chiplets_ - 1] = ChipletPosition(num_chiplets_ - 1, 0);
}

void ChipletTopology::build_ring_topology(const UCIePhyConfig& phy_config) {
    /**
     * Ring Topology: 0 -- 1
     *                 |    |
     *                 3 -- 2
     *
     * Properties:
     * - Reduces diameter vs linear
     * - Good for small-scale tensor parallel
     * - Diameter = N/2
     *
     * Used in: Ring AllReduce (Megatron-LM style)
     */
    for (int i = 0; i < num_chiplets_ - 1; i++) {
        add_bidirectional_link(i, i + 1, phy_config);
    }
    // Close the ring
    add_bidirectional_link(num_chiplets_ - 1, 0, phy_config);

    // Position chiplets in a circle for visualization
    for (int i = 0; i < num_chiplets_; i++) {
        double angle = 2.0 * M_PI * i / num_chiplets_;
        int x = static_cast<int>(100 * cos(angle));
        int y = static_cast<int>(100 * sin(angle));
        chiplet_positions_[i] = ChipletPosition(x, y);
    }
}

void ChipletTopology::build_mesh_2d_topology(const UCIePhyConfig& phy_config) {
    /**
     * 2D Mesh Topology (example 2×2):
     *     0 -- 1
     *     |    |
     *     2 -- 3
     *
     * Properties:
     * - Most common in practice (AMD MI300, Intel Ponte Vecchio)
     * - Scalable to large systems
     * - Diameter = 2×(sqrt(N) - 1)
     * - Good balance of cost and performance
     *
     * Reference: AMD MI300 uses 9 chiplets in 3×3 mesh
     */
    if (mesh_width_ == 0 || mesh_height_ == 0) {
        // Auto-determine mesh dimensions (as square as possible)
        mesh_width_ = static_cast<int>(sqrt(num_chiplets_));
        mesh_height_ = (num_chiplets_ + mesh_width_ - 1) / mesh_width_;
    }

    int chiplet_id = 0;
    for (int y = 0; y < mesh_height_; y++) {
        for (int x = 0; x < mesh_width_; x++) {
            if (chiplet_id >= num_chiplets_) break;

            chiplet_positions_[chiplet_id] = ChipletPosition(x, y);

            // East link
            if (x < mesh_width_ - 1 && chiplet_id + 1 < num_chiplets_) {
                add_bidirectional_link(chiplet_id, chiplet_id + 1, phy_config);
            }

            // South link
            if (y < mesh_height_ - 1 && chiplet_id + mesh_width_ < num_chiplets_) {
                add_bidirectional_link(chiplet_id, chiplet_id + mesh_width_, phy_config);
            }

            chiplet_id++;
        }
    }
}

void ChipletTopology::build_torus_2d_topology(const UCIePhyConfig& phy_config) {
    /**
     * 2D Torus: Mesh with wrap-around edges
     *
     * Properties:
     * - Lower diameter than mesh
     * - More links (more cost)
     * - Better bisection bandwidth
     * - Diameter = 2×(sqrt(N)/2)
     */
    // First build mesh
    build_mesh_2d_topology(phy_config);

    // Add wrap-around links
    // Horizontal wrap
    for (int y = 0; y < mesh_height_; y++) {
        int left = y * mesh_width_;
        int right = left + mesh_width_ - 1;
        if (left < num_chiplets_ && right < num_chiplets_) {
            add_bidirectional_link(left, right, phy_config);
        }
    }

    // Vertical wrap
    for (int x = 0; x < mesh_width_; x++) {
        int top = x;
        int bottom = x + mesh_width_ * (mesh_height_ - 1);
        if (top < num_chiplets_ && bottom < num_chiplets_) {
            add_bidirectional_link(top, bottom, phy_config);
        }
    }
}

void ChipletTopology::build_star_topology(const UCIePhyConfig& phy_config) {
    /**
     * Star Topology:
     *       1
     *       |
     *   2 - 0 - 3
     *       |
     *       4
     *
     * Chiplet 0 is hub (often I/O or memory die)
     * All others connect only to hub
     *
     * Properties:
     * - Centralized (good for I/O-centric designs)
     * - Hub can be bottleneck
     * - Diameter = 2 (always 2 hops except to/from hub)
     *
     * Used in: Some Intel designs with I/O die
     */
    if (num_chiplets_ < 2) {
        std::cerr << "Star topology requires at least 2 chiplets\n";
        return;
    }

    // Chiplet 0 is hub at center
    chiplet_positions_[0] = ChipletPosition(0, 0);

    // Connect all others to hub
    for (int i = 1; i < num_chiplets_; i++) {
        add_bidirectional_link(0, i, phy_config);

        // Position spokes in a circle
        double angle = 2.0 * M_PI * (i - 1) / (num_chiplets_ - 1);
        int x = static_cast<int>(100 * cos(angle));
        int y = static_cast<int>(100 * sin(angle));
        chiplet_positions_[i] = ChipletPosition(x, y);
    }
}

void ChipletTopology::build_tree_topology(const UCIePhyConfig& phy_config) {
    /**
     * Binary Tree Topology:
     *         0
     *       /   \
     *      1     2
     *     / \   / \
     *    3  4  5  6
     *
     * Properties:
     * - Hierarchical structure
     * - Good for memory hierarchies
     * - Diameter = 2×log2(N)
     * - Root can be bottleneck
     */
    for (int i = 0; i < num_chiplets_; i++) {
        int left_child = 2 * i + 1;
        int right_child = 2 * i + 2;

        if (left_child < num_chiplets_) {
            add_bidirectional_link(i, left_child, phy_config);
        }
        if (right_child < num_chiplets_) {
            add_bidirectional_link(i, right_child, phy_config);
        }

        // Position by tree level
        int level = static_cast<int>(log2(i + 1));
        int pos_in_level = i - (static_cast<int>(pow(2, level)) - 1);
        chiplet_positions_[i] = ChipletPosition(pos_in_level * 10, level * 10);
    }
}

void ChipletTopology::build_fully_connected_topology(const UCIePhyConfig& phy_config) {
    /**
     * Fully Connected: All-to-all connections
     *
     * Properties:
     * - Lowest latency (1 hop)
     * - Highest cost (N×(N-1) links)
     * - Not practical for >8 chiplets
     * - Good for small, latency-critical systems
     */
    for (int i = 0; i < num_chiplets_; i++) {
        for (int j = i + 1; j < num_chiplets_; j++) {
            add_bidirectional_link(i, j, phy_config);
        }
        chiplet_positions_[i] = ChipletPosition(i, 0);
    }
}

void ChipletTopology::build_custom_topology(
    const std::vector<std::tuple<int, int, UCIePhyConfig>>& link_specs
) {
    type_ = TopologyType::CUSTOM;

    for (const auto& [src, dst, phy_config] : link_specs) {
        if (src >= num_chiplets_ || dst >= num_chiplets_) {
            std::cerr << "Invalid chiplet ID in custom topology\n";
            continue;
        }
        add_bidirectional_link(src, dst, phy_config);
    }

    compute_hop_distances();
}

// ============================================================================
// Link Management
// ============================================================================

void ChipletTopology::add_bidirectional_link(int src, int dst, const UCIePhyConfig& phy_config) {
    int link_id = links_.size();

    // Forward link (src → dst)
    auto link_fwd = std::make_unique<UCIeLink>(link_id, src, dst, phy_config);
    connectivity_matrix_[src][dst] = link_id;
    links_.push_back(std::move(link_fwd));

    // Reverse link (dst → src)
    link_id = links_.size();
    auto link_rev = std::make_unique<UCIeLink>(link_id, dst, src, phy_config);
    connectivity_matrix_[dst][src] = link_id;
    links_.push_back(std::move(link_rev));
}

UCIeLink* ChipletTopology::get_link(int src, int dst) const {
    int link_id = connectivity_matrix_[src][dst];
    if (link_id < 0 || link_id >= static_cast<int>(links_.size())) {
        return nullptr;
    }
    return links_[link_id].get();
}

const UCIeLink& ChipletTopology::get_link(int link_id) const {
    if (link_id < 0 || link_id >= static_cast<int>(links_.size())) {
        throw std::out_of_range("Invalid link ID: " + std::to_string(link_id));
    }
    return *links_[link_id];
}

UCIeLink& ChipletTopology::get_link_mut(int link_id) {
    if (link_id < 0 || link_id >= static_cast<int>(links_.size())) {
        throw std::out_of_range("Invalid link ID: " + std::to_string(link_id));
    }
    return *links_[link_id];
}

int ChipletTopology::get_link_id(int src, int dst) const {
    if (src < 0 || src >= num_chiplets_ || dst < 0 || dst >= num_chiplets_) {
        return -1;
    }
    return connectivity_matrix_[src][dst];
}

// ============================================================================
// Routing
// ============================================================================

void ChipletTopology::compute_hop_distances() {
    /**
     * Floyd-Warshall Algorithm
     * Computes shortest path distances between all pairs
     *
     * Complexity: O(N³) but runs once at initialization
     */
    // Initialize with direct connections
    for (int i = 0; i < num_chiplets_; i++) {
        for (int j = 0; j < num_chiplets_; j++) {
            if (i == j) {
                hop_distances_[i][j] = 0;
            } else if (connectivity_matrix_[i][j] >= 0) {
                hop_distances_[i][j] = 1;  // Direct link
            } else {
                hop_distances_[i][j] = std::numeric_limits<int>::max();
            }
        }
    }

    // Floyd-Warshall
    for (int k = 0; k < num_chiplets_; k++) {
        for (int i = 0; i < num_chiplets_; i++) {
            for (int j = 0; j < num_chiplets_; j++) {
                if (hop_distances_[i][k] != std::numeric_limits<int>::max() &&
                    hop_distances_[k][j] != std::numeric_limits<int>::max()) {
                    hop_distances_[i][j] = std::min(
                        hop_distances_[i][j],
                        hop_distances_[i][k] + hop_distances_[k][j]
                    );
                }
            }
        }
    }
}

std::vector<int> ChipletTopology::get_route(int src, int dst) const {
    if (src == dst) {
        return {src};
    }

    switch (routing_algo_) {
        case RoutingAlgorithm::XY_ROUTING:
            return route_xy(src, dst);
        case RoutingAlgorithm::YX_ROUTING:
            return route_yx(src, dst);
        case RoutingAlgorithm::SHORTEST_PATH:
            return route_shortest_path(src, dst);
        default:
            return route_shortest_path(src, dst);
    }
}

std::vector<int> ChipletTopology::route_xy(int src, int dst) const {
    /**
     * XY Routing (Dimension-Order)
     * - Route along X dimension first, then Y
     * - Deadlock-free in mesh/torus
     * - Deterministic (always same path)
     */
    if (type_ != TopologyType::MESH_2D && type_ != TopologyType::TORUS_2D) {
        return route_shortest_path(src, dst);  // Fallback
    }

    auto src_pos = chiplet_positions_.at(src);
    auto dst_pos = chiplet_positions_.at(dst);

    std::vector<int> route;
    route.push_back(src);

    int current = src;
    auto current_pos = src_pos;

    // Route X direction
    while (current_pos.x != dst_pos.x) {
        if (current_pos.x < dst_pos.x) {
            current++;
            current_pos.x++;
        } else {
            current--;
            current_pos.x--;
        }
        route.push_back(current);
    }

    // Route Y direction
    while (current_pos.y != dst_pos.y) {
        if (current_pos.y < dst_pos.y) {
            current += mesh_width_;
            current_pos.y++;
        } else {
            current -= mesh_width_;
            current_pos.y--;
        }
        route.push_back(current);
    }

    return route;
}

std::vector<int> ChipletTopology::route_yx(int src, int dst) const {
    if (type_ != TopologyType::MESH_2D && type_ != TopologyType::TORUS_2D) {
        return route_shortest_path(src, dst);
    }

    auto src_pos = chiplet_positions_.at(src);
    auto dst_pos = chiplet_positions_.at(dst);

    std::vector<int> route;
    route.push_back(src);

    int current = src;
    auto current_pos = src_pos;

    // Route Y direction first
    while (current_pos.y != dst_pos.y) {
        if (current_pos.y < dst_pos.y) {
            current += mesh_width_;
            current_pos.y++;
        } else {
            current -= mesh_width_;
            current_pos.y--;
        }
        route.push_back(current);
    }

    // Then route X direction
    while (current_pos.x != dst_pos.x) {
        if (current_pos.x < dst_pos.x) {
            current++;
            current_pos.x++;
        } else {
            current--;
            current_pos.x--;
        }
        route.push_back(current);
    }

    return route;
}

std::vector<int> ChipletTopology::route_shortest_path(int src, int dst) const {
    /**
     * BFS-based shortest path
     * Works for any topology
     */
    if (hop_distances_[src][dst] == std::numeric_limits<int>::max()) {
        return {};  // No path
    }

    std::vector<int> route;
    std::vector<int> parent(num_chiplets_, -1);
    std::queue<int> q;

    q.push(src);
    parent[src] = src;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == dst) break;

        for (int next = 0; next < num_chiplets_; next++) {
            if (connectivity_matrix_[current][next] >= 0 && parent[next] == -1) {
                parent[next] = current;
                q.push(next);
            }
        }
    }

    // Reconstruct path
    int current = dst;
    while (current != src) {
        route.push_back(current);
        current = parent[current];
    }
    route.push_back(src);
    std::reverse(route.begin(), route.end());

    return route;
}

std::vector<UCIeLink*> ChipletTopology::get_links_on_route(const std::vector<int>& route) const {
    std::vector<UCIeLink*> links;
    for (size_t i = 0; i < route.size() - 1; i++) {
        UCIeLink* link = get_link(route[i], route[i + 1]);
        if (link) {
            links.push_back(link);
        }
    }
    return links;
}

// ============================================================================
// Statistics
// ============================================================================

int ChipletTopology::get_hop_count(int src, int dst) const {
    return hop_distances_[src][dst];
}

double ChipletTopology::get_average_hop_count() const {
    double total = 0;
    int count = 0;
    for (int i = 0; i < num_chiplets_; i++) {
        for (int j = i + 1; j < num_chiplets_; j++) {
            if (hop_distances_[i][j] != std::numeric_limits<int>::max()) {
                total += hop_distances_[i][j];
                count++;
            }
        }
    }
    return count > 0 ? total / count : 0.0;
}

double ChipletTopology::get_network_diameter() const {
    int max_hops = 0;
    for (int i = 0; i < num_chiplets_; i++) {
        for (int j = 0; j < num_chiplets_; j++) {
            if (hop_distances_[i][j] != std::numeric_limits<int>::max()) {
                max_hops = std::max(max_hops, hop_distances_[i][j]);
            }
        }
    }
    return max_hops;
}

// ============================================================================
// Simulation
// ============================================================================

void ChipletTopology::tick(uint64_t current_cycle) {
    for (auto& link : links_) {
        link->tick(current_cycle);
    }
}

void ChipletTopology::reset_all_stats() {
    for (auto& link : links_) {
        link->reset_stats();
    }
}

// ============================================================================
// Visualization
// ============================================================================

void ChipletTopology::print_topology(std::ostream& os) const {
    os << "\n=== Chiplet Topology ===\n";
    os << "Type: " << static_cast<int>(type_) << "\n";
    os << "Chiplets: " << num_chiplets_ << "\n";
    os << "Links: " << links_.size() / 2 << " bidirectional\n";
    os << "Diameter: " << get_network_diameter() << " hops\n";
    os << "Avg Hops: " << std::fixed << std::setprecision(2)
       << get_average_hop_count() << "\n\n";

    os << "Connectivity Matrix:\n";
    os << "     ";
    for (int i = 0; i < num_chiplets_; i++) {
        os << std::setw(3) << i << " ";
    }
    os << "\n";
    for (int i = 0; i < num_chiplets_; i++) {
        os << std::setw(3) << i << ": ";
        for (int j = 0; j < num_chiplets_; j++) {
            if (connectivity_matrix_[i][j] >= 0) {
                os << " ✓  ";
            } else {
                os << " ·  ";
            }
        }
        os << "\n";
    }
}

void ChipletTopology::print_status(std::ostream& os, uint64_t current_cycle) const {
    os << "\n=== Chiplet Topology Link Status ===\n";

    const std::string type_names[] = {
        "LINEAR", "RING", "MESH_2D", "TORUS_2D", "STAR", "TREE", "FULLY_CONNECTED", "CUSTOM"
    };
    os << "  Topology: " << type_names[static_cast<int>(type_)] << "\n";
    os << "  Chiplets: " << num_chiplets_
       << " | Bidirectional links: " << (links_.size() / 2) << "\n\n";

    // Per-link stats (forward direction only — skip reverse links, which are odd indices).
    // Forward links are even-indexed: link 0 (0→1 fwd), link 1 (1→0 rev), link 2 (2→3 fwd), …
    double total_util = 0.0;
    double total_energy = 0.0;
    uint64_t total_bytes = 0;
    uint64_t total_pkts = 0;
    int link_count = 0;

    os << "  " << std::left
       << std::setw(12) << "Link"
       << std::setw(10) << "Util%"
       << std::setw(14) << "Bytes TX"
       << std::setw(10) << "Pkts TX"
       << std::setw(12) << "Energy(mJ)"
       << std::setw(10) << "Stalls\n";
    os << "  " << std::string(66, '-') << "\n";

    for (size_t i = 0; i < links_.size(); i += 2) {
        const UCIeLink& fwd = *links_[i];
        UCIeLinkStats ls = fwd.get_stats();
        ls.calculate_derived_stats(current_cycle);

        os << "  " << std::left
           << std::setw(12) << (std::to_string(fwd.get_src_chiplet()) + "->" +
                                std::to_string(fwd.get_dst_chiplet()))
           << std::setw(10) << std::fixed << std::setprecision(1)
                            << (ls.utilization * 100.0)
           << std::setw(14) << ls.bytes_transmitted
           << std::setw(10) << ls.packets_transmitted
           << std::setw(12) << std::setprecision(4) << ls.total_energy_mJ
           << std::setw(10) << (ls.credit_stalls + ls.buffer_full_stalls)
           << "\n";

        total_util   += ls.utilization;
        total_energy += ls.total_energy_mJ;
        total_bytes  += ls.bytes_transmitted;
        total_pkts   += ls.packets_transmitted;
        link_count++;
    }

    if (link_count > 0) {
        os << "\n  Totals:";
        os << "  avg util=" << std::fixed << std::setprecision(1)
           << (total_util / link_count * 100.0) << "%"
           << "  bytes=" << (total_bytes / 1024.0 / 1024.0) << " MB"
           << "  pkts=" << total_pkts
           << "  energy=" << std::setprecision(3) << total_energy << " mJ\n";
    }
}

// ============================================================================
// Convenience Builders
// ============================================================================

namespace topologies {

std::unique_ptr<ChipletTopology> create_2chiplet_linear(const UCIePhyConfig& phy_config) {
    auto topo = std::make_unique<ChipletTopology>(TopologyType::LINEAR, 2);
    topo->build_topology(phy_config);
    return topo;
}

std::unique_ptr<ChipletTopology> create_4chiplet_ring(const UCIePhyConfig& phy_config) {
    auto topo = std::make_unique<ChipletTopology>(TopologyType::RING, 4);
    topo->build_topology(phy_config);
    return topo;
}

std::unique_ptr<ChipletTopology> create_2x2_mesh(const UCIePhyConfig& phy_config) {
    auto topo = std::make_unique<ChipletTopology>(TopologyType::MESH_2D, 4);
    topo->set_mesh_dimensions(2, 2);
    topo->build_topology(phy_config);
    return topo;
}

std::unique_ptr<ChipletTopology> create_4x4_mesh(const UCIePhyConfig& phy_config) {
    auto topo = std::make_unique<ChipletTopology>(TopologyType::MESH_2D, 16);
    topo->set_mesh_dimensions(4, 4);
    topo->build_topology(phy_config);
    return topo;
}

std::unique_ptr<ChipletTopology> create_star(int num_compute_chiplets, const UCIePhyConfig& phy_config) {
    auto topo = std::make_unique<ChipletTopology>(TopologyType::STAR, num_compute_chiplets + 1);
    topo->build_topology(phy_config);
    return topo;
}

} // namespace topologies

} // namespace chiplets
