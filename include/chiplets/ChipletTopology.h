/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef COCOSSIM_CHIPLETTOPOLOGY_H
#define COCOSSIM_CHIPLETTOPOLOGY_H

#include "chiplets/UCIeLink.h"
#include <vector>
#include <map>
#include <memory>
#include <string>

/**
 * Chiplet Topology Modeling
 *
 * Implements various chiplet interconnection topologies using UCIe links.
 *
 * REFERENCES:
 * [1] AMD MI300 Architecture - 9 chiplets in mesh configuration
 * [2] Intel Ponte Vecchio - Tile-based mesh with Foveros 3D
 * [3] "Topology Exploration for Chiplet-based Architectures", ISCA 2023
 * [4] "Network Topology Optimization for AI Accelerators", HPCA 2024
 */

namespace chiplets {

/**
 * Topology Type
 *
 * Common topologies used in chiplet-based systems:
 * - LINEAR: Simple daisy-chain (2-4 chiplets)
 * - RING: Circular connection (reduces diameter)
 * - MESH_2D: 2D mesh/grid (most common, e.g., AMD MI300)
 * - TORUS_2D: 2D torus (wrap-around edges)
 * - STAR: Central hub with spokes (I/O chiplet + compute)
 * - TREE: Hierarchical tree (memory-centric designs)
 * - CUSTOM: User-defined connectivity
 *
 * Reference: [3] Section 3 - Topology Comparison
 */
enum class TopologyType {
    LINEAR,          // 2-4 chiplets in a line
    RING,            // Circular connection
    MESH_2D,         // 2D grid (e.g., 2×2, 4×4)
    TORUS_2D,        // 2D torus with wrap-around
    STAR,            // Central hub + spokes
    TREE,            // Hierarchical tree
    FULLY_CONNECTED, // All-to-all (expensive but low diameter)
    CUSTOM           // User-defined
};

/**
 * Routing Algorithm
 *
 * Reference: [4] Section 4.2 - Routing for Chiplets
 *
 * Routing must be deadlock-free and efficient:
 * - XY/YX: Dimension-order routing (simple, deadlock-free)
 * - Adaptive: Can choose paths dynamically (better load balance)
 * - Shortest Path: Precomputed shortest paths (optimal latency)
 * - Source Routing: Source specifies entire path
 */
enum class RoutingAlgorithm {
    XY_ROUTING,      // Route X then Y (dimension-order)
    YX_ROUTING,      // Route Y then X (dimension-order)
    SHORTEST_PATH,   // Precomputed shortest paths
    ADAPTIVE,        // Dynamic path selection (load-aware)
    SOURCE_ROUTING,  // Source specifies full path
    CUSTOM           // User-defined
};

/**
 * Chiplet Position
 *
 * Physical position in 2D/3D space for mesh-like topologies
 */
struct ChipletPosition {
    int x;           // X coordinate (column)
    int y;           // Y coordinate (row)
    int z;           // Z coordinate (for 3D stacking)

    ChipletPosition() : x(0), y(0), z(0) {}
    ChipletPosition(int x, int y) : x(x), y(y), z(0) {}
    ChipletPosition(int x, int y, int z) : x(x), y(y), z(z) {}

    bool operator==(const ChipletPosition& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    int manhattan_distance(const ChipletPosition& other) const {
        return abs(x - other.x) + abs(y - other.y) + abs(z - other.z);
    }
};

/**
 * Chiplet Topology
 *
 * Manages the physical layout and connectivity of chiplets via UCIe links.
 */
class ChipletTopology {
public:
    ChipletTopology(TopologyType type, int num_chiplets);
    ~ChipletTopology() = default;

    // Configuration
    TopologyType get_type() const { return type_; }
    int get_num_chiplets() const { return num_chiplets_; }
    void set_routing_algorithm(RoutingAlgorithm algo) { routing_algo_ = algo; }

    // Topology Construction
    /**
     * Build topology with UCIe links
     *
     * Creates UCIe links between chiplets according to topology type.
     * Each link is bidirectional (implemented as two unidirectional links).
     */
    void build_topology(const UCIePhyConfig& default_phy_config);

    /**
     * Build custom topology
     *
     * For CUSTOM topology type, user specifies exact connectivity.
     * Format: vector of (src, dst, phy_config) tuples
     */
    void build_custom_topology(
        const std::vector<std::tuple<int, int, UCIePhyConfig>>& links
    );

    // Mesh-specific configuration
    void set_mesh_dimensions(int width, int height);
    void get_mesh_dimensions(int& width, int& height) const;

    // Chiplet Position Management
    void set_chiplet_position(int chiplet_id, const ChipletPosition& pos);
    ChipletPosition get_chiplet_position(int chiplet_id) const;

    // Connectivity Queries
    bool are_connected(int src, int dst) const;
    int get_hop_count(int src, int dst) const;
    std::vector<int> get_neighbors(int chiplet_id) const;

    // Routing
    /**
     * Get route between chiplets
     *
     * Returns sequence of chiplet IDs from src to dst.
     * Route depends on routing_algo_.
     *
     * Example: route from 0 to 3 in 2×2 mesh might be [0, 1, 3] or [0, 2, 3]
     */
    std::vector<int> get_route(int src, int dst) const;

    /**
     * Get UCIe links along a route
     *
     * Returns the actual UCIe link objects that packets traverse.
     */
    std::vector<UCIeLink*> get_links_on_route(const std::vector<int>& route) const;

    // Link Management
    UCIeLink* get_link(int src, int dst) const;
    const UCIeLink& get_link(int link_id) const;
    UCIeLink& get_link_mut(int link_id);
    int get_link_id(int src, int dst) const;
    const std::vector<std::unique_ptr<UCIeLink>>& get_all_links() const { return links_; }
    int get_num_links() const { return links_.size(); }

    // Statistics
    double get_average_hop_count() const;
    double get_network_diameter() const;  // Max hop count
    double get_bisection_bandwidth() const;  // Min-cut bandwidth

    // Simulation
    void tick(uint64_t current_cycle);
    void reset_all_stats();

    // Visualization
    void print_topology(std::ostream& os) const;
    std::string to_string() const;

private:
    TopologyType type_;
    int num_chiplets_;
    RoutingAlgorithm routing_algo_;

    // Mesh parameters
    int mesh_width_;
    int mesh_height_;

    // Chiplet positions (for mesh/torus)
    std::map<int, ChipletPosition> chiplet_positions_;

    // UCIe Links (bidirectional = 2 unidirectional)
    std::vector<std::unique_ptr<UCIeLink>> links_;

    // Connectivity matrix: [src][dst] -> link_id
    // -1 means no direct link
    std::vector<std::vector<int>> connectivity_matrix_;

    // Hop distance matrix: [src][dst] -> hop count
    std::vector<std::vector<int>> hop_distances_;

    // Helper functions
    void initialize_matrices();
    void build_linear_topology(const UCIePhyConfig& phy_config);
    void build_ring_topology(const UCIePhyConfig& phy_config);
    void build_mesh_2d_topology(const UCIePhyConfig& phy_config);
    void build_torus_2d_topology(const UCIePhyConfig& phy_config);
    void build_star_topology(const UCIePhyConfig& phy_config);
    void build_tree_topology(const UCIePhyConfig& phy_config);
    void build_fully_connected_topology(const UCIePhyConfig& phy_config);

    void add_bidirectional_link(int src, int dst, const UCIePhyConfig& phy_config);
    void compute_hop_distances();  // Floyd-Warshall or BFS

    // Routing helpers
    std::vector<int> route_xy(int src, int dst) const;
    std::vector<int> route_yx(int src, int dst) const;
    std::vector<int> route_shortest_path(int src, int dst) const;
};

/**
 * Topology Builder Utilities
 *
 * Convenience functions for creating common topologies
 */
namespace topologies {
    /**
     * Create 2-chiplet linear topology
     * Reference: AMD MI300A style (compute + I/O die)
     */
    std::unique_ptr<ChipletTopology> create_2chiplet_linear(
        const UCIePhyConfig& phy_config
    );

    /**
     * Create 4-chiplet ring topology
     * Reference: Common for small-scale tensor parallel
     */
    std::unique_ptr<ChipletTopology> create_4chiplet_ring(
        const UCIePhyConfig& phy_config
    );

    /**
     * Create 2×2 mesh (4 chiplets)
     * Reference: Balanced topology for hybrid parallelism
     */
    std::unique_ptr<ChipletTopology> create_2x2_mesh(
        const UCIePhyConfig& phy_config
    );

    /**
     * Create 4×4 mesh (16 chiplets)
     * Reference: Large-scale training clusters
     */
    std::unique_ptr<ChipletTopology> create_4x4_mesh(
        const UCIePhyConfig& phy_config
    );

    /**
     * Create star topology (1 hub + N spokes)
     * Reference: Memory-centric or I/O-centric designs
     */
    std::unique_ptr<ChipletTopology> create_star(
        int num_compute_chiplets,
        const UCIePhyConfig& phy_config
    );
}

} // namespace chiplets

#endif // COCOSSIM_CHIPLETTOPOLOGY_H
