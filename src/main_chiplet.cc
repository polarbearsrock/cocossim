/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "chiplets/ChipletArch.h"
#include "chiplets/UCIeConfig.h"
#include "chiplets/TensorPartition.h"
#include "EnergyModel.h"
#include "global.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>

using namespace chiplets;

// Global configuration
struct ChipletConfig {
    int num_chiplets = 4;
    int sa_sz = 64;                    // Systolic array size per chiplet
    double freq_ghz = 1.0;
    int topology = 0;                  // 0=ring, 1=mesh, 2=torus
    int ucie_speed = 1;                // 0=8GT, 1=16GT, 2=24GT, 3=32GT
    uint64_t memory_per_chiplet_gb = 8;
    std::string layer_file;
    std::string output_file;
};

void print_help() {
    std::cerr << "perf_model_chiplet - Multi-Chiplet Architecture Simulator\n\n";
    std::cerr << "Usage: perf_model_chiplet [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  -c <int>      Number of chiplets (default: 4)\n";
    std::cerr << "  -sa_sz <int>  Systolic array size per chiplet (default: 64)\n";
    std::cerr << "  -f <float>    Frequency in GHz (default: 1.0)\n";
    std::cerr << "  -topo <int>   Topology: 0=ring, 1=mesh, 2=torus (default: 0)\n";
    std::cerr << "  -ucie <int>   UCIe speed: 0=8GT, 1=16GT, 2=24GT, 3=32GT (default: 1)\n";
    std::cerr << "  -mem <int>    Memory per chiplet in GB (default: 8)\n";
    std::cerr << "  -i <file>     Layer configuration file\n";
    std::cerr << "  -o <file>     Output file\n";
    std::cerr << "  -h            Show this help\n";
}

ChipletConfig parse_args(int argc, char** argv) {
    ChipletConfig config;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            config.num_chiplets = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-sa_sz") == 0 && i + 1 < argc) {
            config.sa_sz = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            config.freq_ghz = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-topo") == 0 && i + 1 < argc) {
            config.topology = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ucie") == 0 && i + 1 < argc) {
            config.ucie_speed = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-mem") == 0 && i + 1 < argc) {
            config.memory_per_chiplet_gb = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.layer_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            print_help();
            exit(0);
        }
    }

    return config;
}

TopologyType get_topology_type(int topo) {
    switch (topo) {
        case 0: return TopologyType::RING;
        case 1: return TopologyType::MESH_2D;
        case 2: return TopologyType::TORUS_2D;
        default: return TopologyType::RING;
    }
}

UCIePhyConfig get_ucie_config(int speed) {
    switch (speed) {
        case 0: return ucie_configs::low_power_8gt_x8();
        case 1: return ucie_configs::standard_16gt_x16();
        case 2: return ucie_configs::balanced_24gt_x16();
        case 3: return ucie_configs::high_bw_32gt_x32();
        default: return ucie_configs::standard_16gt_x16();
    }
}

struct LayerSpec {
    std::string type;
    std::vector<int> dims;
};

std::vector<LayerSpec> parse_layer_file(const std::string& filename) {
    std::vector<LayerSpec> layers;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open layer file: " << filename << std::endl;
        return layers;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        LayerSpec layer;
        iss >> layer.type;

        int dim;
        while (iss >> dim) {
            layer.dims.push_back(dim);
        }

        if (!layer.dims.empty()) {
            layers.push_back(layer);
        }
    }

    return layers;
}

int main(int argc, char** argv) {
    auto t_start = std::chrono::high_resolution_clock::now();

    ChipletConfig config = parse_args(argc, argv);

    if (config.layer_file.empty()) {
        std::cerr << "Error: Layer file required (-i)\n";
        print_help();
        return 1;
    }

    // Parse layers
    auto layers = parse_layer_file(config.layer_file);
    if (layers.empty()) {
        std::cerr << "Error: No layers parsed from file\n";
        return 1;
    }

    std::cout << "=== Multi-Chiplet Architecture Simulator ===\n\n";

    // Configuration
    std::cout << "Configuration:\n";
    std::cout << "  Chiplets: " << config.num_chiplets << "\n";
    std::cout << "  SA Size: " << config.sa_sz << "x" << config.sa_sz << " per chiplet\n";
    std::cout << "  Frequency: " << config.freq_ghz << " GHz\n";
    std::cout << "  Topology: " << (config.topology == 0 ? "Ring" :
                                    config.topology == 1 ? "Mesh" : "Torus") << "\n";
    std::cout << "  Memory: " << config.memory_per_chiplet_gb << " GB per chiplet\n";
    std::cout << "  Layers: " << layers.size() << "\n\n";

    // Create chiplet architecture
    TopologyType topo_type = get_topology_type(config.topology);
    UCIePhyConfig ucie_config = get_ucie_config(config.ucie_speed);

    ChipletArch arch(config.num_chiplets, topo_type, ucie_config);

    // Configure chiplets
    uint64_t compute_ops_per_cycle = config.sa_sz * config.sa_sz;  // MACs per cycle
    uint64_t memory_bytes = config.memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;
    uint64_t memory_bw_gbps = 400;  // HBM bandwidth

    for (int i = 0; i < config.num_chiplets; i++) {
        arch.configure_chiplet(i, compute_ops_per_cycle, memory_bytes, memory_bw_gbps);
    }

    // Process layers
    std::cout << "Processing layers:\n";
    uint64_t total_macs = 0;

    for (size_t i = 0; i < layers.size(); i++) {
        const auto& layer = layers[i];
        std::cout << "  Layer " << i << ": " << layer.type;
        for (int d : layer.dims) std::cout << " " << d;
        std::cout << "\n";

        // Calculate MACs for matmul: M * K * N
        if (layer.type == "Matmul" && layer.dims.size() >= 3) {
            uint64_t M = layer.dims[0];
            uint64_t K = layer.dims[1];
            uint64_t N = layer.dims[2];
            uint64_t layer_macs = M * K * N;
            total_macs += layer_macs;

            // Estimate compute cycles (distributed across chiplets)
            uint64_t compute_cycles = layer_macs / (config.num_chiplets * compute_ops_per_cycle);
            uint64_t data_size = (M * K + K * N + M * N) * 2;  // FP16

            arch.submit_job(layer.type, i, data_size, compute_cycles);
        }
    }

    // Run simulation
    std::cout << "\nRunning simulation...\n";
    arch.run_until_idle();

    // Get statistics
    const auto& stats = arch.get_stats();

    // Energy model
    double freq_mhz = config.freq_ghz * 1000.0;
    EnergyModel energy_model(TechnologyNode::NM_7, freq_mhz);

    // Calculate energy using precision based on data_type_width
    ComputePrecision precision = get_precision_from_data_width(data_type_width);
    double compute_energy = energy_model.compute_mac_energy_mj(total_macs, precision);
    double ucie_energy = stats.total_energy_mJ;  // From UCIe links
    double total_energy = compute_energy + ucie_energy;

    // Calculate timing
    double time_ms = (double)stats.total_cycles / (freq_mhz * 1000.0);
    double throughput_tops = (double)total_macs / time_ms / 1e9;
    double power_w = total_energy / time_ms;
    double efficiency = throughput_tops / power_w;

    // Print results
    stats.print_summary(std::cout);

    std::cout << "\n=== Energy and Power Analysis ===\n";
    std::cout << "  Technology: " << energy_model.get_technology_string()
              << " @ " << freq_mhz << " MHz\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n  Energy Breakdown:\n";
    std::cout << "    Compute (MACs):    " << compute_energy << " mJ\n";
    std::cout << "    UCIe Interconnect: " << ucie_energy << " mJ\n";
    std::cout << "    ─────────────────────────────\n";
    std::cout << "    TOTAL:             " << total_energy << " mJ\n";

    std::cout << "\n  Performance:\n";
    std::cout << "    Total Cycles:      " << stats.total_cycles << "\n";
    std::cout << "    Latency:           " << time_ms << " ms\n";
    std::cout << "    Total MACs:        " << total_macs << "\n";
    std::cout << "    Throughput:        " << throughput_tops << " TOPS\n";

    std::cout << "\n  Power:\n";
    std::cout << "    Average Power:     " << power_w << " W\n";
    std::cout << "    Energy Efficiency: " << efficiency << " TOPS/W\n";

    // Write output file
    if (!config.output_file.empty()) {
        std::ofstream out(config.output_file);
        out << "Cycles " << stats.total_cycles << "\n";
        out << "Latency_ms " << time_ms << "\n";
        out << "Total_MACs " << total_macs << "\n";
        out << "Throughput_TOPS " << throughput_tops << "\n";
        out << "Energy_mJ " << total_energy << "\n";
        out << "Power_W " << power_w << "\n";
        out << "Efficiency_TOPS_W " << efficiency << "\n";
        out.close();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    std::cout << "\nSimulation took " << duration.count() << " ms\n";

    return 0;
}
