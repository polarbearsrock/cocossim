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
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <queue>

using namespace chiplets;

// Global configuration
struct ChipletConfig {
    int num_chiplets = 4;
    int sa_sz = 64;                    // Systolic array size per chiplet
    double freq_ghz = 1.0;
    int topology = 0;                  // 0=ring, 1=mesh, 2=torus
    int ucie_speed = 1;                // 0=8GT, 1=16GT, 2=24GT, 3=32GT
    uint64_t memory_per_chiplet_gb = 8;
    uint64_t max_cycles = 100000000ULL; // simulation cycle limit
    int dp = 1;                        // Data-parallelism degree (replicas)
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
    std::cerr << "  -dp <int>     Data-parallel degree / replicas (default: 1)\n";
    std::cerr << "  -max_cycles <int> Simulation cycle limit (default: 100000000)\n";
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
        } else if (strcmp(argv[i], "-max_cycles") == 0 && i + 1 < argc) {
            config.max_cycles = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-dp") == 0 && i + 1 < argc) {
            config.dp = std::stoi(argv[++i]);
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

// ── DAG format (# COCOSSim graph v2) ────────────────────────────────────────
// Line format: <id> <type> <dim1> [<dim2> ...] [-> <dep_id1> <dep_id2> ...]

struct DagNode {
    int id;
    std::string type;     // e.g. "Matmul", "LayerNorm", "Softmax", "Activation"
    std::string parallel; // "col", "head", or "" (unknown → fall back to heuristic)
    std::vector<int> dims;
    std::vector<int> deps;  // IDs of predecessor compute nodes
};

// Returns true if the file starts with "# COCOSSim graph v2"
bool is_dag_format(const std::string& filename) {
    std::ifstream f(filename);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        return line.find("# COCOSSim graph v2") != std::string::npos;
    }
    return false;
}

std::vector<DagNode> parse_dag_file(const std::string& filename) {
    std::vector<DagNode> nodes;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open DAG file: " << filename << "\n";
        return nodes;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        DagNode n;
        if (!(iss >> n.id)) continue;

        // Read "type" or "type:parallel" token
        std::string type_tok;
        if (!(iss >> type_tok)) continue;
        auto colon = type_tok.find(':');
        if (colon != std::string::npos) {
            n.type     = type_tok.substr(0, colon);
            n.parallel = type_tok.substr(colon + 1);
        } else {
            n.type = type_tok;
        }

        std::string tok;
        bool in_deps = false;
        while (iss >> tok) {
            if (tok == "->") { in_deps = true; continue; }
            int val = std::stoi(tok);
            if (in_deps) n.deps.push_back(val);
            else         n.dims.push_back(val);
        }
        nodes.push_back(n);
    }
    return nodes;
}

// Kahn's algorithm — returns node IDs in topological order
std::vector<int> topo_sort(const std::vector<DagNode>& nodes) {
    int N = (int)nodes.size();
    std::vector<int> in_degree(N, 0);
    // Build adjacency: dep → dependents
    std::vector<std::vector<int>> children(N);
    for (const auto& n : nodes) {
        for (int dep : n.deps) {
            children[dep].push_back(n.id);
            in_degree[n.id]++;
        }
    }
    std::queue<int> q;
    for (int i = 0; i < N; i++) {
        if (in_degree[i] == 0) q.push(i);
    }
    std::vector<int> order;
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        order.push_back(cur);
        for (int child : children[cur]) {
            if (--in_degree[child] == 0) q.push(child);
        }
    }
    if ((int)order.size() != N) {
        std::cerr << "Error: DAG has a cycle — cannot topologically sort\n";
    }
    return order;
}

int main(int argc, char** argv) {
    auto t_start = std::chrono::high_resolution_clock::now();

    ChipletConfig config = parse_args(argc, argv);

    if (config.layer_file.empty()) {
        std::cerr << "Error: Layer file required (-i)\n";
        print_help();
        return 1;
    }

    // Detect file format
    bool dag_mode = is_dag_format(config.layer_file);

    std::vector<DagNode> dag_nodes;
    std::vector<LayerSpec> layers;

    if (dag_mode) {
        dag_nodes = parse_dag_file(config.layer_file);
        if (dag_nodes.empty()) {
            std::cerr << "Error: No nodes parsed from DAG file\n";
            return 1;
        }
    } else {
        layers = parse_layer_file(config.layer_file);
        if (layers.empty()) {
            std::cerr << "Error: No layers parsed from file\n";
            return 1;
        }
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
    std::cout << "  Parallelism: tp=" << config.num_chiplets
              << "  dp=" << config.dp
              << "  total chiplets=" << (config.num_chiplets * config.dp) << "\n";
    if (dag_mode)
        std::cout << "  Nodes (DAG): " << dag_nodes.size() << "\n\n";
    else
        std::cout << "  Layers: " << layers.size() << "\n\n";

    // Create chiplet architecture
    TopologyType topo_type = get_topology_type(config.topology);
    UCIePhyConfig ucie_config = get_ucie_config(config.ucie_speed);

    const int tp           = config.num_chiplets;  // tensor-parallel degree
    const int dp           = config.dp;             // data-parallel degree
    const int total_chips  = tp * dp;

    ChipletArch arch(total_chips, topo_type, ucie_config);

    arch.set_sa_sz(config.sa_sz);

    uint64_t sa_ops_per_cycle = (uint64_t)config.sa_sz * config.sa_sz;
    uint64_t memory_bytes = config.memory_per_chiplet_gb * 1024ULL * 1024ULL * 1024ULL;

    for (int i = 0; i < total_chips; i++) {
        arch.configure_chiplet(i, sa_ops_per_cycle, memory_bytes, /*bw_gbps=*/0);
    }

    arch.init_dram("../dramsim3/configs/HBM2_8Gb_x128.ini", config.freq_ghz);

    uint64_t total_macs = 0;

    // ── Helper: submit one compute+AllReduce for a Matmul node ───────────────
    // parallel_tag: "col"  → column-parallel: split N across chiplets, AllReduce output
    //               "row"  → row-parallel: split K across chiplets, AllReduce output
    //               "head" → head-parallel: split M (heads) across chiplets, no AllReduce
    //               ""     → unknown: fall back to shape heuristic (N < tp → head)
    //
    // For DP: one job is submitted per replica. Replica r uses chiplets
    //   [r*tp, (r+1)*tp - 1]. Each replica processes M/dp rows independently.
    //   Replicas run in parallel because their chiplet sets are disjoint.
    auto submit_matmul = [&](uint64_t M, uint64_t K, uint64_t N, int node_idx,
                              const std::string& parallel_tag) {
        total_macs += M * K * N;

        // Determine parallelism mode
        bool head_parallel = false;
        bool row_parallel  = false;
        if (parallel_tag == "head") {
            head_parallel = true;
        } else if (parallel_tag == "row") {
            row_parallel = true;
        } else if (parallel_tag == "col") {
            // explicit column-parallel
        } else {
            // heuristic: if N is smaller than tp, treat as head-parallel
            head_parallel = (N < (uint64_t)tp);
        }

        // Per-replica M: DP splits the batch dimension.
        // If dp=1 this is a no-op.
        uint64_t M_replica = std::max(1ULL, M / (uint64_t)dp);

        // effective_tp and per-chip dims depend on parallelism mode.
        uint64_t effective_tp, M_per_chip, K_per_chip, N_per_chip;
        bool needs_allreduce;

        if (head_parallel) {
            effective_tp  = std::min((uint64_t)tp, std::max(1ULL, M_replica));
            M_per_chip    = std::max(1ULL, M_replica / effective_tp);
            K_per_chip    = K;
            N_per_chip    = N;
            needs_allreduce = false;
        } else if (row_parallel) {
            effective_tp  = std::min((uint64_t)tp, std::max(1ULL, K));
            M_per_chip    = M_replica;
            K_per_chip    = std::max(1ULL, K / effective_tp);
            N_per_chip    = N;
            needs_allreduce = (effective_tp > 1);
        } else {
            // col-parallel
            effective_tp  = std::min((uint64_t)tp, std::max(1ULL, N));
            M_per_chip    = M_replica;
            K_per_chip    = K;
            N_per_chip    = std::max(1ULL, N / effective_tp);
            needs_allreduce = (effective_tp > 1);
        }

        if (effective_tp < (uint64_t)tp) {
            const char* dim_name = head_parallel ? "M_replica" : (row_parallel ? "K" : "N");
            uint64_t dim_val     = head_parallel ? M_replica   : (row_parallel ? K   : N);
            std::cout << "    [note] node " << node_idx << " uses " << effective_tp
                      << "/" << tp << " chiplets per replica ("
                      << dim_name << "=" << dim_val << " < tp=" << tp << ")\n";
        }

        uint64_t weight_bytes     = K * N * 2;
        uint64_t activation_bytes = M_replica * K_per_chip * 2 * effective_tp;

        // Submit all replica compute jobs first, then all AllReduces.
        // This lets the scheduler see all compute jobs before any AllReduce
        // blocks the queue, so disjoint-chiplet replicas start in parallel.
        std::vector<std::vector<int>> replica_chip_sets(dp);
        for (int r = 0; r < dp; r++) {
            for (uint64_t c = 0; c < effective_tp; c++)
                replica_chip_sets[r].push_back(r * tp + (int)c);
        }

        for (int r = 0; r < dp; r++) {
            Job* compute_job = nullptr;
            if (M_per_chip == 1) {
                std::vector<std::pair<VectorUnit::VPUPhase, int>> phases = {
                    {VectorUnit::VPUPhase::REDUCE, 1}
                };
                auto* vpu_job = new VectorUnit::VecUnitJob(
                    (int)K_per_chip, (int)N_per_chip,
                    /*is_prebuffered=*/false, phases);
                vpu_job->core_id  = 0;
                vpu_job->task_idx = node_idx;
                vpu_job->rem_deps = 0;
                compute_job = vpu_job;
            } else {
                auto* sa_job = new SystolicArray::SysArrayJob(
                    (int)M_per_chip, (int)K_per_chip, (int)N_per_chip);
                sa_job->core_id  = 0;
                sa_job->task_idx = node_idx;
                sa_job->rem_deps = 0;
                compute_job = sa_job;
            }
            arch.submit_job("Matmul", node_idx,
                            weight_bytes + activation_bytes,
                            compute_job, replica_chip_sets[r]);
        }

        // AllReduces after all compute jobs are in the queue.
        if (needs_allreduce) {
            uint64_t out_bytes = M_replica * N * 2;
            uint64_t ar_bytes  = 2 * (effective_tp - 1) * out_bytes / effective_tp;
            for (int r = 0; r < dp; r++) {
                chiplets::CollectiveOperation ar;
                ar.type = chiplets::CollectiveOperation::Type::ALLREDUCE;
                ar.data_size_bytes = ar_bytes;
                ar.root_chiplet = replica_chip_sets[r][0];
                ar.participating_chiplets = replica_chip_sets[r];
                arch.submit_collective(ar);
            }
        }
    };

    // ── Helper: submit a lightweight VPU job for elementwise/norm/softmax ops ─
    // These are modeled as VPU REDUCE(K=reduction_dim, N=parallel_dim) jobs.
    // DRAM: only activation load+store, no weights.
    // Elementwise ops (LayerNorm, Softmax, Activation) are replicated across DP
    // replicas — each replica processes its own M/dp tokens independently.
    auto submit_vpu_elementwise = [&](const std::string& op_type,
                                       int K, int N, int node_idx) {
        int N_replica = std::max(1, N / dp);   // tokens per replica
        int N_per_chip = std::max(1, N_replica / tp);
        uint64_t data_size = (uint64_t)K * N_replica * 2 * 2;  // read + write, FP16

        for (int r = 0; r < dp; r++) {
            std::vector<int> replica_chips;
            for (int c = 0; c < tp; c++)
                replica_chips.push_back(r * tp + c);

            std::vector<std::pair<VectorUnit::VPUPhase, int>> phases = {
                {VectorUnit::VPUPhase::REDUCE, 1}
            };
            auto* vpu_job = new VectorUnit::VecUnitJob(
                K, N_per_chip,
                /*is_prebuffered=*/false, phases);
            vpu_job->core_id  = 0;
            vpu_job->task_idx = node_idx;
            vpu_job->rem_deps = 0;
            arch.submit_job(op_type, node_idx, data_size, vpu_job, replica_chips);
        }
    };

    if (dag_mode) {
        // ── DAG mode ─────────────────────────────────────────────────────────
        // Topological sort, then submit in order.
        // Each job uses all chiplets simultaneously (tensor parallelism).
        // Non-Matmul ops (LayerNorm, Softmax, Activation) are modeled as
        // lightweight VPU jobs over the activation tensor.
        std::cout << "Processing DAG nodes (topological order):\n";
        auto topo_order = topo_sort(dag_nodes);

        // Build an id→node lookup
        std::vector<const DagNode*> node_by_id(dag_nodes.size(), nullptr);
        for (const auto& n : dag_nodes) node_by_id[n.id] = &n;

        for (int node_id : topo_order) {
            const DagNode& n = *node_by_id[node_id];
            std::cout << "  Node " << n.id << " (" << n.type << ")";
            for (int d : n.dims) std::cout << " " << d;
            if (!n.deps.empty()) {
                std::cout << " -> deps:";
                for (int d : n.deps) std::cout << " " << d;
            }
            std::cout << "\n";

            if (n.type == "Matmul" && n.dims.size() >= 3) {
                submit_matmul(n.dims[0], n.dims[1], n.dims[2], n.id, n.parallel);

            } else if (n.type == "LayerNorm" && n.dims.size() >= 2) {
                // dims: [M, hidden]  — M tokens, reduce over hidden
                submit_vpu_elementwise("LayerNorm", n.dims[1], n.dims[0], n.id);

            } else if (n.type == "Softmax" && n.dims.size() >= 2) {
                // dims: [heads*batch, seq]  — reduce over seq for each head
                submit_vpu_elementwise("Softmax", n.dims[1], n.dims[0], n.id);

            } else if (n.type == "Activation" && n.dims.size() >= 1) {
                // dims: [total_elements]
                int total = n.dims[0];
                // Model as reduce over total/tp elements per chiplet
                submit_vpu_elementwise("Activation", std::max(1, total / tp), 1, n.id);

            } else if (n.type == "Conv" && n.dims.size() >= 5) {
                // dims: [B, C_in, H, W, C_out, k, s, p]
                // Map to Matmul: M = B*H_out*W_out, K = C_in*k*k, N = C_out
                int B    = n.dims[0];
                int C_in = n.dims[1];
                int H    = n.dims[2];
                int W    = n.dims[3];
                int C_out = n.dims[4];
                int k = (n.dims.size() >= 6) ? n.dims[5] : 1;
                int s = (n.dims.size() >= 7) ? n.dims[6] : 1;
                int p = (n.dims.size() >= 8) ? n.dims[7] : 0;
                int H_out = (H + 2 * p - k) / s + 1;
                int W_out = (W + 2 * p - k) / s + 1;
                uint64_t M = (uint64_t)B * H_out * W_out;
                uint64_t K = (uint64_t)C_in * k * k;
                uint64_t N = (uint64_t)C_out;
                submit_matmul(M, K, N, n.id, "col");

            } else {
                std::cout << "    (skipping unsupported op type: " << n.type << ")\n";
            }
        }
    } else {
        // ── Legacy sequential mode ────────────────────────────────────────────
        std::cout << "Processing layers:\n";
        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];
            std::cout << "  Layer " << i << ": " << layer.type;
            for (int d : layer.dims) std::cout << " " << d;
            std::cout << "\n";

            if (layer.type == "Matmul" && layer.dims.size() >= 3) {
                submit_matmul(layer.dims[0], layer.dims[1], layer.dims[2], (int)i, "");
            }
        }
    }

    // Run simulation — all jobs (compute + AllReduce) drain together
    std::cout << "\nRunning simulation...\n";
    arch.run_until_idle(config.max_cycles);

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
    arch.print_status(std::cout);

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
