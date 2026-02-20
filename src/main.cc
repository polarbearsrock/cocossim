/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/Frontend.h"
#include "frontends/standard/StandardLayer.h"
#include "frontends/standard/StandardParser.h"
#include "units/standard/SysArray.h"
#include "EnergyModel.h"

#include "memory.h"
#include "buffers/BufferConfig.h"
#include "buffers/BufferHierarchy.h"
#include <chrono>
#include <iomanip>

std::string layer_file;
std::string ofile;

using namespace frontend::standard;

using MyArchParser = StandardParser;
using MyLayerParser = StandardLayer;

int main(int argc, char **argv) {
  MyArchParser archParser(argc, argv);
  MyLayerParser layerParser;

  auto t1 = std::chrono::high_resolution_clock::now();
  mem::setup();

  // Parse arguments first (sets buffer_size_bytes from -buf_mb arg)
  Arch *arch = archParser.make_arch();

  // Initialize buffer hierarchy using buffer_size_bytes (now set by parser)
  auto buffer_config = buffers::configs::create_simple_two_level(
    buffer_size_bytes,
    "dramsim3/configs/HBM2_8Gb_x128.ini"
  );
  global_buffer_hierarchy = new buffers::BufferHierarchy(buffer_config);

  global_buffer_hierarchy->print_hierarchy(std::cout);

  // Assign buffers to states after hierarchy is created
  arch->assign_buffers();

#ifdef VCD
  vcd = fopen("out.vcd", "w");
#endif

  arch->init_waveforms();

  std::string line;
  std::vector<LayerConfig> layer_configs;

  std::ifstream layer_stream(layer_file);
  if (!layer_stream.is_open()) {
    throw std::runtime_error("Error: Could not open layer configuration file: " + layer_file);
  }

  // Parse layer configuration file (format: layer_name dim1 dim2 ... dim8)
  int n_layers = 0;
  while (std::getline(layer_stream, line)) {
    const char *nt_buff = line.c_str();
    LayerConfig l_config;
    char nm[64] = {0};
    std::cout << "processing " << line << std::endl;
    
    // Parse up to 8 dimensions per layer
    std::vector<int> dims(8);
    int successes = sscanf(nt_buff, "%s %d %d %d %d %d %d %d %d",
                           nm,
                           &dims[0], &dims[1], &dims[2], &dims[3],
                           &dims[4], &dims[5], &dims[6], &dims[7]);
    if (successes < 2) {
      std::cerr << "failed with '" << nt_buff << "'" << std::endl;
      throw std::exception();
    }
    
    l_config.layer_type = std::string(nm);
    int n_dims = successes - 1;
    dims.resize(n_dims);
    l_config.dimensions = dims;
    layer_configs.push_back(l_config);
    n_layers++;
  }
  layer_stream.close();

  // Setup multi-period simulation with time-based job enqueuing
  TimeBasedEnqueue time_enqueues;
  uint64_t t = 0;
  auto dt = 30000000;  // Time between periods
  std::vector<Job *> period_jobs[periods];
  
  std::cout << "Period: " << periods << std::endl;
  for (int i = 0; i < periods; ++i) {
    // Create jobs for each thread in this period
    for (int j = 0; j < n_threads; ++j) {
      auto network = layerParser.make_layers(layer_configs);
      for (auto &layer: network) {
        period_jobs[i].insert(period_jobs[i].end(), layer.first.begin(), layer.first.end());
      }
      alloc_task_idx++;
    }
    time_enqueues.enqueue_at(t, &period_jobs[i]);
    t += dt;

    std::cout << "Jobs for Period " << i << ":" << std::endl;
    for (auto *job: period_jobs[i]) {
      job->printDetails();
    }
  }

  jobs_to_dot(period_jobs[0]);

  mem::mem_sys->ResetStats();

  auto res = arch->get_cycles(time_enqueues);
  FILE *f = fopen(ofile.c_str(), "w");
  for (int p = 0; p < periods; ++p) {
    fprintf(f, "Cycles %llu\n", res[p].cycles);
    for (int i = 0; i < arch->states.size(); ++i) {
      fprintf(f, "%s %f\n", arch->states[i]->get_ty_string().c_str(), res[p].pct_active[i]);
    }
  }


  uint64_t last_cycles = res[periods - 1].cycles;
  auto lc = (double) last_cycles;
  auto expected_c = (double)res[0].cycles;
  double ratio = lc / expected_c;
  printf("Drain Ratio: %f\n", ratio);

  fclose(f);
  mem::mem_sys->PrintEpochStats();

  // Print buffer statistics
  if (global_buffer_hierarchy != nullptr) {
    std::cout << "\n=== Buffer Access Statistics ===\n";
    for (int i = 0; i < global_buffer_hierarchy->get_num_levels(); ++i) {
      const auto* level = global_buffer_hierarchy->get_level(i);
      if (level) {
        level->get_stats().print_summary(std::cout, level->get_config().name);
      }
    }
    global_buffer_hierarchy->print_utilization(std::cout);
  }

  // Print activation vs weight memory statistics
  std::cout << "\n=== Memory Read Breakdown (Activation vs Weight) ===\n";
  uint64_t total_activation_bytes = 0;
  uint64_t total_weight_bytes = 0;
  for (int i = 0; i < arch->states.size(); ++i) {
    auto* sa_state = dynamic_cast<SystolicArray::SysArrayState*>(arch->states[i]);
    if (sa_state) {
      uint64_t act_bytes = sa_state->get_activation_bytes_read();
      uint64_t wgt_bytes = sa_state->get_weight_bytes_read();
      total_activation_bytes += act_bytes;
      total_weight_bytes += wgt_bytes;
      std::cout << "  Core " << i << ": activation=" << act_bytes
                << " bytes, weight=" << wgt_bytes << " bytes\n";
    }
  }
  std::cout << "  TOTAL: activation=" << total_activation_bytes
            << " bytes, weight=" << total_weight_bytes << " bytes\n";
  if (arch_config.shared_weights) {
    std::cout << "  (Shared weight streaming enabled - only core 0 fetches weights from DRAM)\n";
  }

  // Print effective array utilization (accounts for partial filling)
  std::cout << "\n=== Systolic Array Effective Utilization ===\n";
  double total_effective_util = 0.0;
  int sa_count = 0;
  for (int i = 0; i < arch->states.size(); ++i) {
    auto* sa_state = dynamic_cast<SystolicArray::SysArrayState*>(arch->states[i]);
    if (sa_state) {
      double eff_util = sa_state->get_effective_utilization();
      total_effective_util += eff_util;
      sa_count++;
      std::cout << "  Core " << i << ": " << (eff_util * 100.0) << "% effective utilization"
                << " (active MAC-cycles: " << sa_state->get_active_mac_cycles()
                << " / " << sa_state->get_total_mac_capacity_cycles() << ")\n";
    }
  }
  if (sa_count > 0) {
    std::cout << "  Average: " << (total_effective_util / sa_count * 100.0) << "% effective utilization\n";
  }

  // Energy and Power Analysis
  std::cout << "\n=== Energy and Power Analysis ===\n";

  // Initialize energy model (7nm, frequency from command line)
  double freq_mhz = freq_sa * 1000.0;  // Convert GHz to MHz
  EnergyModel energy_model(TechnologyNode::NM_7, freq_mhz);
  std::cout << "  Technology: " << energy_model.get_technology_string()
            << " @ " << freq_mhz << " MHz\n";

  // Calculate total MACs from layer configurations (M * K * N per matmul)
  // Note: get_active_mac_cycles() tracks array utilization, not actual operations
  uint64_t total_macs = 0;
  for (const auto& lc : layer_configs) {
    if (lc.layer_type == "Matmul" && lc.dimensions.size() >= 3) {
      uint64_t M = lc.dimensions[0];
      uint64_t K = lc.dimensions[1];
      uint64_t N = lc.dimensions[2];
      total_macs += M * K * N;
    }
  }
  uint64_t total_flops = 2 * total_macs;  // Each MAC = 1 multiply + 1 add

  // Get SRAM statistics from buffer hierarchy
  uint64_t sram_read_bytes = 0;
  uint64_t sram_write_bytes = 0;
  if (global_buffer_hierarchy != nullptr) {
    const auto* sram_level = global_buffer_hierarchy->get_level(0);
    if (sram_level) {
      sram_read_bytes = sram_level->get_stats().bytes_read;
      sram_write_bytes = sram_level->get_stats().bytes_written;
    }
  }

  // Parse DRAM energy from dramsim3.txt
  DRAMSim3Stats dram_stats = parse_dramsim3_output("dramsim3.txt");

  // Calculate energy components using precision based on data_type_width
  ComputePrecision precision = get_precision_from_data_width(data_type_width);
  double compute_energy = energy_model.compute_mac_energy_mj(total_macs, precision);
  double sram_read_energy = energy_model.sram_read_energy_mj(sram_read_bytes);
  double sram_write_energy = energy_model.sram_write_energy_mj(sram_write_bytes);
  double sram_total_energy = sram_read_energy + sram_write_energy;
  double control_energy = energy_model.control_logic_energy_mj(res[0].cycles);
  double dram_energy = dram_stats.total_energy_mj();

  double total_energy = compute_energy + sram_total_energy + dram_energy + control_energy;

  // Calculate power (energy / time)
  double time_ms = (double)res[0].cycles / (freq_mhz * 1000.0);  // cycles / (MHz * 1000) = ms
  double total_power_w = total_energy / time_ms;  // mJ / ms = W

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n  Energy Breakdown:\n";
  std::cout << "    Compute (MACs):    " << compute_energy << " mJ ("
            << (compute_energy/total_energy*100) << "%)\n";
  std::cout << "    SRAM Read:         " << sram_read_energy << " mJ\n";
  std::cout << "    SRAM Write:        " << sram_write_energy << " mJ\n";
  std::cout << "    SRAM Total:        " << sram_total_energy << " mJ ("
            << (sram_total_energy/total_energy*100) << "%)\n";
  std::cout << "    DRAM (HBM):        " << dram_energy << " mJ ("
            << (dram_energy/total_energy*100) << "%)\n";
  std::cout << "    Control Logic:     " << control_energy << " mJ\n";
  std::cout << "    ─────────────────────────────\n";
  std::cout << "    TOTAL:             " << total_energy << " mJ\n";

  std::cout << "\n  Performance:\n";
  std::cout << "    Cycles:            " << res[0].cycles << "\n";
  std::cout << "    Latency:           " << time_ms << " ms\n";
  std::cout << "    Total MACs:        " << total_macs << " (" << (total_macs / 1e9) << " GMACs)\n";
  std::cout << "    Total FLOPs:       " << total_flops << " (" << (total_flops / 1e9) << " GFLOPs)\n";
  std::cout << "    Throughput:        " << (total_flops / time_ms / 1e9) << " TFLOPS\n";

  std::cout << "\n  Power:\n";
  std::cout << "    Average Power:     " << total_power_w << " W\n";
  std::cout << "    Energy Efficiency: " << (total_flops / total_energy / 1e9) << " TFLOPS/W\n";

  // Cleanup
  if (global_buffer_hierarchy != nullptr) {
    delete global_buffer_hierarchy;
    global_buffer_hierarchy = nullptr;
  }

#ifdef VCD
  fclose(vcd);
#endif
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "\nSimulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}
