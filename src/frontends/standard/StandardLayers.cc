/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "NNLayers.h"
#include "frontends/standard/StandardArch.h"
#include "frontends/standard/StandardLayer.h"
#include "global.h"
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace frontend::standard;

namespace {
using JobCreate_f = std::function<JobPair(const ArchConfig &, const LayerConfig &)>;

const std::vector<std::pair<VectorUnit::VPUPhase, int>> softmax_phases =
    {{VectorUnit::VPUPhase::BROADCAST, 1},
     {VectorUnit::VPUPhase::REDUCE, 1},
     {VectorUnit::VPUPhase::BROADCAST, 1}};

uint64_t checked_add(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    throw std::overflow_error("byte count exceeds 64-bit range");
  }
  return lhs + rhs;
}

int checked_int_product(std::initializer_list<int> factors, const std::string &what) {
  uint64_t product = 1;
  for (const int factor : factors) {
    if (factor <= 0) {
      throw std::invalid_argument(what + " dimensions must be positive");
    }
    product = checked_product({product, static_cast<uint64_t>(factor)});
  }
  if (product > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(what + " dimension exceeds the simulator's integer job limit");
  }
  return static_cast<int>(product);
}

void validate_arch(const ArchConfig &config) {
  if (config.n_cores <= 0 || config.sa_sz_allo <= 0 || config.vu_sz_allo <= 0) {
    throw std::invalid_argument("architecture core and unit sizes must be positive");
  }
}

void validate_positive_dimensions(const LayerConfig &config, const std::string &name) {
  for (const int dimension : config.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(name + " dimensions must be positive");
    }
  }
}

struct CoreSlice {
  int core;
  int width;
};

std::vector<CoreSlice> split_output_dimension(int n, int n_cores) {
  if (n <= 0 || n_cores <= 0) {
    throw std::invalid_argument("output dimension and core count must be positive");
  }
  std::vector<CoreSlice> slices;
  const int base = n / n_cores;
  const int remainder = n % n_cores;
  for (int core = 0; core < n_cores; ++core) {
    const int width = base + (core < remainder ? 1 : 0);
    if (width > 0) {
      slices.push_back({core, width});
    }
  }
  return slices;
}

Job *assign_sa_core(SystolicArray::SysArrayJob *job, int core) {
  job->core_id = core;
  return job;
}

void assign_vector_core(Job *job, const ArchConfig &config, size_t ordinal) {
  // StandardArch stores SA states first and vector states second.
  job->core_id = config.n_cores + static_cast<int>(ordinal % config.n_cores);
}

JobList create_output_stationary_jobs(const ArchConfig &config, int m, int k, int n) {
  JobList jobs;
  for (const auto slice : split_output_dimension(n, config.n_cores)) {
    for (int row = 0; row < m;) {
      const int m_tile = std::min(config.sa_sz_allo, m - row);
      jobs.push_back(assign_sa_core(new SystolicArray::SysArrayJob(m_tile, k, slice.width),
                                    slice.core));
      row += m_tile;
    }
  }
  return jobs;
}

bool weight_stationary_buffer_fits(int m, int k, int n, const ArchConfig &config) {
  const auto output_bytes = bytes_for_elements(
      checked_product({static_cast<uint64_t>(m), static_cast<uint64_t>(n),
                       static_cast<uint64_t>(batch_size)}));
  const auto activation_bytes = bytes_for_elements(
      checked_product({static_cast<uint64_t>(m),
                       static_cast<uint64_t>(std::min(k, config.sa_sz_allo)),
                       static_cast<uint64_t>(batch_size)}));
  return checked_add(output_bytes, activation_bytes) <= buffer_size_bytes;
}

template<typename Predicate>
int largest_fitting_prefix(int limit, Predicate fits) {
  int low = 0;
  int high = limit;
  while (low < high) {
    const int middle = low + (high - low + 1) / 2;
    if (fits(middle)) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return low;
}

JobList create_weight_stationary_jobs(const ArchConfig &config, int m, int k, int n) {
  JobList jobs;
  for (const auto slice : split_output_dimension(n, config.n_cores)) {
    const int max_m = largest_fitting_prefix(
        m, [&](int candidate_m) {
          return weight_stationary_buffer_fits(candidate_m, k, 1, config);
        });
    if (max_m == 0) {
      throw std::runtime_error(
          "buffer is too small for one weight-stationary activation row; increase -buffer_bytes");
    }

    for (int row = 0; row < m;) {
      const int m_chunk = std::min(max_m, m - row);
      const int max_n = largest_fitting_prefix(
          slice.width, [&](int candidate_n) {
            return weight_stationary_buffer_fits(m_chunk, k, candidate_n, config);
          });
      if (max_n == 0) {
        throw std::runtime_error(
            "buffer is too small for one weight-stationary output column; increase -buffer_bytes");
      }
      for (int col = 0; col < slice.width;) {
        const int n_chunk = std::min(max_n, slice.width - col);
        jobs.push_back(assign_sa_core(new SystolicArray::SysArrayJob(m_chunk, k, n_chunk),
                                      slice.core));
        col += n_chunk;
      }
      row += m_chunk;
    }
  }
  return jobs;
}

JobPair make_matmul_jobs(const ArchConfig &config, int m, int k, int n) {
  validate_arch(config);
  if (m <= 0 || k <= 0 || n <= 0) {
    throw std::invalid_argument("Matmul dimensions must be positive");
  }
  JobList jobs = config.ws ? create_weight_stationary_jobs(config, m, k, n)
                           : create_output_stationary_jobs(config, m, k, n);
  if (jobs.empty()) {
    throw std::runtime_error("Matmul lowering produced no jobs");
  }
  return {jobs, jobs};
}

std::tuple<int, int, int> parse_matmul_dimensions(const LayerConfig &config,
                                                  const std::string &name) {
  validate_positive_dimensions(config, name);
  if (config.dimensions.size() == 3) {
    return {config.dimensions[0], config.dimensions[1], config.dimensions[2]};
  }
  if (config.dimensions.size() == 4) {
    return {config.dimensions[1], config.dimensions[2],
            checked_int_product({config.dimensions[0], config.dimensions[3]}, name)};
  }
  throw std::invalid_argument(name + " expects M K N or groups M K N_per_group");
}

JobPair Matmul(const ArchConfig &config, const LayerConfig &layer) {
  const auto [m, k, n] = parse_matmul_dimensions(layer, "Matmul");
  return make_matmul_jobs(config, m, k, n);
}

JobPair Conv(const ArchConfig &config, const LayerConfig &layer) {
  if (layer.dimensions.size() < 5 || layer.dimensions.size() > 8) {
    throw std::invalid_argument(
        "Conv expects batch channels height width output_channels [kernel stride padding]");
  }
  const size_t positive_dimensions = std::min<size_t>(7, layer.dimensions.size());
  for (size_t index = 0; index < positive_dimensions; ++index) {
    if (layer.dimensions[index] <= 0) {
      throw std::invalid_argument("Conv dimensions other than padding must be positive");
    }
  }

  const int conv_batch = layer.dimensions[0];
  const int input_channels = layer.dimensions[1];
  const int input_height = layer.dimensions[2];
  const int input_width = layer.dimensions[3];
  const int output_channels = layer.dimensions[4];
  const int kernel_size = layer.dimensions.size() > 5 ? layer.dimensions[5] : 3;
  const int stride = layer.dimensions.size() > 6 ? layer.dimensions[6] : 1;
  const int padding = layer.dimensions.size() > 7 ? layer.dimensions[7] : 1;
  if (padding < 0) {
    throw std::invalid_argument("Conv padding must be non-negative");
  }

  const int64_t height_numerator = static_cast<int64_t>(input_height) +
                                   2LL * padding - kernel_size;
  const int64_t width_numerator = static_cast<int64_t>(input_width) +
                                  2LL * padding - kernel_size;
  if (height_numerator < 0 || width_numerator < 0) {
    throw std::invalid_argument("Conv kernel is larger than the padded input");
  }
  const int64_t output_height_64 = height_numerator / stride + 1;
  const int64_t output_width_64 = width_numerator / stride + 1;
  if (output_height_64 > std::numeric_limits<int>::max() ||
      output_width_64 > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Conv output extent exceeds the simulator's integer limit");
  }
  const int output_height = static_cast<int>(output_height_64);
  const int output_width = static_cast<int>(output_width_64);
  const int m = checked_int_product({conv_batch, output_height, output_width}, "Conv M");
  const int k = checked_int_product({input_channels, kernel_size, kernel_size}, "Conv K");

  std::cout << "Conv2GEMM: M=" << m << ", K=" << k << ", N=" << output_channels
            << std::endl;
  return make_matmul_jobs(config, m, k, output_channels);
}

JobPair MatmulAct(const ArchConfig &config, const LayerConfig &layer) {
  const auto [m, k, n] = parse_matmul_dimensions(layer, "MatmulAct");
  auto matmul = make_matmul_jobs(config, m, k, n);
  const int output_elements = checked_int_product({m, n}, "MatmulAct output");
  JobList activation = {new VectorUnit::VecUnitJob(
      1, output_elements, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};
  assign_vector_core(activation.front(), config, 0);
  connectJobLists(matmul.second, activation);
  return {matmul.first, activation};
}

JobPair ActMatmul(const ArchConfig &config, const LayerConfig &layer) {
  const auto [m, k, n] = parse_matmul_dimensions(layer, "ActMatmul");
  JobList activation = {new VectorUnit::VecUnitJob(
      1, checked_int_product({m, k}, "ActMatmul input"), true,
      {{VectorUnit::VPUPhase::BROADCAST, 1}})};
  assign_vector_core(activation.front(), config, 0);
  auto matmul = make_matmul_jobs(config, m, k, n);
  connectJobLists(activation, matmul.first);
  return {activation, matmul.second};
}

JobPair LayerNorm(const ArchConfig &config, const LayerConfig &layer) {
  validate_arch(config);
  validate_positive_dimensions(layer, "LayerNorm");
  int linear_dimension;
  int parallel_dimension = 1;
  switch (layer.dimensions.size()) {
    case 1:
      linear_dimension = layer.dimensions[0];
      break;
    case 2:
      parallel_dimension = layer.dimensions[0];
      linear_dimension = layer.dimensions[1];
      break;
    case 3:
      parallel_dimension = checked_int_product(
          {layer.dimensions[0], layer.dimensions[1]}, "LayerNorm parallel");
      if (layer.dimensions[2] % layer.dimensions[0] != 0) {
        throw std::invalid_argument("LayerNorm linear dimension is not divisible by group size");
      }
      linear_dimension = layer.dimensions[2] / layer.dimensions[0];
      break;
    default:
      throw std::invalid_argument("LayerNorm expects one, two, or three dimensions");
  }

  const auto elements_per_row = checked_product(
      {static_cast<uint64_t>(linear_dimension), static_cast<uint64_t>(batch_size)});
  const uint64_t rows_per_job_u64 = elements_fitting_in_bytes(buffer_size_bytes) /
                                    elements_per_row;
  if (rows_per_job_u64 == 0) {
    throw std::runtime_error("LayerNorm row does not fit in -buffer_bytes");
  }
  const int rows_per_job = static_cast<int>(
      std::min<uint64_t>(parallel_dimension, rows_per_job_u64));

  JobList jobs;
  for (int row = 0; row < parallel_dimension; row += rows_per_job) {
    auto *job = new VectorUnit::VecUnitJob(
        linear_dimension, std::min(rows_per_job, parallel_dimension - row), false,
        {{VectorUnit::VPUPhase::REDUCE, 1},
         {VectorUnit::VPUPhase::REDUCE, 4},
         {VectorUnit::VPUPhase::BROADCAST, 1}});
    assign_vector_core(job, config, jobs.size());
    jobs.push_back(job);
  }
  return {jobs, jobs};
}

JobPair Activation(const ArchConfig &config, const LayerConfig &layer) {
  validate_arch(config);
  validate_positive_dimensions(layer, "Activation");
  uint64_t remaining = 1;
  for (const int dimension : layer.dimensions) {
    remaining = checked_product({remaining, static_cast<uint64_t>(dimension)});
  }

  JobList jobs;
  while (remaining > 0) {
    const int chunk = static_cast<int>(std::min<uint64_t>(
        remaining, static_cast<uint64_t>(std::numeric_limits<int>::max())));
    auto *job = new VectorUnit::VecUnitJob(
        1, chunk, false, {{VectorUnit::VPUPhase::BROADCAST, 1}});
    assign_vector_core(job, config, jobs.size());
    jobs.push_back(job);
    remaining -= static_cast<uint64_t>(chunk);
  }
  return {jobs, jobs};
}

JobPair Softmax(const ArchConfig &config, const LayerConfig &layer) {
  validate_arch(config);
  validate_positive_dimensions(layer, "Softmax");
  int heads = 1;
  int sequence_length;
  if (layer.dimensions.size() == 1) {
    sequence_length = layer.dimensions[0];
  } else if (layer.dimensions.size() == 2) {
    heads = layer.dimensions[0];
    sequence_length = layer.dimensions[1];
  } else {
    throw std::invalid_argument("Softmax expects sequence_length or heads sequence_length");
  }

  const auto elements_per_row = checked_product(
      {static_cast<uint64_t>(sequence_length), static_cast<uint64_t>(batch_size)});
  const auto buffer_rows = elements_fitting_in_bytes(buffer_size_bytes) / elements_per_row;
  if (buffer_rows == 0) {
    throw std::runtime_error("one Softmax row does not fit in -buffer_bytes");
  }
  const int rows_per_job = static_cast<int>(std::min<uint64_t>(1024, buffer_rows));
  const uint64_t total_rows = checked_product(
      {static_cast<uint64_t>(heads), static_cast<uint64_t>(sequence_length)});

  JobList jobs;
  for (uint64_t row = 0; row < total_rows; row += static_cast<uint64_t>(rows_per_job)) {
    const int rows = static_cast<int>(
        std::min<uint64_t>(rows_per_job, total_rows - row));
    auto *job = new VectorUnit::VecUnitJob(sequence_length, rows, false, softmax_phases);
    assign_vector_core(job, config, jobs.size());
    jobs.push_back(job);
  }
  std::cout << "Softmax split into " << jobs.size() << " exact row chunk(s)" << std::endl;
  return {jobs, jobs};
}

JobPair SelfAttention(const ArchConfig &config, const LayerConfig &layer) {
  validate_positive_dimensions(layer, "SelfAttention");
  int heads = 1;
  int sequence_length;
  int input_dimension;
  int output_dimension;
  if (layer.dimensions.size() == 3) {
    sequence_length = layer.dimensions[0];
    input_dimension = layer.dimensions[1];
    output_dimension = layer.dimensions[2];
  } else if (layer.dimensions.size() == 4) {
    heads = layer.dimensions[0];
    sequence_length = layer.dimensions[1];
    input_dimension = layer.dimensions[2];
    output_dimension = layer.dimensions[3];
  } else {
    throw std::invalid_argument(
        "SelfAttention expects sequence input output or heads sequence input output");
  }
  if (output_dimension % heads != 0) {
    throw std::invalid_argument("SelfAttention output dimension must be divisible by head count");
  }
  const int head_dimension = output_dimension / heads;

  auto key_projection = make_matmul_jobs(config, sequence_length, input_dimension,
                                         output_dimension);
  auto query_projection = make_matmul_jobs(config, sequence_length, input_dimension,
                                           output_dimension);
  auto value_projection = make_matmul_jobs(config, sequence_length, input_dimension,
                                           output_dimension);
  auto scores = make_matmul_jobs(config, sequence_length, head_dimension,
                                 checked_int_product({heads, sequence_length},
                                                     "SelfAttention score output"));
  auto softmax = Softmax(config, LayerConfig("Softmax", {heads, sequence_length}));
  auto values = make_matmul_jobs(config, sequence_length, sequence_length,
                                 output_dimension);
  auto output_projection = make_matmul_jobs(config, sequence_length, output_dimension,
                                            input_dimension);

  connectJobs(key_projection, query_projection);
  connectJobs(query_projection, value_projection);
  connectJobs(value_projection, scores);
  connectJobs(scores, softmax);
  connectJobs(softmax, values);
  connectJobs(values, output_projection);
  return {key_projection.first, output_projection.second};
}

JobPair MultiHeadSelfAttention(const ArchConfig &config, const LayerConfig &layer) {
  if (layer.dimensions.size() != 4) {
    throw std::invalid_argument(
        "MultiHeadSelfAttention requires explicit heads sequence input output dimensions");
  }
  return SelfAttention(config, layer);
}

JobCreate_f getLayerLambda(const std::string &layer_type) {
  if (layer_type == "Matmul") return Matmul;
  if (layer_type == "Conv") return Conv;
  if (layer_type == "MatmulAct") return MatmulAct;
  if (layer_type == "ActMatmul") return ActMatmul;
  if (layer_type == "Softmax") return Softmax;
  if (layer_type == "Activation") return Activation;
  if (layer_type == "LayerNorm") return LayerNorm;
  if (layer_type == "SelfAttention") return SelfAttention;
  if (layer_type == "MultiHeadSelfAttention") return MultiHeadSelfAttention;
  throw std::runtime_error("Unknown layer type: " + layer_type);
}
}

ArchConfig frontend::standard::arch_config;

std::vector<JobPair> StandardLayer::make_layers(
    const std::vector<LayerConfig> &layer_configs) const {
  std::vector<JobPair> model_heads;
  if (layer_configs.empty()) {
    return model_heads;
  }

  JobList previous_tail;
  for (int replica = 0; replica < model_parallelism; ++replica) {
    std::vector<JobPair> layers;
    layers.reserve(layer_configs.size());
    for (const auto &layer_config : layer_configs) {
      layers.push_back(getLayerLambda(layer_config.layer_type)(arch_config, layer_config));
    }
    for (size_t index = 1; index < layers.size(); ++index) {
      connectJobLists(layers[index - 1].second, layers[index].first);
    }
    if (replica == 0 || do_par) {
      model_heads.push_back(layers.front());
    } else {
      connectJobLists(previous_tail, layers.front().first);
    }
    previous_tail = layers.back().second;
  }
  return model_heads;
}
