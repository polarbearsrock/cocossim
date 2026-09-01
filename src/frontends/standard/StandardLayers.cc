/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "NNLayers.h"
#include "frontends/standard/StandardLayer.h"
#include "global.h"
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"
#include <frontends/standard/StandardArch.h>

#include <functional>
#include <stdexcept>
#include <cmath>

using namespace frontend::standard;

// One id per weight tensor instance (a GEMM invocation, an attention head's
// K/V panel). Jobs sharing an id may reuse each other's VMEM-staged weights.
static int next_weight_tag = 0;

// VMEM fit policy (spec 6.7): a K x N_slice weight slice claims residency iff
// it fits its share of VMEM -- the buffer split across the n_slices cores
// whose slices must co-reside, derated to -vmem_headroom percent. Integer
// math; int64 because K*N*dtw can approach 2^31 at calibration shapes.
static bool weightSliceFitsVmem(int K, int n_slice, int n_slices, int64_t copies = 1) {
  return (int64_t) K * n_slice * data_type_width * 100 * n_slices * copies
         <= (int64_t) vmem_headroom_pct * buffer_size_bytes;
}

// OS-mode job creation. M is the layer's TRUE row count: it is split into
// ceil(M / sa_sz) row-block jobs, the last carrying the remainder, so job
// dims preserve under-fill information (spec 3.5). N is split across cores.
JobList createSAJobs(int M, int K, int N, int sa_sz, int n_cores = 1) {
  JobList jobs;
  int core_n = N / n_cores;
  // A zero-column split would either silently drop the layer's work (OS,
  // masked by max(N/sz, 1)) or trip the WS loop_cols_tiles guard downstream.
  // Reject the input cleanly instead; remainder distribution is not modeled.
  if (core_n < 1) {
    std::cerr << "Error: cannot split N=" << N << " output columns across "
              << n_cores << " cores (need N >= n_cores)" << std::endl;
    exit(1);
  }
  int num_jobs = div_ru(M, sa_sz);
  int weight_tag = next_weight_tag++;
  bool fits = weightSliceFitsVmem(K, core_n, n_cores);
  static std::vector<int> core_task_counters(n_cores, 0);
  for (int core = 0; core < n_cores; ++core) {
    for (int job = 0; job < num_jobs; ++job) {
      int m = std::min(sa_sz, M - job * sa_sz);
      auto sys_job = new SystolicArray::SysArrayJob(m, K, core_n, sa_sz, /*ws=*/false);
      sys_job->core_id = core;
      sys_job->task_idx = core_task_counters[core]++;
      // All row-block jobs of one GEMM read the same weight slice; tags are
      // compared only within one state, so cores can share the invocation id.
      sys_job->weight_tag = weight_tag;
      sys_job->weights_fit_vmem = fits;
      jobs.push_back(sys_job);
    }
  }
  return jobs;
}

// Weight-stationary N-splitting shared by Matmul and Conv: split N across
// cores, then across sequential jobs sized so each job's modeled working set
// (M output columns plus the M x min(K, sa_sz) activation panel) satisfies the
// same inequality as the bufferability test below.
static JobList createWSJobs(const ArchConfig &a_config, int M, int K, int N, const char *label) {
  JobList jl;

  int core_n = N / a_config.n_cores;
  if (core_n < 1) {
    std::cerr << "Error: cannot split N=" << N << " output columns across "
              << a_config.n_cores << " cores (need N >= n_cores)" << std::endl;
    exit(1);
  }
  std::cout << label << " N-splitting: " << N << " output channels across " << a_config.n_cores << " cores" << std::endl;

  static std::vector<int> core_task_counters(a_config.n_cores, 0);
  for (int core = 0; core < a_config.n_cores; ++core) {
    int required_buff_sz_per_core = (M * core_n + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
    bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;

    if (core_is_bufferable) {
      auto job = new SystolicArray::SysArrayJob(M, K, core_n, a_config.sa_sz_allo, /*ws=*/true);
      job->core_id = core;
      job->task_idx = core_task_counters[core]++;
      jl.push_back(job);
      std::cout << "  Core " << core << ": " << core_n << " out dim - bufferable" << std::endl;
    } else {
      // Sequential jobs: reserve room for the activation panel so each job
      // passes the same buffer-fit test that routed us here.
      std::cout << "  Core " << core << ": " << core_n << " out dim - not bufferable, using sequential execution" << std::endl;
      int cap = buffer_size_bytes / (data_type_width * M * batch_size);
      int N_per_job = cap - std::min(K, a_config.sa_sz_allo);
      if (N_per_job < 1) {
        std::cerr << "Warning: " << label << " M=" << M << " K=" << K
                  << " cannot fit one output column plus its activation panel in the "
                  << buffer_size_bytes << "-byte buffer; modeled working set will exceed it"
                  << " (M/K tiling is not implemented)" << std::endl;
        N_per_job = 1;
      }
      int num_sequential_jobs = div_ru(core_n, N_per_job);

      for (int i = 0; i < num_sequential_jobs; ++i) {
        int remaining_N = core_n - i * N_per_job;
        int current_N = std::min(N_per_job, remaining_N);
        auto job = new SystolicArray::SysArrayJob(M, K, current_N, a_config.sa_sz_allo, /*ws=*/true);
        job->core_id = core;
        job->task_idx = core_task_counters[core]++;
        jl.push_back(job);
      }
    }
  }

  return jl;
}

static const std::vector<std::pair<VectorUnit::VPUPhase, int>> softmax_phases =
    {{VectorUnit::VPUPhase::BROADCAST, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}};

static const std::vector<std::pair<VectorUnit::VPUPhase, int>> rmsnorm_phases =
    {{VectorUnit::VPUPhase::REDUCE, 2}, {VectorUnit::VPUPhase::BROADCAST, 1}};

using JobCreate_f = std::function<JobPair(const ArchConfig &, const LayerConfig &)>;

JobPair Matmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    JobList jl = createWSJobs(a_config, M, K, N, "WS");
    return {jl, jl};
  } else {
    // OS: Create sequential jobs per core
    JobList matmul_layers = createSAJobs(M, K, N, a_config.sa_sz_allo, a_config.n_cores);
    return {matmul_layers, matmul_layers};
  }
}


JobPair Conv(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  
  // Conv layer parameters: batch, input_channels, input_height, input_width, output_channels, kernel_size, stride, padding
  if (l_config.dimensions.size() < 5) {
    std::cerr << "Conv expecting at least 5 dimensions: batch, input_channels, input_height, input_width, output_channels" << std::endl;
    throw std::exception();
  }
  
  int batch = l_config.dimensions[0];
  int input_channels = l_config.dimensions[1]; 
  int input_height = l_config.dimensions[2];
  int input_width = l_config.dimensions[3];
  int output_channels = l_config.dimensions[4];
  
  // Default values if not specified
  int kernel_size = (l_config.dimensions.size() > 5) ? l_config.dimensions[5] : 3;
  int stride = (l_config.dimensions.size() > 6) ? l_config.dimensions[6] : 1;
  int padding = (l_config.dimensions.size() > 7) ? l_config.dimensions[7] : 1;
  
  
  // Calculate output dimensions after convolution
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
  
  // Convert to GEMM dimensions (im2col approach):
  // M = batch * output_height * output_width (number of output spatial positions)
  // K = input_channels * kernel_size * kernel_size (input channels * kernel area) 
  // N = output_channels (number of filters)
  M = batch * output_height * output_width;
  K = input_channels * kernel_size * kernel_size;
  N = output_channels;
  
  std::cout << "Conv2GEMM: batch=" << batch << ", in_ch=" << input_channels << ", in_h=" << input_height << ", in_w=" << input_width << std::endl;
  std::cout << "           out_ch=" << output_channels << ", kernel=" << kernel_size << ", stride=" << stride << ", padding=" << padding << std::endl;
  std::cout << "           out_h=" << output_height << ", out_w=" << output_width << std::endl;
  std::cout << "           GEMM dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;

  if (a_config.ws) {
    JobList jl = createWSJobs(a_config, M, K, N, "Conv WS");
    return {jl, jl};
  } else {
    // OS: Create sequential jobs per core
    JobList matmul_layers = createSAJobs(M, K, N, a_config.sa_sz_allo, a_config.n_cores);
    return {matmul_layers, matmul_layers};
  }
}


JobPair MatmulAct(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse matmul dimensions from layer config
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    JobList matmul_layers;
    for (int kb = 0; kb < std::max(1, int(std::ceil(float(K) / a_config.sa_sz_allo))); ++kb) {
      JobList part = createSAJobs(M, a_config.sa_sz_allo, N, a_config.sa_sz_allo);
      matmul_layers.insert(matmul_layers.end(), part.begin(), part.end());
    }
    JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
  } else {
    JobList matmul_layers = createSAJobs(M, K, N, a_config.sa_sz_allo);
    JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
  }
}

JobPair ActMatmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse matmul dimensions from layer config
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "AM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  // Create activation layer job
  JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

  if (a_config.ws) {
    JobList matmul_layers;
    for (int kb = 0; kb < std::max(1, K / a_config.sa_sz_allo); ++kb) {
      JobList part = createSAJobs(M, a_config.sa_sz_allo, N, a_config.sa_sz_allo);
      matmul_layers.insert(matmul_layers.end(), part.begin(), part.end());
    }
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
  } else {
    JobList matmul_layers = createSAJobs(M, K, N, a_config.sa_sz_allo);
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
  }
}

JobPair LayerNorm(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse dimensions for layernorm operation
  int lin_dim, par_dim = 1;
  switch (l_config.dimensions.size()) {
    case 1:
      lin_dim = l_config.dimensions[0];
      break;
    case 2:
      par_dim = l_config.dimensions[0];
      lin_dim = l_config.dimensions[1];
      break;
    case 3:
      par_dim = l_config.dimensions[0] * l_config.dimensions[1];
      lin_dim = l_config.dimensions[2] / l_config.dimensions[0];
      if (l_config.dimensions[2] % l_config.dimensions[0] != 0) {
        std::cerr << "linear dimension is not divisible by group size in layernorm..." << std::endl;
        throw std::exception();
      }
      break;
    default:
      std::cout << "Unexpected number of dimensions received in LayerNorm" << std::endl;
      throw std::exception();
  }
  // Create jobs with buffer size constraints
  JobList jl;
  int par_acc = par_dim;
  int dec_amt = buffer_size_bytes / data_type_width / lin_dim;
  while (par_acc > 0) {
    jl.push_back(new VectorUnit::VecUnitJob(lin_dim, std::min(dec_amt, par_acc), false,
                                            {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 4}, {VectorUnit::VPUPhase::BROADCAST, 1}}));
    par_acc -= dec_amt;
  }
  return {{jl}, {jl}};
}

// RMSNorm over par_dim rows of length lin_dim, chunked to the buffer the
// same way LayerNorm is.
static JobList makeRMSNormJobs(int lin_dim, int par_dim) {
  JobList jl;
  int par_acc = par_dim;
  int dec_amt = buffer_size_bytes / data_type_width / lin_dim;
  while (par_acc > 0) {
    jl.push_back(new VectorUnit::VecUnitJob(lin_dim, std::min(dec_amt, par_acc), false, rmsnorm_phases));
    par_acc -= dec_amt;
  }
  return jl;
}

JobPair RMSNorm(const ArchConfig &a_config, const LayerConfig &l_config) {
  int lin_dim, par_dim = 1;
  if (l_config.dimensions.size() == 1) {
    lin_dim = l_config.dimensions[0];
  } else if (l_config.dimensions.size() == 2) {
    par_dim = l_config.dimensions[0];
    lin_dim = l_config.dimensions[1];
  } else {
    std::cerr << "RMSNorm expects 1 or 2 dimensions, got " << l_config.dimensions.size() << std::endl;
    throw std::exception();
  }
  JobList jl = makeRMSNormJobs(lin_dim, par_dim);
  return {jl, jl};
}

JobPair Activation(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;

  auto job = new VectorUnit::VecUnitJob(1, sz, false, {{VectorUnit::VPUPhase::BROADCAST, 1}});
  return {{job}, {job}};
}

JobPair Add(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;
  auto job = new VectorUnit::VecUnitJob(1, sz, false, {{VectorUnit::VPUPhase::BROADCAST, 1}}, 2);
  return {{job}, {job}};
}

// n_rows independent softmax rows of length row_len, chunked so each job's
// working set fits the buffer and no job exceeds 1024 rows.
static JobList makeSoftmaxJobs(int row_len, int n_rows) {
  int spl = 1;
  int Mp = n_rows;
  if (row_len * n_rows * data_type_width * batch_size > buffer_size_bytes || Mp > 1024) {
    spl = std::max(div_ru(row_len * n_rows * data_type_width * batch_size, buffer_size_bytes),
                   div_ru(Mp, 1024));
    if (spl > Mp) {
      std::cerr << "Can't split this enough to fit inside buffer." << std::endl;
      throw std::exception();
    }
    Mp /= spl;
  }
  std::cout << "Splitting by " << spl << std::endl;// preserved from Softmax: stdout must stay identical
  // One job per Mp-row chunk: every chunk must be modeled, on however many
  // vector units the architecture actually has (the scheduler spreads
  // unpinned jobs across idle units of the type at dispatch time).
  int n_jobs = div_ru(n_rows, Mp);
  JobList softmax_layer;
  for (int i = 0; i < n_jobs; ++i)
    softmax_layer.push_back(new VectorUnit::VecUnitJob(row_len, Mp, false, softmax_phases));
  return softmax_layer;
}

// Llama/Gemma-style decoder stack (spec 3.4).
// Line: Transformer n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch
//   mode 0 = prefill: every GEMM has M = seq_len rows, attention context = seq_len
//   mode 1 = decode:  every GEMM has M = batch rows, KV context = seq_len
// Residual adds depend on BOTH true parents so the simulator is allowed the
// same SA/VPU overlap the hardware has. KV-cache traffic rides the score/AV
// jobs' weight-side reads. For layer 0 the residual's block-input parent is
// approximated by norm1 (the true parent is the external predecessor line).
JobPair Transformer(const ArchConfig &a_config, const LayerConfig &l_config) {
  if (l_config.dimensions.size() != 8 && l_config.dimensions.size() != 9) {
    std::cerr << "Transformer expects 8 or 9 dims: n_layers d_model n_heads n_kv_heads d_ff seq_len mode batch [vocab]" << std::endl;
    throw std::exception();
  }
  const auto &d = l_config.dimensions;
  int n_layers = d[0], d_model = d[1], nh = d[2], nkv = d[3], d_ff = d[4], seq_len = d[5], mode = d[6], batch = d[7];
  int vocab = l_config.dimensions.size() == 9 ? d[8] : 0;// 0 = no LM head
  if (n_layers < 1 || d_model < 1 || nh < 1 || nkv < 1 || d_ff < 1 || seq_len < 1 || batch < 1 ||
      (mode != 0 && mode != 1) || d_model % nh != 0 || nh % nkv != 0 || vocab < 0) {
    std::cerr << "Transformer: invalid dims (need positive sizes, mode 0|1, nh | d_model, nkv | nh, vocab >= 0)" << std::endl;
    throw std::exception();
  }
  int head_dim = d_model / nh;
  int M = (mode == 0) ? seq_len : batch;// rows through every GEMM
  int S = seq_len;                      // attention context length

  auto mk_binary_ew = [&](int rows, int cols) -> JobList {
    auto *jb = new VectorUnit::VecUnitJob(cols, rows, false, {{VectorUnit::VPUPhase::BROADCAST, 1}}, 2);
    return {jb};
  };

  JobList model_head, prev_tail;
  for (int l = 0; l < n_layers; ++l) {
    JobList norm1 = makeRMSNormJobs(d_model, M);
    if (l == 0) model_head = norm1;
    else connectJobLists(prev_tail, norm1);

    auto q = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_model}));
    auto k = Matmul(a_config, LayerConfig("Matmul", {M, d_model, nkv * head_dim}));
    auto v = Matmul(a_config, LayerConfig("Matmul", {M, d_model, nkv * head_dim}));
    connectJobLists(norm1, q.first);
    connectJobLists(norm1, k.first);
    connectJobLists(norm1, v.first);

    JobList rope = {new VectorUnit::VecUnitJob(1, M * (d_model + nkv * head_dim), false,
                                               {{VectorUnit::VPUPhase::BROADCAST, 1}})};
    connectJobLists(q.second, rope);
    connectJobLists(k.second, rope);

    JobList scores, av;
    // KV panels are PER-SEQUENCE state, not shared weights (spec 6.7): in
    // decode, each of the `batch` sequences owns its KV cache, so score/AV
    // jobs stream `batch` copies of their panel (n_weight_streams). GQA
    // sharing: the nh/nkv query heads of one group read the SAME panels, so
    // they share a weight_tag per group and the residency machinery makes
    // the siblings' reads free when the panel set fits VMEM. (Prefill models
    // a single sequence, M = seq_len, so streams stay 1.) Caveat: residency
    // is per-MXU, so a group split across both MXUs fetches its panels once
    // per MXU -- bounded 2x over hardware's shared-VMEM single fetch.
    int kv_streams = (mode == 1) ? batch : 1;
    int group_sz = nh / nkv;
    for (int h = 0; h < nh; ++h) {
      if (h % group_sz == 0) next_weight_tag++;
      auto *sc = new SystolicArray::SysArrayJob(M, head_dim, S, a_config.sa_sz_allo, a_config.ws, kv_streams);
      sc->weight_tag = next_weight_tag;
      sc->weights_fit_vmem = weightSliceFitsVmem(head_dim, S, a_config.n_cores, kv_streams);
      scores.push_back(sc);
    }
    next_weight_tag++;
    for (int h = 0; h < nh; ++h) {
      if (h % group_sz == 0) next_weight_tag++;
      auto *avj = new SystolicArray::SysArrayJob(M, S, head_dim, a_config.sa_sz_allo, a_config.ws, kv_streams);
      avj->weight_tag = next_weight_tag;
      avj->weights_fit_vmem = weightSliceFitsVmem(S, head_dim, a_config.n_cores, kv_streams);
      av.push_back(avj);
    }
    next_weight_tag++;
    connectJobLists(rope, scores);

    JobList sm = makeSoftmaxJobs(S, M * nh);
    connectJobLists(scores, sm);
    connectJobLists(sm, av);
    connectJobLists(v.second, av);

    auto o = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_model}));
    connectJobLists(av, o.first);

    JobList block_in = (l == 0) ? norm1 : prev_tail;
    JobList res1;
    if (fuse_epilogue) {
      // Residual absorbed into the projection epilogue: downstream work
      // depends on both contributors directly, no VPU job.
      res1 = o.second;
      res1.insert(res1.end(), block_in.begin(), block_in.end());
    } else {
      res1 = mk_binary_ew(M, d_model);
      connectJobLists(o.second, res1);
      connectJobLists(block_in, res1);
    }

    JobList norm2 = makeRMSNormJobs(d_model, M);
    connectJobLists(res1, norm2);

    auto gate = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_ff}));
    auto up = Matmul(a_config, LayerConfig("Matmul", {M, d_model, d_ff}));
    connectJobLists(norm2, gate.first);
    connectJobLists(norm2, up.first);

    JobList silu_mul = mk_binary_ew(M, d_ff);
    connectJobLists(gate.second, silu_mul);
    connectJobLists(up.second, silu_mul);

    auto down = Matmul(a_config, LayerConfig("Matmul", {M, d_ff, d_model}));
    connectJobLists(silu_mul, down.first);

    JobList res2;
    if (fuse_epilogue) {
      res2 = down.second;
      res2.insert(res2.end(), res1.begin(), res1.end());
    } else {
      res2 = mk_binary_ew(M, d_model);
      connectJobLists(down.second, res2);
      connectJobLists(res1, res2);
    }

    prev_tail = res2;
  }
  if (vocab > 0) {
    // LM head (full-model coverage): final RMSNorm -> unembedding GEMM
    // (M x d_model x vocab) -> vocabulary softmax. The head weight is the
    // model's largest tensor (~1 GB at Llama-8B) and can never be
    // VMEM-resident -- the fit check says so on its own -- so it streams
    // from HBM every step. The embedding gather at the model's front is
    // deliberately unmodeled: a few KB per token, no kernel worth a job.
    // Serving computes logits for the LAST token of each sequence only, so
    // the head chain carries batch rows in BOTH modes (in decode M == batch
    // already; in prefill using M = seq_len would inflate the head ~seq_len/
    // batch-fold vs. real serving - session-3 calibration finding).
    int head_rows = batch;
    JobList final_norm = makeRMSNormJobs(d_model, head_rows);
    connectJobLists(prev_tail, final_norm);
    auto head = Matmul(a_config, LayerConfig("Matmul", {head_rows, d_model, vocab}));
    connectJobLists(final_norm, head.first);
    JobList logits_sm = makeSoftmaxJobs(vocab, head_rows);
    connectJobLists(head.second, logits_sm);
    prev_tail = logits_sm;
  }
  return {model_head, prev_tail};
}

JobPair Softmax(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M;
  int heads = 1;
  if (l_config.dimensions.size() == 1) {
    M = l_config.dimensions[0];
  } else if (l_config.dimensions.size() == 2) {
    heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
  } else {
    std::cerr << "SM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  JobList softmax_layer = makeSoftmaxJobs(M, M * heads);
  return {softmax_layer, softmax_layer};
}

JobPair SelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  // Parse self-attention dimensions (batch, seq_len, hidden_dim, n_heads)
  int M, K, N;
  int n_heads = 1;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    n_heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3];
  } else {
    std::cerr << "SA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    // Create K, Q, V projection matrices
    auto K_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    auto Q_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    auto V_proj = Matmul(a_config, LayerConfig("Matmul", {M, K, N}));
    
    // Attention mechanism: Q*K^T and softmax(QK^T)*V
    auto Dot1 = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, M}));
    auto Dot2 = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, M, N / n_heads}));
    auto O_proj = Matmul(a_config, LayerConfig("Matmul", {n_heads, M, N / n_heads, K}));
    auto softmax_layer = Softmax(a_config, LayerConfig("Softmax", {8, M}));

    // Chain operations: K->Q->V->QK^T->softmax->attention*V->output
    connectJobs(K_proj, Q_proj);
    connectJobs(Q_proj, V_proj);
    connectJobs(V_proj, Dot1);
    connectJobs(Dot1, softmax_layer);
    connectJobs(softmax_layer, Dot2);
    connectJobs(Dot2, O_proj);
    return {K_proj.first, O_proj.second};
  } else {
    JobList K_proj = createSAJobs(M,
                                  K,
                                  N, a_config.sa_sz_allo);
    JobList Q_proj = createSAJobs(M,
                                  K,
                                  N, a_config.sa_sz_allo);
    JobList V_proj = createSAJobs(M,
                                  K,
                                  N, a_config.sa_sz_allo);
    JobList Dot1 = createSAJobs(M,
                                K,// / m_config.n_heads,
                                M, a_config.sa_sz_allo);
    JobList Dot2 = createSAJobs(M,
                                M,
                                N, a_config.sa_sz_allo);/// m_config.n_heads
    JobList O_proj = createSAJobs(M,
                                  K,
                                  N, a_config.sa_sz_allo);

    JobList softmax_layer = {new VectorUnit::VecUnitJob(M, M, true,
                                                        {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}})};
    connectJobLists(K_proj, Q_proj);
    connectJobLists(Q_proj, Dot1);
    connectJobLists(Dot1, softmax_layer);
    connectJobLists(softmax_layer, V_proj);
    connectJobLists(V_proj, Dot2);
    connectJobLists(Dot2, O_proj);


    return {K_proj, O_proj};
  }
}

JobPair MultiHeadSelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  JobList all_heads, all_tails;

  int N = ceil(static_cast<double>(n_heads) / a_config.n_cores);
  std::vector<JobPair> head_jobs;
  for (int i = 0; i < N; ++i) {
    head_jobs.push_back(SelfAttention(a_config, l_config));
  }
  for (int i = 0; i < N - 1; ++i) {
    connectJobLists(head_jobs[i].second, head_jobs[i + 1].first);
  }
  return {head_jobs.front().first, head_jobs.back().second};
}


JobCreate_f getLayerLambda(const std::string &layer_type) {
  if (layer_type == "Matmul")
    return Matmul;
  if (layer_type == "Conv")
    return Conv;
  if (layer_type == "MatmulAct")
    return MatmulAct;
  if (layer_type == "Softmax")
    return Softmax;
  if (layer_type == "Activation")
    return Activation;
  if (layer_type == "Add")
    return Add;
  if (layer_type == "LayerNorm")
    return LayerNorm;
  if (layer_type == "RMSNorm")
    return RMSNorm;
  if (layer_type == "SelfAttention")
    return SelfAttention;
  if (layer_type == "MultiHeadSelfAttention")
    return MultiHeadSelfAttention;
  if (layer_type == "Transformer")
    return Transformer;
  throw std::runtime_error("Unknown layer type: " + layer_type);
}

ArchConfig frontend::standard::arch_config;

std::vector<JobPair> StandardLayer::make_layers(const std::vector<LayerConfig> &layer_configs) const {
  std::vector<JobPair> model_heads;
  JobList jp;
  for (int m = 0; m < model_parallelism; ++m) {
    std::vector<JobPair> lists;
    for (const auto &layer_config: layer_configs) {
      auto layer_f = getLayerLambda(layer_config.layer_type);
      lists.push_back(layer_f(arch_config, layer_config));
    }
    std::cout << "list size: " << lists.size() << std::endl;
    for (int i = 1; i < lists.size(); ++i) {
      connectJobLists(lists[i - 1].second, lists[i].first);
      std::cout << "connect " << i << " to " << (i - 1) << std::endl;
    }
    if (m == 0 || do_par) {
      model_heads.push_back(lists[0]);
      jp = lists.back().second;
    } else {
      connectJobLists(jp, lists[0].first);
      jp = lists.back().second;
    }
  }

  return model_heads;
}
