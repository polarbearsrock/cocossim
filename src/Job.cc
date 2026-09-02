/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "Job.h"
#include "global.h"
#include <string>
#include <unordered_map>

uint64_t alloc_addr = 0;
static int job_identifier = 0;

const char *const OP_CLASS_NAMES[N_OP_CLASSES] = {
    "OTHER", "QKV", "O", "GATE_UP", "DOWN", "HEAD", "ATTN", "VPU_NORM", "VPU_EW"};

Job::Job(uint64_t alloc_sz) : addr(alloc_addr), addr_hold(alloc_addr), alloc_size(alloc_sz), rem_deps(0) {
  alloc_addr += alloc_sz;
  total_jobs++;
  task_idx = alloc_task_idx;
  job_idx = job_identifier++;
}

int64_t Job::take_prefetch_credit_beats(int64_t want) {
  int64_t take = std::min(prefetch_credit_beats, want);
  prefetch_credit_beats -= take;
  pf_outstanding_beats -= take;
  return take;
}

void jobs_to_dot(std::vector<Job *> &jobs, const std::string &fname) {
  // Generate DOT graph file for job dependency visualization
  FILE *f = fopen(fname.c_str(), "w");
  fprintf(f, "digraph G {\n");
  fprintf(f, "frontier [label=\"frontier\"];\n");
  
  // Traverse job dependency graph and assign unique names
  std::unordered_map<Job *, std::string> job_names;
  std::vector<Job *> to_visit = jobs;
  while (!to_visit.empty()) {
    Job *job = to_visit.back();
    to_visit.pop_back();
    if (job_names.find(job) == job_names.end()) {
      std::string name = "job" + std::to_string(job_names.size());
      job_names[job] = name;
      fprintf(f, "  %s [label=\"%s\"];\n", name.c_str(), job->get_job_dims_string().c_str());
      for (auto *child: job->children) {
        to_visit.push_back(child);
      }
    }
  }
  for (auto &pair: job_names) {
    Job *job = pair.first;
    std::string name = pair.second;
    for (auto *child: job->children) {
      fprintf(f, "  %s -> %s;\n", name.c_str(), job_names[child].c_str());
    }
  }
  for (auto *job: jobs) {
    fprintf(f, "  frontier -> %s;\n", job_names[job].c_str());
  }

  fprintf(f, "}\n");
  fclose(f);
}
