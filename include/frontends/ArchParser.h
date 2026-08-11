/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_ARCHPARSER_H
#define PERF_MODEL_ARCHPARSER_H
#include "Arch.h"
#include "global.h"
#include <cmath>
#include <cstring>
#include <limits>

extern std::string layer_file;
extern std::string ofile;

struct ArchParser {
  virtual Arch *make_arch() {
    throw std::runtime_error("make_arch() not implemented for the given frontend.");
  }
  void parse_args(const std::vector<std::pair<std::string, int *>> &input_prs, const std::string &help_str = "") const {
    const auto next_value = [&](int &index, const std::string &flag) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for flag '" + flag + "'");
      }
      return argv[++index];
    };
    const auto parse_int = [](const std::string &value, const std::string &flag) -> int {
      size_t parsed = 0;
      long long result;
      try {
        result = std::stoll(value, &parsed);
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid integer for " + flag + ": '" + value + "'");
      }
      if (parsed != value.size() || result < std::numeric_limits<int>::min() ||
          result > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Invalid integer for " + flag + ": '" + value + "'");
      }
      return static_cast<int>(result);
    };
    const auto parse_u64 = [](const std::string &value, const std::string &flag) -> uint64_t {
      if (value.empty() || value.front() == '-') {
        throw std::runtime_error("Invalid non-negative integer for " + flag + ": '" + value + "'");
      }
      size_t parsed = 0;
      unsigned long long result;
      try {
        result = std::stoull(value, &parsed);
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid non-negative integer for " + flag + ": '" + value + "'");
      }
      if (parsed != value.size()) {
        throw std::runtime_error("Invalid non-negative integer for " + flag + ": '" + value + "'");
      }
      return static_cast<uint64_t>(result);
    };
    const auto parse_float = [](const std::string &value, const std::string &flag) -> float {
      size_t parsed = 0;
      float result;
      try {
        result = std::stof(value, &parsed);
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid floating-point value for " + flag + ": '" + value + "'");
      }
      if (parsed != value.size() || !std::isfinite(result)) {
        throw std::runtime_error("Invalid floating-point value for " + flag + ": '" + value + "'");
      }
      return result;
    };

    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "-i") {
        layer_file = next_value(i, flag);
      } else if (flag == "-o") {
        ofile = next_value(i, flag);
      } else if (flag == "-f") {
        freq_sa = parse_float(next_value(i, flag), flag);
      } else if (flag == "-batch_size" || flag == "--batch-size") {
        batch_size = parse_int(next_value(i, flag), flag);
      } else if (flag == "-data_bits" || flag == "--data-bits") {
        data_type_bits = parse_int(next_value(i, flag), flag);
      } else if (flag == "-buffer_bytes" || flag == "--buffer-bytes") {
        buffer_size_bytes = parse_u64(next_value(i, flag), flag);
      } else if (flag == "-compute_only" || flag == "--compute-only") {
        const int parsed = parse_int(next_value(i, flag), flag);
        if (parsed != 0 && parsed != 1) {
          throw std::runtime_error(flag + " must be either 0 or 1");
        }
        compute_only = parsed == 1;
      } else if (flag == "-h" || flag == "--help") {
        std::cerr << "Global Options:\n"
                     "-i <file>                 layer input file\n"
                     "-o <file>                 output statistic file\n"
                     "-f <float>                compute frequency (GHz)\n"
                     "-batch_size <int>         runtime batch size (default: 1)\n"
                     "-data_bits <int>           packed scalar width in bits (default: 16)\n"
                     "-buffer_bytes <uint64>     per-core buffer capacity (default: 8388608)\n"
                     "-compute_only <0|1>        suppress memory waits when 1 (default: 0)\n";
        if (help_str != "") {
          std::cerr << "Arch Specific Options:\n"
                    << help_str << std::endl;
        }
        exit(0);
      } else {
        bool found = false;
        for (auto &pr: input_prs) {
          if (flag == pr.first) {
            *pr.second = parse_int(next_value(i, flag), flag);
            found = true;
            break;
          }
        }
        if (not found) {
          throw std::runtime_error("Failed to parse passed flag: '" + flag + "'\n");
        }
      }
    }
    if (freq_sa <= 0.0F) {
      throw std::runtime_error("-f must be a positive number");
    }
    if (batch_size <= 0) {
      throw std::runtime_error("-batch_size must be a positive integer");
    }
    if (data_type_bits <= 0 || data_type_bits > 64) {
      throw std::runtime_error("-data_bits must be in the range [1, 64]");
    }
    if (buffer_size_bytes == 0) {
      throw std::runtime_error("-buffer_bytes must be positive");
    }
    if (layer_file.empty()) {
      throw std::runtime_error("-i <file> is required");
    }
    if (ofile.empty()) {
      throw std::runtime_error("-o <file> is required");
    }
  }

  int argc;
  char **argv;
  ArchParser(int argc, char **argv) : argc(argc), argv(argv) {}
  ArchParser() = delete;
};
#endif//PERF_MODEL_ARCHPARSER_H
