/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 *
 * Copyright (c) 2025 APEX Lab, Duke University
 *
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_SYSARRAY_H
#define PERF_MODEL_SYSARRAY_H
#include "State.h"
#include "frontends/standard/StandardUnits.h"
#include "EnergyModel.h"

namespace SystolicArray {
  struct SysArrayJob : Job {

    int get_type() const override {
      return SYSTOLIC_ARRAY_IDX;
    }
    int M, K, N;

    SysArrayJob(int m, int k, int n);

    [[nodiscard]] std::string get_job_dims_string() const override;
  };

  enum ExState {
    idle = 0,
    prefetch,//only WS
    read,
    shift,
    write
  };

  struct SysArrayState : State {
public:
    explicit SysArrayState(int sz, bool ws, EnergyModel* energy_model = nullptr,
                          ComputePrecision precision = ComputePrecision::FP32);
    bool ws = false;// Flag for weight-stationary systolic array

    void init_row_loop(bool new_row);
    void init() override;
    int sz;
    ExState state = idle;

    bool increment(const std::function<void(Job *)> &enqueue_job,
                   int &total_idle,
                   int *n_idle_units) override;

    void set_state(int st) override {
      state = (ExState) st;
    };

    int get_state() override {
      return state;
    }


    int get_ty_idx() override {
      return SYSTOLIC_ARRAY_IDX;
    }
    std::string get_ty_string() override {
      return SYSTOLIC_ARRAY_STRING;
    }

    // Energy tracking
    void record_compute_energy(uint64_t num_macs);
    double get_total_compute_energy_mj() const { return total_compute_energy_mj_; }
    void reset_energy_stats() { total_compute_energy_mj_ = 0.0; weight_bytes_read_ = 0; activation_bytes_read_ = 0; }

    // Memory read tracking (activation vs weight)
    uint64_t get_weight_bytes_read() const { return weight_bytes_read_; }
    uint64_t get_activation_bytes_read() const { return activation_bytes_read_; }

    // Array utilization tracking (accounts for partial filling)
    double get_effective_utilization() const;
    uint64_t get_active_mac_cycles() const { return active_mac_cycles_; }
    uint64_t get_total_mac_capacity_cycles() const { return total_mac_capacity_cycles_; }

private:
    int beats_per_wb;

    // Energy model
    EnergyModel* energy_model_;
    ComputePrecision precision_;
    double total_compute_energy_mj_;

    // Memory tracking
    uint64_t weight_bytes_read_ = 0;
    uint64_t activation_bytes_read_ = 0;

    // Utilization tracking (partial array filling)
    uint64_t active_mac_cycles_ = 0;      // Sum of (active_macs × cycles)
    uint64_t total_mac_capacity_cycles_ = 0;  // Sum of (sz×sz × cycles)
  };

};// namespace SystolicArray


#endif//PERF_MODEL_SYSARRAY_H
