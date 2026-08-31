/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "memory.h"
#include "global.h"
#include "frontends/ArchParser.h"
#include <deque>
#include <fstream>

using namespace mem;

namespace mem {
  mem_ty *mem_sys;
  dramsim3::Config *dramsim3config;
  std::unordered_map<uint64_t, std::deque<State *>> address_reads_bkwds_lookup;
  std::unordered_map<uint64_t, std::deque<State *>> address_writes_bkwds_lookup;
}

static int q = 0;

// Priority comparator for memory transactions (lower priority number = higher priority)
struct PrioritySorter {
    bool operator()(const std::tuple<uint64_t, bool, int, State *> &first, const std::tuple<uint64_t, bool, int, State *> &second) const {
        return std::get<2>(first) < std::get<2>(second);
    }
};

bool mem::try_enqueue_tx() {
    // -mem_prio 1: serve systolic-array transactions (priority 1) ahead of
    // vector-unit ones (priority 2) within the scan window -- the ISPASS'25
    // case-study-A fix. PrioritySorter above documents the ordering; a
    // two-pass scan applies it without reordering the queue. Default (0)
    // keeps the historical FIFO scan.
    if (mem_prio) {
        for (int i = 0; i < to_enqueue.size() && i < 64; ++i) {
            auto &pair = to_enqueue[i];
            if (std::get<2>(pair) > 1) continue;
            if (mem_sys->WillAcceptTransaction(std::get<0>(pair), std::get<1>(pair))) {
                mem_sys->AddTransaction(std::get<0>(pair), std::get<1>(pair));
                if (std::get<1>(pair)) {
                    address_writes_bkwds_lookup[std::get<0>(pair)].push_back(std::get<3>(pair));
                } else {
                    address_reads_bkwds_lookup[std::get<0>(pair)].push_back(std::get<3>(pair));
                }
                if (i != to_enqueue.size() - 1) {
                    std::swap(pair, to_enqueue.back());
                }
                to_enqueue.pop_back();
                return true;
            }
        }
    }
    for (int i = 0; i < to_enqueue.size() && i < 64; ++i) {
        auto &pair = to_enqueue[i];
        uint64_t addr = std::get<0>(pair);
        bool is_write = std::get<1>(pair);
        int priority = std::get<2>(pair);
        State *state = std::get<3>(pair);
        
        if (mem_sys->WillAcceptTransaction(addr, is_write)) {
            mem_sys->AddTransaction(addr, is_write);
            // Register callback mapping for completion notification
            if (is_write) {
                address_writes_bkwds_lookup[addr].push_back(state);
            } else {
                address_reads_bkwds_lookup[addr].push_back(state);
            }
            // Remove processed transaction from queue
            if (i != to_enqueue.size() - 1) {
                std::swap(pair, to_enqueue.back());
            }
            to_enqueue.pop_back();
            return true;
        }
    }
    return false;
}

void mem::setup() {
  std::ifstream probe(dram_ini_path);
  if (!probe.good()) {
    std::cerr << "Error: DRAM config file not found: " << dram_ini_path << std::endl;
    exit(1);
  }
  probe.close();
  dramsim3config = new dramsim3::Config(dram_ini_path, "./");
  mem_sys = new mem_ty(*dramsim3config, "./", [](uint64_t addr) {
        auto it = address_reads_bkwds_lookup.find(addr);
        if (it != address_reads_bkwds_lookup.end() && !it->second.empty()) {
            State *q = it->second.front();
            it->second.pop_front();
            if (it->second.empty()) {
                address_reads_bkwds_lookup.erase(it);
            }
            q->mem_read_left -= 1;
        } else {
            std::cerr << "Error: Address " << std::hex << addr << " not found in address_reads_bkwds_lookup" << std::endl;
        } }, [](uint64_t addr) {
        auto it = address_writes_bkwds_lookup.find(addr);
        if (it != address_writes_bkwds_lookup.end() && !it->second.empty()) {
            State *q = it->second.front();
            it->second.pop_front();
            if (it->second.empty()) {
                address_writes_bkwds_lookup.erase(it);
            }
            q->mem_write_left -= 1;
        } else {
            std::cerr << "Error: Address " << addr << " not found in address_writes_bkwds_lookup" << std::endl;
        } });
  bytes_per_tx = dramsim3config->request_size_bytes;
  std::cout << "REQUEST SIZE BYTES " << bytes_per_tx << std::endl;

}
