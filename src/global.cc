/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "global.h"
#include <limits>
#include <stdexcept>

int total_jobs = 0;
int jobs_finished = 0;
int bytes_per_tx;
std::vector<std::tuple<uint64_t, bool, int, State *>> to_enqueue;
FILE *vcd = nullptr;
uint64_t gcycles = 0;
int alloc_task_idx = 0;
int model_parallelism = 1;
float freq_sa = 1;
float freq_vu = 1;
int batch_size = 1;
int data_type_bits = 16;
uint64_t buffer_size_bytes = 8ULL * 1024ULL * 1024ULL;
bool compute_only = false;

bool do_par = false;
char const *rand_chars[] = {"a", "b", "c", "d", "e", "f",
                            "g", "h", "i", "j", "k", "l",
                            "m", "n", "o", "p", "q", "r",
                            "s", "t", "u", "v", "w", "x",
                            "y", "z", "A", "B", "C", "D",
                            "E", "F", "G", "H", "I", "J",
                            "K", "L", "M", "N", "O", "P",
                            "Q", "R", "S", "T", "U", "V",
                            "W", "X", "Y", "Z", "0", "1",
                            "2", "3", "4", "5", "6", "7",
                            "aa", "ab", "ac", "ad", "ae",
                            "af", "ag", "ah", "ai", "aj",
                            "ak", "al", "am", "an", "ao",
                            "ap", "aq", "ar", "as", "at",
                            "au", "av", "aw", "ax", "ay"};

uint64_t div_ru(uint64_t q, uint64_t r) {
    if (r == 0) {
        throw std::invalid_argument("division by zero in div_ru");
    }
    return q / r + static_cast<uint64_t>(q % r != 0);
}

uint64_t checked_product(std::initializer_list<uint64_t> factors) {
    uint64_t product = 1;
    for (const auto factor : factors) {
        if (factor != 0 && product > std::numeric_limits<uint64_t>::max() / factor) {
            throw std::overflow_error("dimension product exceeds 64-bit range");
        }
        product *= factor;
    }
    return product;
}

uint64_t bytes_for_elements(uint64_t element_count) {
    if (data_type_bits <= 0) {
        throw std::invalid_argument("data_type_bits must be positive");
    }
    const auto total_bits = static_cast<unsigned __int128>(element_count) *
                            static_cast<unsigned int>(data_type_bits);
    const auto total_bytes = (total_bits + 7U) / 8U;
    if (total_bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("tensor storage exceeds 64-bit byte range");
    }
    return static_cast<uint64_t>(total_bytes);
}

uint64_t elements_fitting_in_bytes(uint64_t byte_count) {
    if (data_type_bits <= 0) {
        throw std::invalid_argument("data_type_bits must be positive");
    }
    const auto available_bits = static_cast<unsigned __int128>(byte_count) * 8U;
    const auto elements = available_bits / static_cast<unsigned int>(data_type_bits);
    if (elements > std::numeric_limits<uint64_t>::max()) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(elements);
}
