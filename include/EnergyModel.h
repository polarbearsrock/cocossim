#ifndef COCOSSIM_ENERGY_MODEL_H
#define COCOSSIM_ENERGY_MODEL_H

#include <cstdint>
#include <string>
#include <ostream>

enum class TechnologyNode {
    NM_16,
    NM_7
};

enum class ComputePrecision {
    INT8,
    FP16,
    FP32
};

// Helper function to convert data_type_width to ComputePrecision
// data_type_width: 1 = INT8, 2 = FP16, 4 = FP32
inline ComputePrecision get_precision_from_data_width(int data_type_width) {
    switch (data_type_width) {
        case 1: return ComputePrecision::INT8;
        case 2: return ComputePrecision::FP16;
        case 4: return ComputePrecision::FP32;
        default: return ComputePrecision::FP16;  // Default to FP16 for unknown widths
    }
}

struct TechnologyParams16nm {
    static constexpr double INT8_MAC_PJ  = 0.15;
    static constexpr double FP16_MAC_PJ  = 0.50;
    static constexpr double FP32_MAC_PJ  = 0.80;

    static constexpr double SRAM_READ_PJ_PER_BYTE  = 0.80;
    static constexpr double SRAM_WRITE_PJ_PER_BYTE = 1.20;
    static constexpr double SRAM_LEAKAGE_MW_PER_KB = 0.05;

    static constexpr double RF_READ_PJ  = 0.010;
    static constexpr double RF_WRITE_PJ = 0.015;

    static constexpr double CONTROL_LOGIC_MW = 30.0;
};

struct TechnologyParams7nm {
    static constexpr double INT8_MAC_PJ  = 0.06;
    static constexpr double FP16_MAC_PJ  = 0.20;
    static constexpr double FP32_MAC_PJ  = 0.40;

    static constexpr double SRAM_READ_PJ_PER_BYTE  = 0.50;
    static constexpr double SRAM_WRITE_PJ_PER_BYTE = 0.80;
    static constexpr double SRAM_LEAKAGE_MW_PER_KB = 0.09;

    static constexpr double RF_READ_PJ  = 0.010;
    static constexpr double RF_WRITE_PJ = 0.015;

    static constexpr double CONTROL_LOGIC_MW = 22.0;
};

struct DRAMSim3Stats {
    int num_channels = 0;

    uint64_t total_reads  = 0;
    uint64_t total_writes = 0;

    double total_read_energy_pj  = 0.0;
    double total_write_energy_pj = 0.0;
    double total_energy_pj       = 0.0;

    double total_power_mw = 0.0;
    uint64_t dram_cycles  = 0;

    static constexpr int BYTES_PER_TRANSACTION = 16;

    uint64_t total_bytes_read() const {
        return total_reads * BYTES_PER_TRANSACTION;
    }

    uint64_t total_bytes_written() const {
        return total_writes * BYTES_PER_TRANSACTION;
    }

    uint64_t total_bytes() const {
        return total_bytes_read() + total_bytes_written();
    }

    double read_energy_pj_per_byte() const {
        auto b = total_bytes_read();
        return b ? total_read_energy_pj / b : 0.0;
    }

    double write_energy_pj_per_byte() const {
        auto b = total_bytes_written();
        return b ? total_write_energy_pj / b : 0.0;
    }

    double effective_energy_pj_per_byte() const {
        auto b = total_bytes();
        return b ? total_energy_pj / b : 0.0;
    }

    double total_energy_mj() const {
        return total_energy_pj * 1e-9;
    }

    bool is_valid() const {
        return num_channels > 0 && total_energy_pj > 0.0;
    }

    void print_summary(std::ostream& os) const;
};

DRAMSim3Stats parse_dramsim3_output(const std::string& filename);

class EnergyModel {
public:
    explicit EnergyModel(TechnologyNode tech,
                         double frequency_mhz = 1000.0);

    double compute_mac_energy_mj(uint64_t ops,
                                 ComputePrecision p) const;

    double sram_read_energy_mj(uint64_t bytes) const;
    double sram_write_energy_mj(uint64_t bytes) const;

    double sram_leakage_energy_mj(uint64_t size_kb,
                                  uint64_t cycles) const;

    static double hbm_read_energy_mj(uint64_t bytes,
                                    const DRAMSim3Stats& s);

    static double hbm_write_energy_mj(uint64_t bytes,
                                     const DRAMSim3Stats& s);

    static double hbm_total_energy_mj(uint64_t rbytes,
                                      uint64_t wbytes,
                                      const DRAMSim3Stats& s);

    double rf_read_energy_mj(uint64_t n) const;
    double rf_write_energy_mj(uint64_t n) const;

    double control_logic_energy_mj(uint64_t cycles) const;

    TechnologyNode get_technology() const { return tech_; }
    double get_frequency_mhz() const { return frequency_mhz_; }

    std::string get_technology_string() const;

    void print_parameters(std::ostream& os) const;

private:
    TechnologyNode tech_;
    double frequency_mhz_;

    double int8_mac_pj_;
    double fp16_mac_pj_;
    double fp32_mac_pj_;

    double sram_read_pj_;
    double sram_write_pj_;
    double sram_leakage_mw_;

    double rf_read_pj_;
    double rf_write_pj_;

    double control_logic_mw_;

    static constexpr double PJ_TO_MJ = 1e-9;

    double mw_to_mj_per_cycle() const {
        return 1e-3 / (frequency_mhz_ * 1e6);
    }
};

#endif
