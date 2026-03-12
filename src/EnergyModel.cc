#include "EnergyModel.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>
#include <vector>

// ============================================================================
// Constructor
// ============================================================================

EnergyModel::EnergyModel(TechnologyNode tech, double frequency_mhz)
    : tech_(tech),
      frequency_mhz_(frequency_mhz) {

    if (tech_ == TechnologyNode::NM_16) {

        int8_mac_pj_   = TechnologyParams16nm::INT8_MAC_PJ;
        fp16_mac_pj_   = TechnologyParams16nm::FP16_MAC_PJ;
        fp32_mac_pj_   = TechnologyParams16nm::FP32_MAC_PJ;

        sram_read_pj_  = TechnologyParams16nm::SRAM_READ_PJ_PER_BYTE;
        sram_write_pj_ = TechnologyParams16nm::SRAM_WRITE_PJ_PER_BYTE;
        sram_leakage_mw_ = TechnologyParams16nm::SRAM_LEAKAGE_MW_PER_KB;

        rf_read_pj_  = TechnologyParams16nm::RF_READ_PJ;
        rf_write_pj_ = TechnologyParams16nm::RF_WRITE_PJ;

        control_logic_mw_ = TechnologyParams16nm::CONTROL_LOGIC_MW;

    } else {

        int8_mac_pj_   = TechnologyParams7nm::INT8_MAC_PJ;
        fp16_mac_pj_   = TechnologyParams7nm::FP16_MAC_PJ;
        fp32_mac_pj_   = TechnologyParams7nm::FP32_MAC_PJ;

        sram_read_pj_  = TechnologyParams7nm::SRAM_READ_PJ_PER_BYTE;
        sram_write_pj_ = TechnologyParams7nm::SRAM_WRITE_PJ_PER_BYTE;
        sram_leakage_mw_ = TechnologyParams7nm::SRAM_LEAKAGE_MW_PER_KB;

        rf_read_pj_  = TechnologyParams7nm::RF_READ_PJ;
        rf_write_pj_ = TechnologyParams7nm::RF_WRITE_PJ;

        control_logic_mw_ = TechnologyParams7nm::CONTROL_LOGIC_MW;
    }
}

// ============================================================================
// Compute
// ============================================================================

double EnergyModel::compute_mac_energy_mj(uint64_t ops,
                                         ComputePrecision p) const {

    double pj = fp32_mac_pj_;

    if (p == ComputePrecision::INT8)  pj = int8_mac_pj_;
    if (p == ComputePrecision::FP16)  pj = fp16_mac_pj_;

    return ops * pj * PJ_TO_MJ;
}

// ============================================================================
// SRAM
// ============================================================================

double EnergyModel::sram_read_energy_mj(uint64_t bytes) const {
    return bytes * sram_read_pj_ * PJ_TO_MJ;
}

double EnergyModel::sram_write_energy_mj(uint64_t bytes) const {
    return bytes * sram_write_pj_ * PJ_TO_MJ;
}

double EnergyModel::sram_leakage_energy_mj(uint64_t size_kb,
                                          uint64_t cycles) const {

    double power_mw = size_kb * sram_leakage_mw_;
    double time_s   = cycles / (frequency_mhz_ * 1e6);

    return power_mw * time_s;
}

// ============================================================================
// HBM (DRAMSim3-Calibrated)
// ============================================================================

double EnergyModel::hbm_read_energy_mj(uint64_t bytes,
                                      const DRAMSim3Stats& s) {

    if (!s.is_valid()) return 0.0;

    return bytes * s.read_energy_pj_per_byte() * PJ_TO_MJ;
}

double EnergyModel::hbm_write_energy_mj(uint64_t bytes,
                                       const DRAMSim3Stats& s) {

    if (!s.is_valid()) return 0.0;

    return bytes * s.write_energy_pj_per_byte() * PJ_TO_MJ;
}

double EnergyModel::hbm_total_energy_mj(uint64_t rbytes,
                                       uint64_t wbytes,
                                       const DRAMSim3Stats& s) {

    if (!s.is_valid()) return 0.0;

    return (rbytes + wbytes) *
           s.effective_energy_pj_per_byte() * PJ_TO_MJ;
}

// ============================================================================
// Register File
// ============================================================================

double EnergyModel::rf_read_energy_mj(uint64_t n) const {
    return n * rf_read_pj_ * PJ_TO_MJ;
}

double EnergyModel::rf_write_energy_mj(uint64_t n) const {
    return n * rf_write_pj_ * PJ_TO_MJ;
}

// ============================================================================
// Control
// ============================================================================

double EnergyModel::control_logic_energy_mj(uint64_t cycles) const {

    double time_s = cycles / (frequency_mhz_ * 1e6);

    return control_logic_mw_ * time_s;
}

// ============================================================================
// DRAMSim3 Parsing
// ============================================================================

DRAMSim3Stats parse_dramsim3_output(const std::string& filename) {

    DRAMSim3Stats s;

    std::ifstream file(filename);
    if (!file) return s;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    std::regex ch_re("Statistics of Channel (\\d+)");
    std::sregex_iterator it(content.begin(), content.end(), ch_re);
    std::sregex_iterator end;

    std::vector<size_t> pos;

    for (; it != end; ++it) {
        pos.push_back(it->position());
        s.num_channels++;
    }

    if (s.num_channels == 0) return s;

    for (size_t i = 0; i < pos.size(); ++i) {

        size_t start = pos[i];
        size_t stop  = (i + 1 < pos.size()) ? pos[i + 1] : content.size();

        std::string sec = content.substr(start, stop - start);

        std::smatch m;

        if (std::regex_search(sec, m, std::regex("num_reads_done\\s*=\\s*(\\d+)")))
            s.total_reads += std::stoull(m[1]);

        if (std::regex_search(sec, m, std::regex("num_writes_done\\s*=\\s*(\\d+)")))
            s.total_writes += std::stoull(m[1]);

        if (std::regex_search(sec, m, std::regex("read_energy\\s*=\\s*([\\deE+.-]+)")))
            s.total_read_energy_pj += std::stod(m[1]);

        if (std::regex_search(sec, m, std::regex("write_energy\\s*=\\s*([\\deE+.-]+)")))
            s.total_write_energy_pj += std::stod(m[1]);

        if (std::regex_search(sec, m, std::regex("total_energy\\s*=\\s*([\\deE+.-]+)")))
            s.total_energy_pj += std::stod(m[1]);

        if (std::regex_search(sec, m, std::regex("average_power\\s*=\\s*([\\deE+.-]+)")))
            s.total_power_mw += std::stod(m[1]);

        if (s.dram_cycles == 0 &&
            std::regex_search(sec, m, std::regex("num_cycles\\s*=\\s*(\\d+)")))
            s.dram_cycles = std::stoull(m[1]);
    }

    if (s.total_read_energy_pj == 0 &&
        s.total_write_energy_pj == 0 &&
        s.total_energy_pj > 0) {

        uint64_t t = s.total_reads + s.total_writes;

        if (t > 0) {
            double r = double(s.total_reads) / t;
            s.total_read_energy_pj  = s.total_energy_pj * r;
            s.total_write_energy_pj = s.total_energy_pj * (1.0 - r);
        }
    }

    return s;
}

// ============================================================================
// DRAMSim3 Summary
// ============================================================================

void DRAMSim3Stats::print_summary(std::ostream& os) const {

    os << "\n=== DRAMSim3 Statistics ===\n";
    os << "Channels:    " << num_channels << "\n";
    os << "Cycles:      " << dram_cycles << "\n\n";

    os << "Reads:       " << total_reads  << "\n";
    os << "Writes:      " << total_writes << "\n\n";

    os << std::fixed << std::setprecision(2);

    os << "Total:       " << total_energy_mj() << " mJ\n";
    os << "Read:        " << total_read_energy_pj  << " pJ\n";
    os << "Write:       " << total_write_energy_pj << " pJ\n\n";

    os << "Read pJ/B:   " << read_energy_pj_per_byte() << "\n";
    os << "Write pJ/B:  " << write_energy_pj_per_byte() << "\n";
    os << "Eff pJ/B:    " << effective_energy_pj_per_byte() << "\n\n";

    os << "Power:       " << total_power_mw << " mW\n";
}

// ============================================================================
// Utilities
// ============================================================================

std::string EnergyModel::get_technology_string() const {

    if (tech_ == TechnologyNode::NM_16) return "16nm";
    if (tech_ == TechnologyNode::NM_7)  return "7nm";

    return "Unknown";
}

void EnergyModel::print_parameters(std::ostream& os) const {

    os << "\n=== Energy Model ===\n";

    os << "Tech:   " << get_technology_string() << "\n";
    os << "Freq:   " << frequency_mhz_ << " MHz\n\n";

    os << "MAC (pJ):  "
       << "INT8=" << int8_mac_pj_
       << " FP16=" << fp16_mac_pj_
       << " FP32=" << fp32_mac_pj_ << "\n";

    os << "SRAM (pJ/B): "
       << "R=" << sram_read_pj_
       << " W=" << sram_write_pj_ << "\n";

    os << "Leakage: " << sram_leakage_mw_ << " mW/KB\n";

    os << "RF (pJ): "
       << "R=" << rf_read_pj_
       << " W=" << rf_write_pj_ << "\n";

    os << "Control: " << control_logic_mw_ << " mW\n\n";

    os << "1M INT8 MAC: "
       << compute_mac_energy_mj(1e6, ComputePrecision::INT8)
       << " mJ\n";

    os << "1MB SRAM R:  "
       << sram_read_energy_mj(1024 * 1024)
       << " mJ\n";
}
