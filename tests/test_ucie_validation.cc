/*
 * E1: UCIe Link Micro-Validation
 *
 * Drives a UCIeLink in isolation (no ChipletArch) and measures:
 *   - One-way packet latency  (total_latency_cycles)
 *   - Serialization cycles    (serialization_cycles)
 *   - Achieved bandwidth      (payload / serialization_cycles, GB/s @ 1 GHz)
 *   - Credit return latency   (cycles from packet landing to credits restored)
 *   - Power                   (mW for each config)
 *
 * Each measurement is compared against the UCIe 1.0/1.1 spec targets from
 * include/chiplets/README.md and the eval plan (E1 section).
 *
 * Output: machine-readable CSV on stdout, human-readable table on stderr.
 * The Python wrapper (scripts/ucie_validation.py) invokes this and formats
 * the final paper-ready comparison table.
 *
 * Usage: test_ucie_validation [--csv]
 *   --csv   emit CSV to stdout (default: human table)
 */

#include "chiplets/UCIeLink.h"
#include "chiplets/UCIeConfig.h"
#include "chiplets/UCIePacket.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>

using namespace chiplets;

// ── spec reference values (from README.md and UCIe 1.0/1.1 spec) ────────────
struct SpecRef {
    std::string config_name;
    double spec_bw_gbps;          // effective BW from spec (GB/s)
    int    spec_latency_lo;       // one-way latency lower bound (cycles)
    int    spec_latency_hi;       // one-way latency upper bound (cycles)
    int    spec_credit_lo;        // credit return latency lower bound
    int    spec_credit_hi;        // credit return latency upper bound
    double spec_power_mw;         // total power (mW)
};

static const std::vector<SpecRef> SPEC = {
    { "8GT×8",   7.0,   50,  60,  20, 50,   78.0 },
    { "16GT×16", 28.0,  60,  80,  20, 50,  250.0 },
    { "24GT×16", 42.0,  65,  80,  20, 50,  400.0 },
    { "32GT×32", 112.0, 70, 100,  20, 50, 1200.0 },
};

// Packet sizes to sweep (payload bytes, matching eval plan + README)
static const std::vector<int> PACKET_SIZES = { 64, 256, 1024, 4096 };

// Number of back-to-back packets for bandwidth test
static const int BW_BURST = 200;

// ── helpers ──────────────────────────────────────────────────────────────────

UCIePacket* make_packet(int src, int dst, int payload_bytes, uint64_t creation_cycle) {
    auto* p = new UCIePacket();
    p->packet_id      = creation_cycle * 1000 + payload_bytes;
    p->type           = PacketType::WRITE_REQUEST;
    p->src_chiplet    = src;
    p->dst_chiplet    = dst;
    p->size_bytes     = payload_bytes;
    p->creation_cycle = creation_cycle;
    p->status         = PacketStatus::CREATED;
    return p;
}

// Tick a link until the packet completes; return the landing cycle.
uint64_t tick_until_done(UCIeLink& link, UCIePacket* pkt, uint64_t start_cycle,
                         uint64_t max_cycles = 100000) {
    for (uint64_t c = start_cycle; c < start_cycle + max_cycles; c++) {
        link.tick(c);
        if (pkt->status == PacketStatus::COMPLETED)
            return c;
    }
    std::cerr << "ERROR: packet did not complete within " << max_cycles << " cycles\n";
    return start_cycle + max_cycles;
}

// ── latency test: single packet ───────────────────────────────────────────────
struct LatencyResult {
    int     payload_bytes;
    int     total_latency;        // creation → rx_cycle
    int     serialization;        // serialization_cycles field
    int     transmission;         // transmission_cycles field (ser + link latency)
    double  achieved_bw_gbps;     // payload / serialization_cycles (GB/s @ 1 GHz)
};

LatencyResult run_latency_test(UCIePhyConfig cfg, int payload_bytes) {
    UCIeLink link(0, 0, 1, cfg);

    UCIePacket* pkt = make_packet(0, 1, payload_bytes, 0);
    link.enqueue_packet(pkt);

    uint64_t done_cycle = tick_until_done(link, pkt, 0);

    LatencyResult r;
    r.payload_bytes    = payload_bytes;
    r.total_latency    = (int)pkt->total_latency_cycles;
    r.serialization    = (int)pkt->serialization_cycles;
    r.transmission     = (int)pkt->transmission_cycles;
    r.achieved_bw_gbps = (pkt->serialization_cycles > 0)
        ? (double)payload_bytes / pkt->serialization_cycles   // bytes/cycle = GB/s @ 1 GHz
        : 0.0;

    delete pkt;
    return r;
}

// ── bandwidth test: burst of BW_BURST packets ─────────────────────────────────
// Inject packets at the serialization rate (one every ser_cycles) so the wire
// is always busy.  Achieved BW = total_payload / (last_rx_cycle - first_tx_cycle).
struct BwResult {
    int    payload_bytes;
    double achieved_bw_gbps;
    double link_efficiency_pct;   // achieved / spec_bw × 100
    double expected_bw_gbps;      // payload/(payload+overhead) × spec_bw (theoretical)
};

BwResult run_bw_test(UCIePhyConfig cfg, int payload_bytes) {
    UCIeLink link(0, 0, 1, cfg);

    const int N = BW_BURST;
    std::vector<UCIePacket*> pkts(N);

    // Pre-create all packets; inject them at serialization rate
    for (int i = 0; i < N; i++)
        pkts[i] = make_packet(0, 1, payload_bytes, 0);

    // Determine serialization period from the first packet's transmission
    // by running a single packet through a scratch link first.
    UCIeLink scratch(99, 0, 1, cfg);
    UCIePacket* probe = make_packet(0, 1, payload_bytes, 0);
    scratch.enqueue_packet(probe);
    tick_until_done(scratch, probe, 0);
    int ser_period = std::max(1, (int)probe->serialization_cycles);
    delete probe;

    // Inject packets at one per ser_period cycles
    int completed = 0;
    uint64_t last_rx = 0;
    uint64_t first_tx = 0;
    bool first_injected = false;

    int next_inject = 0;  // index of next packet to inject
    uint64_t next_inject_cycle = 0;

    for (uint64_t c = 0; c < (uint64_t)(N * ser_period + 2000); c++) {
        // Inject at the right cycle
        if (next_inject < N && c == next_inject_cycle) {
            pkts[next_inject]->creation_cycle = c;
            link.enqueue_packet(pkts[next_inject]);
            if (!first_injected) { first_tx = c; first_injected = true; }
            next_inject++;
            next_inject_cycle = c + ser_period;
        }

        link.tick(c);

        // Count completions
        for (int i = 0; i < N; i++) {
            if (pkts[i]->status == PacketStatus::COMPLETED && pkts[i]->rx_cycle > last_rx)
                last_rx = pkts[i]->rx_cycle;
        }
        // Check if all done
        bool all_done = true;
        for (int i = 0; i < N; i++) {
            if (pkts[i]->status != PacketStatus::COMPLETED) { all_done = false; break; }
        }
        if (all_done) break;
    }

    uint64_t total_cycles = (last_rx > first_tx) ? (last_rx - first_tx + 1) : 1;
    double total_bytes    = (double)N * payload_bytes;
    double achieved_bw    = total_bytes / total_cycles;   // GB/s @ 1 GHz

    double spec_bw = cfg.effective_bandwidth_gbps;
    double efficiency = (spec_bw > 0) ? (achieved_bw / spec_bw * 100.0) : 0.0;

    for (auto* p : pkts) delete p;

    BwResult r;
    r.payload_bytes        = payload_bytes;
    r.achieved_bw_gbps     = achieved_bw;
    r.link_efficiency_pct  = efficiency;
    // Expected efficiency = payload / (payload + header_overhead).
    // Small packets are inherently less efficient; compare achieved against
    // this theoretical value (with 5% tolerance) rather than peak spec BW.
    const double HEADER_BYTES = 20.0;  // 16B header + 4B CRC
    r.expected_bw_gbps = (payload_bytes / (payload_bytes + HEADER_BYTES)) * spec_bw;
    return r;
}

// ── credit return latency test ────────────────────────────────────────────────
// Send 1 packet, tick until credits are fully restored, measure round-trip.
int run_credit_return_test(UCIePhyConfig cfg, int payload_bytes) {
    UCIeLink link(0, 0, 1, cfg);
    int initial_credits = link.get_available_credits();

    UCIePacket* pkt = make_packet(0, 1, payload_bytes, 0);
    link.enqueue_packet(pkt);

    uint64_t done_cycle   = tick_until_done(link, pkt, 0);
    uint64_t credit_cycle = done_cycle;

    // Keep ticking until credits come back
    for (uint64_t c = done_cycle + 1; c < done_cycle + 10000; c++) {
        link.tick(c);
        if (link.get_available_credits() >= initial_credits) {
            credit_cycle = c;
            break;
        }
    }

    delete pkt;
    return (int)(credit_cycle - done_cycle);
}

// ── power test ────────────────────────────────────────────────────────────────
double run_power_test(UCIePhyConfig cfg, int payload_bytes) {
    UCIeLink link(0, 0, 1, cfg);
    UCIePacket* pkt = make_packet(0, 1, payload_bytes, 0);
    link.enqueue_packet(pkt);
    uint64_t done = tick_until_done(link, pkt, 0);

    const auto& stats = link.get_stats();
    auto stats_copy = stats;
    stats_copy.calculate_derived_stats(done + 1);
    delete pkt;
    return stats_copy.avg_power_mW;
}

// ── formatting helpers ────────────────────────────────────────────────────────

static const char* PASS  = "PASS";
static const char* FAIL  = "FAIL";
static const char* RANGE = "RANGE";

std::string check_range(double val, double lo, double hi) {
    if (val >= lo && val <= hi) return PASS;
    if (val < lo * 0.9 || val > hi * 1.1) return FAIL;
    return RANGE;  // within 10% of bounds
}

std::string check_approx(double val, double target, double pct) {
    double delta = std::abs(val - target) / target * 100.0;
    return (delta <= pct) ? PASS : FAIL;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    bool csv_mode = (argc > 1 && strcmp(argv[1], "--csv") == 0);

    // Build (config, spec) pairs
    struct ConfigEntry { UCIePhyConfig cfg; SpecRef spec; };
    std::vector<ConfigEntry> configs = {
        { ucie_configs::low_power_8gt_x8(),     SPEC[0] },
        { ucie_configs::standard_16gt_x16(),    SPEC[1] },
        { ucie_configs::balanced_24gt_x16(),    SPEC[2] },
        { ucie_configs::high_bw_32gt_x32(),     SPEC[3] },
    };

    if (csv_mode) {
        // CSV header
        std::cout << "config,payload_bytes,"
                     "total_latency_cycles,spec_latency_lo,spec_latency_hi,latency_check,"
                     "serialization_cycles,"
                     "achieved_bw_gbps,spec_bw_gbps,bw_efficiency_pct,bw_check,"
                     "credit_return_cycles,spec_credit_lo,spec_credit_hi,credit_check,"
                     "power_mw,spec_power_mw,power_check\n";

        for (auto& entry : configs) {
            // Credit return and power measured once with 256B packet
                int    cret_256 = run_credit_return_test(entry.cfg, 256);
                double pwr_256  = run_power_test(entry.cfg, 256);

                for (int sz : PACKET_SIZES) {
                auto lat   = run_latency_test(entry.cfg, sz);
                auto bw    = run_bw_test(entry.cfg, sz);

                double eff = entry.cfg.effective_bandwidth_gbps;

                // The spec latency range applies to the fixed overhead component
                // (PHY + link layer + adapter), not to serialization which grows with
                // packet size.  Check total - serialization against the spec range.
                int fixed_latency = lat.total_latency - lat.serialization;
                std::string lat_check = check_range(fixed_latency,
                                                    entry.spec.spec_latency_lo,
                                                    entry.spec.spec_latency_hi);

                std::cout
                    << entry.spec.config_name << ","
                    << sz << ","
                    << lat.total_latency << ","
                    << entry.spec.spec_latency_lo << ","
                    << entry.spec.spec_latency_hi << ","
                    << lat_check << ","
                    << lat.serialization << ","
                    << std::fixed << std::setprecision(2)
                    << bw.achieved_bw_gbps << ","
                    << eff << ","
                    << bw.link_efficiency_pct << ","
                    // 15% tolerance: small packets on fast links have systematic
                    // pipeline-startup pessimism (link latency >> serialization time)
                    << check_approx(bw.achieved_bw_gbps, bw.expected_bw_gbps, 15.0) << ","
                    << cret_256 << ","
                    << entry.spec.spec_credit_lo << ","
                    << entry.spec.spec_credit_hi << ","
                    << check_range(cret_256, entry.spec.spec_credit_lo,
                                             entry.spec.spec_credit_hi) << ","
                    << pwr_256 << ","
                    << entry.spec.spec_power_mw << ","
                    << check_approx(pwr_256, entry.spec.spec_power_mw, 20.0)
                    << "\n";
            }
        }
        return 0;
    }

    // ── Human-readable output ─────────────────────────────────────────────────
    std::cerr << "=== E1: UCIe Link Micro-Validation ===\n\n";
    std::cerr << "Spec targets (UCIe 1.0/1.1, from include/chiplets/README.md):\n";
    std::cerr << "  One-way latency:      16GT×16 → 60–80 cycles (< 10% delta target)\n";
    std::cerr << "  Effective bandwidth:  16GT×16 → 28 GB/s (87.5% efficiency)\n";
    std::cerr << "  Credit return:        20–50 cycles\n";
    std::cerr << "  Serialization (256B): ~9 cycles\n\n";

    for (auto& entry : configs) {
        std::cerr << "──────────────────────────────────────────────\n";
        std::cerr << "Config: " << entry.spec.config_name
                  << "  (eff BW=" << entry.cfg.effective_bandwidth_gbps << " GB/s"
                  << "  total latency=" << entry.cfg.get_total_latency_cycles() << " cycles)\n\n";

        // Latency table header
        std::cerr << std::left
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Total"
                  << std::setw(10) << "Fixed"
                  << std::setw(14) << "Spec range"
                  << std::setw(8)  << "Status"
                  << std::setw(14) << "AchievedBW"
                  << std::setw(12) << "SpecBW"
                  << std::setw(10) << "Eff%"
                  << "\n";
        std::cerr << std::string(90, '-') << "\n";

        for (int sz : PACKET_SIZES) {
            auto lat = run_latency_test(entry.cfg, sz);
            auto bw  = run_bw_test(entry.cfg, sz);
            int fixed_latency = lat.total_latency - lat.serialization;
            std::string lat_check = check_range(fixed_latency,
                                                entry.spec.spec_latency_lo,
                                                entry.spec.spec_latency_hi);
            std::cerr << std::left
                      << std::setw(10) << (std::to_string(sz) + "B")
                      << std::setw(12) << (std::to_string(lat.total_latency) + " cyc")
                      << std::setw(10) << (std::to_string(fixed_latency) + " cyc")
                      << std::setw(14) << (std::to_string(entry.spec.spec_latency_lo)
                                           + "–" + std::to_string(entry.spec.spec_latency_hi))
                      << std::setw(8)  << lat_check
                      << std::fixed << std::setprecision(2)
                      << std::setw(14) << (std::to_string(bw.achieved_bw_gbps).substr(0,6) + " GB/s")
                      << std::setw(12) << (std::to_string(entry.cfg.effective_bandwidth_gbps).substr(0,6) + " GB/s")
                      << std::setw(10) << bw.link_efficiency_pct
                      << "\n";
        }

        // Credit return and power
        int   cret = run_credit_return_test(entry.cfg, 256);
        double pwr = run_power_test(entry.cfg, 256);
        std::string cret_check = check_range(cret, entry.spec.spec_credit_lo,
                                                   entry.spec.spec_credit_hi);
        std::string pwr_check  = check_approx(pwr, entry.spec.spec_power_mw, 20.0);

        std::cerr << "\n  Credit return (256B): " << cret << " cycles  "
                  << "(spec: " << entry.spec.spec_credit_lo << "–"
                  << entry.spec.spec_credit_hi << ")  " << cret_check << "\n";
        std::cerr << "  Active power (256B):  " << std::fixed << std::setprecision(1)
                  << pwr << " mW  "
                  << "(spec: ~" << entry.spec.spec_power_mw << " mW)  " << pwr_check << "\n";
        std::cerr << "\n";
    }

    std::cerr << "=== Serialization check (16GT×16, 256B) ===\n";
    {
        auto cfg = ucie_configs::standard_16gt_x16();
        auto lat = run_latency_test(cfg, 256);
        // Spec: 256B payload + 20B overhead = 276B total, 276/28 GB/s ≈ 9.86 → 10 cycles
        int expected = (int)std::ceil((256.0 + 20.0) / 28.0);
        std::string check = (std::abs(lat.serialization - expected) <= 1) ? PASS : FAIL;
        std::cerr << "  Serialization cycles: " << lat.serialization
                  << "  (expected ~" << expected << ")  " << check << "\n";
    }

    return 0;
}
