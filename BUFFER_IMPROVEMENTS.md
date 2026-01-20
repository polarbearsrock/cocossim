# Buffer Flexibility and Power Analysis Improvements

**Branch:** `feature/flexible-buffer-power-analysis`
**Status:** Phase 1 Complete ✓

## Overview

This document tracks the improvements to COCOSSim's buffer/SRAM structure to enable flexible hierarchies and power analysis capabilities.

## Motivation

The original COCOSSim implementation had:
- Single global buffer size constant (`buffer_size_bytes = 64MB`)
- No hierarchical buffer modeling (L1/L2/unified)
- No power/energy tracking for on-chip buffers
- Limited flexibility for architectural exploration

These improvements enable:
- **Power analysis** for buffer accesses (critical for accelerator design)
- **Flexible buffer hierarchies** (L1/L2/unified/scratchpad architectures)
- **Multi-core buffer partitioning**
- **Detailed access statistics** for optimization studies

---

## Phase 1: Infrastructure (✓ COMPLETED)

### What Was Added

#### 1. Core Buffer Classes

**`include/buffers/BufferConfig.h`**
- `PowerConfig`: Read/write energy, static power parameters
- `BufferLevelConfig`: Per-level configuration (size, bandwidth, latency, banking, power)
- `BufferHierarchyConfig`: Full hierarchy configuration
- Preset builders: `create_tpu_v3_style()`, `create_eyeriss_style()`, `create_simple_two_level()`

**`include/buffers/BufferStats.h`**
- `BufferAccessStats`: Tracks reads, writes, bytes, bank conflicts, occupancy
- Per-operation breakdown (matmul, conv, softmax)
- `EnergyBreakdown`: Total energy, per-level, per-operation

**`include/buffers/BufferHierarchy.h`**
- `BufferLevel`: Single buffer level with access tracking
- `BufferHierarchy`: Multi-level hierarchy manager
- Capacity queries, allocation tracking, utilization reporting

**`include/buffers/PowerModel.h`**
- `PowerModel`: Energy calculation utilities
- Read/write/static energy computation
- Technology scaling (7nm, 5nm, etc.)
- Power report generation

#### 2. Integration Points

- **`global.h`**: Added `global_buffer_hierarchy` pointer (forward-compatible)
- **`RuntimeStats_t.h`**: Added `energy` and `buffer_stats` fields
- **`CMakeLists.txt`**: Included new buffer source files

#### 3. Build System

All files compile successfully with no errors:
```bash
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j4
# ✓ perf_model builds successfully
```

### Preset Configurations

Three architectural templates provided:

1. **TPU v3 Style**: 64MB unified buffer + HBM2 DRAM
2. **Eyeriss Style**: 256KB L1 scratchpad + 16MB L2 + DDR4 DRAM
3. **Simple Two-Level**: Configurable SRAM + DRAM

### Backward Compatibility

- `buffer_size_bytes` constant preserved
- Existing code continues to work unchanged
- New buffer hierarchy is opt-in

---

## Phase 2: Integration with Simulation (TODO)

### 2.1 Update State Class

**File:** `include/State.h`, `src/State.cc`

```cpp
struct State {
    // Add buffer tracking
    BufferLevel* assigned_buffer;     // Which buffer this unit uses
    std::string current_op_type;      // "matmul", "conv", etc.

    // Modified methods
    void enqueue_reads() {
        // Track buffer accesses
        if (global_buffer_hierarchy) {
            assigned_buffer->record_read(addr, bytes, current_op_type);
        }
        // ... existing DRAM logic
    }

    void enqueue_writes() {
        // Similar buffer tracking
    }
};
```

**Effort:** 1-2 days
**Benefit:** Actual buffer access tracking during simulation

### 2.2 Update Layer Parser

**File:** `src/frontends/standard/StandardLayers.cc`

```cpp
// Line 73: Replace hardcoded buffer check
// OLD:
bool core_is_bufferable = required_buff_sz_per_core <= buffer_size_bytes;

// NEW:
bool core_is_bufferable = global_buffer_hierarchy->fits_in_level(0, required_buff_sz_per_core);
```

**Effort:** 1 day
**Benefit:** Use actual buffer hierarchy for capacity checks

### 2.3 Initialize Buffer Hierarchy in Main

**File:** `src/main.cc`

```cpp
int main(int argc, char **argv) {
    // Early initialization
    auto buffer_config = buffers::configs::create_tpu_v3_style();
    global_buffer_hierarchy = new buffers::BufferHierarchy(buffer_config);

    global_buffer_hierarchy->print_hierarchy(std::cout);

    // ... existing simulation code ...

    // After simulation
    auto energy = buffers::PowerModel::calculate_energy(
        *global_buffer_hierarchy,
        gcycles,
        freq_sa
    );

    buffers::PowerModel::print_power_report(
        std::cout, energy, *global_buffer_hierarchy, gcycles, freq_sa
    );

    delete global_buffer_hierarchy;
}
```

**Effort:** 1 day
**Benefit:** Power reports in every simulation run

### 2.4 Update Arch::get_cycles

**File:** `src/Arch.cc`

```cpp
RuntimeStats_t *Arch::get_cycles(TimeBasedEnqueue &time_enqueues) {
    // ... existing simulation loop ...

    // Per-cycle: update buffer hierarchy
    for (auto* state : states) {
        if (state->assigned_buffer) {
            state->assigned_buffer->tick(gcycles);
        }
    }

    // End of phase: collect stats
    if (global_buffer_hierarchy) {
        stats[phase_idx].buffer_stats = new buffers::BufferAccessStats(
            global_buffer_hierarchy->get_total_stats()
        );

        stats[phase_idx].energy = new buffers::EnergyBreakdown(
            buffers::PowerModel::calculate_energy(
                *global_buffer_hierarchy, phase_cycles, freq_sa
            )
        );
    }
}
```

**Effort:** 1 day
**Benefit:** Per-phase energy tracking

---

## Phase 3: Command-Line Interface (TODO)

### 3.1 Add Arguments

**File:** `src/frontends/standard/StandardParser.cc`

New command-line options:
```bash
-buffer_config <file>     # Load buffer hierarchy from JSON/config file
-power_report <file>      # Save detailed power report to file
-buffer_trace <file>      # Save detailed access trace (debugging)
-buffer_preset <name>     # Use preset: "tpu", "eyeriss", "simple"
```

**Effort:** 2 days
**Benefit:** Easy configuration without recompilation

### 3.2 JSON Configuration Loading

Create `configs/tpu_v3.json`:
```json
{
    "buffer_hierarchy": {
        "use_unified_buffer": true,
        "levels": [
            {
                "name": "UnifiedBuffer",
                "size_MB": 64,
                "bandwidth_GB_per_s": 2000,
                "power": {
                    "read_energy_pJ_per_byte": 2.5,
                    "write_energy_pJ_per_byte": 3.0,
                    "static_power_mW": 500
                }
            }
        ]
    }
}
```

**Effort:** 2 days (with JSON library integration)
**Benefit:** Easy experimentation with different configs

---

## Phase 4: Validation and Testing (TODO)

### 4.1 Unit Tests

Create `tests/buffer_tests.cc`:
- Test buffer allocation/deallocation
- Test access tracking
- Test energy calculation
- Test multi-core partitioning

**Effort:** 2 days

### 4.2 Validation Against Known Hardware

Compare power estimates to published data:
- Google TPU v3 power numbers
- Eyeriss energy efficiency
- CACTI estimates for SRAM at various sizes

**Effort:** 2-3 days
**Benefit:** Confidence in power model accuracy

### 4.3 Regression Testing

Ensure existing simulations produce same cycle counts:
```bash
# Run all examples with new buffer infrastructure
for example in examples/*.txt; do
    ./perf_model -c 1 -sa_sz 64 -vu_sz 64 -i $example -o results.txt
done
```

**Effort:** 1 day

---

## Phase 5: Advanced Features (FUTURE)

### 5.1 CACTI Integration

Automatically generate power parameters from CACTI:
```cpp
auto config = buffers::PowerModel::estimate_from_cacti(
    32 * 1024 * 1024,  // 32MB
    7,                 // 7nm technology
    1.0                // 1 GHz frequency
);
```

**Effort:** 1 week

### 5.2 Cache Modeling

Add cache replacement policies:
- LRU (Least Recently Used)
- FIFO (First In First Out)
- Optimal (for analysis)

**Effort:** 1 week

### 5.3 Data Movement Optimization

Suggest optimal data placement:
```cpp
hierarchy.suggest_placement(layer_size, access_pattern);
```

**Effort:** 2 weeks

### 5.4 Thermal Modeling

Add temperature tracking:
- Power → Heat generation
- Temperature impact on leakage
- Throttling simulation

**Effort:** 2 weeks

---

## Usage Examples

### Basic Usage (Phase 2)

```bash
# Use default TPU-style buffer with power analysis
./perf_model -c 1 -sa_sz 64 -vu_sz 64 -f 1.0 \
    -i examples/holonet_p1.txt \
    -o results.txt

# Output includes:
# - Standard cycle counts
# - Buffer access statistics
# - Energy breakdown (read/write/static)
# - Per-operation energy
```

### Advanced Usage (Phase 3)

```bash
# Custom buffer configuration
./perf_model -c 1 -sa_sz 64 -vu_sz 64 \
    -buffer_config configs/custom_buffer.json \
    -power_report power_analysis.txt \
    -i model.txt -o results.txt

# Compare different buffer sizes
./perf_model -buffer_preset tpu -i model.txt -o tpu_results.txt
./perf_model -buffer_preset eyeriss -i model.txt -o eyeriss_results.txt
```

---

## Estimated Timeline

| Phase | Description | Effort | Status |
|-------|-------------|--------|--------|
| 1 | Infrastructure | 1 week | ✓ DONE |
| 2 | Integration | 1 week | TODO |
| 3 | CLI & Config | 1 week | TODO |
| 4 | Validation | 1 week | TODO |
| 5 | Advanced Features | 4-6 weeks | FUTURE |

**Total for Phases 1-4:** ~4 weeks
**Total including Phase 5:** ~10 weeks

---

## Testing Checklist

- [ ] Phase 1 builds successfully (✓ DONE)
- [ ] Unit tests for buffer classes pass
- [ ] Integration with State class tracks accesses correctly
- [ ] Power calculations match hand calculations
- [ ] Existing simulations produce same cycle counts
- [ ] Power numbers are reasonable (compare to literature)
- [ ] JSON configuration loading works
- [ ] Command-line arguments function correctly
- [ ] Documentation is complete

---

## References

### Papers
- ISPASS 2025: COCOSSim paper
- ISCA 2016: Eyeriss paper
- ISCA 2017: Google TPU v1 paper

### Tools
- CACTI: Cache power/area/timing modeling
- DRAMSim3: DRAM power modeling (already integrated)

### Accelerator Power Data
- TPU v3: ~250W total (includes compute + memory)
- Eyeriss: 278 mW @ 200 MHz
- SRAM: 1-5 pJ/byte typical for 7nm

---

## Questions / Decisions

1. **Power parameter sources**: Use CACTI? Literature? Measurements?
   - **Decision**: Start with literature values, validate with CACTI later

2. **DRAMSim3 integration**: Extract power from DRAMSim3?
   - **Decision**: Phase 5 - requires DRAMSim3 power model understanding

3. **Operation type tracking**: How granular?
   - **Decision**: Start with layer types (matmul, conv, softmax)

4. **Multi-level data movement**: Model transfers between levels?
   - **Decision**: Phase 5 - requires more complex tracking

---

## Git Workflow

```bash
# Current branch
git branch
# * feature/flexible-buffer-power-analysis

# View changes
git log --oneline

# Push to remote (when ready)
git push -u origin feature/flexible-buffer-power-analysis

# Create pull request for review
```

---

## Contact / Notes

- Implementation by: Claude + Mansi
- Date started: 2026-01-20
- Branch: `feature/flexible-buffer-power-analysis`
- Base commit: 8da7e20 (cnn example fixed)

---

## Summary

**What works now:**
- ✓ Complete buffer hierarchy infrastructure
- ✓ Power modeling framework
- ✓ Access statistics tracking
- ✓ Preset configurations
- ✓ Compiles successfully

**Next steps to make it fully functional:**
1. Integrate with State class to actually track accesses (Phase 2)
2. Add command-line arguments (Phase 3)
3. Generate power reports automatically (Phase 2-3)
4. Validate against known hardware (Phase 4)

**Estimated time to production-ready:** 3-4 weeks
