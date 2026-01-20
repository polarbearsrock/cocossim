# Buffer Hierarchy and Power Analysis

This directory contains the flexible buffer hierarchy and power modeling infrastructure for COCOSSim.

## Overview

The buffer subsystem enables:
- **Hierarchical buffer modeling** (L1, L2, unified buffer, DRAM)
- **Per-level power analysis** with read/write energy tracking
- **Multi-core partitioning** support
- **Access statistics** for performance analysis
- **Configurable architectures** via preset or custom configurations

## Architecture

```
BufferHierarchy
├── BufferLevel 0 (e.g., L1 Scratchpad)
│   ├── Configuration (size, bandwidth, latency, banking)
│   ├── Power params (read/write energy, static power)
│   └── Statistics (accesses, occupancy, conflicts)
├── BufferLevel 1 (e.g., Unified Buffer)
└── BufferLevel 2 (e.g., DRAM)
```

## Files

- **BufferConfig.h/cc**: Configuration structures and preset builders
- **BufferStats.h/cc**: Access tracking and statistics
- **BufferHierarchy.h/cc**: Main hierarchy manager
- **PowerModel.h/cc**: Energy calculation utilities

## Usage

### Creating a Buffer Hierarchy

```cpp
#include "buffers/BufferConfig.h"
#include "buffers/BufferHierarchy.h"

// Use preset configuration
auto config = buffers::configs::create_tpu_v3_style();
buffers::BufferHierarchy hierarchy(config);

// Or build custom configuration
buffers::BufferHierarchyConfig custom_config;
custom_config.use_unified_buffer = true;

buffers::BufferLevelConfig sram;
sram.name = "OnChipSRAM";
sram.size_bytes = 32 * 1024 * 1024;  // 32MB
sram.power.read_energy_pJ_per_byte = 2.5;
// ... configure other parameters

custom_config.levels.push_back(sram);
```

### Recording Accesses

```cpp
// Get buffer level
buffers::BufferLevel* buffer = hierarchy.get_level(0);

// Allocate space
buffer->allocate(0x1000, 1024 * 1024);  // 1MB at address 0x1000

// Record accesses
buffer->record_read(0x1000, 4096, "matmul");
buffer->record_write(0x2000, 8192, "conv");

// Check capacity
if (buffer->can_allocate(size_needed)) {
    // ...
}
```

### Calculating Power

```cpp
#include "buffers/PowerModel.h"

// After simulation
auto energy = buffers::PowerModel::calculate_energy(
    hierarchy,
    total_cycles,
    frequency_GHz
);

// Print report
buffers::PowerModel::print_power_report(
    std::cout,
    energy,
    hierarchy,
    total_cycles,
    frequency_GHz
);
```

## Preset Configurations

### TPU v3 Style
```cpp
auto config = buffers::configs::create_tpu_v3_style();
```
- 64MB unified buffer
- HBM2 DRAM
- 32 banks
- 7nm power parameters

### Eyeriss Style
```cpp
auto config = buffers::configs::create_eyeriss_style();
```
- 256KB per-PE scratchpad
- 16MB global buffer
- DDR4 DRAM
- Per-core partitioned

### Simple Two-Level
```cpp
auto config = buffers::configs::create_simple_two_level(
    32 * 1024 * 1024,  // 32MB SRAM
    "../dramsim3/configs/HBM2_8Gb_x128.ini"
);
```

## Power Parameters

Energy values are in **picojoules per byte (pJ/byte)**:
- **SRAM (on-chip)**: 1-5 pJ/byte
- **DRAM (off-chip)**: 20-100 pJ/byte (handled by DRAMSim3)

Static power is in **milliwatts (mW)**.

### Technology Scaling

```cpp
buffers::PowerModel::apply_technology_scaling(config, 5);  // Scale to 5nm
```

## Statistics Collected

Per buffer level:
- Number of reads/writes
- Bytes read/written
- Bank conflicts
- Stall cycles
- Peak/average occupancy
- Per-operation breakdown (matmul, conv, softmax, etc.)

## Multi-Core Support

```cpp
config.partitioned_per_core = true;
config.num_partitions = 4;

// Access per-core partition
buffers::BufferLevel* core0_buffer = hierarchy.get_partition(0, 0);
```

## Example Output

```
=== Power Analysis Report ===

Simulation Time: 1247832 cycles @ 1.0 GHz = 1.248 ms

Total Energy: 1166.7 mJ
Average Power: 0.935 W

Energy Breakdown:
  Read Energy:   785.4 mJ (67.3%)
  Write Energy:  219.1 mJ (18.8%)
  Static Energy: 162.2 mJ (13.9%)

Energy per Buffer Level:
  UnifiedBuffer: 274.2 mJ (23.5%)
  DRAM: 892.5 mJ (76.5%)

=== Buffer Access Stats: UnifiedBuffer ===
  Reads:        49800000 (119.5 MB)
  Writes:       29100000 (87.3 MB)
  Peak Occupancy: 54.3 MB
```

## Integration with COCOSSim

The buffer hierarchy is integrated into COCOSSim through:

1. **Global hierarchy** (`global_buffer_hierarchy` in global.h)
2. **Per-unit access tracking** (in State class)
3. **Runtime statistics** (in RuntimeStats_t)
4. **Command-line configuration** (upcoming)

## Future Enhancements

- [ ] CACTI integration for automatic power parameter generation
- [ ] Cache replacement policies (LRU, FIFO)
- [ ] Multi-level data movement optimization
- [ ] Thermal modeling
- [ ] JSON configuration file loading
- [ ] Visualization of buffer access patterns
- [ ] Integration with DRAMSim3 power models

## References

- CACTI: Cache access time, cycle time, area, leakage, and dynamic power
- DRAMSim3: Memory system simulator
- TPU v3: Google's third-generation Tensor Processing Unit
- Eyeriss: Energy-efficient deep learning accelerator
