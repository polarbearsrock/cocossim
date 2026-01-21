# Buffer Hierarchy Infrastructure

This directory contains the flexible buffer hierarchy infrastructure for COCOSSim.

## Overview

The buffer subsystem enables:
- **Hierarchical buffer modeling** (L1, L2, unified buffer, DRAM)
- **Access statistics tracking** for reads, writes, occupancy, and bank conflicts
- **Multi-core partitioning** support
- **Performance analysis** of buffer utilization
- **Configurable architectures** via preset or custom configurations

**Note:** Power modeling will be added later in a separate subsystem when power data is available for all simulator components (systolic arrays, vector units, buffers, etc.).

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

### Viewing Statistics

```cpp
// Print buffer access statistics
auto stats = hierarchy.get_total_stats();
stats.print_summary(std::cout, "Total");

// Per-level statistics
for (int i = 0; i < hierarchy.get_num_levels(); ++i) {
    auto level_stats = hierarchy.get_level_stats(i);
    level_stats.print_summary(std::cout, hierarchy.get_level(i)->get_config().name);
}

// Print utilization
hierarchy.print_utilization(std::cout);
```

## Preset Configurations

### TPU Style (Default)
```cpp
auto config = buffers::configs::create_tpu_style();
```
- 64MB unified buffer
- HBM2 DRAM
- 32 banks

### Simple Two-Level
```cpp
auto config = buffers::configs::create_simple_two_level(
    32 * 1024 * 1024,  // 32MB SRAM
    "../dramsim3/configs/HBM2_8Gb_x128.ini"
);
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
=== Buffer Access Stats: UnifiedBuffer ===
  Reads:        49800000 (119.5 MB)
  Writes:       29100000 (87.3 MB)
  Total Bytes:  206800000
  Bank Conflicts: 1284
  Peak Occupancy: 54.3 MB

  Reads by Operation:
    matmul: 38500000 reads (92.1 MB)
    conv: 8300000 reads (19.9 MB)
    softmax: 3000000 reads (7.2 MB)

  Writes by Operation:
    matmul: 22100000 writes (52.9 MB)
    conv: 5200000 writes (12.4 MB)
    softmax: 1800000 writes (4.3 MB)

=== Buffer Utilization ===
UnifiedBuffer: 84.8% (54.3 / 64.0 MB)
```

## Integration with COCOSSim

The buffer hierarchy is integrated into COCOSSim through:

1. **Global hierarchy** (`global_buffer_hierarchy` in global.h)
2. **Per-unit access tracking** (in State class)
3. **Runtime statistics** (in RuntimeStats_t)
4. **Command-line configuration** (upcoming)

## Future Enhancements

- [ ] Power modeling integration (separate subsystem)
- [ ] Cache replacement policies (LRU, FIFO)
- [ ] Multi-level data movement optimization
- [ ] JSON configuration file loading
- [ ] Visualization of buffer access patterns
- [ ] Command-line configuration options

## References

- DRAMSim3: Memory system simulator (integrated)
- ISPASS 2025: COCOSSim paper
