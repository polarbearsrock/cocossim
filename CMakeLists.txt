cmake_minimum_required(VERSION 3.10)
project(perf_model)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(dramsim3)

include_directories(include)

# Buffer sources (shared)
set(BUFFER_SOURCES
    src/buffers/BufferConfig.cc
    src/buffers/BufferStats.cc
    src/buffers/BufferHierarchy.cc
)

# Core simulation sources
set(CORE_SOURCES
    src/Job.cc
    src/global.cc
    src/perf_enums.cc
    src/State.cc
    src/Arch.cc
    src/memory.cc
    src/EnqueueStructures.cc
    src/NNLayers.cc
    src/EnergyModel.cc
)

# Standard frontend (non-chiplet)
set(FRONTEND_SOURCES
    src/frontends/standard/StandardLayers.cc
    src/frontends/standard/StandardParser.cc
    src/frontends/standard/StandardArch.cc
    src/units/standard/SysArray.cc
    src/units/standard/VectorUnit.cc
)

# Chiplet sources
set(CHIPLET_SOURCES
    src/chiplets/UCIeConfig.cc
    src/chiplets/UCIePacket.cc
    src/chiplets/UCIeLink.cc
    src/chiplets/ChipletTopology.cc
    src/chiplets/TensorPartition.cc
    src/chiplets/ChipletArch.cc
)

# Main perf model (non-chiplet)
add_executable(perf_model
    src/main.cc
    ${CORE_SOURCES}
    ${BUFFER_SOURCES}
    ${FRONTEND_SOURCES}
)
target_link_libraries(perf_model PRIVATE dramsim3 args)

option(USE_VCD "Enable per-cycle VCD dumping of unit states" OFF)
if (${USE_VCD})
    message("VCD Dumping enabled!")
    add_compile_definitions(VCD=1)
endif()

# Chiplet perf model
# Links the core globals, buffer system, SA/VPU compute units (cycle-accurate),
# and chiplet infrastructure.  StandardParser.cc is NOT included — the chiplet
# main has its own argument parser.
add_executable(perf_model_chiplet
    src/main_chiplet.cc
    ${CORE_SOURCES}
    ${BUFFER_SOURCES}
    src/frontends/standard/StandardArch.cc
    src/frontends/standard/StandardLayers.cc
    src/units/standard/SysArray.cc
    src/units/standard/VectorUnit.cc
    ${CHIPLET_SOURCES}
)
target_include_directories(perf_model_chiplet PRIVATE include dramsim3/ext/headers)
target_link_libraries(perf_model_chiplet PRIVATE dramsim3)

# E1: UCIe link micro-validation (standalone, no DRAMSim3)
add_executable(test_ucie_validation
    tests/test_ucie_validation.cc
    src/chiplets/UCIeConfig.cc
    src/chiplets/UCIePacket.cc
    src/chiplets/UCIeLink.cc
)
target_include_directories(test_ucie_validation PRIVATE include)
