cmake_minimum_required(VERSION 3.16)
project(cocossim_compute_ratio LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT COCOSSIM_SOURCE)
  message(FATAL_ERROR "Set COCOSSIM_SOURCE to the extended cocossim-official checkout")
endif()
add_executable(compute_ratio
  runner.cc compute_state.cc
  ${COCOSSIM_SOURCE}/src/units/standard/SysArray.cc
  ${COCOSSIM_SOURCE}/src/units/standard/VectorUnit.cc
  ${COCOSSIM_SOURCE}/src/Job.cc
  ${COCOSSIM_SOURCE}/src/global.cc
  ${COCOSSIM_SOURCE}/src/perf_enums.cc)
target_include_directories(compute_ratio PRIVATE ${COCOSSIM_SOURCE}/include)
target_compile_options(compute_ratio PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(compute_ratio PRIVATE stdc++fs)
endif()
