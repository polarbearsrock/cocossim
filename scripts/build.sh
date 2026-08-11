#!/bin/bash

# COCOSSim Build Script
# Copyright (c) 2025 APEX Lab, Duke University

set -e

echo "Building COCOSSim..."

# Resolve the project directory so this script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

# Some managed environments expose compilers through ccache while the default
# cache directory is not writable. Keep cache and temporary files in TMPDIR
# when it is available.
if command -v ccache >/dev/null 2>&1 && [ -n "${TMPDIR:-}" ]; then
    export CCACHE_DIR="${CCACHE_DIR:-$TMPDIR/cocossim-ccache}"
    export CCACHE_TEMPDIR="${CCACHE_TEMPDIR:-$TMPDIR/cocossim-ccache-tmp}"
    mkdir -p "$CCACHE_DIR" "$CCACHE_TEMPDIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring build..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the project
echo "Compiling..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build complete! Executable: $BUILD_DIR/perf_model"
