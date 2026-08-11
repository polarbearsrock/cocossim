#!/bin/bash

# COCOSSim Example Runner
# Copyright (c) 2025 APEX Lab, Duke University

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
PERF_MODEL="$BUILD_DIR/perf_model"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
RESULTS_DIR="$PROJECT_ROOT/results"

# Check if build exists
if [ ! -x "$PERF_MODEL" ]; then
    echo "Error: COCOSSim not built. Run 'scripts/build.sh' first."
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Keep auxiliary DRAMSim3 statistics and jobs.dot in the build directory.
run_model() {
    (
        cd "$BUILD_DIR"
        ./perf_model "$@"
    )
}

echo "Running COCOSSim examples..."

# Run simple matrix multiplication
echo "1. Simple Matrix Multiplication"
run_model -c 1 -sa_sz 64 -vu_sz 64 -ws 0 -f 1 \
    -i "$EXAMPLES_DIR/simple_matmul.txt" \
    -o "$RESULTS_DIR/simple_matmul_results.txt"

# Run CNN model
echo "2. CNN Model"
run_model -c 1 -sa_sz 64 -vu_sz 64 -ws 0 -f 1 \
    -i "$EXAMPLES_DIR/cnn_model.txt" \
    -o "$RESULTS_DIR/cnn_results.txt"

# Run transformer model
echo "3. Transformer Model"
run_model -c 1 -sa_sz 64 -vu_sz 64 -ws 0 -f 1 \
    -i "$EXAMPLES_DIR/basic_transformer.txt" \
    -o "$RESULTS_DIR/transformer_results.txt"

# Compare different dataflow modes
echo "4. Dataflow Comparison (Output Stationary vs Weight Stationary)"
run_model -c 1 -sa_sz 64 -vu_sz 64 -ws 0 -f 1 \
    -i "$EXAMPLES_DIR/simple_matmul.txt" \
    -o "$RESULTS_DIR/os_dataflow_results.txt"

run_model -c 1 -sa_sz 64 -vu_sz 64 -ws 1 -f 1 \
    -i "$EXAMPLES_DIR/simple_matmul.txt" \
    -o "$RESULTS_DIR/ws_dataflow_results.txt"

echo "Examples completed! Results saved in $RESULTS_DIR/"
