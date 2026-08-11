#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAMSIM_DIR="$PROJECT_ROOT/dramsim3"

if [ -d "$DRAMSIM_DIR/.git" ]; then
    echo "DRAMSim3 is already installed; refreshing its nested submodules."
    git -C "$DRAMSIM_DIR" submodule update --init --recursive
elif [ -e "$DRAMSIM_DIR" ]; then
    echo "Error: $DRAMSIM_DIR exists but is not a Git checkout." >&2
    exit 1
else
    git clone --recursive https://github.com/umd-memsys/DRAMsim3.git "$DRAMSIM_DIR"
fi
