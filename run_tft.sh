#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate TFT
else
    echo "Error: conda is not available. Please install conda and ensure the TFT environment exists."
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/MonteCarloTreeSearch:$SCRIPT_DIR/TFTSet4Gym${PYTHONPATH:+:$PYTHONPATH}"

exec "$@"
