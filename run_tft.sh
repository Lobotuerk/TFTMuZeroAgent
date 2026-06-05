#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/MonteCarloTreeSearch:$SCRIPT_DIR/TFTSet4Gym${PYTHONPATH:+:$PYTHONPATH}"

exec "$@"
