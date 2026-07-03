#!/bin/bash
# ---------------------------------------------------------------------------
# Distributed GPU Training Server Launcher
# ---------------------------------------------------------------------------
set -e

START_EPISODE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -se|--starting_episode) START_EPISODE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1. Usage: $0 [-se|--starting_episode <step>]"; exit 1 ;;
    esac
    shift
done

EXTRA_ARGS=""
if [ -n "$START_EPISODE" ]; then
    EXTRA_ARGS="--starting_episode $START_EPISODE"
    echo "Configured cluster to load/resume from checkpoint step $START_EPISODE"
else
    echo "No explicit starting episode specified. Server will auto-detect the latest checkpoint under ./checkpoint if available."
fi

# Ensure clean setup
mkdir -p ./checkpoint
mkdir -p ./data/gameplay
mkdir -p ./data/combats

echo "Clearing stale experience logs to prevent training data pollution..."
rm -f ./data/gameplay/*.pkl
rm -f ./data/combats/*.pkl

# Gracefully terminate all child processes on exit/signals
cleanup() {
    # Immediately disable traps to prevent recursive triggering
    trap - SIGINT SIGTERM EXIT
    echo "============================================================"
    echo "Terminating all distributed training processes..."
    echo "============================================================"
    kill 0 2>/dev/null
}

# Trap Ctrl+C (SIGINT), SIGTERM, and EXIT to trigger cleanup
trap cleanup SIGINT SIGTERM EXIT

# 1. Start the GPU Training Server
echo "============================================================"
echo "Starting GPU Training Server..."
echo "============================================================"
PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode train_server --checkpoint_interval 200 $EXTRA_ARGS &
SERVER_PID=$!

echo "============================================================"
echo "GPU Training Server is fully online!"
echo "Press Ctrl+C to terminate the server."
echo "============================================================"

# Wait for background processes
wait
