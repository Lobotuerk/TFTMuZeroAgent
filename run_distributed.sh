#!/bin/bash
# ---------------------------------------------------------------------------
# Distributed Process-Level Training Orchestrator (Option A)
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

# Allow the server a brief moment to initialize the weights file
sleep 4

# 2. Start the Evaluator Worker (Worker ID 0)
echo "============================================================"
echo "Starting Evaluator Worker (Worker 0)..."
echo "============================================================"
PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode worker --worker_id 0 --worker_role evaluator --eval_games 8 --eval_concurrent 2 $EXTRA_ARGS &

# Give the evaluator a second to spawn
sleep 2

# 3. Start 6 self-play Collection Workers
for i in {1..6}
do
    echo "============================================================"
    echo "Starting Collection Worker $i..."
    echo "============================================================"
    # Each worker runs 1 concurrent game locally
    PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode worker --worker_id $i --worker_role collector --concurrent_games 2 &
    sleep 1
done

echo "============================================================"
echo "Distributed Training Cluster is fully online!"
echo "Press Ctrl+C to terminate the cluster."
echo "============================================================"

# Wait for background processes
wait
