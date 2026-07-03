#!/bin/bash
# ---------------------------------------------------------------------------
# Distributed Worker Launcher (Evaluator + Collectors)
# ---------------------------------------------------------------------------
set -e

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXEC="$(which python)"
fi

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
    echo "No explicit starting episode specified."
fi

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

# 1. Start the Evaluator Worker (Worker ID 0)
echo "============================================================"
echo "Starting Evaluator Worker (Worker 0)..."
echo "============================================================"
PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode worker --worker_id 0 --worker_role evaluator --eval_games 9 --eval_concurrent 3 $EXTRA_ARGS &

# Give the evaluator a second to spawn
sleep 2

# 2. Start 6 self-play Collection Workers
for i in {1..6}
do
    echo "============================================================"
    echo "Starting Collection Worker $i..."
    echo "============================================================"
    # Each worker runs 1 concurrent game locally
    PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode worker --worker_id $i --worker_role collector --concurrent_games 2 &
    sleep 1
done

echo "============================================================"
echo "Distributed Training Cluster workers are fully online!"
echo "Press Ctrl+C to terminate the cluster."
echo "============================================================"

# Wait for background processes
wait
