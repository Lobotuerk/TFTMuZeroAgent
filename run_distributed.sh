#!/bin/bash
# ---------------------------------------------------------------------------
# Distributed Process-Level Training Orchestrator (Option A)
# ---------------------------------------------------------------------------
set -e

# Ensure clean setup
mkdir -p ./checkpoint
mkdir -p ./data/gameplay
mkdir -p ./data/combats

echo "Clearing stale experience logs to prevent training data pollution..."
rm -f ./data/gameplay/*.pkl
rm -f ./data/combats/*.pkl

# Trap Ctrl+C (SIGINT), SIGTERM, and EXIT to gracefully terminate all child processes
trap 'echo "Terminating all distributed training processes..."; kill 0' SIGINT SIGTERM EXIT

# 1. Start the GPU Training Server
echo "============================================================"
echo "Starting GPU Training Server..."
echo "============================================================"
PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode train_server --checkpoint_interval 200 &
SERVER_PID=$!

# Allow the server a brief moment to initialize the weights file
sleep 4

# 2. Start the Evaluator Worker (Worker ID 0)
echo "============================================================"
echo "Starting Evaluator Worker (Worker 0)..."
echo "============================================================"
PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode worker --worker_id 0 --worker_role evaluator --eval_games 8 --eval_concurrent 2 &

# Give the evaluator a second to spawn
sleep 2

# 3. Start 3 self-play Collection Workers
for i in {1..3}
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
