# Teamfight Tactics (TFT) MuZero Agent

This repository contains a sophisticated Reinforcement Learning (RL) architecture for playing Teamfight Tactics (TFT) Set 4. It leverages the MuZero algorithm, optimized for a multi-agent environment, to learn complex game strategies through self-play and evaluation.

## Key Capabilities

- **High-Fidelity Simulation**: A complete simulation of TFT Set 4 (all rounds, units, items, and mechanics) implemented as a PettingZoo multi-agent environment (`TFTSet4Gym`).
- **MuZero Reinforcement Learning**: An advanced implementation of the MuZero algorithm, featuring:
    - Parallel game collection without the overhead of heavy frameworks like Ray.
    - Batched inference for efficient action selection across multiple concurrent games.
    - Decoupled model training and experience collection.
- **Architectural Depth**: A modular design that separates the training orchestration, environment management, and agent logic.
- **Diverse Agent Support**: Native support for various agent types, including:
    - **MuZero RL Agent**: The primary learning agent.
    - **Rule-based Agents**: Heuristic-driven agents (e.g., Cultist, Divine, Reroll, FastLevel) for evaluation and training diversity.
    - **Random Agent**: A baseline for initial performance comparison.

## System Architecture

The project is structured around a central **Training Orchestrator** that drives the RL lifecycle:

1.  **COLLECT**: Run multiple concurrent games using the `ParallelEnvironmentManager`. Experiences are gathered into a `GlobalBuffer`.
2.  **TRAIN**: Sample experiences from the buffer and update the MuZero model via the `Trainer`.
3.  **SYNC**: Distribute updated model weights to the active collection agents for immediate policy improvement.
4.  **EVALUATE**: Periodically pit the current model against previous versions or rule-based baselines to track progress and retain the best-performing weights.

## Usage Instructions

### Prerequisites
- **Python**: ~3.8
- **GCC**: version 7.5+ (required for building Cython/C++ extensions)

### Installation
1. **Clone the repository and its submodules**:
   ```bash
   git clone --recursive https://github.com/Lobotuerk/TFTMuZeroAgent.git
   cd TFTMuZeroAgent
   ```

2. **Setup the environment**:
   It is highly recommended to use the provided Conda environment to ensure compatibility with Python 3.13 and all dependencies. The `env.yml` includes the `python-freethreading` build, which disables the GIL for improved multi-threaded performance.
   ```bash
   conda env create -f env.yml
   conda activate TFT
   ```

3. **Build the C++ MCTS extensions**:
   ```bash
   cd MonteCarloTreeSearch
   python3.13t setup.py build_ext --inplace
   cd ..
   ```

### Running the System (Distributed Workflow)

Always run the project via the `run_tft.sh` wrapper script. It activates the `TFT` conda environment (with the free-threaded Python build) and sets up `PYTHONPATH` automatically.

The primary entry point is `main.py`, which supports two modes of operation:

- **Run the HTTP Training Server** (GPU-bound process that trains the model and serves an HTTP API):
  ```bash
  ./run_server_distributed.sh
  ```

- **Run the cluster workers** (CPU-bound processes for game collection and evaluation):
  ```bash
  ./run_workers_distributed.sh
  ```

### Configuration
Hyperparameters, training settings, and environment constants are located in `config.py`. Key settings include:
- `CONCURRENT_GAMES`: Number of games to run in parallel.
- `BATCH_SIZE`: Training batch size.
- `NUM_SIMULATIONS`: Number of MCTS simulations per move.
- `SYNC_STEPS`: Interval (in training steps) for saving model checkpoints.

## Architecture

CPU parallelism during environment simulation uses separate OS processes via `multiprocessing` (`_MultiProcessEnvManager`). Each environment runs in a dedicated subprocess, bypassing the Python GIL for CPU-bound game simulation while inference runs asynchronously on the main process.

## Community and Contributions
This project aims to push the boundaries of AI in complex games like TFT. Contributions, questions, and discussions are welcome!
- **Discord**: [Join our community](https://discord.gg/cPKwGU7dbU)
- **Contact**: slucoris@gmail.com