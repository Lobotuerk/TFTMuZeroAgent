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
   python setup.py build_ext --inplace
   cd ..
   ```

### Running the System (Free-Threading)
Always run the project via the `run_tft.sh` wrapper script. It activates the `TFT` conda environment (with the free-threaded Python build) and sets up `PYTHONPATH` automatically.

```bash
./run_tft.sh python main.py --mode training --run_name "my_experiment"
```

The primary entry point is `main.py`, which supports several modes of operation:

- **Training**: Run the full RL lifecycle.
  ```bash
  ./run_tft.sh python main.py --mode training --run_name "my_experiment"
  ```
- **Evaluation**: Pit the agent against baselines.
  ```bash
  ./run_tft.sh python main.py --mode eval --eval_games 10
  ```
- **Demo**: Run parallel games without training to observe agent behavior.
  ```bash
  ./run_tft.sh python main.py --mode demo --demo_episodes 5
  ```
- **Debug**: Test neural network architecture or run a single debug episode.
  ```bash
  ./run_tft.sh python main.py --mode debug --debug_network
  ./run_tft.sh python main.py --mode debug --debug_single_episode
  ```

Running `python main.py` or `python benchmark_training.py` directly will trigger a fail-fast error if `FORCE_THREADING_ENV_MANAGER` is `True` but the GIL is still enabled.

### Benchmarking
A standalone benchmark script is provided to profile the training pipeline's performance:

```bash
./run_tft.sh python benchmark_training.py
```

This runs the `TrainingOrchestrator` for 100 training steps with 18 concurrent games and 10 evaluation games, then outputs a detailed performance breakdown:

- **Environment stepping time**: time spent executing game logic in the simulation
- **Inference wait time**: time spent waiting for batched GPU inference
- **Training time**: time spent updating model weights
- **Idle time**: time waiting for sufficient experience to be collected

Advanced options:
```bash
./run_tft.sh python benchmark_training.py --steps 200 --concurrent 8 --eval-games 5
```

Run `python benchmark_training.py --help` for all options.

### Configuration
Hyperparameters, training settings, and environment constants are located in `config.py`. Key settings include:
- `CONCURRENT_GAMES`: Number of games to run in parallel.
- `BATCH_SIZE`: Training batch size.
- `NUM_SIMULATIONS`: Number of MCTS simulations per move.
- `CHECKPOINT_STEPS`: Interval for saving models and running evaluations.

## Migration to Python 3.13+ Free-Threading

A design for migrating from `multiprocessing`-based parallelism to Python 3.13+ free-threading (PEP 703) is available.

### Overview

Currently, CPU parallelism during environment simulation uses separate OS processes via `multiprocessing` (`_MultiProcessEnvManager`). This introduces IPC overhead from pickling NumPy arrays across pipes, complexity in process management, and higher memory footprint.

With Python 3.13+ Free-Threading (GIL disabled), we can replace heavy multiprocessing workers with lightweight threads sharing the same memory address space, eliminating serialization overhead.

### Architecture

A thread-based environment manager `_ThreadEnvManager` mirrors the public API of `_MultiProcessEnvManager`. It uses `asyncio.run_coroutine_threadsafe` to bridge synchronous CPU-bound simulation threads with the async inference server on the main thread:

1. **Background Thread (CPU Simulation):** Runs synchronous game loops, scheduling async action selection on the main event loop via `asyncio.run_coroutine_threadsafe`.
2. **Main Thread (Async Event Loop):** Awaits and batches GPU inference, then resolves the future, unblocking the simulator thread.

### Thread Safety

- PyTorch model execution and `EnhancedAgentManager` async states run on the main event loop thread only.
- `GameplayBuffer` and `CombatBuffer` already use `threading.Lock()` for thread-safe access.
- Each background thread has its own isolated environment instance with no shared state.

### Migration Tasks

1. **GIL Check & Config Toggle:** Programmatic detection of free-threading status (`sys._is_gil_enabled()`).
2. **Implement `_ThreadEnvManager` & Worker Loop:** Thread-based manager with pause/resume/drain support.
3. **Integrate into `TrainingOrchestrator`:** Auto-select `_ThreadEnvManager` when free-threading is active.
4. **Comprehensive Testing:** Unit and integration tests for the thread-based manager.

## Community and Contributions
This project aims to push the boundaries of AI in complex games like TFT. Contributions, questions, and discussions are welcome!
- **Discord**: [Join our community](https://discord.gg/cPKwGU7dbU)
- **Contact**: slucoris@gmail.com
