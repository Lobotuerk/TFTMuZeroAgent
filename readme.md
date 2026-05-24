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
1. Clone the repository and its submodules (TFTSet4Gym, MonteCarloTreeSearch).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Build the C++/Cython external packages:
   ```bash
   cd core/ctree
   bash make.sh
   cd ../..
   ```

### Running the System
The primary entry point is `main.py`, which supports several modes of operation:

- **Training**: Run the full RL lifecycle.
  ```bash
  python main.py --mode training --run_name "my_experiment"
  ```
- **Evaluation**: Pit the agent against baselines.
  ```bash
  python main.py --mode evaluation --eval_games 10
  ```
- **Demo**: Run parallel games without training to observe agent behavior.
  ```bash
  python main.py --mode demo --demo_episodes 5
  ```
- **Debug**: Test neural network architecture or run a single debug episode.
  ```bash
  python main.py --mode debug --debug_network
  ```

### Configuration
Hyperparameters, training settings, and environment constants are located in `config.py`. Key settings include:
- `CONCURRENT_GAMES`: Number of games to run in parallel.
- `BATCH_SIZE`: Training batch size.
- `NUM_SIMULATIONS`: Number of MCTS simulations per move.
- `CHECKPOINT_STEPS`: Interval for saving models and running evaluations.

## Community and Contributions
This project aims to push the boundaries of AI in complex games like TFT. Contributions, questions, and discussions are welcome!
- **Discord**: [Join our community](https://discord.gg/cPKwGU7dbU)
- **Contact**: slucoris@gmail.com
