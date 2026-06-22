# Technical Specification for TFT-194: Architecture Refactor and N-Step Bootstrap

## Overview
This specification outlines the implementation details for aligning the current model architecture more closely with the MuZero paper where beneficial, fixing core temporal credit assignment bugs, and cleaning up unused code components as discussed in TFT-194.

## Scope of Changes

### 1. Remove Reward Head from Dynamics Network (Bug Fix)
In terminal-only environments like TFT Set 4, intermediate rewards are strictly 0, meaning the reward head has no useful learning signal until the final step (where it is redundant with the value head).
* **Target File**: `Models/MuZero_torch_model.py`
* **Changes**:
  * Remove `reward_head` from `DynNetwork`.
  * Modify `DynNetwork.forward` to only return the `next_hidden_state`.
  * Update `MuZeroNetwork.dynamics` to synthesize a dummy zero reward instead of expecting one from `DynNetwork`.

### 2. Implement N-Step Bootstrap via Recomputing (Core Bug)
Currently, target values for all unrolled steps are strictly the `final_val`, destroying the temporal distance signal. We will implement proper n-step bootstrap targets ($z_t = \gamma^n \cdot v_{t+n}$) using recomputing during training.
* **Target Files**: `Models/replay_buffer.py`, `Models/global_buffer.py`, `Models/MuZero_torch_trainer.py`
* **Changes**:
  * **`replay_buffer.py`**: Fix the iteration to span all steps bounded by game length. The loop should cap `unroll_steps = min(config.UNROLL_STEPS, len(self.observations) - t)`. It must append `target_obs` (the observation at step $t+n$) and the `bootstrap_depth` ($n$) to each stored experience.
  * **`global_buffer.py`**: Update `_convert_sample_if_needed` and `GameplayBuffer.sample` to handle the extended tuple elements (`target_obs` and `bootstrap_depth`).
  * **`MuZero_torch_trainer.py`**: Update `compute_loss` to unpack the updated batch. Run a single batched `initial_inference` on `target_obs` to obtain the value network's predictions for step $t+n$. Compute the target values as `(config.DISCOUNT ** bootstrap_depth) * v_{t+n}`. For terminal states, use the final value directly.

### 3. Refactor Network Constructors (Cleanup)
The network components currently ignore their constructor parameters and hardcode configurations internally.
* **Target File**: `Models/MuZero_torch_model.py`
* **Changes**:
  * Refactor `PredNetwork`, `DynNetwork`, and `RepNetwork` to utilize `input_size`, `layer_sizes`, `output_size`, and `encoding_size` from their constructor arguments.
  * Remove internal hardcoding of these variables, delegating to the `MuZeroNetwork` constructor to pass `config` values accurately.

### 4. Remove LSTM Flatteners (Cleanup)
Functions mapping between RNN representations and flat vectors exist but are dead code.
* **Target File**: `Models/MuZero_torch_model.py`
* **Changes**:
  * Remove `rnn_to_flat` and `flat_to_lstm_input` from `MuZeroNetwork`.

### 5. Fix Config Value Bounds and Unroll Steps (Config Fix)
The configuration limits don't reflect the TFT environment, and the unroll length is too short to capture meaningful sequences.
* **Target File**: `config.py`
* **Changes**:
  * Change `MINIMUM_REWARD` from `-300.0` to `0.0`.
  * Change `MAXIMUM_REWARD` from `300.0` to `250.0` (matching the environment's true limits).
  * Change `UNROLL_STEPS` from `5` to `25` (representing approximately one full round of actions).

### 6. Add Evaluation Metrics (Enhancement)
Enhance metric reporting to provide deeper insights into learning stability.
* **Target Files**: `Models/MuZero_torch_trainer.py`, `training_orchestrator.py`
* **Changes**:
  * **Value regression error**: Track MAE/MSE of predicted values vs final targets.
  * **Policy entropy**: Log policy distribution entropy to verify action diversity and prevent premature convergence.
  * **Elo rating**: Record moving Elo where evaluations occur.

## Execution Strategy
These changes represent a cohesive refactor and can be performed incrementally by `SDD-Implementer` in the `sdd/feature-TFT-194` branch. They introduce no major new dependencies.