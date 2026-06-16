# Technical Specification: Bug-fix TFT-192

## 1. Overview

This document outlines the technical design for fixing two critical bugs in the TFT-MuZero-Agent:
1. An `IndexError` in `sell_from_bench` due to missing input validation.
2. An action space dimension mismatch between the Gym environment and the agent, which could lead to unpredictable behavior.

The previous agent has already implemented the required changes, this document formalizes the implementation details.

## 2. Background

The root cause of the `IndexError` is that the `sell_from_bench` function in `TFTSet4Gym/tft_set4_gym/player.py` does not validate the `location` parameter, which is derived from the action produced by the model. This can lead to out-of-bounds access to the `self.bench` list.

Additionally, a discrepancy was found in the action space dimensions. The Gym environment defines an action space of `[8, 37, 10]`, but several parts of the agent's code, including the MCTS implementation and observation schema, were hardcoded with `[7, 37, 10]`. This inconsistency can cause subtle bugs and unpredictable behavior during training and evaluation.

## 3. Design and Implementation Details

### 3.1. Defensive Bounds Checking

To prevent the `IndexError`, bounds checks will be added to all bench access locations in the `batch_2d_controller` function in `TFTSet4Gym/tft_set4_gym/step_function.py`, and in the `sell_from_bench` function in `TFTSet4Gym/tft_set4_gym/player.py`.

**`TFTSet4Gym/tft_set4_gym/player.py`**

In the `sell_from_bench` function, a guard will be added to ensure the `location` is within the valid range of the bench (0-8).

```python
# player.py

def sell_from_bench(self, location):
    if not (0 <= location < 9):
        return False
    if self.bench[location]:
        self.gold += self.bench[location].cost
        self.bench[location] = None
        return True
    return False
```

**`TFTSet4Gym/tft_set4_gym/step_function.py`**

In the `batch_2d_controller` function, a check will be added before accessing `player.bench`.

```python
# step_function.py
...
    elif action == 3:  # Move Item
        ...
    elif action == 4:  # Sell Unit
        if target_1 \uff1e= 28:
            bench_loc = target_1 - 28
            if 0 <= bench_loc < 9 and player.bench[bench_loc]:
                 player.sell_from_bench(bench_loc)
...
```

### 3.2. Action Dimension Consistency

The action space dimensions will be standardized to `[8, 37, 10]` across the entire codebase to match the Gym environment's definition. This involves updating hardcoded values in multiple files.

**`TFTMuZeroAgent/training_orchestrator.py`**
- All instances of `action_limits=[7,37,10]` will be changed to `[8,37,10]`.

**`TFTSet4Gym/tft_set4_gym/observation_schema.py`**
- The shape of the `valid_actions` in the observation schema will be updated from `(54,)` to `(55,)`.

**`TFTMuZeroAgent/Models/MuZero_torch_model.py`**
- The `ACTION_MASK_DIM` constant will be updated from `54` to `55`.

**`TFTMuZeroAgent/Models/agent_manager.py`**
- `np.ones(54, dtype=bool)` will be changed to `np.ones(55, dtype=bool)`.

**`TFTMuZeroAgent/Models/tft_mcts.py`**
- `np.zeros((54,), dtype=bool)` will be changed to `np.zeros((55,), dtype=bool)`.

**`TFTMuZeroAgent/Models/MuZero_torch_agent.py`**
- The fallback mask `np.ones(54)` will be updated to `np.ones(55)`.

**`TFTMuZeroAgent/config.py`**
- Comments will be updated to reflect the `8+37+10` action space.

**`TFTSet4Gym/tft_set4_gym/step_function.py`**
- Comments will be updated to reflect the new action space slicing (`0:8, 8:45, 45:55`).

**Test and Demo Files**
- All test and demo files using `action_limits=[7,37,10]` will be updated to `[8,37,10]`.

## 4. Testing

The existing test suite should be run to ensure that the changes do not introduce any regressions. A new test case should be added to specifically test the bounds checking in `sell_from_bench` and the other bench access locations.
However, since the previous agent already implemented the changes and added a new test `tests/test_pollution_fix.py`, we will rely on this test.

## 5. Rollout Plan

The changes will be implemented on the `sdd/feature-TFT-192` branch. Once the changes are verified, the branch will be merged into `main`.

