# Technical Specification: TFT-201 Buffer and Trainer Fixes

## 1. Objective
Address the 6 confirmed bugs in the MuZero data pipeline that cause `ValueError` and `IndexError` during distributed training. The core issues stem from varying sequence lengths in the replay buffer and a broken n-step bootstrap implementation in the trainer.

## 2. Background
The training orchestrator crashes when `GameplayBuffer.sample()` attempts to create a numpy array from variable-length value lists. Further analysis revealed that even if this was fixed, the trainer's `compute_loss` method contains a bug where the `target_value` tensor is overwritten by a 1D bootstrap scalar, causing a crash on the next line. Additionally, the bootstrap logic ignores intermediate rewards, `global_buffer.py` corrupts the bootstrap target observation, and unit tests are misaligned with the trainer's 7-element batch signature.

## 3. Implementation Steps

### Step 1: Fix Target Observation Corruption (Issue 1)
**File:** `Models/global_buffer.py`
**Location:** `_convert_sample_if_needed` (approx. line 165)
- **Change:** Remove the assignment `extended[5] = obs`. 
- **Reason:** The comment states `target_obs stays as-is`, but the code mistakenly overwrites the future target observation (index 5) with the current observation (index 0), breaking n-step bootstrapping. `extended[6]` assignment is also redundant and can be removed.

### Step 2: Fix Value Sequence Length Mismatch (Issue 3)
**File:** `Models/replay_buffer.py`
**Location:** `move_buffer_to_global` (sync and async, approx. lines 88 and 122)
- **Change:** Update the value target list from `[final_val] * unroll_steps` to `[final_val] * config.UNROLL_STEPS`.
- **Reason:** `actions`, `rewards`, and `policys` are correctly sliced to exactly `UNROLL_STEPS` length (25 steps). `value` must match this length to satisfy the trainer's `num_target_steps` expectation and to allow `np.array()` stacking in `global_buffer.py`.

*Note on Issue 6:* The slicing logic `actions[start:t]` (where `start = t - config.UNROLL_STEPS`) is functionally correct because `t - start` is always exactly `UNROLL_STEPS`. Modifying this to `unroll_steps` would break the recurrent inference loop in the trainer. Thus, no changes are required for the actions/rewards slicing.

### Step 3: Fix Bootstrapping Shape and Reward Accumulation (Issue 2 & 4)
**File:** `Models/MuZero_torch_trainer.py`
**Location:** `compute_loss` (approx. lines 99-104)
- **Current Behavior:** Overwrites the `(B, 25)` `target_value` tensor with a `(B,)` scalar `bootstrap_targets = gamma_n * v_t_plus_n.squeeze()`.
- **Change:** Implement proper backwards accumulation of n-step returns to create a correct `(B, 25)` target tensor.
- **Implementation:**
  ```python
  # Instead of replacing target_value with a 1D tensor:
  discount_tensor = torch.tensor(config.DISCOUNT, device=device)
  
  # target_idx is unroll_steps - 1 steps ahead of the unroll window's end
  gamma_n = discount_tensor ** (bootstrap_depth - 1)
  z = gamma_n * v_t_plus_n
  
  new_target_value = torch.zeros_like(target_value)
  target_reward_tensor = torch.from_numpy(np.array(target_reward)).float().to(device)
  
  # Accumulate backwards through the unroll window
  for i in reversed(range(num_target_steps)):
      z = target_reward_tensor[:, i] + config.DISCOUNT * z
      new_target_value[:, i] = z
      
  target_value = new_target_value
  ```
- **Reason:** This preserves the `(B, UNROLL_STEPS)` shape, preventing the `IndexError` at line 121. It also mathematically incorporates the intermediate rewards from the unroll window, resolving Issue 4.

### Step 4: Fix Unit Test Signatures (Issue 5)
**Files:** `tests/test_muzero_trainer.py` and `tests/test_training_step.py`
**Location:** Mock batch creation (approx. line 62 and 54)
- **Change:** Expand the `batch` tuple from 5 elements to 7 elements by appending mock data for `target_obs` and `bootstrap_depth`.
- **Implementation:**
  ```python
  target_obs = [None] * batch_size
  bootstrap_depth = np.array([unroll_steps] * batch_size)
  batch = (observations, actions, values, rewards, policies, target_obs, bootstrap_depth)
  ```
- **Reason:** `Trainer.train_network` unpacks 7 variables. Passing only 5 causes a `ValueError: not enough values to unpack`.

## 4. Testing & Validation
- **Unit Tests:** Run `pytest tests/test_muzero_trainer.py` and `pytest tests/test_training_step.py` to ensure the trainer accepts the batch and computes loss without crashing.
- **Integration:** Run `python main.py --train` (or the equivalent local script) to verify that `GameplayBuffer.sample()` successfully constructs a batch and that the recurrent step executes without shape mismatch errors.