# Technical Specification: Fix Mock Batches for MuZero Trainer Tests

## 1. Overview
The `train_network` method in `Models/MuZero_torch_trainer.py` expects a 7-element `batch` tuple (`observation, action, value, reward, policy, target_obs, bootstrap_depth`), but the testing scripts currently supply only 5 elements. This discrepancy causes an unpacking error during test execution. 

This specification dictates the necessary updates to the mock training data generation in the test suite to conform to the 7-element contract.

## 2. Implementation Details

### 2.1. Update `tests/test_muzero_trainer.py`
**File:** `tests/test_muzero_trainer.py`
**Location:** Around line 62

**Changes:**
1. Generate `target_obs` using `np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)`.
2. Generate `bootstrap_depth` using `np.random.randint(1, 10, (batch_size,)).astype(np.float32)` (or simply `np.zeros`). Since depth is often an integer cast to float or just a float, we can use `np.random.rand(batch_size).astype(np.float32)` or `np.random.randint(1, config.UNROLL_STEPS, (batch_size,)).astype(np.float32)`. Let's use `np.random.rand(batch_size).astype(np.float32)` to be safe, or simply `np.zeros((batch_size,)).astype(np.float32)`. Actually, `bootstrap_depth` is typically an array of integers representing depth, let's use `np.ones((batch_size,)).astype(np.float32)`. Let's just follow the existing randomization pattern: `np.random.rand(batch_size).astype(np.float32)`.
3. Update the `batch` assignment to include these two new variables:
   ```python
   target_obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
   bootstrap_depth = np.random.rand(batch_size).astype(np.float32)
   
   batch = (observations, actions, values, rewards, policies, target_obs, bootstrap_depth)
   ```

### 2.2. Update `tests/test_training_step.py`
**File:** `tests/test_training_step.py`
**Location:** Around line 54

**Changes:**
1. Generate `target_obs` with shape `(batch_size, config.OBSERVATION_SIZE)`.
2. Generate `bootstrap_depth` with shape `(batch_size,)`.
3. Update the `batch` assignment similarly:
   ```python
   target_obs = np.random.rand(batch_size, config.OBSERVATION_SIZE).astype(np.float32)
   bootstrap_depth = np.random.rand(batch_size).astype(np.float32)
   
   batch = (observations, actions, values, rewards, policies, target_obs, bootstrap_depth)
   ```

## 3. Design Philosophy Alignment
- **Simple Interfaces:** Aligning the test mock input strictly with the trainer's expected signature resolves the unpacking crash while maintaining a clean boundary.
- **Deep Modules:** The tests should transparently construct full valid input states required by the deep module (`Trainer`).

## 4. Test Verification
Running `python tests/test_muzero_trainer.py` and `python tests/test_training_step.py` should no longer throw the `ValueError: not enough values to unpack (expected 7, got 5)` and should succeed assuming the rest of the pipeline functions correctly.