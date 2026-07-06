# Technical Specification: Target Value Collapse Fix (TFT-209)

## 1. Overview
The target value in the MuZero trainer collapses to 0 after ~50 training steps because the n-step bootstrap mechanism interacts poorly with the terminal-only reward structure of the TFT environment. Intermediate rewards are 0, and bootstrapping from the model's own initial random predictions (which are near 0) results in zeroed-out value targets. The model subsequently trains on these zeros, predicting zeros continuously and never recovering.

To resolve this, we will replace the bootstrap target computation with Monte Carlo returns computed at the end of the episode using the actual trajectory rewards. 

## 2. Design Details

### 2.1 Replay Buffer Updates (`Models/replay_buffer.py`)
- Modify `move_buffer_to_global` and `move_buffer_to_global_async` methods.
- Currently, they store `[final_val] * config.UNROLL_STEPS` as value targets. Instead, compute the actual discounted Monte Carlo returns for each step in the episode.
- The return from step `i` should be calculated backwards from the end of the episode: `returns[i] = reward[i] + config.DISCOUNT * returns[i+1]`. For the terminal step, the return is the `final_val`.
- Replace `[final_val] * config.UNROLL_STEPS` with the computed slice of returns for the unroll window, padded appropriately if the window exceeds the episode length.
- Set `target_obs=None` when appending to `replay_set`. This signals to the trainer that bootstrapping from a future observation is unnecessary.

### 2.2 Trainer Updates (`Models/MuZero_torch_trainer.py`)
- In the `compute_loss` method, the system already checks `if target_obs[0] is not None:` before applying the n-step bootstrap. By passing `None` from the replay buffer, we safely bypass the bootstrap logic, and the trainer will directly use our pre-computed Monte Carlo returns in `target_value` as the true targets for the MSE loss.
- Add new validation metrics to TensorBoard to verify that the target values no longer collapse:
  - After accumulating metrics in the `accs` dict (e.g. `accs['value']`), compute and log the absolute mean of the predicted value.
  - In the `summary_writer` section, add `summary_writer.add_scalar('prediction/mean_abs_value', torch.mean(torch.abs(sum_accs['value'])), train_step)` or calculated over the stacked `accs['value']` to ensure it is distinctly tracked.

## 3. Implementation Steps

1. In `Models/replay_buffer.py`:
   - At the beginning of `move_buffer_to_global` and `move_buffer_to_global_async`, import `config` or utilize `config.DISCOUNT` to calculate the returns array.
   - `returns = np.zeros(max_obs)`
   - `returns[-1] = final_val`
   - `for i in reversed(range(max_obs - 1)):`
       `returns[i] = self.rewards[i] + config.DISCOUNT * returns[i+1]`
   - Inside the loop, extract `unroll_returns = returns[start:start+unroll_steps]`. If `unroll_steps < config.UNROLL_STEPS`, pad `unroll_returns` with zeros or `returns[-1]` as previously done with `[final_val]`.
   - Update the `replay_set.append(...)` call to use the padded `unroll_returns` instead of `[final_val] * config.UNROLL_STEPS`.
   - Pass `None` in place of `target_obs` in the tuple.

2. In `Models/MuZero_torch_trainer.py`:
   - Within `compute_loss()`, add `summary_writer.add_scalar('prediction/mean_abs_value', torch.mean(torch.abs(sum_accs['value'])), train_step)` below the existing `prediction/value` log.
   
## 4. Risks and Mitigation
- Ensure that the Monte Carlo returns list computation correctly utilizes NumPy arrays or Python lists, perfectly mirroring `config.UNROLL_STEPS` padding logic.
- The `move_buffer_to_global_async` must identically mirror the return calculation added to `move_buffer_to_global`.