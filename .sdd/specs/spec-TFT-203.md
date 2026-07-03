# Technical Specification for TFT-203

## 1. Overview
The issue is an `IndexError` during training caused by overwriting the `target_value` tensor. During the n-step bootstrap calculation in `Models/MuZero_torch_trainer.py`, the 2D tensor `target_value` of shape `(batch_size, UNROLL_STEPS)` is incorrectly replaced by a 1D tensor of shape `(batch_size,)`. When computing the value loss over the unroll steps, 2D indexing on this 1D tensor causes a crash.

## 2. Root Cause
In `Models/MuZero_torch_trainer.py` (lines 93-104), the n-step return simply calculates the discounted value of the target observation and overwrites `target_value`. It fails to account for the intermediate rewards inside the unroll steps, and incorrectly reduces the target value tensor dimensionality from 2D to 1D, removing the unroll steps dimension. 

## 3. Implementation Plan
We must implement a backwards accumulation over the unroll steps to compute the correct n-step returns, thereby preserving the `(batch_size, UNROLL_STEPS)` shape.

1.  **Modify `Models/MuZero_torch_trainer.py`:**
    Locate the section computing bootstrap targets around line 99:
    ```python
            # Compute bootstrap targets: z_t = gamma^n * v_{t+n}
            discount = torch.tensor(config.DISCOUNT, device=device)
            gamma_n = discount ** bootstrap_depth
            bootstrap_targets = gamma_n * v_t_plus_n.squeeze()
            # Replace placeholder target_value with bootstrap targets
            target_value = bootstrap_targets
    ```
    Replace it with the backwards accumulation logic:
    ```python
            # Implement backwards accumulation of n-step returns preserving (B, UNROLL_STEPS) shape
            discount_tensor = torch.tensor(config.DISCOUNT, device=device)
            gamma_n = discount_tensor ** (bootstrap_depth - 1)
            z = gamma_n * v_t_plus_n.squeeze()
            new_target_value = torch.zeros_like(target_value)
            target_reward_tensor = torch.from_numpy(np.array(target_reward)).float().to(device)
            for i in reversed(range(num_target_steps)):
                z = target_reward_tensor[:, i] + config.DISCOUNT * z
                new_target_value[:, i] = z
            target_value = new_target_value
    ```

    Note: `num_target_steps` is already defined earlier in the method (`num_target_steps = target_value.shape[-1]`). `v_t_plus_n.squeeze()` is used to ensure shape compatibility with `target_reward_tensor[:, i]`.

## 4. Dependencies & Impact
- **Dependencies**: This change introduces a dependency on `target_reward` inside the loss computation function. The function signature of `compute_loss` already includes `target_reward`, so no signature change is necessary.
- **Impact**: Fixes a critical crash loop in the MuZero training logic, unblocking the pipeline.