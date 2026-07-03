# Technical Specification for TFT-204

## Overview
The `move_buffer_to_global` and `move_buffer_to_global_async` methods in `Models/replay_buffer.py` have a bug that causes an inhomogeneous shape mismatch. During the tail of an episode, `unroll_steps` becomes less than `config.UNROLL_STEPS`. However, other arrays in the appended replay data (like `actions`, `rewards`, `policys`) always take a full `config.UNROLL_STEPS` length slice because they are sliced as `[start:t]` where `start = t - config.UNROLL_STEPS`. This mismatch causes NumPy array conversion failures in the sampler.

## Implementation Plan
1. **Target File:** `Models/replay_buffer.py`
2. **Modifications:**
   - In the `move_buffer_to_global` method (around line 88), change `[final_val] * unroll_steps` to `[final_val] * config.UNROLL_STEPS`.
   - In the `move_buffer_to_global_async` method (around line 122), change `[final_val] * unroll_steps` to `[final_val] * config.UNROLL_STEPS`.
3. **Rationale:** Since `final_val` is a scalar corresponding to the final reward of the episode, padding it to `config.UNROLL_STEPS` ensures all array slices appended to `replay_set` match in dimension, resolving the inhomogeneous shape error. This does not alter the underlying game semantics, as the final reward remains constant.
