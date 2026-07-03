# Technical Specification: Fix `actions[start:t]` upper bound near episode end (TFT-207)

## Overview
In `Models/replay_buffer.py`, during the conversion of episode trajectories into training samples (`move_buffer_to_global` and `move_buffer_to_global_async`), the code slices `self.actions`, `self.rewards`, and `self.policys` with bounds `[start:t]`. Since `start = t - config.UNROLL_STEPS`, the length of these slices is exactly `config.UNROLL_STEPS` at all times.

However, near the end of the episode (when `t` is close to `max_obs`), `unroll_steps = min(config.UNROLL_STEPS, max_obs - t)` becomes less than `config.UNROLL_STEPS`. When this happens, the later entries of the slice contain actions, rewards, and policies from positions beyond `start + unroll_steps`. Since the trainer only expects `unroll_steps` valid transitions, these out-of-window entries cause subtle bugs during sample unrolling and loss calculation.

The chosen solution (Approach B) is to skip these "tail samples" entirely. By ensuring only samples with a full `config.UNROLL_STEPS` horizon are added to the replay buffer, we maintain uniform shapes across all generated samples without modifying the trainer logic.

## Changes Required

### 1. `Models/replay_buffer.py`

Modify both `move_buffer_to_global` and `move_buffer_to_global_async` methods.

**Logic adjustment:**
Inside the loop over `t`:
```python
for t in range(config.UNROLL_STEPS, max_obs):
    unroll_steps = min(config.UNROLL_STEPS, max_obs - t)
    
    # ADDED: Skip incomplete tail samples
    if unroll_steps < config.UNROLL_STEPS:
        continue
```

**Context in `move_buffer_to_global`:**
```python
        for t in range(config.UNROLL_STEPS, max_obs):
            unroll_steps = min(config.UNROLL_STEPS, max_obs - t)
            if unroll_steps < config.UNROLL_STEPS:
                continue

            start = t - config.UNROLL_STEPS
```

**Context in `move_buffer_to_global_async`:**
```python
        for t in range(config.UNROLL_STEPS, max_obs):
            unroll_steps = min(config.UNROLL_STEPS, max_obs - t)
            if unroll_steps < config.UNROLL_STEPS:
                continue

            start = t - config.UNROLL_STEPS
```

## Architectural Justification
- **Consistency:** Ensures `GameplayBuffer.sample()` returns batches where all dimensions match the expected `UNROLL_STEPS`.
- **Complexity:** Avoiding ragged arrays allows `numpy.array()` stacking to remain straightforward. The trainer remains unmodified.
- **Data Impact:** The loss of `UNROLL_STEPS - 1` samples at the very end of the episode is minimal (~24 steps) and removes transitions that have high termination bias and uncertain bootstrap targets.

## Edge Cases Handled
- Very short episodes (where `max_obs < 2 * config.UNROLL_STEPS`) will result in 0 samples generated. This is normal and already the intended behavior for episodes that are too short to provide a full training horizon.
