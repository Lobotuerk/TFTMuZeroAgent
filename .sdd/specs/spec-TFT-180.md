# TFT-180: Fix Policy Softmax — 3-Block Independent Distributions & Full 3D Action Encoding

## Problem

1. **Wrong softmax axis**: The 111-dim policy head output is treated as one joint distribution (`log_softmax(dim=-1)` over all 111 dims). It actually represents 3 independent choices (action_type, target_1, target_2) and should use 3 separate `log_softmax` calls per block.

2. **Incomplete policy target**: `TFTMove.to_numpy()` at `tft_mcts.py:113-119` creates a 111-dim one-hot using `move.index` (0-6, action type only). The full 3D action `[action_type, target_1, target_2]` is never encoded. Combined with `action_to_policy_if_needed` returning early for non-trivial policies, the training target never includes `target_1` or `target_2` information.

3. **Uniform block sizes**: `action_3d_to_policy` hardcodes `num_slots=37` for all 3 blocks, but `ACTION_DIM = [7, 37, 10]`. The third dimension only has 10 values, not 37.

## Solution

### A. Dimension Adjustment

Change `config.py`:

- `ACTION_CONCAT_SIZE = sum(ACTION_DIM)` = 7 + 37 + 10 = **54**
- `ACTION_ENCODING_SIZE = sum(ACTION_DIM)` = **54** (must match `ACTION_CONCAT_SIZE`)

The policy head output changes from 111 → 54 dimensions.

### B. Variable-Block `action_3d_to_policy`

Change `action_3d_to_policy` in `action_conversion.py`:

- Accept optional `dim_sizes` parameter (default: `config.ACTION_DIM`)
- Create 3 one-hot blocks with variable sizes: `[7, 37, 10]`
- Block 0 (action type): offset 0, size 7 → indices 0..6
- Block 1 (target_1): offset 7, size 37 → indices 7..43
- Block 2 (target_2): offset 44, size 10 → indices 44..53

Remove hardcoded `num_slots=37` default.

### C. Full 3D Encoding in `TFTMove.to_numpy()`

Change `to_numpy()` in `tft_mcts.py`:

- Encode the full `[action_type, target_1, target_2]` as a 54-dim vector
- Use the same scheme as `action_3d_to_policy` (one-hot per block)
- Remove the old `self.index`-based single-position encoding

### D. 3-Block Softmax in Trainer

Change `compute_loss()` in `MuZero_torch_trainer.py`:

- Split `policy_logits` (batch, 54) into 3 blocks by `ACTION_DIM`
- Apply `log_softmax` independently to each block
- Split `target_policy` (batch, 54) into corresponding blocks
- Compute KL divergence per block, sum across blocks
- Aggregate the 3-block loss into the total policy loss

### E. 3-Block Action Priors in MCTS

Change `get_action_probabilities()` in `tft_mcts.py`:

- Apply softmax independently to each of the 3 blocks from `self.policy`
- For each move `[a, b, c]`, compute log-prior = `log_softmax_block0[a] + log_softmax_block1[b] + log_softmax_block2[c]`
- Use these log-priors as scores in the softmax over all legal moves

### F. Policy Size Calculation

Fix `policy_size` in `MuZero_torch_agent.py`:

- Change from `self.action_limits[1] * self.action_size` to `sum(self.action_limits)`

## Files Changed

| File | Change |
|------|--------|
| `config.py` | `ACTION_CONCAT_SIZE=54`, `ACTION_ENCODING_SIZE=54` |
| `Models/action_conversion.py` | Variable block sizes from `ACTION_DIM` |
| `Models/tft_mcts.py` | Full 3D `to_numpy()`, factored `get_action_probabilities()` |
| `Models/MuZero_torch_model.py` | `action_to_3d()` uses 54 dims |
| `Models/MuZero_torch_trainer.py` | 3 independent `log_softmax` calls |
| `Models/MuZero_torch_agent.py` | `policy_size = sum(ACTION_DIM)` |

## Backward Compatibility

- Existing checkpoints have 111-dim policy head weights. The model architecture changes (Linear(512 → 54) instead of Linear(512 → 111)), so old checkpoints are incompatible. The training must start from scratch.
- This is acceptable because existing checkpoints encoded incorrect (action-type-only) policy targets, making them unrecoverable for correct behavior.
