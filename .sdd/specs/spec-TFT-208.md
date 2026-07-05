# Technical Specification: Vectorized Triplet Extraction and Replay Buffer Increase

## 1. File Structure Changes
- **Modified:** `config.py`
- **Modified:** `Models/MuZero_torch_trainer.py`

## 2. Interfaces & Signatures

### `config.py`
- Update `REPLAY_BUFFER_SIZE` variable from `10000` to `20000`.

### `Models/MuZero_torch_trainer.py`
- Ensure `import random` is present at the top of the file.
- Refactor the `Trainer.compute_loss` method:
  - Inside the `if len(combats) > 0:` block, replace the O(n²) tensor-based combinatorial triplet generation logic with a CPU-side vectorized approach using stratified sampling (1 positive + 1 negative per anchor).
  - Initialize `combat_board_loss = None` at the top of the `if len(combats) > 0:` block.
  - Flatten `results` to CPU: `results_cpu = results.flatten()`.
  - Iterate through `results_cpu`, find valid positive and negative indices using `np.where`, and use `random.choice` to pick exactly 1 positive and 1 negative index per anchor.
  - Construct a single PyTorch tensor from the generated `triplet_candidates` and send it to the GPU: `triplets_tensor = torch.tensor(triplet_candidates, dtype=torch.long, device=device)`.
  - Use batched indexing (`hidden_flat[triplets_tensor[:, 0]]`, etc.) to extract `anchors`, `positives`, and `negatives` in one GPU operation.
  - If `len(triplet_candidates) > 0`, compute `combat_board_loss`.
  - Remove the fallback assigning `combat_board_loss = torch.tensor(0.0, ...)` when no combats/triplets exist.
- Update downstream logic handling `combat_board_loss` in `Trainer.compute_loss`:
  - Change `if len(combats) > 0:` immediately after `mean_loss = value_loss.mean() + policy_loss.mean()` to `if len(combats) > 0 and combat_board_loss is not None:`.
  - Change `if len(combats) > 0:` in the `summary_writer` section to `if len(combats) > 0 and combat_board_loss is not None:`.

## 3. Edge Cases & Concurrency
- **No Combats:** If `len(combats) == 0`, `combat_board_loss` is completely bypassed. `mean_loss` only aggregates `value_loss` and `policy_loss`.
- **Missing Positives or Negatives:** Stratified sampling checks `if len(pos_indices) > 0 and len(neg_indices) > 0:` to ensure an anchor is only added if a valid positive and negative pair can be sampled. If no labels meet this criterion, `triplet_candidates` may remain empty.
- **Empty Triplets List:** If `len(triplet_candidates) == 0`, `combat_board_loss` remains `None` and downstream accumulations check for `is not None`, preventing `UnboundLocalError` or zero-gradient tensors.
- **OOM Avoidance:** The previous `max_triplets` random sampling is naturally replaced by the 1-pos-1-neg sampling, which creates at most `batch_size` triplets (128 max), safely avoiding any OOM scenario. 

## 4. Testing Strategy
- **Correctness:** Run a single training step locally or via test suite to verify that `compute_loss` processes an experience batch without errors and that `mean_loss.backward()` properly propagates gradients.
- **Performance:** Verify that the GPU-to-CPU synchronization bottleneck is eliminated by profiling step time; it should take milliseconds rather than minutes.
- **Null states:** Check that the training step naturally succeeds without `combat_board_loss` when `len(combats) == 0` or when `len(triplet_candidates) == 0`.
- **Variable validation:** Check `global_buffer.py` correctly uses `config.REPLAY_BUFFER_SIZE` set to `20000`.