# Technical Specification for TFT-202: Fix `_convert_sample_if_needed` corrupting `target_obs`

## 1. File Structure Changes
- **Modified:** `Models/global_buffer.py`

## 2. Interfaces & Signatures
- **Method Modified:** `_convert_sample_if_needed` within `GlobalBuffer` class
- **Change Details:** 
  - Remove the assignments `extended[5] = obs` and `extended[6] = extended[6]` inside the condition `if len(extended) >= 7:`.
  - Add `extended[4] = policy` to ensure that any translated/converted policy is correctly saved in the extended list before appending it to the `converted` list.
  - The logic should append the tuple of the updated `extended` list to preserve `target_obs` (at index 5) and `bootstrap_depth` (at index 6) intact.

## 3. Edge Cases
- **Samples with len < 7:** Handled by the existing `else` clause which uses `(obs, action, value, reward, policy)` where `policy` has already been correctly updated.
- **`target_obs` preservation:** By not overwriting `extended[5]`, the actual `target_obs` originally recorded in the sample is kept for the n-step bootstrap mechanism, preventing value target corruption during training.
- **Policy persistence:** Assigning `extended[4] = policy` ensures 3d action transformations apply correctly regardless of sample length.

## 4. Testing Strategy
- Create a mock sample with 7 elements where `extended[5]` (the `target_obs`) is distinctly different from `obs` (the start observation).
- Pass this sample through `_convert_sample_if_needed`.
- Assert that the returned sample's index 5 remains the original `target_obs`, rather than being overwritten by `obs`.
- Assert that the returned sample's index 4 reflects any policy conversion performed by `action_to_policy_if_needed`.
