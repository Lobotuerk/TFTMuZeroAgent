# Technical Specification: TFT-195 (Fix DynNetwork Dimension Mismatch)

## 1. Context and Problem Statement
The `DynNetwork` inside `Models/MuZero_torch_model.py` raises a dimension mismatch `RuntimeError` during the residual connection step in the `forward` pass (`x = x + residual`). 

The root cause is in `DynNetwork.__init__`:
- `self.dense1` is instantiated as `torch.nn.Linear(input_size, hidden)` where `hidden = input_size`.
- The residual layers (`dense2` through `dense7`) are instantiated as `torch.nn.Linear(hidden, size)` where `size` comes from `layer_sizes` (e.g., 512).
- When `input_size` (e.g., 2129) is different from the values in `layer_sizes` (e.g., 512), `dense1` outputs a tensor of size 2129, while the first residual layer expects 2129 and outputs 512. The subsequent addition `x + residual` fails because `x` is 512 and `residual` is 2129.

## 2. Proposed Solution
We need to refactor `DynNetwork.__init__` so that it matches the architecture of `PredNetwork`, where `dense1` acts as a projection layer that down-samples or up-samples the input into the size expected by the residual layers. The residual layers themselves should maintain the dimension size.

### `DynNetwork.__init__` Changes:
1. Initialize `layer_sizes = layer_sizes if layer_sizes else [input_size] * 6`
2. Change `self.dense1` to `torch.nn.Linear(input_size, layer_sizes[0])`.
3. In the loop to generate `dense2`...`denseN`, the input size of each layer should be the size of the previous layer, and the output size should be its target size.
   `torch.nn.Linear(layer_sizes[i - 1] if i > 0 else layer_sizes[0], size)`

## 3. Backwards Compatibility
Since checkpoints with `layer_sizes=[512, 512...]` and `input_size=2129` would crash instantly during the first forward pass, no valid training checkpoints exist with this configuration. 
For configurations where `layer_sizes=None`, `hidden` was set to `input_size`, and all layers were dynamically sized to `input_size`. Our proposed logic will evaluate `layer_sizes[i]` as `input_size`, producing exactly identically shaped weights and ensuring seamless checkpoint loading for previously working configurations.

## 4. Implementation Steps
1. Modify `Models/MuZero_torch_model.py` inside `DynNetwork.__init__`.
2. Replace:
   ```python
        hidden = input_size

        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.dense1 = torch.nn.Linear(input_size, hidden)
        layer_sizes = layer_sizes if layer_sizes else [hidden] * 6
        for i, size in enumerate(layer_sizes):
            setattr(self, f'dense{i + 2}', torch.nn.Linear(hidden, size))
   ```
   With:
   ```python
        self.relu = torch.nn.LeakyReLU(inplace=True)
        layer_sizes = layer_sizes if layer_sizes else [input_size] * 6
        self.dense1 = torch.nn.Linear(input_size, layer_sizes[0])
        for i, size in enumerate(layer_sizes):
            prev_size = layer_sizes[i - 1] if i > 0 else layer_sizes[0]
            setattr(self, f'dense{i + 2}', torch.nn.Linear(prev_size, size))
   ```
3. No changes to the `forward` function are needed since the dynamic `getattr` logic remains the same, but the shapes will now align.

## 5. Verification
- Validate the module instantiation by passing dummy tensors of shape `(batch_size, 2048)` for `x` and `(batch_size, 81)` for `action`. 
- Ensure no dimension mismatch occurs during the forward pass.
