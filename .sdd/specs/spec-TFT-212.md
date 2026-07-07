# 📋 Technical Specification: Memory Leak Resolution (TFT-212)

This specification addresses the two major memory leaks identified in the training cluster:
1. **Server-side:** PyTorch Computational Graph Leak in `MuZero_torch_trainer.py`.
2. **Worker-side:** Combat Buffer Reference Leak in `main.py` and `Models/global_buffer.py`.

---

## 🔍 Complete Dependency Audit

Before designing these modifications, we analyzed downstream dependents of the target symbols using CodeGraph:
- `Models/global_buffer.py`:
  - `CombatBuffer`: Internal circular reservoir buffer used exclusively inside `GlobalBuffer`.
  - `GlobalBuffer`: Initialized in `main.py` (via `create_global_buffer`), referenced extensively during training steps, gameplay compilation, and workers' storage.
  - `GlobalBuffer.clear_combat_buffer()`: Found to be a no-op (`pass`). A dependency audit via grep and CodeGraph shows it is only referenced in a single test: `tests/test_global_buffer.py:test_clear_combat_buffer_is_noop`.
- `Models/MuZero_torch_trainer.py`:
  - `compute_loss()`: Invoked solely by `train_network()` to compute gradient loss for training steps and log metrics to TensorBoard if a `summary_writer` is active.

---

## 📁 File Structure Changes

The following files will be created or modified:

| Action | Path | Description |
| :--- | :--- | :--- |
| **Modify** | `Models/MuZero_torch_trainer.py` | Detach and convert logged tensors to Python floats within `compute_loss()`. |
| **Modify** | `Models/global_buffer.py` | Implement `CombatBuffer.clear()`, and delete the unused, deprecated `GlobalBuffer.clear_combat_buffer()`. |
| **Modify** | `main.py` | Update worker experience-sending logic to call `combat_buffer.clear()`. |
| **Modify** | `tests/test_global_buffer.py` | Remove the deprecated no-op test, and add characterization tests for `CombatBuffer.clear()`. |

---

## ⚙️ Interfaces & Signatures

To maintain structural integrity, high modular depth, and clean encapsulated interfaces (conforming to *A Philosophy of Software Design*), we define the following changes:

### 1. `Models/global_buffer.py`

#### `CombatBuffer.clear(self) -> None`
- **Purpose:** Resets the buffer state, releases all references to stored combat samples, and prevents memory accumulation.
- **Thread Safety:** Must acquire `self._lock` before modifying any state variables.
- **State Changes:**
  - Overwrite `self._buffer` with a list of `None` of length `self._capacity`.
  - Set `self._size` to `0`.
  - Set `self._pos` to `0`.

#### `GlobalBuffer`
- **Method Deletion:** Remove `clear_combat_buffer(self) -> None` entirely to eliminate dead code and shallow wrapper interfaces. Callers should interact directly with `GlobalBuffer.combat_buffer` if they need to operate on it, or we can use the explicit encapsulation if required. Since the worker already directly accesses `orch.global_buffer.combat_buffer`, deleting this unused, no-op proxy aligns with simplifying the interface.

### 2. `main.py`

#### Worker Combat Experience Send Loop (`worker_mode`)
- **Modification:** Replace the direct, non-encapsulated assignment of private properties `combat_buffer._size = 0` and `combat_buffer._pos = 0` with a clean call to `combat_buffer.clear()`. This respects information hiding principles and ensures memory is actually reclaimed.

### 3. `Models/MuZero_torch_trainer.py`

#### `compute_loss` (specifically the TensorBoard logging block under `if summary_writer is not None:`)
- **Logging Values:** Ensure all tensors passed to `summary_writer.add_scalar()` are decoupled from PyTorch's autograd computation graph.
- **Conversion Helper `get_mean(k)`:**
  - **Modification:** Convert the returned value of `get_mean` to a standard Python float by applying `.detach().cpu().item()` on the resulting tensor.
- **Logged Scalars:**
  - For each `summary_writer.add_scalar(tag, tensor_value, train_step)` call, ensure `tensor_value` is processed to be a pure Python float.
  - Apply `.detach().cpu().item()` (or `.item()` if already detached/reduced) to all live PyTorch tensors, including variance, mean-absolute values, losses (`value_loss`, `policy_loss`, `combat_board_loss`, `mean_loss`), entropies, and MAE.

---

## 🛡️ Edge Cases & Concurrency

### 1. Multi-threaded Worker Buffer Reset
- **Problem:** Workers run asynchronous loops and access buffers concurrently with gameplay collection.
- **Solution:** `CombatBuffer.clear()` must run entirely within `self._lock` to avoid race conditions where `add()` is called while the buffer is being overwritten.

### 2. Live Gradient Retention for Training
- **Problem:** If we detach a tensor that needs backpropagation (e.g., `mean_loss`), PyTorch will raise an error during `loss.backward()`.
- **Solution:** Only detach the *copied* values passed to `summary_writer.add_scalar`. The original `mean_loss` returned by `compute_loss()` must remain fully attached to the computation graph.

### 3. Device Allocation for Logged Tensors
- **Problem:** Logged tensors may reside on a GPU or other accelerator device.
- **Solution:** Explicitly call `.cpu()` before `.item()` to ensure compatibility and prevent device-to-host memory/driver failures.

---

## 🧪 Testing Strategy

To verify the changes and prevent future regressions, the following tests must be updated or added:

### 1. Verification of `CombatBuffer.clear()`
- **Path:** `tests/test_global_buffer.py`
- **Test cases:**
  1. `test_combat_buffer_clear_resets_pointers`:
     - Store several combat samples in a `GlobalBuffer` instance.
     - Assert `buffer.get_combat_buffer_size() > 0`.
     - Call `buffer.combat_buffer.clear()`.
     - Assert `buffer.get_combat_buffer_size() == 0` and `buffer.combat_buffer._pos == 0`.
  2. `test_combat_buffer_clear_releases_references`:
     - Populated the buffer with large mock objects or arrays.
     - Call `buffer.combat_buffer.clear()`.
     - Assert that every element in `buffer.combat_buffer._buffer` is `None`.

### 2. Validation of TensorBoard Detachment (Unit Level)
- **Path:** `tests/test_training_step.py`
- **Verification:** Run `pytest tests/test_training_step.py` to confirm that the standard training loop continues to execute correctly, backpropagates successfully, and runs without warnings or computational graph errors.
