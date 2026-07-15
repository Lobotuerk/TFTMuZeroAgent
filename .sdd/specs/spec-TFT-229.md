# Technical Specification: Memory Pressure and Spill-to-Disk Logic (TFT-229)

## 1. File Structure Changes
The following files will be modified in the repository:
- `Models/global_buffer.py`: Re-implement `read_gameplay_batch`, `available_gameplay_batch`, and `add_gameplay_experience`. Add `drain_memory_to_disk` and `_load_from_disk_until_batch`.
- `main.py`: Update `handle_experience` to invoke `drain_memory_to_disk` when the memory threshold is exceeded.

## 2. Interfaces & Signatures

### `Models/global_buffer.py`
- **`GlobalBuffer.drain_memory_to_disk(self)`**:
  - **Purpose**: Flushes the entire in-memory `gameplay_buffer` to disk as `.pkl` files and clears the buffer.
  - **Logic**: Converts `list(self.gameplay_buffer)` to a list. If not empty, chunks the list into batches of `self.batch_size` and saves them to `config.GAMEPLAY_BUFFER_PATH` using UUIDs for safety.
- **`GlobalBuffer._load_from_disk_until_batch(self)`**:
  - **Purpose**: Reads `.pkl` files from disk and adds them to `self.gameplay_buffer` until `len(self.gameplay_buffer) >= self.batch_size` or files are exhausted.
  - **Logic**: A `while` loop that pops files from disk, deserializes them, and calls `self.gameplay_buffer.add()`. Handles exceptions and cleans up corrupted files.
- **`GlobalBuffer.available_gameplay_batch(self)`**:
  - **Logic**: Calls `self._load_from_disk_until_batch()` first, then returns `len(self.gameplay_buffer) >= self.batch_size`.
- **`GlobalBuffer.read_gameplay_batch(self)`**:
  - **Logic**: Calls `self._load_from_disk_until_batch()` first, then returns `self.gameplay_buffer.sample(self.batch_size)`. (This naturally prioritizes memory because if memory already has a batch, disk is not read).
- **`GlobalBuffer.add_gameplay_experience(self, samples, skip_memory_buffer: bool = False)`**:
  - **Logic**: When `leftover > 0`, if `skip_memory_buffer` is `True`, the leftover items are serialized to disk instead of being discarded.

### `main.py`
- **`handle_experience(request)`**:
  - **Logic**: Inside the threshold check (`if mem.percent > threshold:`), call `orch.global_buffer.drain_memory_to_disk()` immediately before setting `skip_memory_buffer = True`.

## 3. Edge Cases & Concurrency
- **Idempotency of Drain**: `drain_memory_to_disk` checks if `gameplay_buffer` is empty before writing to avoid creating empty files. It's safe to call continuously during memory pressure.
- **Partial Batches on Disk**: Because leftovers are written to disk, `.pkl` files might contain fewer than `batch_size` elements. `_load_from_disk_until_batch` correctly accumulates items in memory until `batch_size` is reached.
- **File System Conflicts**: Pickle files use `time.time_ns()` and `uuid.uuid4().hex` to prevent collisions from async concurrent `handle_experience` handlers.
- **Training Stability**: Because `read_gameplay_batch()` requires `batch_size` to be fulfilled in memory, the training loop will never receive partial batches unless configured otherwise, preventing downstream shape mismatch errors.

## 4. Testing Strategy
- **Drain Test**: Insert items into memory, call `drain_memory_to_disk()`, assert memory buffer is empty and files appear in `GAMEPLAY_BUFFER_PATH`.
- **Priority Test**: With items in both memory and disk, `read_gameplay_batch()` should consume memory first before unpickling disk files.
- **Leftovers Test**: Call `add_gameplay_experience(skip_memory_buffer=True)` with a non-multiple of `batch_size` and assert the leftover count appears as a `.pkl` file on disk.
