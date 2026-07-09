# SDD Technical Specification — TFT-216: Batched Combat Step POSTing for WorkerGlobalBuffer

## 1. File Structure Changes
The following files in the `TFTMuZeroAgent` repository are affected by this design:
- **Modified:**
  - `Models/global_buffer.py`:
    - Refactor `WorkerCombatBuffer` from a stub into a real thread-safe accumulator.
    - Modify `WorkerGlobalBuffer.__init__` to initialize `self.combat_buffer` with `batch_size`.
    - Modify `WorkerGlobalBuffer.store_combat` to accumulate samples in `WorkerCombatBuffer` and flush only when `batch_size` is reached.
    - Implement `WorkerGlobalBuffer._flush_combat` to pop and POST batches of `batch_size` samples.
    - Implement `WorkerGlobalBuffer.clear_combat_buffer` to clear the underlying combat buffer.
  - `tests/test_global_buffer.py`:
    - Add unit tests for `WorkerCombatBuffer` to verify accumulation, popping, sizing, and clearing.
    - Add mock-based integration tests for `WorkerGlobalBuffer.store_combat` to verify batched POST behavior.

---

## 2. Interfaces & Signatures

### `WorkerCombatBuffer` Class
A thread-safe accumulator class designed to collect worker combat steps and extract them in fixed-size batches.

- **`__init__(self, batch_size: int = config.BATCH_SIZE)`**:
  - Initializes an empty list `self._buffer`.
  - Stores `self.batch_size`.
  - Initializes a thread lock `self._lock` (`threading.Lock()`).
- **`add(self, sample: Any) -> bool`**:
  - Acquires the thread lock.
  - Appends the sample to `self._buffer`.
  - Returns `True` if `len(self._buffer) >= self.batch_size`, otherwise `False`.
- **`pop(self) -> List[Any]`**:
  - Acquires the thread lock.
  - Pops exactly `self.batch_size` elements from the front of `self._buffer`.
  - Retains any remaining leftover elements in `self._buffer`.
  - Returns the list of popped elements.
- **`clear(self) -> None`**:
  - Acquires the thread lock.
  - Clears `self._buffer`.
- **`size (property) -> int`**:
  - Acquires the thread lock.
  - Returns the length of `self._buffer`.

### `WorkerGlobalBuffer` Class
A client-side worker buffer that coordinates upload to the server.

- **`__init__(self, action_to_policy: Optional[Callable] = None)`**:
  - Updates `self.combat_buffer` to be instantiated with `WorkerCombatBuffer(batch_size=self.batch_size)`.
- **`store_combat(self, sample: Any) -> None`**:
  - Adds the sample to `self.combat_buffer` via `.add()`.
  - If the addition triggers a full batch (`True`), calls `self._flush_combat()`.
- **`_flush_combat(self) -> None`**:
  - Retrieves a full batch of size `self.batch_size` from `self.combat_buffer` using `.pop()`.
  - Detects if an asyncio event loop is currently active.
    - If active, enqueues `_post_to_server(batch, "combat")` as a background task.
    - If inactive, executes the coroutine using `asyncio.run()`.
- **`clear_combat_buffer(self) -> None`**:
  - Clears the underlying `self.combat_buffer` via `.clear()`.

---

## 3. Edge Cases

- **Leftover Samples (< batch_size):**
  - Samples numbering less than a full `batch_size` must persist across combat rounds and only be flushed once the accumulator reaches the target size. There is no forced flush mechanism at the end of the simulation; leftovers are permitted to remain in-memory upon worker shutdown per product design requirements.
- **Thread Safety / Concurrency:**
  - Since multiple environment runner threads or async tasks may interact with `WorkerGlobalBuffer` and `WorkerCombatBuffer`, thread locks must protect all read and write operations on `WorkerCombatBuffer`'s internal list.
- **Event Loop Coexistence:**
  - `store_combat` is a synchronous method that must coexist peacefully inside active asyncio loops (utilizing `loop.create_task`) and outside them (utilizing `asyncio.run`).
- **Post Failures & Retry Safety:**
  - The `_post_to_server` method's built-in retry backoff strategy remains unchanged but will process a batch of size `self.batch_size` rather than single-item arrays. If all retries fail, errors are logged, but the data is safely dropped to avoid blocking training workers.

---

## 4. Testing Strategy

We will implement the following verification suite in `tests/test_global_buffer.py`:

### 1. `test_worker_combat_buffer_accumulation`
- Instantiate `WorkerCombatBuffer(batch_size=4)`.
- Add 3 mock samples; verify that `add()` returns `False` and size is `3`.
- Add the 4th sample; verify that `add()` returns `True` and size is `4`.
- Pop the batch; verify that the returned list contains all 4 samples in order, and the size becomes `0`.
- Add 5 samples; verify that the 4th returns `True`, the 5th returns `False`, and size is `5`.
- Pop the batch; verify that exactly 4 samples are returned and 1 leftover sample remains in the buffer (size is `1`).
- Clear the buffer; verify that size becomes `0`.

### 2. `test_worker_global_buffer_store_combat_batching`
- Initialize `WorkerGlobalBuffer` with `batch_size = 4`.
- Patch `WorkerGlobalBuffer._post_to_server` using `unittest.mock.patch` with `AsyncMock`.
- Call `store_combat` with 3 separate samples; verify that `_post_to_server` has not been called.
- Call `store_combat` with a 4th sample; verify that `_post_to_server` was called exactly once with the correct array of 4 samples and the experience type `"combat"`.
