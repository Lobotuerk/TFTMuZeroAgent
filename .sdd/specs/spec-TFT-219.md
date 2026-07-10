# Technical Specification - TFT-219: Bug in worker combat buffer

## Overview
A bug was identified in `main.py` where the worker collection loop attempted to access the non-existent private attribute `_size` of `WorkerCombatBuffer`, leading to an `AttributeError`. Additionally, the previous implementation had a potential race condition where any samples appended to the buffer during the asynchronous upload process would be lost when `combat_buffer.clear()` was called after a successful upload.

This specification outlines the design to:
1. Fix the `AttributeError` by using the thread-safe `size` property of `WorkerCombatBuffer`.
2. Redesign the buffer access to use clean public interfaces conforming to our Information Hiding and Deep Modules principles.
3. Solve the concurrency race condition by implementing a safe, partial removal mechanism for successfully uploaded samples.
4. Add comprehensive unit tests to prevent regressions.

---

## 1. File Structure Changes
The following files will be modified:
* **`Models/global_buffer.py`**: Extend `WorkerCombatBuffer` with safe public query and slice methods.
* **`main.py`**: Update `worker_mode`'s combat experience collection step to use the new public API and avoid race conditions.
* **`tests/test_global_buffer.py`**: Add automated test coverage verifying the correctness of `WorkerCombatBuffer`'s new methods and concurrency behavior.

---

## 2. Interfaces & Signatures

### `WorkerCombatBuffer` (in `Models/global_buffer.py`)
To prevent external callers from directly accessing or slicing the private `_buffer` list, we will introduce two thread-safe public methods:

1. **`get_all(self) -> List[Any]`**
   * **Description**: Returns a shallow copy of all elements currently residing in the buffer.
   * **Thread-Safety**: Executed within the instance's lock (`self._lock`).

2. **`remove_front(self, count: int) -> None`**
   * **Description**: Removes the first `count` elements from the front of the buffer, shifting the remaining elements forward.
   * **Thread-Safety**: Executed within the instance's lock (`self._lock`).
   * **Safety Constraint**: If `count` is negative or exceeds the current buffer size, it must be handled gracefully (e.g. by using Python list slicing which naturally handles out-of-bound indices safely).

### `main.py` (within `worker_mode`)
Inside the `worker_role == "collector"` branch of the worker loop:
* **Check**: Access the thread-safe `combat_buffer.size` property instead of `_size` to determine if there are leftovers to send.
* **Retrieval**: Use the public `combat_buffer.get_all()` method to retrieve the pending samples, rather than slicing `_buffer` directly.
* **Cleanup**: On successful POST (HTTP 200), call `combat_buffer.remove_front(len(combat_samples))` instead of `combat_buffer.clear()`.

---

## 3. Edge Cases & Concurrency

### Concurrency Race Condition
During the asynchronous HTTP POST request (`await _request_with_retry(...)`), the event loop yields execution. In a multi-threaded or multi-process environment, concurrent game tasks could append new combat samples to the `WorkerCombatBuffer` before the HTTP request completes.
* **Old Behavior**: Calling `clear()` would wipe the entire buffer, including any newly added samples, causing silent data loss.
* **New Behavior**: Calling `remove_front(len(combat_samples))` ensures that only the successfully uploaded samples are removed. Any samples appended during the upload are preserved at the end of the buffer and will be sent in the next iteration.

### Empty or Small Buffer
* If the buffer is empty, `size` is `0`, and the upload step is skipped entirely.
* If `remove_front` is called with `0`, the buffer remains unchanged.

### Failure of Upload
* If the HTTP upload fails (non-200 / non-503), the loop breaks without calling `remove_front`. The samples remain in the buffer and are retried on the next iteration.

---

## 4. Testing Strategy

We will introduce specific unit tests in `tests/test_global_buffer.py`:

1. **`test_worker_combat_buffer_get_all`**
   * **Verification**: Add items, call `get_all()`, and assert that the returned list matches the added items and has the correct size.

2. **`test_worker_combat_buffer_remove_front`**
   * **Verification**: Add multiple items, call `remove_front(N)`, and assert that the first `N` items are removed while the remaining items are correctly preserved.

3. **`test_worker_combat_buffer_concurrency_simulation`**
   * **Verification**:
     * Add initial items.
     * Retrieve items using `get_all()`.
     * Simulate an in-flight operation by appending new items.
     * Call `remove_front(len(initial_items))`.
     * Assert that the buffer contains exactly and only the newly added items.
