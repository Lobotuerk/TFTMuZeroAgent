# Technical Specification for TFT-213: Saving batched data to disk (Memory Deadlock Fix)

## 1. Overview & Root Cause Diagnosis
In the initial implementation of the distributed spill-to-disk feature for `TFTMuZeroAgent` (PR #31), a server-side memory check was introduced at the entry point of `/api/v1/experience` (`main.py:155`):
```python
threshold = getattr(config, "MEMORY_THRESHOLD", 85.0)
if mem.percent > threshold:
    return web.Response(status=503, reason="Service Unavailable (High Memory)", text="Memory usage too high")
```
### The Deadlock
Under normal distributed execution with 5 collection workers, a training orchestrator, and an evaluation worker, the host system's RAM (32GB) naturally stabilizes above 85% (only ~4.8GB of headroom remains). 
Once memory crosses 85%, the server immediately returns a `503 Service Unavailable` status code to all worker experience uploads.
This results in a total deadlock:
1. **No files on disk**: Because the request is rejected at the handler entrance, `orch.global_buffer.add_gameplay_experience(data)` is never invoked, meaning zero batch files are ever written to disk.
2. **Infinite retries**: Workers retry indefinitely in a loop, holding all accumulated data in their own memory.
3. **No training progress**: Since no data is accepted, the server cannot sample any new batches to train. Because training is blocked, the model never progresses and memory is never cleared or decreased.
4. **Console Spam**: The server continuously spams "Memory threshold exceeded" warnings.

### The Two-Tiered Solution
To resolve this deadlock while still protecting the server against OOM crashes, we will implement a **Two-Tiered Memory Control Structure**:
1. **Tier 1: Spill-to-Disk Active Warning (`MEMORY_THRESHOLD = 85.0`)**
   When host RAM usage is above 85.0% but below 95.0%, the server **STILL ACCEPTS** the incoming experience payload. Disk writes do not consume significant transient or permanent RAM. 
   However, the server calls `add_gameplay_experience(data, skip_memory_buffer=True)`. This ensures that any complete batches (size `config.BATCH_SIZE`) are safely persisted as `.pkl` files on disk, but any leftover samples are **completely discarded** rather than appended to the in-memory RAM deque. This halts any RAM-based buffer growth immediately.
2. **Tier 2: Hard OOM Protection (`CRITICAL_MEMORY_THRESHOLD = 95.0`)**
   If host RAM usage climbs above 95.0%, the server triggers real backpressure by returning `503 Service Unavailable`. This is a last-resort safety valve to prevent system-wide OOM crashes. Workers will back off, wait, and retry.

This tiered system guarantees that:
- Workers can successfully upload experience data when memory is moderately high.
- The local worker buffers are cleared, reducing total memory across the environment.
- Disk files accumulate instead of server RAM.
- The training loop continuously drains and deletes batch files from disk, allowing RAM to stabilize.

---

## 2. Architecture & Design Philosophy
- **Modular Depth**: The training loop and orchestrator interfaces remain completely unchanged. The `GlobalBuffer` encapsulates the routing and spill-to-disk details. The new `skip_memory_buffer` parameter is fully backward-compatible.
- **Information Hiding**: Callers and workers remain entirely agnostic to whether their data is kept in memory or written to disk. All disk-file management (naming, scanning, deleting) is hidden inside `GlobalBuffer`.
- **Zero-Cost Exception Handling**: If a disk batch file is corrupted, the server catches the exception, deletes the invalid file, logs the warning, and immediately falls back to the in-memory queue or next file.
- **Testability**: The design allows full, side-effect-free testing of both healthy and high-memory code paths using mock environments.

---

## 3. Detailed Implementation Plan

### A. Configuration (`config.py`)
1. Define the hard critical threshold next to `MEMORY_THRESHOLD`:
   ```python
   MEMORY_THRESHOLD = 85.0
   CRITICAL_MEMORY_THRESHOLD = 95.0
   ```

### B. GlobalBuffer Modifications (`Models/global_buffer.py`)
1. Update `add_gameplay_experience` signature and behavior to accept `skip_memory_buffer`:
   ```python
   def add_gameplay_experience(self, samples, skip_memory_buffer: bool = False):
       converted = self._convert_sample_if_needed(samples)
       batch_size = self.batch_size
       num_batches = len(converted) // batch_size
       leftover = len(converted) % batch_size

       if num_batches > 0:
           os.makedirs(config.GAMEPLAY_BUFFER_PATH, exist_ok=True)
           for i in range(num_batches):
               batch_data = converted[i * batch_size : (i + 1) * batch_size]
               filename = f"batch_{time.time_ns()}_{uuid.uuid4().hex}.pkl"
               filepath = os.path.join(config.GAMEPLAY_BUFFER_PATH, filename)
               with open(filepath, "wb") as f:
                   pickle.dump(batch_data, f)

       if leftover > 0 and not skip_memory_buffer:
           leftover_data = converted[num_batches * batch_size:]
           self.gameplay_buffer.add(leftover_data)
   ```

### C. Server Endpoints & Route Control (`main.py`)
1. Refactor `handle_experience` to apply the two-tiered threshold logic:
   ```python
   async def handle_experience(request):
       experience_type = request.headers.get("X-Experience-Type", "")
       if experience_type not in ("gameplay", "combat"):
           return web.Response(status=400, text="Invalid or missing X-Experience-Type header")

       mem = psutil.virtual_memory()
       threshold = getattr(config, "MEMORY_THRESHOLD", 85.0)
       critical_threshold = getattr(config, "CRITICAL_MEMORY_THRESHOLD", 95.0)

       # Tier 2: Hard OOM Protection
       if mem.percent > critical_threshold:
           print(f"[Server] Critical memory threshold exceeded ({mem.percent}% > {critical_threshold}%). Triggering backpressure (503).")
           return web.Response(status=503, reason="Service Unavailable (Critical Memory)", text="Critical memory usage too high")

       # Tier 1: Spill-to-Disk Active Warning
       skip_memory_buffer = False
       if mem.percent > threshold:
           skip_memory_buffer = True
           print(f"[Server] Memory threshold exceeded ({mem.percent}% > {threshold}%). Spill-to-disk active (bypassing in-memory buffer).")

       body = await request.read()
       try:
           data = pickle.loads(body)
       except Exception:
           return web.Response(status=400, text="Invalid pickle data")

       if experience_type == "gameplay":
           orch.global_buffer.add_gameplay_experience(data, skip_memory_buffer=skip_memory_buffer)
       else:
           for sample in data:
               orch.global_buffer.store_combat(sample)
       return web.Response(status=200)
   ```

---

## 4. Unit & Integration Testing Strategy
We will add new tests to `tests/test_global_buffer.py` to exhaustively verify the new parameters and disk-writing logic.

### Test Cases to Implement:
1. **`test_add_gameplay_experience_spill_to_disk`**:
   - Create a `GlobalBuffer` with `batch_size = 4`.
   - Call `add_gameplay_experience` with 10 samples and `skip_memory_buffer = False`.
   - Verify that 2 batch files are written to disk under `config.GAMEPLAY_BUFFER_PATH`.
   - Verify that 2 leftovers are added to the in-memory `gameplay_buffer`.
   - Clean up the disk files afterwards.

2. **`test_add_gameplay_experience_skip_memory_buffer`**:
   - Create a `GlobalBuffer` with `batch_size = 4`.
   - Call `add_gameplay_experience` with 10 samples and `skip_memory_buffer = True`.
   - Verify that 2 batch files are written to disk.
   - Verify that 0 leftovers are added to the in-memory `gameplay_buffer` (i.e. size of `gameplay_buffer` remains 0).
   - Clean up the disk files.

3. **`test_read_gameplay_batch_from_disk`**:
   - Call `add_gameplay_experience` to write batches to disk.
   - Verify that calling `read_gameplay_batch()` returns a batch loaded from disk, and that the read file is automatically deleted.
   - Verify that subsequent calls read from the remaining files, and eventually fallback to the in-memory buffer once all files are consumed.