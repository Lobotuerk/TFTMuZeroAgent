# Technical Specification for TFT-213: Saving batched data to disk

## 1. Overview
In the distributed training setup, when multiple collectors finish environments at similar times, they POST large volumes of experience data simultaneously to the training server. Due to each gameplay experience entry containing massive multidimensional observation arrays (~10–50 MB per entry), deserializing and storing thousands of these entries directly into the server's in-memory `GameplayBuffer` (`deque`) can quickly consume all available RAM (32GB) and trigger Out-Of-Memory (OOM) crashes on the server.

To guarantee high system reliability and memory stability without sacrificing worker productivity, we will implement:
1. **Primary - Memory-aware flow control**: The server will monitor its host virtual memory usage. When memory usage crosses a threshold (85%), it will respond to worker POSTs with HTTP 503 (Retry). Workers will intercept the 503 status code, back off, sleep, and safely retry sending the local queue without executing more games.
2. **Secondary - Disk persistence (spill-to-disk)**: When memory is healthy, the server will partition incoming gameplay POSTs into complete batches of size `config.BATCH_SIZE` and write them directly to disk as `.pkl` files in `config.GAMEPLAY_BUFFER_PATH`. Leftover samples (less than a complete batch) will remain in the in-memory buffer. During training, the server will prioritize consuming, deserializing, training, and deleting these batch files from disk.
3. **Model promotion buffer purging**: On model promotion (evaluation model beats the best model), all gameplay data (in-memory buffer and disk batch files) will be deleted to avoid training on outdated policy data.
4. **Worker buffer synchronization**: The server will track a stateless `promotion_count`. When a worker pulls best weights, it receives the `promotion_count`. If this count has incremented, the worker will load the new weights and immediately clear its local gameplay and combat buffers to discard old data.

## 2. Architecture & Design Philosophy
- **Modular Depth**: Encapsulate disk-writing, file-scanning, and batch-formatting logic completely inside the `GameplayBuffer` and `GlobalBuffer` classes in `Models/global_buffer.py`. The training loop in `TrainingOrchestrator` will continue calling `available_gameplay_batch()` and `read_gameplay_batch()` transparently without needing to know whether the data is coming from RAM or disk.
- **Simple Interfaces**: Avoid adding complex parameters or new classes. The client-server communication leverages HTTP 503 for backpressure and a single `promotion_count` field in the existing weight JSON envelope for synchronization.
- **Clean Exception Paths**: File reads/writes are wrapped in robust try-except blocks. If a `.pkl` file is corrupted, it is automatically removed from disk and the server moves on to the next file or memory queue.
- **Separation of Concerns**: Keep workers simple and stateless relative to the server. Workers do not need local disk persistence; they simply hold data in memory and wait to transmit if backpressured.

## 3. Detailed Implementation Plan

### A. Configuration & Memory Threshold (`config.py`)
1. Define a default memory-usage percentage threshold:
   ```python
   MEMORY_THRESHOLD = 85.0
   ```
2. Confirm that `GAMEPLAY_BUFFER_PATH` (default `./data/gameplay/`) is defined and referenced.

### B. GlobalBuffer Modifications (`Models/global_buffer.py`)
1. **Extend `GameplayBuffer`**:
   - Extract the numpy array construction and unroll padding logic from `sample` into a helper method:
     ```python
     def _format_batch(self, samples: List[Any]) -> List[np.ndarray]:
         observation_batch = []
         action_batch = []
         value_batch = []
         reward_batch = []
         policy_batch = []
         target_obs_batch = []
         bootstrap_depth_batch = []
         for sample in samples:
             observation_batch.append(sample[0])
             action_batch.append(sample[1])
             value_batch.append(sample[2])
             reward_batch.append(sample[3])
             policy_batch.append(sample[4])
             if len(sample) >= 7:
                 target_obs_batch.append(sample[5])
                 bootstrap_depth_batch.append(sample[6])
             else:
                 target_obs_batch.append(None)
                 bootstrap_depth_batch.append(config.UNROLL_STEPS)
         result = [
             np.array(observation_batch),
             np.array(action_batch),
             np.array(value_batch),
             np.array(reward_batch),
             np.array(policy_batch)
         ]
         result.append(np.array(target_obs_batch))
         result.append(np.array(bootstrap_depth_batch))
         return result
     ```
   - Update `sample(self, batch_size)` to invoke `_format_batch(samples)`.

2. **Extend `GlobalBuffer`**:
   - Implement `add_gameplay_experience(self, samples)`:
     ```python
     import uuid
     import time
     import pickle
     import os

     def add_gameplay_experience(self, samples):
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
         
         if leftover > 0:
             leftover_data = converted[num_batches * batch_size:]
             self.gameplay_buffer.add(leftover_data)
     ```
   - Update `available_gameplay_batch(self)`:
     ```python
     def available_gameplay_batch(self):
         # Check disk files first
         if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
             files = [f for f in os.listdir(config.GAMEPLAY_BUFFER_PATH) if f.endswith(".pkl")]
             if len(files) > 0:
                 return True
         # Fallback to memory
         return len(self.gameplay_buffer) >= self.batch_size
     ```
   - Update `read_gameplay_batch(self)`:
     ```python
     def read_gameplay_batch(self):
         if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
             files = sorted([f for f in os.listdir(config.GAMEPLAY_BUFFER_PATH) if f.endswith(".pkl")])
             if len(files) > 0:
                 filepath = os.path.join(config.GAMEPLAY_BUFFER_PATH, files[0])
                 try:
                     with open(filepath, "rb") as f:
                         batch_samples = pickle.load(f)
                     os.remove(filepath)
                     return self.gameplay_buffer._format_batch(batch_samples)
                 except Exception as e:
                     print(f"Error reading/deleting batch file {filepath}: {e}")
                     if os.path.exists(filepath):
                         try:
                             os.remove(filepath)
                         except:
                             pass
         # Fallback to memory
         return self.gameplay_buffer.sample(self.batch_size)
     ```
   - Implement `clear_all_gameplay_data(self)`:
     ```python
     def clear_all_gameplay_data(self):
         self.clear_gameplay_buffer()
         if os.path.exists(config.GAMEPLAY_BUFFER_PATH):
             for f in os.listdir(config.GAMEPLAY_BUFFER_PATH):
                 if f.endswith(".pkl"):
                     try:
                         os.remove(os.path.join(config.GAMEPLAY_BUFFER_PATH, f))
                     except Exception as e:
                         print(f"Error deleting file {f}: {e}")
     ```

### C. Server Endpoints & Flow Control (`main.py`)
1. **Initialize Promotion State** in `train_server_mode`:
   - Add `orch.promotion_count = 0` prior to starting the HTTP server.
2. **Implement Memory Protection in `handle_experience`**:
   - Check RAM status before processing any POST payload:
     ```python
     import psutil
     mem = psutil.virtual_memory()
     threshold = getattr(config, "MEMORY_THRESHOLD", 85.0)
     if mem.percent > threshold:
         print(f"[Server] Memory threshold exceeded ({mem.percent}% > {threshold}%). Triggering backpressure (503).")
         return web.Response(status=503, reason="Service Unavailable (High Memory)", text="Memory usage too high")
     ```
   - If acceptable, route experience:
     ```python
     if experience_type == "gameplay":
         orch.global_buffer.add_gameplay_experience(data)
     ```
3. **Include `promotion_count` in weight JSON** in `handle_weights`:
   - Append the current state: `"promotion_count": getattr(orch, "promotion_count", 0)` in the response body.
4. **Flush Old Buffers on Model Promotion** in `handle_promote_best`:
   - Purge memory and disk caches upon new model setup:
     ```python
     orch.global_buffer.clear_all_gameplay_data()
     orch.promotion_count += 1
     ```

### D. Worker-side Backpressure & Weight Synchronization (`main.py`)
1. **Sync + Clear local worker buffers in `worker_mode` (collectors)**:
   - Handle the new weights payload envelope:
     ```python
     server_promo_count = resp_json.get("promotion_count", 0)
     if not hasattr(orch, "last_promotion_count"):
         orch.last_promotion_count = server_promo_count
     elif server_promo_count > orch.last_promotion_count:
         print(f"[Worker {worker_id}] New best model promoted. Clearing local buffers.")
         orch.global_buffer.clear_gameplay_buffer()
         orch.global_buffer.clear_combat_buffer()
         orch.last_promotion_count = server_promo_count
     ```
2. **Handle HTTP 503 backpressure in worker upload phase**:
   - Introduce infinite retry loops with 10s wait intervals:
     ```python
     while True:
         resp = await _request_with_retry(
             "POST", f"{server_url}/api/v1/experience",
             data=data,
             headers={"Content-Type": "application/octet-stream", "X-Experience-Type": "gameplay"}
         )
         async with resp:
             if resp.status == 200:
                 print(f"[Worker {worker_id}] Sent {len(samples)} gameplay steps")
                 orch.global_buffer.clear_gameplay_buffer()
                 break
             elif resp.status == 503:
                 print(f"[Worker {worker_id}] Server reported high memory (503). Retrying in 10s...")
                 await asyncio.sleep(10.0)
             else:
                 body = await resp.text()
                 print(f"[Worker {worker_id}] Failed to send gameplay steps (status {resp.status}): {body[:200]}")
                 break
     ```
   - Apply the same looping pattern for combat experience POSTs.

## 4. Edge Cases and Resilience
- **High Concurrency Disk Access**: `available_gameplay_batch` and `read_gameplay_batch` scan the file system. Disk lookups are sorted to preserve FIFO (first-in, first-out) order of training batches.
- **Write Failures**: Disk-write operations are guarded against full-disk or permissions issues, failing gracefully back to in-memory buffers where possible.
- **Worker Starvation**: Exponential retry logic is not required for 503 handling, as workers should maintain a steady 10s sleep cycle. This ensures they don't flood the server with spam retries.
- **Test Integrity**: Unit tests will programmatically mock the filesystem and `psutil` virtual memory queries to guarantee correct branching behavior.
