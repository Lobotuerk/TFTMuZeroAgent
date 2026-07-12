# Technical Specification: Double-buffered Batch Fetch

## 1. Goal

Overlap CPU-side batch construction with GPU-side training in `TrainingOrchestrator._train_step()` to eliminate GPU idle time. This double-buffering pre-fetches the next batch on the CPU via the asyncio event loop while the current batch is being trained on the GPU via the thread pool executor.

## 2. Design Considerations

* **Buffer Locking**: The current architecture uses a separate process for the server and single event loop for training. Lock mechanisms (`threading.Lock`) within `GlobalBuffer` are obsolete single-process artifacts that offer no concurrency protection in pure asyncio, yet add overhead and complexity. They must be removed.
* **Prefetch Execution**: Pre-fetching the next batch involves synchronous CPU work (`read_gameplay_batch`). By scheduling it via `asyncio.create_task` combined with an `await asyncio.sleep(0)`, the pre-fetch task can run synchronously on the event loop concurrently with the GPU training thread (launched via `run_in_executor`), effectively overlapping CPU and GPU time without needing an additional thread pool.
* **Training Loop Clean-up**: `_train_step()` will be refactored to manage the pre-fetch lifecycle and return a boolean indicating success. `main.py` will use an orchestrator method `has_batch_ready()` to check loop conditions without accidentally ignoring pre-fetched batches already popped from the buffer.

## 3. Implementation Steps

### Step 1: Remove Obsolete Locks in `Models/global_buffer.py`
The locks are no longer needed and should be stripped.
1. Remove `import threading`.
2. In `CombatBuffer`: Remove `self._lock = threading.Lock()` from `__init__`. Remove all `with self._lock:` contexts from `add`, `clear`, and `sample` methods. Keep their inner logic intact.
3. In `GameplayBuffer`: Remove `self._lock = threading.Lock()` from `__init__`. Remove all `with self._lock:` contexts from `add`, `sample`, and `clear` methods. Keep their inner logic intact.
4. In `WorkerCombatBuffer`: Remove `self._lock = threading.Lock()` from `__init__`. Remove all `with self._lock:` contexts from `add`, `pop`, `clear`, `get_all`, `remove_front`, and `size` methods. Keep their inner logic intact.

### Step 2: Refactor `TrainingOrchestrator` in `training_orchestrator.py`
1. **Initialize State**: Add `self._next_batch_task: Optional["asyncio.Task"] = None` to `TrainingOrchestrator.__init__` (requires `import asyncio` if not present, though it's already used).
2. **Add Batch Preparation Helper**: Extract the synchronous batch reading into `_prepare_batch_sync(self)`:
   ```python
   def _prepare_batch_sync(self):
       if not self.global_buffer or not self.global_buffer.available_gameplay_batch():
           return None

       batch = self.global_buffer.read_gameplay_batch()
       combat_batch = []
       if hasattr(self.global_buffer, "available_combat_batch") and self.global_buffer.available_combat_batch():
           cb = self.global_buffer.read_combat_batch()
           if cb is not None:
               combat_batch = cb
       return batch, combat_batch
   ```
3. **Add Pre-fetch Task Helper**: Add `_fetch_next_batch_task(self)`:
   ```python
   async def _fetch_next_batch_task(self):
       # Yield to the event loop so the thread executor can launch
       await asyncio.sleep(0)
       return self._prepare_batch_sync()
   ```
4. **Refactor `_train_step`**: Change the signature to `async def _train_step(self) -> bool:` and implement the double-buffered logic:
   ```python
   async def _train_step(self) -> bool:
       """Perform a single training step using double-buffered batch pre-fetching."""
       try:
           # 1. Initialize prefetch task if none exists
           if getattr(self, "_next_batch_task", None) is None:
               self._next_batch_task = asyncio.create_task(self._fetch_next_batch_task())

           # 2. Wait for the pre-fetched batch
           batch_data = await self._next_batch_task
           if batch_data is None:
               self._next_batch_task = None
               return False

           batch, combat_batch = batch_data

           # 3. Immediately schedule the NEXT prefetch task
           self._next_batch_task = asyncio.create_task(self._fetch_next_batch_task())

           # 4. Perform GPU training (runs in thread pool)
           loop = asyncio.get_event_loop()
           await loop.run_in_executor(
               self._train_executor,
               self.trainer.train_network,
               batch,
               combat_batch,
               self.current_model.model,
               self.training_step,
               self.summary_writer,
           )
           self.training_step += 1

           # ... Keep existing checkpoint, benchmark, and metric logic ...

           return True
       except Exception as e:
           import traceback
           traceback.print_exc()
           print(f"Error in _train_step: {e}")
           raise
   ```
5. **Add State Checker**: Add `has_batch_ready(self) -> bool` to check if a task exists or if the buffer has data:
   ```python
   def has_batch_ready(self) -> bool:
       if getattr(self, "_next_batch_task", None) is not None:
           return True
       return bool(self.global_buffer and self.global_buffer.available_gameplay_batch())
   ```

### Step 3: Update Training Server Loop in `main.py`
Update `train_server_mode` to utilize `has_batch_ready()` and handle the boolean return value from `_train_step()`.

**From:**
```python
        while orch.training_active:
            trained = False
            while orch.training_active and orch.global_buffer.available_gameplay_batch() and orch.training_step < args.max_steps:
                await orch._train_step()
                trained = True
                await asyncio.sleep(0.001)
            if not trained:
                await asyncio.sleep(1.0)
```

**To:**
```python
        while orch.training_active:
            trained = False
            while orch.training_active and orch.has_batch_ready() and orch.training_step < args.max_steps:
                trained_this_step = await orch._train_step()
                if trained_this_step:
                    trained = True
                    await asyncio.sleep(0.001)
                else:
                    break
            if not trained:
                await asyncio.sleep(1.0)
```

## 4. Expected Impact
By stripping the obsolete locks, the CPU batch creation can securely and sequentially pop items from the deques. Pre-fetching the next batch allows the synchronous sampling overhead to finish alongside the multithreaded tensor crunching taking place inside `Trainer.train_network`. As a result, the next iteration of the loop should find its `_next_batch_task` already completed, maximizing GPU utilization.