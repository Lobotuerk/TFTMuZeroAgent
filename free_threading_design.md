# Design Document: Migration to Python 3.13+ Free-Threading

## 1. Overview & Rationale

Currently, `TFTMuZeroAgent` achieves CPU parallelism during environment simulation by using separate OS processes via `multiprocessing` (`_MultiProcessEnvManager`). This bypasses the traditional Python Global Interpreter Lock (GIL) but introduces significant architectural and performance overheads:
- **Inter-Process Communication (IPC) Overhead:** All observations (including high-dimensional NumPy arrays), rewards, actions, and scores must be serialized (pickled) and deserialized across pipes, which bottlenecks scaling.
- **Complexity in Process Management:** Managing child processes, pipelines, EOF errors, and cleaning up crashed workers adds considerable cognitive load and fragility.
- **Memory Footprint:** Running multiple complete Python OS processes duplicates the memory space.

With **Python 3.13+ Free-Threading (PEP 703)**, the GIL can be disabled entirely. This enables true multi-threaded CPU execution within a single Python process. Migrating to free-threading allows us to:
1. Replace heavy, pipe-communicating multiprocessing workers with lightweight threads.
2. Share the same memory address space, passing observations and actions as zero-copy Python references.
3. Eliminate all IPC serialization overhead, leading to faster games/sec.
4. Align with future high-performance Python patterns.

---

## 2. Architectural Design

We introduce a thread-based environment manager `_ThreadEnvManager` that mirrors the public API of `_MultiProcessEnvManager` and `_ParallelEnvManager`. This maintains a **deep module** interface, ensuring that `TrainingOrchestrator` does not need to know how environments are managed under the hood.

### 2.1 The Sync-Async Bridge via Thread-Safe Futures

Since `EnhancedAgentManager.get_actions` is an asynchronous coroutine that executes on the main thread (coordinating GPU batch inference), and the environment simulations are synchronous CPU-bound loops, we bridge them safely without any custom lock contention or pipe overhead.

We leverage `asyncio.run_coroutine_threadsafe(coro, loop)` from the background thread:
1. **Background Thread (CPU Simulation):**
   Runs a synchronous game loop (`env.step` / `env.reset`). When it needs action selection, it schedules the async `get_actions` coroutine on the main event loop and waits for the result:
   ```python
   coro = agent_manager.get_actions(observations, rewards, terminated, game_id)
   future = asyncio.run_coroutine_threadsafe(coro, loop)
   actions = future.result()  # Standard concurrent.futures.Future blocking call
   ```
2. **Main Thread (Async Event Loop):**
   Awaits and batches the inference request on the GPU, then resolves the future, unblocking the simulator thread.

This design completely decouples the threading execution model of simulations from the async batch inference server, preventing concurrent access to PyTorch or asyncio structures.

---

## 3. Thread Safety & Locking

1. **Inference Server & Model:**
   Since PyTorch model execution and `EnhancedAgentManager` async states are evaluated exclusively on the main event loop thread, they are inherently thread-safe.
2. **Experiences & Buffers:**
   `GameplayBuffer` and `CombatBuffer` inside `Models/global_buffer.py` already protect all entry additions and sampling with a `threading.Lock()`. This is fully thread-safe without the GIL.
3. **Environment Instances:**
   Each background thread instantiates its own dedicated, isolated `parallel_env` instance. There is zero shared state between simulator environments.

---

## 4. Proposed Classes & APIs

### `_ThreadEnvManager` (in `training_orchestrator.py`)

```python
import threading
import asyncio

class _ThreadEnvManager:
    """
    Manages N concurrent game workers running in separate threads.
    Bypasses serialization overhead in Python 3.13+ Free-Threading mode.
    """
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self._threads: List[threading.Thread] = []
        self._tasks: List[asyncio.Task] = []
        self.should_continue = True
        self.should_spawn = True

    def stop(self):
        self.should_continue = False

    def pause(self):
        self.should_spawn = False

    def resume(self):
        self.should_spawn = True

    async def wait_for_drain(self):
        # Cooperatively wait for active worker threads to terminate
        pass

    async def run_continuously(self, agent_manager: EnhancedAgentManager, on_game_done: Optional[Callable] = None):
        # Starts threads, registers on_game_done callbacks, and monitors execution
        pass

    async def run_fixed_games(self, agent_manager: EnhancedAgentManager, num_games: int) -> List[GameResult]:
        # Runs exact number of games via thread pool / managed threads
        pass
```

---

## 5. Migration Sub-Tasks (Doable, Testable, Verifiable)

To ensure high-quality construction, the work is divided into the following sequential sub-tasks:

### Task 1: Environment GIL Check & Config Toggle
- **Objective:** Add programmatic detection of free-threading / GIL status.
- **Details:** 
  - Update `config.py` (or `training_orchestrator.py`) to auto-detect if the GIL is disabled:
    ```python
    import sys
    IS_GIL_DISABLED = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
    ```
  - Allow forcing the threading manager via a configuration flag (e.g., `FORCE_THREADING_ENV_MANAGER = False`).
- **Verification:** Unit test checking correct detection of the GIL state.

### Task 2: Implement `_ThreadEnvManager` & Worker Thread Loop
- **Objective:** Implement `_ThreadEnvManager` and `_thread_worker_main` in `training_orchestrator.py`.
- **Details:**
  - Implement `_thread_worker_main(worker_id, loop, agent_manager, should_continue, should_spawn, queue/callback)` which runs the environment simulation.
  - Bridge to `agent_manager.get_actions(...)` via `asyncio.run_coroutine_threadsafe(...)`.
  - Handle termination, pause, resume, and errors gracefully.
- **Verification:** Ensure it adheres to the same public interface as `_MultiProcessEnvManager`.

### Task 3: Integrate into `TrainingOrchestrator`
- **Objective:** Enable `TrainingOrchestrator` to automatically use `_ThreadEnvManager` if `IS_GIL_DISABLED` or forced via config.
- **Details:**
  - Update `self.env_manager` initialization in `TrainingOrchestrator.setup()` to instantiate `_ThreadEnvManager` instead of `_MultiProcessEnvManager` if Free-Threading is active.
- **Verification:** The entire orchestrator initialization and lifecycle runs without change.

### Task 4: Comprehensive Unit & Integration Testing
- **Objective:** Verify correctness, lack of deadlocks, and performance.
- **Details:**
  - Create `tests/test_thread_env_manager.py` (modeling after `test_multiprocess_env_manager.py`) to test fixed games, continuous play, drain logic, and exception handling of the thread-based manager.
  - Validate that the existing test suite (particularly orchestrator integration tests) passes cleanly when threading is enabled.

---

## 6. Definition of Done (Verification Checklist)

- [ ] Free-threading/GIL status is programmatically detectable.
- [ ] `_ThreadEnvManager` implements the exact public contract as `_MultiProcessEnvManager`.
- [ ] Direct memory references are passed to the async agent manager instead of pickled data over pipes.
- [ ] No deadlocks occur under parallel thread-to-async loop interaction.
- [ ] Robust error-handling prevents a single worker thread crash from hanging the training run.
- [ ] `tests/test_thread_env_manager.py` verifies both `run_fixed_games` and `run_continuously` work perfectly.
- [ ] All unit and integration tests run and pass successfully.
