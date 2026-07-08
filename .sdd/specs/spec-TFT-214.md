# Technical Specification for TFT-214: Workers allocate all memory on start

## 1. Overview & Root Cause Diagnosis
When launching distributed collection and evaluation workers via `run_workers_distributed.sh`, they immediately consume excessive system memory, potentially exhausting all available RAM and leaving no headroom for the training server process.

Investigation into worker startup identified four major memory-allocation hotspots:
1. **GlobalBuffer in every worker process**:
   On startup, each worker initializes a full `GlobalBuffer`, which internally instantiates:
   - `GameplayBuffer(maxlen=20000)`
   - `CombatBuffer(capacity=12800)`
   This pre-allocates massive arrays and deques, taking up to ~6.1 GB of RAM per process. However, collector workers do not need to store 20,000 gameplay episodes or 12,800 combats in memory since they are designed to POST data to the training server immediately upon completing games.
2. **Collection Agent in Evaluator Worker**:
   During `orch.setup()`, a redundant `collection_agent` is always created and registered under `self._training_agents` and `self.agent_manager`. This consumes ~332 MB of memory for unused model weights. The evaluator worker only runs standalone comparisons between `best_model` and `current_model` and never performs experience collection.
3. **ThreadPoolExecutor in MuZeroAgent**:
   Every `MuZeroAgent` instance creates a persistent `ThreadPoolExecutor` with `max_workers=BATCH_SIZE=128`. With multiple agents running, this creates hundreds of threads, adding significant virtual and physical memory overhead (~8 MB stack size per thread). Since distributed training is the only supported path, local batch inference threading inside `MuZeroAgent` is redundant.
4. **MCTS Queue Leak**:
   Dynamic `BlockingBatchInferenceQueue` queues are created per `game_id` within the `EnhancedMCTS` class during parallel gameplay but are never removed or cleaned up, causing unbounded memory accumulation over long-running worker processes.

This specification outlines a clean, production-ready solution to address all four memory hotspots.

---

## 2. Architecture & Design Philosophy
- **Modular Depth**: The changes are encapsulated within the existing buffer, agent, and orchestrator boundaries. Replay buffers and individual agents remain fully decoupled from the internal routing details.
- **Lightweight Worker Mode**: We introduce a high-performance, low-overhead `WorkerGlobalBuffer` class that intercepts storage calls and POSTs them directly to the server, removing the ~6.1 GB memory overhead entirely.
- **Redundant Component Stripping**: By introducing an `is_evaluator` flag to `setup()`, the orchestrator completely bypasses collection agent, agent manager, environment manager, and benchmarking setups for evaluation workers.
- **Sequential MCTS execution**: We replace the heavy thread pool in `MuZeroAgent` with synchronous list comprehensions, eliminating the virtual memory explosion.
- **Deterministic Cleanup**: We implement clean-up paths for batch queues inside MCTS upon game completion.

---

## 3. Detailed Implementation Plan

### A. WorkerGlobalBuffer (`Models/global_buffer.py`)
We will implement a lightweight, low-memory `WorkerGlobalBuffer` inside `Models/global_buffer.py` to act as a zero-allocation proxy.

1. Implement `WorkerCombatBuffer`:
   ```python
   class WorkerCombatBuffer:
       def __init__(self):
           self._size = 0
           self._buffer = []
       def clear(self):
           pass
   ```
2. Implement `WorkerGlobalBuffer`:
   ```python
   class WorkerGlobalBuffer:
       """Lightweight buffer for worker processes that POSTs experiences directly to the server.
       Does not allocate massive pre-allocated queues, avoiding ~6.1 GB of per-process overhead.
       """
       def __init__(self, action_to_policy: Optional[Callable] = None):
           self.action_to_policy = action_to_policy
           self.batch_size = config.BATCH_SIZE
           self.gameplay_buffer = []
           self.combat_buffer = WorkerCombatBuffer()

       def _convert_sample_if_needed(self, sample):
           if self.action_to_policy is None:
               return sample
           converted = []
           for item in sample:
               obs, action, value, reward, policy = item[:5]
               from Models.action_conversion import action_to_policy_if_needed, is_3d_action
               if is_3d_action(action):
                   policy = action_to_policy_if_needed(action, policy, self.action_to_policy)
               extended = list(item)
               if len(extended) >= 7:
                   extended[4] = policy
                   converted.append(tuple(extended))
               else:
                   converted.append((obs, action, value, reward, policy))
           return converted

       async def store_episode_async(self, sample):
           """POST gameplay episode directly to the training server."""
           converted = self._convert_sample_if_needed(sample)
           await self._post_to_server(converted, "gameplay")

       def store_episode(self, sample):
           """Sync wrapper to run async store_episode in the running loop."""
           try:
               loop = asyncio.get_running_loop()
               if loop.is_running():
                   loop.create_task(self.store_episode_async(sample))
                   return
           except RuntimeError:
               pass
           asyncio.run(self.store_episode_async(sample))

       def store_episode_sync(self, sample):
           self.store_episode(sample)

       def store_combat(self, sample):
           """POST combat experience directly to the training server."""
           try:
               loop = asyncio.get_running_loop()
               if loop.is_running():
                   loop.create_task(self._post_to_server([sample], "combat"))
                   return
           except RuntimeError:
               pass
           asyncio.run(self._post_to_server([sample], "combat"))

       def clear_gameplay_buffer(self):
           pass

       def clear_combat_buffer(self):
           pass

       async def _post_to_server(self, data, experience_type: str):
           import aiohttp
           import pickle
           import random
           
           url = f"http://{config.WORKERS_HOST}:{config.SERVER_PORT}/api/v1/experience"
           payload = pickle.dumps(data)
           timeout = aiohttp.ClientTimeout(total=30, connect=10)
           async with aiohttp.ClientSession(timeout=timeout) as session:
               for attempt in range(5):
                   try:
                       async with session.post(url, data=payload, headers={
                           "Content-Type": "application/octet-stream",
                           "X-Experience-Type": experience_type
                       }) as resp:
                           if resp.status == 200:
                               print(f"[WorkerGlobalBuffer] Successfully POSTed {len(data)} {experience_type} steps")
                               return
                           elif resp.status == 503:
                               # Backpressure: wait and retry
                               print(f"[WorkerGlobalBuffer] Server reported 503 on {experience_type} upload. Retrying in 10s...")
                               await asyncio.sleep(10.0)
                           else:
                               body = await resp.text()
                               print(f"[WorkerGlobalBuffer] Failed to upload {experience_type} (status {resp.status}): {body[:200]}")
                               return
                   except Exception as e:
                       print(f"[WorkerGlobalBuffer] Connection error on {experience_type} upload (attempt {attempt+1}): {e}")
                       if attempt < 4:
                           await asyncio.sleep(2.0 + random.random() * 2.0)
   ```

### B. Setup Optimization (`training_orchestrator.py`)
Modify `setup()` to accept `is_evaluator: bool = False`, initialize `WorkerGlobalBuffer` for all workers, and completely skip redundant allocations if `is_evaluator=True`.

1. Import `WorkerGlobalBuffer` from `Models.global_buffer`.
2. Update the `setup` method signature and body:
   ```python
   def setup(self, is_collector: bool = False, is_evaluator: bool = False):
       """Create all components: buffer, agents, batch processor, trainer."""
       self.trainer = Trainer()
       self.summary_writer = self._build_logger()
       
       # Use lightweight WorkerGlobalBuffer if running as a worker process
       if is_collector or is_evaluator:
           from Models.global_buffer import WorkerGlobalBuffer
           self.global_buffer = WorkerGlobalBuffer(action_to_policy=action_3d_to_policy)
       else:
           self.global_buffer = GlobalBuffer(config.BATCH_SIZE, action_to_policy=action_3d_to_policy)

       # --- agent config -------------------------------------------------
       if not is_collector:
           self.best_model = MuZeroAgent(
               action_size=3,
               action_limits=config.ACTION_DIM,
               obs_size=config.OBSERVATION_SIZE,
               simulations=config.NUM_SIMULATIONS,
               global_buffer=self.global_buffer,
               config_obj=self.cfg,
           )

           self.current_model = MuZeroAgent(
               action_size=3,
               action_limits=config.ACTION_DIM,
               obs_size=config.OBSERVATION_SIZE,
               simulations=config.NUM_SIMULATIONS,
               global_buffer=self.global_buffer,
               weights=copy.deepcopy(self.best_model.get_weights()),
               config_obj=self.cfg,
           )

           if self.training_step > 0:
               ckpt = f"./checkpoint/best_{self.training_step}"
               if os.path.isfile(ckpt):
                   state = torch.load(ckpt)
                   self.best_model.model.load_state_dict(state)
                   self.current_model.model.load_state_dict(state)

       # Evaluators do NOT need collection agents, agent manager, env manager, or benchmarking!
       if is_evaluator:
           print("Evaluator worker setup complete (skipping collection agent and environment manager).")
           return

       # MuZero agents for *collection* – start with best model weights
       if is_collector:
           collection_agent = MuZeroAgent(
               action_size=3,
               action_limits=config.ACTION_DIM,
               obs_size=config.OBSERVATION_SIZE,
               simulations=config.NUM_SIMULATIONS,
               global_buffer=self.global_buffer,
               config_obj=self.cfg,
           )
       else:
           collection_agent = MuZeroAgent(
               action_size=3,
               action_limits=config.ACTION_DIM,
               obs_size=config.OBSERVATION_SIZE,
               simulations=config.NUM_SIMULATIONS,
               global_buffer=self.global_buffer,
               weights=copy.deepcopy(self.best_model.get_weights()),
               config_obj=self.cfg,
           )
       self._training_agents = [collection_agent]

       agent_configs: List[Tuple[Any, int]] = [
           (collection_agent, 8)
       ]

       # --- batch processor + agent manager -------------------------------
       self.agent_manager, _ = create_custom_agent_setup(
           agent_configs,
           max_batch_size=self.cfg.max_batch_size,
           batch_timeout_ms=self.cfg.batch_timeout_ms,
           gpu_memory_fraction=self.cfg.gpu_memory_fraction,
           metrics_collector=self.metrics_collector,
       )

       # --- parallel env manager -----------------------------------------
       self.env_manager = self._create_env_manager(
           self.cfg.concurrent_games,
           profiling=self.profiling,
           metrics_collector=self.metrics_collector
       )

       # --- benchmark ----------------------------------------------------
       self.benchmark = EnvironmentBenchmark(parallel_env)
       self._run_benchmark()

       print(f"TrainingOrchestrator setup complete:")
       print(f"  Concurrent games : {self.cfg.concurrent_games}")
       print(f"  Batch size       : {self.cfg.max_batch_size}")
       print(f"  Training step    : {self.training_step}")
       print(f"  GPU available    : {torch.cuda.is_available()}")
   ```

### C. Remove ThreadPoolExecutor (`Models/MuZero_torch_agent.py`)
1. Remove `from concurrent.futures import ThreadPoolExecutor`.
2. Remove the executor instantiation:
   ```python
   # Delete this line:
   # self._executor = ThreadPoolExecutor(max_workers=getattr(self.config, 'BATCH_SIZE', 8))
   ```
3. Update `_batch_select_action_impl` to run sequentially:
   ```python
   # Replace:
   # results = list(self._executor.map(run_mcts_item, range(batch_size)))
   # With:
   results = [run_mcts_item(i) for i in range(batch_size)]
   ```

### D. MCTS Queue Leak Cleanup (`Models/MCTS_torch.py` & `Models/MuZero_torch_agent.py`)
1. Add `cleanup_game` to `EnhancedMCTS` in `Models/MCTS_torch.py`:
   ```python
   def cleanup_game(self, game_id: str):
       """Remove batch queue for a game_id to prevent memory leak."""
       with self.batch_queues_lock:
           if game_id in self.batch_queues:
               del self.batch_queues[game_id]
   ```
2. Implement `terminate` override in `MuZeroAgent` inside `Models/MuZero_torch_agent.py` to trigger the cleanup:
   ```python
   def terminate(self, final_value, player_id=None):
       """Handle game/episode termination and clean up MCTS queues to prevent memory leak."""
       super().terminate(final_value, player_id)
       if hasattr(self, 'mcts') and self.mcts is not None:
           if player_id is None and isinstance(final_value, dict):
               # Clean up for any game IDs found in the dict keys (bulk flush)
               game_ids = set()
               for key in final_value.keys():
                   if "thread_env_" in key:
                       game_ids.add(key.split("_player")[0])
                   elif "env_" in key:
                       game_ids.add(key.split("_player")[0])
               for g_id in game_ids:
                   self.mcts.cleanup_game(g_id)
   ```

### E. Worker Role Setup in `main.py`
Modify `main.py` to correctly identify the evaluator worker role and pass `is_evaluator` into `orch.setup()`:
```python
    is_collector = (worker_role == "collector")
    is_evaluator = (worker_role == "evaluator")
    if is_collector:
        orch._build_logger = lambda: None
        
    orch.setup(is_collector=is_collector, is_evaluator=is_evaluator)
```

---

## 4. Unit & Integration Testing Strategy
We will add new tests to verify that these changes are functioning as designed.

### Test Cases to Implement:
1. **`test_worker_global_buffer_posting`** (`tests/test_global_buffer.py`):
   - Instantiate a `WorkerGlobalBuffer`.
   - Mock `aiohttp.ClientSession.post` using `unittest.mock`.
   - Call `store_episode_async` and verify that a POST request is sent to the correct endpoint with the pickled gameplay payload.
   - Call `store_combat` and verify that a POST request is sent to the correct endpoint with the pickled combat payload.
2. **`test_mcts_queue_cleanup`** (`tests/test_batched_agent_refactor.py`):
   - Instantiate a `MuZeroAgent` and run a simulated game termination with multiple player keys in `final_value`.
   - Verify that the MCTS queue for that game_id is successfully cleaned up from `mcts.batch_queues`.
