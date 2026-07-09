# Technical Specification: TFT Gym Unique Logging IDs (TFT-215)

## Overview
Currently, in distributed training setups, multiple parallel collector workers are executed. Each worker process manages a few environment instances (e.g., `--concurrent_games 2`), passing a process-local worker index (like `0` or `1`) as `rank` to `parallel_env()`. Consequently, workers across different processes attempt to write to the same log files (`log_0.txt` and `log_1.txt`), causing file clobbering, race conditions, and corrupted logs.

This specification addresses the problem by:
1. Generating a globally unique identifier for each environment instance using the worker's subprocess OS process ID (`os.getpid()`) combined with its local environment index (`env_id`).
2. Cleaning up the legacy, unused thread-based environment management infrastructure (`_ThreadEnvManager` and `_thread_worker_main`), which has been superseded by multiprocessing isolation.

---

## 1. File Structure Changes

### Files to Delete
* `tests/test_thread_env_manager.py` (Contains tests specifically for the deleted thread env manager)
* `tests/test_gil_config_toggle.py` (Contains tests for GIL-checking config switches tied to threading managers)

### Files to Modify
* `config.py`
  - Remove `FORCE_THREADING_ENV_MANAGER` configuration variable.
* `training_orchestrator.py`
  - Modify `_env_worker_main` to supply a unique `rank` to `parallel_env`.
  - Remove `_thread_worker_main` function.
  - Remove `_ThreadEnvManager` class.
  - Simplify `TrainingOrchestrator._create_env_manager` to instantiate `_MultiProcessEnvManager` unconditionally.
* `main.py`
  - Remove GIL/thread check function (`_check_gil`) and its invocation.
* `benchmark_training.py`
  - Remove GIL/thread check function (`_check_gil`) and its invocation.
  - Simplify benchmark console output to remove reference to `FORCE_THREADING_ENV_MANAGER` and thread-based managers.
* `readme.md`
  - Remove references/documentation describing `_ThreadEnvManager` and GIL-free threading configurations.
* `tests/test_benchmark_training.py`
  - Clean up mock patches of `_ThreadEnvManager`.
* `tests/test_non_blocking_training.py`
  - Clean up mock patches of `_ThreadEnvManager`.

---

## 2. Interfaces & Signatures

### Unique Rank Identifier
Within the subprocess target `_env_worker_main` in `training_orchestrator.py`:
* Instead of initiating `parallel_env` with `rank=env_id`, construct a unique string rank incorporating `os.getpid()` and `env_id`.
* The updated signature for initializing the environment will remain:
  ```python
  env = parallel_env(rank=unique_rank)
  ```
  where `unique_rank` is a string (e.g., `"12345_0"`).

### Removal of Threading-related Interfaces
* `_ThreadEnvManager` class and its entire public interface is fully deprecated and removed from `training_orchestrator.py`.
* `_thread_worker_main` function is fully removed.
* `TrainingOrchestrator._create_env_manager(num_workers, profiling, metrics_collector)` static method will have its signature unchanged but its interior simplified:
  - It will no longer inspect `config.FORCE_THREADING_ENV_MANAGER`.
  - It will unconditionally instantiate and return a `_MultiProcessEnvManager`.

---

## 3. Edge Cases

### String Type Compatibility
* **Risk:** The `rank` parameter of `parallel_env()` in `TFTSet4Gym` might require an integer.
* **Mitigation:** CodeGraph and file-level inspections of the `TFTSet4Gym` submodule confirm that `rank` is exclusively used as a string interpolation token to build log file names (i.e. `f'log_{self.rank}.txt'`). Therefore, passing a string formatted as `f"{pid}_{env_id}"` is fully compatible and raises no type or runtime exceptions.

### Process Fork vs. Spawn Safety
* **Risk:** Subprocesses on Linux might inherit thread contexts if fork is used.
* **Mitigation:** The multiprocessing context in `training_orchestrator.py` is explicitly configured to use `spawn` (`MP_CONTEXT = mp.get_context('spawn')`). This ensures a completely clean, isolated OS process space with a fresh OS PID for each worker, guaranteeing PID uniqueness across parallel workers.

### Concurrent Log Writing
* **Risk:** Multiple workers on different physical servers or containers might have conflicting PIDs.
* **Mitigation:** Because each worker instance operates on its own dedicated storage or local container volume, localized PID uniqueness is sufficient to guarantee file name segregation. If multi-node storage is shared, the combination of OS PID and local index ensures distinct log file names.

---

## 4. Testing Strategy

### Verification Assertions
1. **Subprocess Unique Log Identification:**
   - Execute the multiprocessing environment manager and assert that individual subprocesses receive distinct, unique string `rank` arguments incorporating their own OS PID.
   - Verify that log files of the format `log_<pid>_<env_id>.txt` are successfully created and populated, rather than competing for a shared `log_0.txt`/`log_1.txt` handle.
2. **Backward Compatibility of Logging Output:**
   - Confirm that the submodule environment initializes and executes standard gameplay steps successfully when supplied with a string rank.
3. **No Threading Regressions:**
   - Run the full existing suite of non-blocking and benchmark tests (`pytest tests/`) to ensure all remaining test suites pass perfectly without reference to `_ThreadEnvManager` or `FORCE_THREADING_ENV_MANAGER`.
