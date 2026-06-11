### 📋 Technical Specification (Corrected)

**Issue:** [TFT-188](mention://issue/7a988e28-2278-434c-ab52-ba9bbfcc419b) - Current happy path

**Objective:** Refactor the `TFTMuZeroAgent` codebase to remove dead code, unused features, and unnecessary complexity, aligning it with the production "happy path" defined by the `run_distributed.sh` script. This specification corrects a previously flawed version based on detailed implementation feedback.

**Repository:** `https://github.com/Lobotuerk/TFTMuZeroAgent.git`

---

### 1. Overview

The `run_distributed.sh` script defines the production training loop. Analysis by the `SDD-Implementer` agent revealed that the initial specification was incorrect and would have broken the training pipeline by attempting to remove critical, in-use code.

This corrected specification outlines a safe refactoring plan. The core of the change is to untangle the `enhanced_agent_interface.py` file by moving essential components to a new, dedicated file, and then proceeding with the removal of genuinely dead code.

---

### 2. Refactoring and Code Removal Plan

#### 2.1. CRITICAL Refactoring: Extract Live Code from `enhanced_agent_interface.py`

The file `Models/enhanced_agent_interface.py` contains classes that are essential to `training_orchestrator.py`. It cannot be deleted directly.

1.  **Create a new file**: `Models/agent_manager.py`.
2.  **Move the following classes and functions** from `Models/enhanced_agent_interface.py` into the new `Models/agent_manager.py`:
    *   `BatchInferenceServer`
    *   `EnhancedAgentManager`
    *   `AsyncGameEnvironment`
    *   `EnvironmentPool`
    *   `InferenceRequest` (dataclass)
    *   `BatchedInferenceRequest` (dataclass)
    *   `create_enhanced_setup`
    *   `_create_default_agent_configs`
    *   `create_custom_agent_setup`
    *   All other factory/helper functions at the bottom of the file (`create_muzero_vs_random_setup`, etc.)
3.  **Update Imports**: Modify `training_orchestrator.py` to import these classes from `Models.agent_manager` instead of `Models.enhanced_agent_interface`.
4.  **Delete the File**: After moving the live code and updating the imports, the file `Models/enhanced_agent_interface.py` will be largely empty or contain only dead code. It can now be safely deleted.

#### 2.2. Classes and Functions to be Removed from Live Files

The following are confirmed dead code within files that are otherwise in use.

-   **In `training_orchestrator.py`**:
    -   Remove the `_ParallelEnvManager` class. It is unreachable as the factory function `_create_env_manager` never returns it.
-   **In `Models/MuZero_torch_agent.py`**:
    -   Remove the type alias `EnhancedMuZeroAgent = MuZeroAgent`.
    -   Remove the factory function `create_enhanced_muzero_agent`.

#### 2.3. Test Files to be Deleted

The following test files cover the now-deleted `_ParallelEnvManager` or the parts of the `enhanced_agent_interface` that were truly dead.

-   `tests/test_enhanced_agent.py`
-   `tests/test_enhanced_ai_interface.py`
-   `tests/test_enhanced_eval.py`
-   `tests/test_enhanced_episode.py`
-   `tests/test_enhanced_interface.py`
-   `tests/test_environment_pool.py`
-   `tests/test_parallel_training.py`

*(Note to Implementer: The original spec listed more test files. Double-check that the remaining tests for `MCTS_torch`, `batched_inference`, etc., are still relevant and passing after the refactor.)*

---

### 3. Code to be EXPLICITLY Kept

Based on the re-evaluation, the following files and functions are critical to the happy path and **MUST NOT be removed**:

-   **`Models/batched_inference.py`**: Kept. It is a dependency of `Models/MCTS_torch.py`.
-   **Functions in `Models/action_conversion.py`**:
    -   `action_3d_to_policy`
    -   `is_3d_action`
    -   `action_to_policy_if_needed`
    -   These are used throughout the model, replay buffer, and MCTS.
-   **`benchmark_training.py`**: Kept. It uses the `TrainingOrchestrator` and is therefore aligned with the happy path, per the user's request.
-   **In `Common_agents.py`**:
    -   `RerollAgent` and `FastLevelAgent` are to be kept for future use.
-   **Silent Fallbacks**:
    -   The `try/except pass` patterns for checkpoint loading and experience file reading will be kept as-is.

---

### 4. Implementation Steps

1.  **Checkout Fresh Branch**: Start from the `main` branch of the `TFTMuZeroAgent` repository.
2.  **Perform Refactoring**: Create `Models/agent_manager.py` and move the classes/functions as described in section 2.1.
3.  **Update Imports**: Change `training_orchestrator.py` to import from the new `Models.agent_manager.py`.
4.  **Delete File**: Delete `Models/enhanced_agent_interface.py`.
5.  **Modify Files**: Edit the files listed in section 2.2 to remove the specified dead code.
6.  **Delete Tests**: Delete the test files listed in section 2.3.
7.  **Run Tests**: Execute the entire remaining test suite to ensure that the happy path is still functional and no regressions have been introduced.
8.  **Static Analysis**: Run linters and other static analysis tools to ensure code quality.
9.  **Commit and Push**: Commit the changes and open a Pull Request.

---

### 5. Validation

The successful implementation of this specification will be validated by:

-   **CI Pipeline**: All remaining tests and static analysis checks must pass.
-   **Manual Verification**: Running the `run_distributed.sh` script should successfully start the training process without errors.
-   **Code Review**: The final pull request should be reviewed to confirm that the refactoring was done correctly and only the specified dead code has been removed.
