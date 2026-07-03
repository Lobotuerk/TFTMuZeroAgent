# Technical Specification: TFT-199 (Hardcoded paths)

## 1. Context and Problem Statement
The shell scripts `run_server_distributed.sh` and `run_workers_distributed.sh` within the `TFTMuZeroAgent` repository contain absolute, hardcoded paths to a specific user's local Python interpreter (`/home/lobo/miniconda3/envs/TFT/bin/python`). This prevents the distributed training scripts from running on other machines or in different environments where the `TFT` conda environment might be installed differently.

## 2. Proposed Solution
We need to replace the hardcoded interpreter paths with a robust, dynamically resolved Python path. The user's directive is to "Resolve it via which python conda prefix". This implies the scripts should leverage `$CONDA_PREFIX` if the script is running inside a Conda environment, or fallback to the system's active `python`. 

The target scripts call `run_tft.sh` and pass the Python interpreter path to it. We will declare a variable `PYTHON_EXEC` at the start of these scripts that resolves the proper Python interpreter path, and then replace the hardcoded strings with `"$PYTHON_EXEC"`.

### Logic for `PYTHON_EXEC` Resolution:
```bash
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXEC="$(which python)"
fi
```

## 3. Backwards Compatibility
Since the assumption (as clarified by the user) is that "the readme has been followed" (meaning the conda environment is expected to be properly configured and activated before running the script), resolving via `$CONDA_PREFIX/bin/python` will perfectly replicate the behavior without failing on missing environments. If the user uses a non-conda environment, the fallback to `$(which python)` will successfully point to the appropriate virtual environment. No backwards compatibility issues are expected.

## 4. Implementation Steps
1. **Modify `run_server_distributed.sh`:**
   - Add the `PYTHON_EXEC` resolution logic below the initial configurations.
   - Replace:
     ```bash
     PYTHON_GIL=0 ./run_tft.sh /home/lobo/miniconda3/envs/TFT/bin/python main.py --mode train_server --checkpoint_interval 200 $EXTRA_ARGS &
     ```
     With:
     ```bash
     PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode train_server --checkpoint_interval 200 $EXTRA_ARGS &
     ```
     
2. **Modify `run_workers_distributed.sh`:**
   - Add the `PYTHON_EXEC` resolution logic below the initial configurations.
   - Replace the hardcoded path for the evaluator worker:
     ```bash
     PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode worker --worker_id 0 --worker_role evaluator --eval_games 9 --eval_concurrent 3 $EXTRA_ARGS &
     ```
   - Replace the hardcoded path for the collector workers:
     ```bash
     PYTHON_GIL=0 ./run_tft.sh "$PYTHON_EXEC" main.py --mode worker --worker_id $i --worker_role collector --concurrent_games 2 &
     ```

## 5. Verification
- Verify the scripts run successfully without errors when the `TFT` conda environment is active.
- Verify that `which python` or `$CONDA_PREFIX` correctly evaluates and runs `main.py`.
- No new automated tests are needed for this change, as the fix is entirely at the bash script level and does not affect the core application logic.
