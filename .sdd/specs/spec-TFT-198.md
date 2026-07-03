### 📋 Technical Specification

#### 1. Overview
The server address is currently hardcoded to `127.0.0.1:8081` for the workers. To allow distributed training across multiple machines in a network, this hardcoded URL must be replaced. We will introduce `WORKERS_HOST` into `config.py` so workers can configure which machine is hosting the server. We will also remove the command line parameter `--server-url` and instead rely entirely on the configuration file, and finally, split `run_distributed.sh` into two distinct scripts (one for the server and one for the workers).

#### 2. Structural & File Changes
- **`config.py`**:
  - Add `WORKERS_HOST = "127.0.0.1"` right after `SERVER_HOST` and `SERVER_PORT`.
- **`main.py`**:
  - Remove the `--server-url` argument definition from the argparse setup.
  - In the worker loop, dynamically construct `server_url = f"http://{config.WORKERS_HOST}:{config.SERVER_PORT}"` instead of reading `--server-url` from `args` or using a hardcoded string.
- **Shell Scripts**:
  - Delete `run_distributed.sh`.
  - Create `run_server_distributed.sh`: Contains argument parsing, cleanup traps, data clearance, and solely starts the GPU Training Server using `--mode train_server`.
  - Create `run_workers_distributed.sh`: Contains argument parsing, cleanup traps, and starts the Evaluator worker and 6 Collection workers.

#### 3. Implementation Steps
1. In `config.py`, around line 76:
   ```python
   SERVER_HOST = "0.0.0.0"
   SERVER_PORT = 8081
   WORKERS_HOST = "127.0.0.1"
   ```
2. In `main.py`, around line 504:
   - Remove:
     ```python
     parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8081",
                         help="URL of the training server (worker mode)")
     ```
3. In `main.py`, around line 234 (inside worker execution branch):
   - Replace:
     ```python
     server_url = getattr(args, "server_url", "http://127.0.0.1:8081").rstrip("/")
     ```
   - With:
     ```python
     server_url = f"http://{config.WORKERS_HOST}:{config.SERVER_PORT}"
     ```
4. Extract server logic from `run_distributed.sh` to a new script `run_server_distributed.sh` which executes `main.py --mode train_server`. Keep the directory creation and experience clearing in the server script (as the server acts as the primary data orchestrator). Ensure it has executable permissions (`chmod +x`).
5. Extract worker logic from `run_distributed.sh` to a new script `run_workers_distributed.sh` which executes `main.py --mode worker` for both the evaluator and the collectors. Remove the initial `sleep 4` wait for the server (since it's a separate script and expected to be running already). Ensure it has executable permissions (`chmod +x`).
6. Ensure both new shell scripts retain the `trap cleanup SIGINT SIGTERM EXIT` pattern so that subprocesses properly shut down when interrupted.
7. Finally, safely remove `run_distributed.sh` via git.

#### 4. Design Principles
- **Separation of Concerns:** By cleanly splitting `run_distributed.sh` into client (worker) and server shells, it naturally mirrors a true distributed architecture.
- **Single Source of Truth:** Centralizing the worker connection address into `config.py` enforces consistency and ensures that users don't have to specify `--server-url` repeatedly on the command line when starting different workers.
