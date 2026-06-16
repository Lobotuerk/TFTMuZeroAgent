# Technical Specification: TFT-193 - Networked Distributed Training

**Author:** SDD-Designer
**Date:** 2026-06-16
**Status:** Draft

## 1. Overview

This document outlines the technical design for enabling distributed training and data collection for the TFT-MuZero agent across multiple computers on a local area network (LAN). The current implementation relies on a shared filesystem, which limits all processes (training, evaluation, data collection) to a single machine. This redesign will replace the file-based communication protocol with a lightweight, HTTP-based protocol, decoupling the components and allowing them to run on separate machines.

The core principle is to introduce a centralized **Training Server** that exposes an HTTP API for workers. **Collector Workers** and **Evaluator Workers** will transition from filesystem operations to HTTP clients, sending experience data and retrieving model weights over the network.

This approach was chosen for its simplicity, minimal dependency footprint (relying on Python's built-in `http.server`), and alignment with the user's requirements for a LAN-only, infrastructure-free solution.

## 2. Proposed Changes

### 2.1. Training Server: HTTP API Implementation

The `train_server` mode in `main.py` will be enhanced to run a background HTTP server in a separate thread. This server will handle API requests from all worker processes.

**File to Modify:** `TFTMuZeroAgent/main.py` (within the `train_server` function)

**API Endpoints:**

| Method | Endpoint | Description | Request Body | Response Body |
| --- | --- | --- | --- | --- |
| `POST` | `/api/experience` | Submits a batch of gameplay or combat experience. | `pickle`-serialized data tuple: `(type, data)` where `type` is 'gameplay' or 'combat'. | `{"status": "ok"}` |
| `GET` | `/api/weights/latest` | Retrieves the most recent model weights for evaluation. | N/A | Raw `model.pth` file content. (`Content-Type: application/octet-stream`) |
| `GET` | `/api/weights/best` | Retrieves the best-performing model weights for collectors. | N/A | Raw `model.pth` file content. (`Content-Type: application/octet-stream`) |

**Implementation Details:**
- A new class, `TrainingAPIServer`, will be created in a new file `TFTMuZeroAgent/distributed/http_server.py`.
- This class will use Python's `http.server.HTTPServer` and `http.server.BaseHTTPRequestHandler`.
- The request handler will parse URLs to route requests to the correct logic (experience submission or weight retrieval).
- Experience data received via `POST` will be placed into the same in-memory buffers currently used by the file-polling mechanism. The file-polling loop will be removed.
- The server will be started in a `threading.Thread` to run concurrently with the main training loop.

### 2.2. Worker Processes: HTTP Client Implementation

The `worker` mode in `main.py` will be updated to act as an HTTP client. A new command-line argument will control the communication method.

**Files to Modify:**
- `TFTMuZeroAgent/main.py` (within `worker` function)
- `TFTMuZeroAgent/distributed/worker.py` (or a new `http_client.py` module)

**Command-Line Argument:**
- A new argument `--server-url <URL>` will be added to `main.py`.
- If `--server-url` is provided, the worker will operate in HTTP mode.
- If it is omitted, the system can either default to a local URL (e.g., `http://localhost:8080`) or maintain the legacy file-based mode for backward compatibility. The primary implementation will target the HTTP-only path, with a default to `localhost`.

**Implementation Details:**
- Logic within the worker will be refactored. Instead of writing `.pkl` files to disk, it will `pickle.dumps()` the experience data and send it via an HTTP `POST` request to the `/api/experience` endpoint.
- To get model weights, the worker will send `GET` requests to `/api/weights/best` (for collectors) or `/api/weights/latest` (for evaluators) and load the received model data directly.
- The `requests` library (or `urllib.request`) will be used for making HTTP calls. To avoid adding a new dependency, `urllib.request` is preferred.

### 2.3. Configuration Changes

To support the new networking capabilities, the central configuration will be updated.

**File to Modify:** `TFTMuZeroAgent/config.py`

**New Parameters:**
- `SERVER_HOST`: The IP address the training server will bind to. Default: `"0.0.0.0"` (listens on all available interfaces).
- `SERVER_PORT`: The port the training server will listen on. Default: `8080`.
- `DEFAULT_SERVER_URL`: The default URL for workers to connect to if not specified. Default: `"http://localhost:8080"`.

### 2.4. Removal of Filesystem Polling

The existing code responsible for polling the `./data/gameplay/` and `./data/combats/` directories in the `train_server` will be removed entirely, as it is being replaced by the `/api/experience` endpoint. The corresponding buffer paths in `config.py` (`GAMEPLAY_BUFFER_PATH`, `COMBAT_BUFFER_PATH`) will be deprecated and removed.

## 3. Deprecations & Removals

- **Removed `config.py` variables:**
  - `GAMEPLAY_BUFFER_PATH`
  - `COMBAT_BUFFER_PATH`
- **Removed Logic:**
  - Filesystem polling loop in `train_server`.
  - `.pkl` file writing in `worker`.
- **Modified Scripts:**
  - `run_distributed.sh` will be updated to launch workers with the appropriate `--server-url` argument if they are intended to run in the new distributed mode, even if on the same machine.

## 4. Test Plan

1.  **Unit Tests:**
    - A test for the `TrainingAPIServer` request handler to verify correct routing and data handling for both `POST` and `GET` requests.
    - A test for the worker's client-side logic, mocking `urllib.request` to ensure it correctly formats and sends requests.
2.  **Integration Test (Single Machine):**
    - Run `train_server` and one of each worker type (`collector`, `evaluator`) on the same machine.
    - Verify that experience is successfully transmitted from the collector to the server's buffer.
    - Verify that both worker types can successfully download the latest and best models.
    - This test validates that the default `localhost` configuration works as a direct replacement for the old file-based system.
3.  **Integration Test (Multi-Machine):**
    - Manually run `train_server` on one machine.
    - On a separate machine on the same LAN, run a `worker` process, providing the IP address of the server machine via the `--server-url` argument.
    - Verify end-to-end data flow as in the single-machine test.

## 5. Rollout & Compatibility

The introduction of the `--server-url` argument with a `localhost` default ensures that existing scripts like `run_distributed.sh` will continue to function with minimal modification. The core change is non-destructive in that it replaces one local communication method (filesystem) with another (local HTTP), while simultaneously enabling remote communication. The same codebase will work for both local and remote workers, fulfilling a key user requirement.
