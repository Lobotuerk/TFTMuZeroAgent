# SDD Technical Specification: TFT-193 - Networked Workers

**Author:** SDD-Designer
**Date:** 2026-06-16
**Status:** DRAFT

## 1. Overview

This document outlines the technical design to transition the `TFTMuZeroAgent` distributed training architecture from a local filesystem-based protocol to a network-based (HTTP) protocol. This will allow collector and evaluator "workers" to run on different machines across a local area network (LAN) and communicate with a central "training server".

The current implementation requires all processes (`train_server`, `worker`) to share a common filesystem, limiting training to a single machine. The approved solution is to introduce a lightweight HTTP server within the `train_server` process and have workers communicate with it over the network.

## 2. Guiding Principles

*   **Decoupling:** Components (server, workers) should not depend on a shared filesystem.
*   **Simplicity:** The solution will use Python's standard libraries (`http.server`, `socketserver`) or minimal dependencies like `aiohttp` to avoid introducing heavy frameworks.
*   **Backward Compatibility:** The existing single-machine orchestration script (`run_distributed.sh`) must continue to function with minimal changes. The same code must support both local-only and multi-machine deployments, selectable via configuration.

## 3. System Architecture Changes

The core change is the removal of file-based polling and the introduction of a simple RESTful API.

### 3.1. Training Server (`train_server_mode`)

The `train_server` process will host an HTTP server.

*   **HTTP Server:** An `aiohttp.web` application will be integrated into the `train_server_mode` asyncio loop.
*   **Configuration:** The server will listen on `config.SERVER_HOST` and `config.SERVER_PORT` (e.g., `0.0.0.0:8080`).
*   **Logic Change:** The existing file polling loop in `train_server_mode` (which scans for `.pkl` files) will be **removed**. Data will now arrive via the API endpoint and be placed directly into the in-memory buffers.

### 3.2. Worker (`worker_mode`)

Workers will be modified to act as HTTP clients.

*   **Configuration:** A new command-line argument, `--server-url`, will be introduced for all modes. It will default to `http://127.0.0.1:8080`.
*   **Logic Change (Collector):**
    *   Instead of writing experience `.pkl` files to disk, collectors will serialize their experience data and `POST` it to the training server's API.
    *   Instead of reading `best_model.pth` from disk, collectors will `GET` the model weights from the server's API.
*   **Logic Change (Evaluator):**
    *   Instead of checking the file modification time of `latest_model.pth`, the evaluator will `GET` it periodically from the server's API. `If-Modified-Since` headers should be used for efficiency.
    *   When an evaluation determines a new model is "best", the evaluator will `POST` a request to the server's API to promote the current `latest` model to `best`.

## 4. API Specification

All endpoints will be prefixed with `/api/v1`.

### 4.1. `POST /api/v1/experience`

*   **Purpose:** Allows a collector worker to submit generated experience.
*   **Method:** `POST`
*   **Headers:**
    *   `Content-Type: application/octet-stream`
    *   `X-Experience-Type: <'gameplay'|'combat'>` (Required)
*   **Body:** Raw bytes of the pickled experience object (a list of samples).
*   **Server Action:**
    1.  Read the `X-Experience-Type` header.
    2.  Deserialize the request body using `pickle.loads()`.
    3.  Add the resulting object to the appropriate in-memory buffer (`global_buffer.gameplay_buffer` or `global_buffer.combat_buffer`).
*   **Response:**
    *   `200 OK`: On success.
    *   `400 Bad Request`: If header is missing or body is malformed.

### 4.2. `GET /api/v1/weights/{name}`

*   **Purpose:** Allows workers to download model weights.
*   **Method:** `GET`
*   **URL Parameters:**
    *   `name`: The name of the weights to fetch. Must be one of `best` or `latest`.
*   **Server Action:**
    1.  Identify the requested weights file (`./checkpoint/best_model.pth` or `./checkpoint/latest_model.pth`).
    2.  Return the file contents.
*   **Response:**
    *   `200 OK` with `Content-Type: application/octet-stream` and the file bytes in the body.
    *   `404 Not Found`: If the requested weights file does not exist.

### 4.3. `POST /api/v1/weights/promote_best`

*   **Purpose:** Allows the evaluator worker to signal that the current `latest` model should become the new `best` model.
*   **Method:** `POST`
*   **Body:** Empty.
*   **Server Action:**
    1.  Copy the file `./checkpoint/latest_model.pth` to `./checkpoint/best_model.pth`.
*   **Response:**
    *   `200 OK`: On success.
    *   `500 Internal Server Error`: If the copy operation fails.

## 5. File and Code Modifications

### 5.1. `config.py`

*   Add new configuration variables:
    ```python
    # Networked Distributed Training
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8080
    ```

### 5.2. `main.py`

*   **Imports:** Add imports for `aiohttp` (or other HTTP library).
*   **Argument Parsing:** Add the `--server-url` argument with a default value.
*   **`train_server_mode(args)`:**
    *   Instantiate and run the `aiohttp` web application.
    *   Register handlers for the API endpoints defined above.
    *   **Remove** the `while True:` loop that performs `glob.glob` to find and read `.pkl` files. The training loop (`orch._train_step()`) will now be driven by the size of the in-memory buffer, which is populated by the API.
*   **`worker_mode(args)`:**
    *   **Collector:**
        *   Replace `pickle.dump()` to a file with an HTTP `POST` request to `{server_url}/api/v1/experience`.
        *   Replace `torch.load()` from `./checkpoint/best_model.pth` with an HTTP `GET` request to `{server_url}/api/v1/weights/best`, saving the response body to a temporary file before loading.
    *   **Evaluator:**
        *   Replace `os.path.getmtime()` check with periodic HTTP `GET` or `HEAD` requests to `{server_url}/api/v1/weights/latest`.
        *   Replace `torch.save(latest_weights, best_weights_path)` with an HTTP `POST` request to `{server_url}/api/v1/weights/promote_best`.

### 5.3. `run_distributed.sh`

*   No changes are strictly required. The default `--server-url` of `http://127.0.0.1:8080` will ensure all locally-spawned processes connect to the local server, preserving the script's functionality.

## 6. Dependencies

*   The `aiohttp` library will be added to the project's dependencies (e.g., `requirements.txt`).

## 7. Risks and Mitigation

*   **Network Latency:** Transferring large experience objects or model weights over the network will introduce latency.
    *   **Mitigation:** This is an inherent trade-off. Payloads should be kept as efficient as possible. For initial rollout, LAN performance is expected to be acceptable.
*   **Error Handling:** Network errors (timeouts, disconnects) must be handled gracefully.
    *   **Mitigation:** Workers should implement retry logic with backoff for API calls. The server should handle client disconnects without crashing.
*   **Security:** The API is unauthenticated.
    *   **Mitigation:** This is acceptable for the stated use case (trusted LAN environment). The server should be bound to the local network interface, not exposed to the public internet. This will be the default.
