# Technical Specification: Centralize GPU Inference (TFT-225)

## 1. Overview
The current architecture creates redundant CPU copies of the MuZero neural network in all 6 worker processes (1 evaluator + 5 collectors), leading to high RAM usage and missing out on GPU acceleration for inference. 
This specification outlines the centralization of PyTorch inference into the existing GPU Training Server process. Workers will connect to the server via Unix Domain Sockets (UDS) and forward their batched inference requests, enabling zero-copy network hops and keeping all models in GPU VRAM.

## 2. Architecture Design

### Server: `Models/inference_server.py`
We will introduce a `UDSInferenceServer` running asynchronously alongside the HTTP server in `main.py` (when in `train_server_mode`).
- **Functionality**: Listens on a Unix Domain Socket (e.g., `/tmp/tft_muzero_inference.sock`). Accepts connections from local worker processes.
- **Worker Isolation**: Cross-worker batching is explicitly excluded. The server will process each worker's incoming batch independently.
- **Concurrency**: The server will maintain separate CUDA streams to process inference requests concurrently alongside training. To support higher throughput, the server can round-robin requests across multiple duplicated `MuZeroNetwork` instances if VRAM allows, but initially, it will route requests to the canonical "latest" and "best" models maintained by the orchestrator.
- **Protocol**: Length-prefixed binary framing. Payloads will be serialized using `pickle` (or optimized tensor byte buffers). Requests will specify:
  - `model_version`: `"latest"` or `"best"`
  - `method`: `"initial_inference"` or `"recurrent_inference"`
  - `args`: The batched tensors.

### Client: `Models/inference_client.py`
We will create a `RemoteMuZeroNetwork` class that adheres to the exact same interface as the local `MuZeroNetwork`.
- **Functionality**: Acts as a proxy. When `initial_inference(obs)` or `recurrent_inference(hidden, action)` is called, it serializes the inputs, sends them over UDS to the server, blocks until the response is received, and returns the deserialized tensors.
- **Integration**: The client operates synchronously, which fits perfectly into the existing `ThreadPoolExecutor` workers used by `BatchInferenceServer` and `BlockingBatchInferenceQueue`. 

### `MuZeroAgent` Modifications
- Add a configuration flag `use_remote_inference` (or detect if UDS path is provided).
- If enabled, bypass local PyTorch model initialization and use `RemoteMuZeroNetwork`.
- Avoid calling `model.to('cuda')` or `model.to('cpu')` on the remote proxy.

### `main.py` Modifications
- **Train Server**: Instantiate and start `UDSInferenceServer`. Pass references to `orch.current_model.model` (latest) and `orch.best_model.model` (best).
- **Workers**: Modify `worker_mode` to skip HTTP polling for weights. Instantiate `MuZeroAgent` with `use_remote_inference=True`. The evaluator will instantiate two agents using `model_version="latest"` and `"best"`.

## 3. Design Philosophy & Guidelines
- **Deep Modules**: The UDS IPC details (sockets, serialization, length prefixing) will be completely hidden behind `RemoteMuZeroNetwork`. The rest of the codebase (MCTS, Agent Manager) will remain unaware that the network is remote.
- **Fail-Hard Exception Paths**: If the UDS server goes down, the client socket will disconnect or timeout. The `RemoteMuZeroNetwork` will throw an unhandled `ConnectionError`, intentionally crashing the worker process, satisfying the "fail-hard" fallback requirement.
- **Simplicity**: By maintaining the existing per-worker batching (`BatchInferenceServer`), we avoid complex cross-worker queueing logic in the UDS server. The server simply receives a tensor of shape `[N, ...]`, runs it, and returns a tensor of shape `[N, ...]`.

## 4. Execution Plan
1. Create `Models/inference_client.py` and `Models/inference_server.py`.
2. Update `Models/MuZero_torch_agent.py` to support `RemoteMuZeroNetwork`.
3. Modify `main.py` `train_server_mode` to start the server and bind models.
4. Modify `main.py` `worker_mode` to use remote inference and remove HTTP weight polling logic.
