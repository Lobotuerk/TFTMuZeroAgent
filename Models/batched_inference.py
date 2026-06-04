import threading
import queue
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional


class InferenceRequest:
    __slots__ = ['hidden_state', 'action', 'event', 'result']

    def __init__(self, hidden_state: torch.Tensor, action: np.ndarray):
        self.hidden_state = hidden_state
        self.action = action
        self.event = threading.Event()
        self.result: Dict[str, Any] = {}


class BlockingBatchInferenceQueue:
    def __init__(self, network, batch_size: int = 64, timeout_seconds: float = 0.005):
        self.network = network
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.worker_thread.start()

    def predict(self, hidden_state: torch.Tensor, action: np.ndarray) -> Dict[str, Any]:
        req = InferenceRequest(hidden_state, action)
        self._queue.put(req)
        req.event.wait()
        return req.result

    def flush(self):
        remaining = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            self._run_batch(remaining)

    def shutdown(self):
        self.flush()
        self._stop_event.set()

    def _batch_processor(self):
        device = next(self.network.parameters()).device
        while not self._stop_event.is_set():
            requests: List[InferenceRequest] = []
            try:
                first_req = self._queue.get(timeout=0.1)
                requests.append(first_req)
            except queue.Empty:
                continue
            start_time = time.time()
            while len(requests) < self.batch_size:
                elapsed = time.time() - start_time
                remaining = self.timeout_seconds - elapsed
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    requests.append(req)
                except queue.Empty:
                    break
            self._run_batch(requests, device)

    def _run_batch(self, requests: List[InferenceRequest], device: Optional[torch.device] = None):
        if not requests:
            return
        if device is None:
            device = next(self.network.parameters()).device
        h_list = []
        for req in requests:
            hs = req.hidden_state
            if not isinstance(hs, torch.Tensor):
                hs = torch.tensor(hs, dtype=torch.float32)
            h_list.append(hs)
        h_states = torch.stack(h_list).to(device)
        actions_np = np.array([req.action for req in requests])
        actions = torch.tensor(actions_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            network_output = self.network.recurrent_inference(h_states, actions)
        for i, req in enumerate(requests):
            req.result = {
                "hidden_state": network_output["hidden_state"][i].detach().cpu(),
                "policy_logits": network_output["policy_logits"][i].detach().cpu().numpy(),
                "value": network_output["value"][i].detach().cpu().numpy(),
            }
            req.event.set()
