import threading
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
    """
    Dynamic leader-follower batch inference queue.
    Eliminates fixed wait times by tracking active searches and flushing
    immediately when all active searches are blocked waiting for inference.
    """
    def __init__(self, network, batch_size: int = 64, timeout_seconds: float = 0.05):
        self.network = network
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        
        self._lock = threading.Lock()
        self.active_count = 0
        self.current_batch: List[InferenceRequest] = []
        
    def register(self):
        """Register an active MCTS search thread."""
        with self._lock:
            self.active_count += 1
            
    def deregister(self):
        """Deregister an active MCTS search thread and flush if necessary."""
        batch_to_process = None
        with self._lock:
            self.active_count -= 1
            # If the barrier is now met because a thread left, trigger the flush
            if self.active_count > 0 and len(self.current_batch) >= self.active_count:
                batch_to_process = self.current_batch
                self.current_batch = []
                
        if batch_to_process:
            self._run_batch(batch_to_process)

    @property
    def _queue(self):
        class DummyQueue:
            def empty(self):
                return True
        return DummyQueue()

    def predict(self, hidden_state: torch.Tensor, action: np.ndarray) -> Dict[str, Any]:
        req = InferenceRequest(hidden_state, action)
        
        with self._lock:
            self.current_batch.append(req)
            
            # If we are the last active thread to arrive, or we hit the hard batch limit
            if len(self.current_batch) >= self.active_count or len(self.current_batch) >= self.batch_size:
                batch_to_process = self.current_batch
                self.current_batch = []
                is_leader = True
            else:
                is_leader = False
                
        if is_leader:
            self._run_batch(batch_to_process)
        else:
            # We are a follower. Wait for the leader to process, or timeout.
            success = req.event.wait(timeout=self.timeout_seconds)
            if not success:
                # Timeout! We become the leader for whatever is in the batch right now.
                with self._lock:
                    if req in self.current_batch:
                        batch_to_process = self.current_batch
                        self.current_batch = []
                        is_leader = True
                    else:
                        is_leader = False
                
                if is_leader:
                    self._run_batch(batch_to_process)
                else:
                    # Someone else took the batch and is processing it.
                    # We must wait indefinitely for them to finish.
                    req.event.wait()
                    
        return req.result

    def flush(self):
        batch_to_process = None
        with self._lock:
            if self.current_batch:
                batch_to_process = self.current_batch
                self.current_batch = []
        if batch_to_process:
            self._run_batch(batch_to_process)

    def shutdown(self):
        self.flush()

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
