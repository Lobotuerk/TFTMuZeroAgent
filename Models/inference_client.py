"""
Remote GPU Inference Client for TFT MuZero.

Provides a synchronous client that proxies inference requests to a
centralized UDS inference server. The RemoteMuZeroNetwork class
implements the same interface as the local MuZeroNetwork.
"""

import os
import pickle
import socket
import struct
import torch
import numpy as np
from typing import Dict, Any


class _TensorEncoder(pickle.Pickler):
    """Custom pickle pickler that serializes tensors as (dtype, shape, bytes) tuples."""

    def save_tensor(self, obj):
        state = obj.state()
        self.save_reduce(_reconstruct_tensor, (state["dtype"], state["shape"], state["data"]), obj=obj)

    dispatch = pickle.Pickler.dispatch.copy()
    dispatch[torch.Tensor] = save_tensor


def _reconstruct_tensor(dtype, shape, data):
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def _dumps_request(data: Any) -> bytes:
    """Serialize a request payload using pickle with tensor support."""
    buf = bytearray()
    _TensorEncoder(buf).dump(data)
    return buf


def _loads_request(data: bytes) -> Any:
    """Deserialize a request payload."""
    return pickle.loads(data)


def _pack_frame(payload: bytes) -> bytes:
    """Pack a length-prefixed frame: 4-byte big-endian length + payload."""
    return struct.pack("!I", len(payload)) + payload


def _unpack_frame(data: bytes) -> bytes:
    """Unpack a length-prefixed frame, returning the payload."""
    if len(data) < 4:
        raise ValueError("Incomplete frame header")
    length = struct.unpack("!I", data[:4])[0]
    if len(data) < 4 + length:
        raise ValueError(f"Incomplete frame: expected {length} bytes, got {len(data) - 4}")
    return data[4:4 + length]


def _tensor_to_bytes(t):
    """Convert a tensor to (dtype, shape, bytes) for serialization."""
    if isinstance(t, dict):
        return {k: _tensor_to_bytes(v) for k, v in t.items()}
    if isinstance(t, torch.Tensor):
        return {
            "dtype": str(t.dtype),
            "shape": t.shape,
            "data": t.detach().cpu().numpy().tobytes(),
        }
    if isinstance(t, np.ndarray):
        return {
            "dtype": str(t.dtype),
            "shape": t.shape,
            "data": t.tobytes(),
        }
    return t


def _bytes_to_tensor(obj):
    """Convert a (dtype, shape, bytes) dict back to a torch.Tensor."""
    if isinstance(obj, dict) and "dtype" in obj and "shape" in obj and "data" in obj:
        dtype = getattr(torch, obj["dtype"])
        return torch.frombuffer(bytearray(obj["data"]), dtype=dtype).reshape(obj["shape"])
    if isinstance(obj, dict):
        return {k: _bytes_to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bytes_to_tensor(v) for v in obj]
    return obj


class RemoteMuZeroNetwork:
    """Synchronous proxy for a remote GPU inference server.

    Implements the same interface as MuZeroNetwork (initial_inference,
    recurrent_inference) by forwarding requests over a Unix Domain
    Socket to a centralized inference server.

    Args:
        socket_path: Path to the UDS socket on the server.
        model_version: Which model to use ("latest" or "best").
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, socket_path: str, model_version: str = "latest", timeout: float = 30.0):
        self.socket_path = socket_path
        self.model_version = model_version
        self.timeout = timeout

    def initial_inference(self, observation):
        """Run initial inference via the remote server.

        Args:
            observation: Input observation (numpy array or torch tensor).

        Returns:
            Dict with keys: value, reward, policy_logits, hidden_state.
        """
        if isinstance(observation, torch.Tensor):
            obs_serializable = _tensor_to_bytes(observation)
        elif isinstance(observation, np.ndarray):
            obs_serializable = _tensor_to_bytes(observation)
        else:
            obs_serializable = _tensor_to_bytes(np.array(observation, dtype=np.float32))

        request = {
            "model_version": self.model_version,
            "method": "initial_inference",
            "args": {"observation": obs_serializable},
        }

        payload = _dumps_request(request)
        frame = _pack_frame(payload)

        response_data = self._send_request(frame)
        response = _loads_request(response_data)

        if "error" in response:
            raise ConnectionError(f"Remote inference error: {response['error']}")

        return _bytes_to_tensor(response)

    def recurrent_inference(self, hidden_state, action):
        """Run recurrent inference via the remote server.

        Args:
            hidden_state: Current hidden state (torch tensor).
            action: Action to apply (numpy array or torch tensor).

        Returns:
            Dict with keys: value, reward, policy_logits, hidden_state.
        """
        hs_serializable = _tensor_to_bytes(hidden_state)
        if isinstance(action, torch.Tensor):
            action_serializable = _tensor_to_bytes(action)
        elif isinstance(action, np.ndarray):
            action_serializable = _tensor_to_bytes(action)
        else:
            action_serializable = _tensor_to_bytes(np.array(action, dtype=np.float32))

        request = {
            "model_version": self.model_version,
            "method": "recurrent_inference",
            "args": {
                "hidden_state": hs_serializable,
                "action": action_serializable,
            },
        }

        payload = _dumps_request(request)
        frame = _pack_frame(payload)

        response_data = self._send_request(frame)
        response = _loads_request(response_data)

        if "error" in response:
            raise ConnectionError(f"Remote inference error: {response['error']}")

        return _bytes_to_tensor(response)

    def _send_request(self, frame: bytes) -> bytes:
        """Send a request frame and receive the response."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            sock.sendall(frame)

            # Read response: 4-byte header + payload
            header = self._recv_exact(sock, 4)
            length = struct.unpack("!I", header)[0]

            if length > 100 * 1024 * 1024:
                raise ConnectionError(f"Response too large: {length} bytes")

            payload = self._recv_exact(sock, length)
            return payload

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from a socket."""
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data.extend(chunk)
        return bytes(data)
