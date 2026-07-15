"""
Remote GPU Inference Server for TFT MuZero.

Provides a UDS-based inference server that centralizes PyTorch model
inference in the training server process. Workers connect via Unix
Domain Sockets and forward batched inference requests.
"""

import asyncio
import pickle
import struct
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _tensor_to_serializable(t):
    """Convert a tensor and its surrounding dict to pickle-friendly form."""
    if isinstance(t, dict):
        return {k: _tensor_to_serializable(v) for k, v in t.items()}
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


def _serializable_to_tensor(obj):
    """Convert a pickled tensor dict back to a torch.Tensor."""
    if isinstance(obj, dict) and "dtype" in obj and "shape" in obj and "data" in obj:
        dtype = getattr(torch, obj["dtype"])
        return torch.frombuffer(bytearray(obj["data"]), dtype=dtype).reshape(obj["shape"])
    if isinstance(obj, dict):
        return {k: _serializable_to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable_to_tensor(v) for v in obj]
    return obj


def _dumps_request(data: Any) -> bytes:
    """Serialize a request payload using pickle."""
    return pickle.dumps(data)


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


def _execute_initial_inference_sync(model, observation):
    """Synchronous initial inference (for asyncio.to_thread)."""
    if model._cuda_stream:
        with torch.cuda.stream(model._cuda_stream):
            with torch.no_grad():
                result = model.initial_inference(observation)
        torch.cuda.current_stream().synchronize()
    else:
        with torch.no_grad():
            result = model.initial_inference(observation)
    return result


def _execute_recurrent_inference_sync(model, hidden_state, action):
    """Synchronous recurrent inference (for asyncio.to_thread)."""
    if model._cuda_stream:
        with torch.cuda.stream(model._cuda_stream):
            with torch.no_grad():
                result = model.recurrent_inference(hidden_state, action)
        torch.cuda.current_stream().synchronize()
    else:
        with torch.no_grad():
            result = model.recurrent_inference(hidden_state, action)
    return result


class UDSInferenceServer:
    """Unix Domain Socket inference server for centralized GPU inference.

    Listens on a Unix socket and accepts inference requests from worker
    processes. Routes requests to the provided model instances based on
    model_version ("latest" or "best").

    Args:
        socket_path: Path for the Unix domain socket.
        models: Dict mapping version name ("latest", "best") to MuZeroNetwork instances.
    """

    def __init__(self, socket_path: str, models: Dict[str, torch.nn.Module]):
        self.socket_path = socket_path
        self.models = models
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False

        # Attach CUDA stream to each model for inference concurrency
        for model in models.values():
            if torch.cuda.is_available():
                model._cuda_stream = torch.cuda.Stream()
            else:
                model._cuda_stream = None

    async def start(self):
        """Start the UDS server, blocking until stopped."""
        import os
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        loop = asyncio.get_event_loop()
        self._server = await loop.create_unix_listener(
            self.socket_path,
            self._handle_connection,
        )
        self._running = True
        logger.info(f"UDS Inference server listening on {self.socket_path}")

        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Stop the UDS server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        import os
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        logger.info(f"UDS Inference server stopped")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection."""
        try:
            while self._running:
                header = await self._read_exact(reader, 4)
                if not header:
                    break

                length = struct.unpack("!I", header)[0]
                if length > 100 * 1024 * 1024:
                    writer.close()
                    await writer.wait_closed()
                    raise ValueError(f"Payload too large: {length} bytes")

                payload = await self._read_exact(reader, length)
                if not payload:
                    break

                try:
                    request = _loads_request(payload)
                except Exception as e:
                    logger.error(f"Failed to deserialize request: {e}")
                    await self._send_error(writer, f"Invalid request: {e}")
                    continue

                model_version = request.get("model_version", "latest")
                method = request.get("method", "")
                args = request.get("args", {})

                model = self.models.get(model_version)
                if model is None:
                    await self._send_error(writer, f"Unknown model version: {model_version}")
                    continue

                try:
                    result = await self._run_inference(model, method, args)
                    response = _tensor_to_serializable(result)
                    serialized = _dumps_request(response)
                    frame = _pack_frame(serialized)
                    writer.write(frame)
                    await writer.drain()
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    await self._send_error(writer, str(e))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _run_inference(self, model, method: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the model, dispatching to thread to avoid blocking event loop."""
        if method == "initial_inference":
            observation = _serializable_to_tensor(args.get("observation"))
            result = await asyncio.to_thread(self._execute_initial_inference, model, observation)
            return result
        elif method == "recurrent_inference":
            hidden_state = _serializable_to_tensor(args.get("hidden_state"))
            action = _serializable_to_tensor(args.get("action"))
            result = await asyncio.to_thread(self._execute_recurrent_inference, model, hidden_state, action)
            return result
        else:
            raise ValueError(f"Unknown method: {method}")

    def _execute_initial_inference(self, model, observation):
        """Run initial_inference (called via asyncio.to_thread to avoid blocking event loop)."""
        return _execute_initial_inference_sync(model, observation)

    def _execute_recurrent_inference(self, model, hidden_state, action):
        """Run recurrent_inference (called via asyncio.to_thread to avoid blocking event loop)."""
        return _execute_recurrent_inference_sync(model, hidden_state, action)

    @staticmethod
    async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
        """Read exactly n bytes from the reader."""
        data = bytearray()
        while len(data) < n:
            chunk = await reader.read(n - len(data))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    async def _send_error(writer: asyncio.StreamWriter, error_msg: str):
        """Send an error response."""
        error_payload = _dumps_request({"error": error_msg})
        frame = _pack_frame(error_payload)
        writer.write(frame)
        await writer.drain()
