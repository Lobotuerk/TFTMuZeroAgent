# Technical Specification for TFT-211: Evaluate stuck at step 0

## 1. Overview
In the distributed architecture, the evaluator worker runs an infinite loop calling `orch.evaluate()`. Because the evaluator does not execute `_train_step()`, its internal `self.training_step` remains `0`. Consequently, all TensorBoard metrics log at step 0 and overwrite one another. 

To resolve this, the training server will broadcast the current `training_step` alongside the model weights via a JSON envelope. The evaluator worker will then parse the step and explicitly pass it into `evaluate()`, ensuring logs reflect the actual progression of the training server.

## 2. Architecture & Design Philosophy
- **Modular Depth**: The `evaluate()` method is simplified to accept an explicit `step` rather than relying on its internal (and in distributed mode, incorrect) state.
- **Data Extensibility**: As agreed upon, we introduce a JSON envelope for the `/api/v1/weights/{name}` endpoint. This replaces raw byte streaming and allows us to easily add more metadata in the future.
- **Clean Exception Paths**: Graceful fallbacks (such as falling back to `self.training_step` if `step` is not provided) keep backward-compatibility intact for any remaining non-distributed calls.

## 3. Implementation Steps

### A. Server Side (`main.py` -> `train_server_mode`)
Modify the `handle_weights(request)` endpoint to return a JSON envelope.
1. Read the raw weights from the `.pth` file.
2. Encode the weights in Base64: `encoded_weights = base64.b64encode(data).decode('utf-8')`.
3. Construct the JSON dictionary: 
   ```json
   {
       "step": orch.training_step,
       "weights": encoded_weights
   }
   ```
4. Return it via `web.json_response(body_dict, headers={"Last-Modified": last_modified})`.

### B. Client Side (`main.py` -> `worker_mode`)
Modify both the `evaluator` and `collector` branches to parse the new JSON envelope.
1. **Evaluator Worker (`/api/v1/weights/latest` & `/best`)**:
   - Parse the response: `resp_json = await resp.json()`.
   - Extract the training step: `step = resp_json.get("step", 0)`.
   - Decode the weights: `weights_bytes = base64.b64decode(resp_json["weights"])`.
   - Feed `weights_bytes` into `torch.load(io.BytesIO(weights_bytes), map_location="cpu")`.
   - Update the call to evaluate: `results = await orch.evaluate(step=step)`.
2. **Collector Worker (`/api/v1/weights/best`)**:
   - Apply the same JSON parsing and Base64 decoding for fetching the weights, as the endpoint format is updated across the board.

### C. Logic Side (`training_orchestrator.py` -> `evaluate`)
Allow `evaluate` to accept and use the explicit step parameter.
1. Update signature: `async def evaluate(self, step: Optional[int] = None) -> Dict[str, float]:`.
2. Define the execution step: `current_step = step if step is not None else self.training_step`.
3. Update print statements: `print(f"\nEVALUATE at step {current_step}")`.
4. Update TensorBoard logs to use `current_step`:
   - `self.summary_writer.add_scalar("evaluation/current_model", current_mean, current_step)`
   - `self.summary_writer.add_scalar("evaluation/best_model", best_mean, current_step)`

## 4. Edge Cases and Resilience
- **Model Size Overhead**: Base64 encoding inflates payload size by ~33%. Assuming a MuZero network size in the low MBs, this is acceptable over a local HTTP connection.
- **Local Fallback**: `evaluate(step=None)` ensures that `evaluation_mode` or the deprecated non-distributed `training_mode` won't crash when invoking `evaluate()`.