# Cross-container CRIU checkpoint restore

## Goal

A CRIU checkpoint created in container A restores successfully in container B
on the same machine. The eviction strategy is `discard + checkpoint` — weights
are NOT baked into the checkpoint image. On restore, vLLM reloads weights from
the HuggingFace cache on disk.

Shared state between containers is limited to:

- `/tmp/llmux-checkpoints` — CRIU checkpoint images
- `~/.cache/huggingface` — model weights

Everything else (vLLM code, compiled caches, llmux binary) comes from the
container image.

## Success test

```bash
# Container A: cold-start, inference, then checkpoint via control API
docker run -d --name llmux --gpus all --privileged --pid=host --ipc=host --network=host \
  -v ./config.json:/etc/llmux/config.json:ro \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/doublewordai/llmux:latest

curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}'
# → valid response (cold start)

curl -X POST http://localhost:3000/control/sleep \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b"}'
# → checkpoint created

# Stop container A, start container B
docker stop llmux && docker rm llmux
docker run -d --name llmux --gpus all --privileged --pid=host --ipc=host --network=host \
  -v ./config.json:/etc/llmux/config.json:ro \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/doublewordai/llmux:latest

# Container B: restore from checkpoint, then inference
curl -X POST http://localhost:3000/control/wake \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b"}'
# → restores from checkpoint (NOT cold start)

curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}'
# → valid response
```

Config must include `checkpoint_path` so llmux knows where to find existing
checkpoints on startup:

```json
{
  "models": {
    "qwen3-0.6b": {
      "model_path": "Qwen/Qwen3-0.6B",
      "port": 8001,
      "eviction": { "weights": "discard", "process": "checkpoint" },
      "checkpoint_path": "/tmp/llmux-checkpoints/qwen3-0.6b/images"
    }
  },
  "checkpoint": {
    "criu_path": "/usr/local/bin/criu",
    "cuda_plugin_dir": "/usr/lib/criu/",
    "images_dir": "/tmp/llmux-checkpoints",
    "cuda_checkpoint_path": "/usr/local/bin/cuda-checkpoint",
    "keep_images": true
  },
  "port": 3000
}
```

## What works today

- In-container checkpoint/restore works for all models (~11-16s restore vs
  ~60-120s cold start)
- O_APPEND fix (v0.7.13) means log file size changes between containers no
  longer break CRIU restore
- CRIU restore itself succeeds across containers
- Health check passes after cross-container restore
- `/wake_up` endpoint works after cross-container restore

## The one blocker: ZMQ IPC

vLLM v0.15.1 uses ZMQ over Unix domain sockets for APIServer ↔ EngineCore
communication. The socket files live in `/tmp` with UUID-based names.

After cross-container CRIU restore:

1. CRIU restores the Unix domain socket file descriptors as connected pairs
2. ZMQ's internal state machine doesn't match the restored kernel socket state
3. The first ZMQ operation (triggered by `collective_rpc` / `reload_weights`)
   blocks the asyncio event loop forever
4. Once the event loop is blocked, ALL HTTP endpoints stop responding

Attempted fixes that didn't work:

- Persisting `/tmp` as a Docker named volume (socket files persist but ZMQ
  state still broken)
- Cleaning stale socket files before restore
- Sending SIGWINCH to kick the event loop
- Mounting `/root` as a named volume (persists compiled caches but creates
  upgrade/double-storage problems)

## Dev loop

To iterate quickly without pushing Docker images:

1. Install Rust on the VM (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
2. Clone repo on the VM
3. Build on bare metal: `cargo build --release`
4. Mount binary into the Docker container:
   ```bash
   docker run -d --name llmux --gpus all --privileged --pid=host --ipc=host --network=host \
     -v $(pwd)/target/release/llmux:/usr/local/bin/llmux:ro \
     -v ./config.json:/etc/llmux/config.json:ro \
     -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     ghcr.io/doublewordai/llmux:latest
   ```
5. Use a single model (qwen3-0.6b) for fast iteration
6. Use the control API (`/control/sleep`, `/control/wake`) to checkpoint/restore
   — no need to switch models

## Constraints

1. **No vLLM patches** — vLLM is a black box, we don't touch its code
2. **No changes to what we checkpoint** — `discard + checkpoint` stays,
   weights stay outside the image
3. **The fix lives in CRIU configuration and llmux's orchestration around
   it** — we can adjust CRIU flags, and if CRIU needs certain files to exist
   in the new container's filesystem for restore to succeed, we can save them
   into the persisted checkpoint directory during dump and copy them back into
   place before calling CRIU restore

## Investigation plan

Figure out exactly what CRIU needs in the new container's filesystem for the
ZMQ Unix domain sockets to restore correctly. Possible factors:

- CRIU flags for external Unix sockets (`--ext-unix-sk`, etc.)
- Socket files in `/tmp` that need to exist before restore
- Other filesystem state the sockets depend on

The approach: inspect what CRIU is doing with the sockets during dump,
identify what's missing or mismatched on restore, and make llmux handle it —
copy the right files out on dump, put them back on restore, pass the right
flags to CRIU.
