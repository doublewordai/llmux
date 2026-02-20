# Getting started

This guide walks through deploying llmux with CRIU checkpointing on a GPU
server. By the end you'll have multiple models cycling on a single GPU with
fast checkpoint/restore switching.

## Prerequisites

- NVIDIA GPU with driver 580+ (CUDA 12.9)
- Docker with `nvidia-container-toolkit`
- HuggingFace model cache populated (or internet access to download on first use)

## 1. Write a config

Create `config.json` with your models. Each model needs a unique port and an
eviction policy. For checkpointing, use `discard + checkpoint`:

```json
{
  "models": {
    "qwen3-0.6b": {
      "model_path": "Qwen/Qwen3-0.6B",
      "port": 8001,
      "eviction": { "weights": "discard", "process": "checkpoint" }
    },
    "qwen3-1.7b": {
      "model_path": "Qwen/Qwen3-1.7B",
      "port": 8002,
      "eviction": { "weights": "discard", "process": "checkpoint" }
    },
    "qwen3-4b": {
      "model_path": "Qwen/Qwen3-4B",
      "port": 8003,
      "eviction": { "weights": "discard", "process": "checkpoint" }
    }
  },
  "policy": {
    "request_timeout_secs": 300
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

### Eviction policy cheat sheet

The eviction policy has two axes -- what happens to weights and what happens to
the OS process:

| Weights | Process | GPU freed | CPU freed | Wake time | CRIU image | Use case |
|---------|---------|-----------|-----------|-----------|------------|----------|
| `offload` | `keep_running` | ~97% | No | ~1s | N/A | Fast switching, have spare CPU RAM |
| `discard` | `keep_running` | ~97% | Yes | ~15s | N/A | Fast switching, no spare CPU RAM |
| `retain` | `cuda_suspend` | 100% | No | ~2s | N/A | Full GPU freed, state in CPU RAM |
| `discard` | `checkpoint` | 100% | 100% | ~20s | Small (~5 GB) | Full memory freed, weights reload from disk |
| `offload` | `checkpoint` | 100% | 100% | ~5s | Large (~40 GB) | Full memory freed, weights in checkpoint |

`discard + checkpoint` is the most memory-efficient: zero GPU and CPU overhead
while the model is evicted. The trade-off is wake time (~20s to reload weights
from the HF cache). If you have the disk space, `offload + checkpoint` gives
faster wakes because the weights are baked into the CRIU image.

## 2. Start the container

```bash
docker run -d --name llmux --gpus all \
  --privileged --pid=host --ipc=host --network=host \
  -v ./config.json:/etc/llmux/config.json:ro \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/doublewordai/llmux:latest
```

The Docker flags:

- `--privileged --pid=host` -- CRIU and cuda-checkpoint need ptrace access
  to vLLM worker processes
- `--ipc=host` -- NCCL shared memory (required for TP>1 and CRIU)
- `--network=host` -- llmux and vLLM talk over localhost ports
- `-v /tmp/llmux-checkpoints:...` -- checkpoint images can be tens of GB;
  this mount lets them persist across container restarts

**Do NOT use `--init`** with CRIU checkpointing. Docker's init process (tini)
redirects stdin to the host's `/dev/null`, which CRIU can't resolve inside
the container.

## 3. Send your first request

llmux starts with no models loaded. The first request triggers a cold start:

```bash
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10
  }'
```

This cold-starts the 0.6B model (takes ~30s). Subsequent requests to the same
model are instant.

## 4. Switch models

Send a request to a different model. llmux will:

1. Drain in-flight requests to the active model
2. Apply the weight strategy (`discard` -- drop weights from GPU)
3. Apply the process strategy (`checkpoint` -- CRIU dump to disk, kill process)
4. Cold-start the new model (or restore from checkpoint if one exists)
5. Proxy the request

```bash
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10
  }'
```

The first switch is slow because no checkpoint exists yet. llmux creates a
checkpoint during eviction, so the *second* time you switch to that model it
restores from checkpoint instead of cold-starting.

## 5. Verify checkpoints

After cycling through all models at least once (to create checkpoints), switch
back to a previously checkpointed model. You should see CRIU restore in the
logs instead of a cold start:

```bash
docker logs llmux --tail 20
```

Look for:

```
Restoring qwen3-0.6b from checkpoint
criu restore: success
```

vs. a cold start which shows:

```
Starting qwen3-0.6b cold (no checkpoint available)
Spawning vLLM process
```

Restore is typically 3-5x faster than a cold start.

## 6. Check checkpoint state

Checkpoint images are stored under the `images_dir` you configured:

```bash
ls -la /tmp/llmux-checkpoints/
```

Each model gets a directory with CRIU images and log files:

```
/tmp/llmux-checkpoints/
  qwen3-0.6b/
    stdout.log
    stderr.log
    images/       # CRIU checkpoint images
  qwen3-1.7b/
    ...
```

With `keep_images: true`, checkpoints persist across container restarts. This
means you can stop the container, restart it, and the first request for a
previously checkpointed model will restore from the checkpoint instead of
cold-starting.

## 7. Pre-create checkpoints (optional)

Instead of waiting for the first eviction cycle to create checkpoints, you can
pre-create them:

```bash
docker exec llmux llmux --config /etc/llmux/config.json \
  --checkpoint qwen3-0.6b

docker exec llmux llmux --config /etc/llmux/config.json \
  --checkpoint qwen3-1.7b
```

Or from the host using the Docker image directly:

```bash
docker run --rm --gpus all --privileged --pid=host --ipc=host --network=host \
  -v ./config.json:/etc/llmux/config.json:ro \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/doublewordai/llmux:latest \
  --config /etc/llmux/config.json --checkpoint qwen3-0.6b
```

This starts the model, runs a warmup inference, then creates the CRIU
checkpoint. Skip warmup with `--no-warmup` if the model doesn't need it.

## 8. Lazy restore from pre-built checkpoints

If you pre-created checkpoints (step 7), add `checkpoint_path` to your model
config to restore from them on first request:

```json
{
  "models": {
    "qwen3-0.6b": {
      "model_path": "Qwen/Qwen3-0.6B",
      "port": 8001,
      "eviction": { "weights": "discard", "process": "checkpoint" },
      "checkpoint_path": "/tmp/llmux-checkpoints/qwen3-0.6b/images"
    }
  }
}
```

Now when the daemon starts and receives a request for `qwen3-0.6b`, it
restores from the checkpoint immediately instead of cold-starting.

## Monitoring

### Logs

```bash
docker logs -f llmux
```

For verbose output including vLLM's own logs:

```bash
docker run ... -e RUST_LOG=info,vllm=debug ghcr.io/doublewordai/llmux:latest
```

### Prometheus metrics

llmux exposes metrics on port 9090 by default:

```bash
curl -s http://localhost:9090/metrics
```

Key metrics:

- `llmux_switches_total` -- total model switches
- `llmux_switch_total_seconds` -- switch duration histogram
- `llmux_request_queue_wait_seconds` -- how long requests wait in queue

Disable with `"metrics_port": 0` in config.

## Troubleshooting

### "Can't lookup mount=N for fd=0 path=/dev/null"

You're using `--init` with CRIU. Remove `--init` from your Docker run command.

### "File has bad size N (expect M)"

Checkpoint images were created in a previous container but the log files were
overwritten by a cold start in the current container. This is fixed in v0.7.13+
which opens log files with `O_APPEND` so CRIU skips size validation. Upgrade
your llmux image, delete old checkpoints, and re-create them.

### Checkpoint restore hangs

Check that `--pid=host` is set. Without it, CRIU can't find the process tree
after restore.

### "Could not find restore thread for process ID"

This usually means the CRIU CUDA plugin couldn't handle a process in the tree.
Make sure you're using the Docker image (which includes the correct CRIU and
CUDA plugin versions). On bare metal, ensure CRIU 4.x with `libcuda_plugin.so`
and nvidia-driver-580+.

### vLLM doesn't free GPU memory on sleep

This is a known vLLM v0.14+ regression. The Docker image includes a patch.
For bare-metal installs, apply `patches/fix-sleep-mode-v0.15.1.patch` or use
`cuda_suspend`/`checkpoint` instead of `offload`/`discard`.
