# llmux

[![Crates.io](https://img.shields.io/crates/v/llmux)](https://crates.io/crates/llmux)
[![GitHub](https://img.shields.io/badge/GitHub-doublewordai%2Fllmux-blue)](https://github.com/doublewordai/llmux)

LLM multiplexer for vLLM. Host multiple models on a single GPU, switching
between them on demand using vLLM's sleep/wake API.

When a request arrives for a model that isn't currently loaded, llmux puts the
active model to sleep (freeing GPU memory) and wakes the requested model. The
OpenAI-compatible API stays up throughout - clients just change the `model`
field.

## How it works

```
                    Client requests
                         |
                    +---------+
                    |  llmux  |   port 3000 (OpenAI-compatible)
                    +---------+
                    /         \
            [vLLM:8001]    [vLLM:8002]
             (active)       (sleeping)
```

llmux spawns vLLM processes lazily on first request and manages their
lifecycle. Only one model is active at a time - the rest are sleeping
(weights offloaded to CPU or discarded) or stopped entirely.

### Sleep levels

| Level | Sleep | Wake | GPU freed | CPU RAM | State preserved | Use case |
|-------|-------|------|-----------|---------|-----------------|----------|
| **L1** | Slow (offload to CPU) | Fast (~1s) | Most | High (holds weights) | Partial | Model you expect to return to soon |
| **L2** | Fast (~1s) | Slow (reload from disk) | Most | None | No (KV cache, CUDA graphs lost) | Model you may not need for a while |
| **L3** (CUDA suspend) | Fast (~3s) | Fast (~3s) | All (100%) | High (holds VRAM) | Full | Like L1, but frees 100% GPU; works with vLLM v0.14+. **TP=1 only.** |
| **L4** (CRIU) | ~27s (checkpoint to disk) | ~15s (restore) | All (100%) | None | Full (KV cache, CUDA graphs, allocator) | Many models; works with vLLM v0.14+. **TP=1 only.** |
| **L5** | Kill process | Cold start | All | None | No | Fallback / cleanup |

If L1-L4 sleep fails, llmux automatically escalates to L5 (kill) to guarantee
GPU memory is freed.

#### CUDA suspend (level 3)

Uses `cuda-checkpoint --toggle` to suspend CUDA state and copy VRAM to host
RAM. The process stays alive — no serialization, no CRIU. Wake is just another
toggle to copy state back to GPU.

Like L1, this holds state in CPU RAM. Unlike L1, it frees 100% of GPU memory
(L1 keeps ~500 MiB for CUDA context) and preserves full state. Doesn't require
vLLM's sleep API, so it works with vLLM v0.14+ where sleep mode is broken.

**Requirements:**
- `cuda-checkpoint` utility
- Root access (or passwordless `sudo` for `cuda-checkpoint`)

**Limitations:**
- **TP=1 only** — `cuda-checkpoint` does not support NCCL IPC memory, so it cannot
  checkpoint/restore processes that communicate via NCCL (i.e., tensor parallel workers).
  See [NVIDIA/cuda-checkpoint#30](https://github.com/NVIDIA/cuda-checkpoint/issues/30).
  For TP>1 models, use L1/L2 (vLLM sleep) or L5 (kill).

#### CRIU checkpoint (level 4)

CRIU checkpointing uses `cuda-checkpoint` and `criu` to snapshot the entire
vLLM process tree to disk, then kill it. On restore, CRIU brings the process
back with all state intact — including GPU VRAM contents, KV cache, CUDA
graphs, and the warmed memory allocator. First inference after restore is ~30ms
(no warmup needed).

**Requirements:**
- CRIU 4.x with the CUDA plugin (`libcuda_plugin.so`)
- `cuda-checkpoint` utility
- Root access (or passwordless `sudo` for `criu` and `cuda-checkpoint`)
- vLLM process must not use `io_uring` or `libuv` (set automatically)

**Limitations:**
- **TP=1 only** — same `cuda-checkpoint` NCCL limitation as L3.

**Trade-offs vs L1/L2:**
- Slower sleep/wake than L1, but no CPU RAM cost and full state preservation
- Slower wake than L1 but faster first-inference (no warmup)
- Works with vLLM v0.14+ where sleep mode is broken
- No need for `--enable-sleep-mode` or `VLLM_SERVER_DEV_MODE`

## Quickstart

Create a `config.json`:

```json
{
  "models": {
    "qwen-14b": {
      "model_path": "Qwen/Qwen3-14B",
      "port": 8001,
      "sleep_level": 1
    },
    "gemma-12b": {
      "model_path": "google/gemma-3-12b-it",
      "port": 8002,
      "sleep_level": 2
    }
  },
  "port": 3000
}
```

### With Docker (recommended)

The Docker image bundles vLLM v0.13.0:

```bash
docker run --gpus all --init \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.json:/etc/llmux/config.json:ro \
  -p 3000:3000 \
  ghcr.io/doublewordai/llmux:latest
```

### From source

Requires vLLM installed and available as `vllm` on PATH:

```bash
cargo install llmux
llmux --config config.json
```

### Send requests

```bash
# First request starts vLLM for qwen-14b
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-14b", "messages": [{"role": "user", "content": "Hello"}]}'

# Switching: sleeps qwen-14b, starts gemma-12b
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-12b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

### Model options

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | *required* | HuggingFace model ID or local path |
| `port` | *required* | Port for this model's vLLM instance |
| `sleep_level` | `5` | Sleep level (1-2: vLLM, 3: CUDA suspend, 4: CRIU, 5: stop) |
| `extra_args` | `[]` | Additional vLLM CLI arguments |

All vLLM-specific flags (e.g. `--gpu-memory-utilization`, `--tensor-parallel-size`,
`--dtype`) should be passed via `extra_args`:

```json
{
  "model_path": "Qwen/Qwen3-14B",
  "port": 8001,
  "extra_args": ["--gpu-memory-utilization", "0.9", "--tensor-parallel-size", "2"]
}
```

### Top-level options

| Field | Default | Description |
|-------|---------|-------------|
| `port` | `3000` | Proxy listen port |
| `metrics_port` | `9090` | Prometheus metrics port (0 to disable) |
| `vllm_command` | `"vllm"` | vLLM binary path |

### Checkpoint config (for sleep levels 3 and 4)

To use CRIU checkpointing, add a `checkpoint` section to your config:

```json
{
  "models": {
    "qwen-14b": {
      "model_path": "Qwen/Qwen3-14B",
      "port": 8001,
      "sleep_level": 4
    }
  },
  "checkpoint": {
    "criu_path": "/tmp/criu/criu/criu",
    "cuda_plugin_dir": "/tmp/criu/plugins/cuda/",
    "images_dir": "/tmp/llmux-checkpoints",
    "cuda_checkpoint_path": "cuda-checkpoint"
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `criu_path` | `"criu"` | Path to the criu binary |
| `cuda_plugin_dir` | `"/usr/lib/criu/"` | Directory containing `libcuda_plugin.so` |
| `images_dir` | `"/tmp/llmux-checkpoints"` | Base directory for checkpoint images |
| `cuda_checkpoint_path` | `"cuda-checkpoint"` | Path to the cuda-checkpoint utility |

### vLLM logging

vLLM process output (stdout/stderr) is always captured and forwarded to the
`vllm` tracing target at `debug` level. Use `RUST_LOG` to control visibility:

```bash
# Default: only llmux info logs, vLLM output hidden
llmux --config config.json

# Show vLLM output
RUST_LOG=info,vllm=debug llmux --config config.json

# --verbose includes vLLM output automatically
llmux --config config.json --verbose
```

ANSI color codes are stripped from vLLM output. The `NO_COLOR=1` environment
variable is also set on spawned vLLM processes.

### Policy options

| Field | Default | Description |
|-------|---------|-------------|
| `policy_type` | `"fifo"` | Switching policy |
| `request_timeout_secs` | `60` | Request timeout |
| `drain_before_switch` | `true` | Wait for in-flight requests before sleeping |
| `sleep_level` | `5` | Default sleep level for policy |

## Validation

llmux includes a built-in validation tool that tests sleep/wake cycles
against a running model, verifying GPU memory is freed and responses are
deterministic after wake:

```bash
llmux --config config.json --validate qwen-14b --levels 1,2,3,4 --verbose
```

Output:

```
Level     Sleep (s)   Wake (s)   GPU Before    GPU After     GPU Wake   Response   Pass
----------------------------------------------------------------------------------------
L1             35.9        1.2      45899 MiB       1341 MiB      44033 MiB      match     OK
L2              0.3        8.2      44033 MiB       1341 MiB      44033 MiB      match     OK

Result: ALL PASSED
```

## Docker Compose with onwards

For production, put [onwards](https://github.com/doublewordai/onwards) in
front for API key authentication:

```yaml
services:
  llmux:
    image: ghcr.io/doublewordai/llmux:latest
    init: true
    command: ["--config", "/etc/llmux/config.json"]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./config.json:/etc/llmux/config.json:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  onwards:
    image: ghcr.io/doublewordai/onwards:latest
    command: ["--targets", "/etc/onwards/targets.json"]
    volumes:
      - ./targets.json:/etc/onwards/targets.json:ro
    ports:
      - "3000:3000"
```

Where `targets.json` maps model names to llmux with API keys:

```json
{
  "targets": {
    "qwen-14b": {
      "url": "http://llmux:3000/v1",
      "keys": ["sk-your-api-key"],
      "onwards_model": "qwen-14b"
    },
    "gemma-12b": {
      "url": "http://llmux:3000/v1",
      "keys": ["sk-your-api-key"],
      "onwards_model": "gemma-12b"
    }
  }
}
```

## Known issues

The `--validate` flag exists specifically to catch these kinds of problems
before they hit production.

### vLLM v0.13.0

- **`openai/gpt-oss-20b` L2 reload fails.** The MXFP4 weight loader crashes on
  wake with `default_weight_loader() got an unexpected keyword argument
  'weight_name'`. L1 works fine (19.6s sleep, 0.6s wake). Use L1 for this
  model.
- L1 and L2 both work correctly for `Qwen/Qwen3-14B` and
  `google/gemma-3-12b-it`.

### vLLM v0.14+

Sleep mode (L1/L2) is broken — weights are not discarded from GPU memory
regardless of sleep level ([vllm#32714](https://github.com/vllm-project/vllm/issues/32714)).
Use CUDA suspend (L3) or CRIU checkpointing (L4) on v0.14+ instead, or stick with v0.13.x for L1/L2.

## Compatibility

- **L1/L2 sleep:** Requires vLLM v0.13.x (broken on v0.14+)
- **L3 CUDA suspend / L4 CRIU checkpoint:** Works with any vLLM version (tested on v0.13.x and v0.14+). **TP=1 only.**

## Tensor parallelism and sleep levels

L3 (CudaSuspend) and L4 (CRIU checkpoint) only work with TP=1. Both depend on
`cuda-checkpoint`, which [does not support NCCL IPC memory](https://github.com/NVIDIA/cuda-checkpoint/issues/30).
With TP>1, suspend works (GPU memory is freed), but restore fails because the
inter-process NCCL communication channels cannot be reconstructed — workers end
up in a broken spin-loop.

**What works today for TP>1:**

| Level | TP>1 | Notes |
|-------|------|-------|
| L1 | Yes | vLLM manages NCCL teardown/rebuild internally |
| L2 | Yes | Same — vLLM handles it |
| L3 | No | `cuda-checkpoint` can't restore NCCL IPC handles |
| L4 | No | Same — CRIU uses `cuda-checkpoint` internally |
| L5 | Yes | Kill + cold restart always works |

### Future: hybrid L2 + cuda-checkpoint for TP>1

A promising approach is to combine vLLM's sleep API (which handles NCCL
teardown/rebuild at the application level) with `cuda-checkpoint` (which frees
the remaining GPU memory that vLLM doesn't release). The orchestrator would
coordinate a two-phase sleep/wake:

**Sleep sequence:**
1. `POST /sleep?level=2` — vLLM tears down NCCL communicators, discards weights.
   This handles the cross-process coordination that `cuda-checkpoint` can't.
2. `cuda-checkpoint --toggle` on each TP worker — frees remaining CUDA context,
   allocator state, and any GPU memory vLLM didn't release (especially on v0.14+
   where sleep is broken for memory freeing).

**Wake sequence:**
1. `cuda-checkpoint --toggle` on each TP worker — restores CUDA context. No NCCL
   IPC to worry about because vLLM already tore it down.
2. `POST /wake_up` — vLLM re-initializes NCCL communicators, reloads weights
   from disk, rebuilds KV cache.

This gives us:
- **100% GPU freed** (cuda-checkpoint handles what vLLM misses)
- **TP>1 support** (vLLM manages NCCL lifecycle, not cuda-checkpoint)
- **Works on v0.14+** (cuda-checkpoint compensates for broken sleep memory freeing)
- **Trade-off**: wake is slower than pure L3 (needs weight reload + NCCL reinit),
  similar to L2 wake time. But unlike pure L2, actually frees all GPU memory.

**Open questions:**
- Does vLLM's `/sleep` on v0.14+ still correctly tear down NCCL, even if it
  fails to free GPU memory? If the NCCL teardown works, the rest follows.
- After `cuda-checkpoint` restore + `/wake_up`, does NCCL re-init succeed on
  the restored CUDA contexts? Or does it need fresh contexts?
- Could we skip the `/sleep` call and instead use vLLM's internal NCCL destroy/
  init APIs directly (e.g., via a custom vLLM plugin or env var)?

### Future: cooperative TP checkpoint via vLLM hooks

A more complete approach — validated in proof-of-concept testing
(see `~/checkpointing/report.md`, Experiment 3). The key insight: NCCL can be
destroyed before checkpoint and reinitialized after restore, and CUDA tensors
(weights, KV cache) survive CRIU independently of NCCL state.

**Validated primitives:**
1. CUDA tensors survive CRIU checkpoint/restore (proven with TP=1 vLLM)
2. NCCL can be reinitialized on a CRIU-restored process — `destroy_process_group()`,
   checkpoint, restore, `init_process_group()`, then `all_reduce` works on original tensors

**vLLM already has the hooks:**
- `vllm.distributed.parallel_state.destroy_model_parallel()` (parallel_state.py) —
  tears down all NCCL process groups (world, TP, PP)
- `vllm.v1.worker.gpu_worker.Worker.reinitialize_distributed()` (gpu_worker.py) —
  destroys and recreates distributed state; currently used for NIXL bootstrapping
  but the pattern is exactly what checkpoint/restore needs
- `vllm.v1.engine.core.EngineCore` — has request scheduling that can be paused

**Checkpoint sequence (TP=N):**
1. **Drain** — stop accepting requests, wait for in-flight completions
2. **Barrier** — all ranks synchronize (no pending NCCL ops)
3. **Destroy NCCL** — each rank calls `destroy_model_parallel()`
4. **CUDA suspend** — `cuda-checkpoint --toggle` on each rank (no IPC memory left)
5. **CRIU dump** — checkpoint each rank process (can be parallel)

**Restore sequence:**
1. **CRIU restore** — restore each rank (CUDA plugin copies memory back to GPU)
2. **Reinit NCCL** — each rank calls `reinitialize_distributed()` with same rank/world_size
3. **Resume** — weights and KV cache are intact, start accepting requests

**Implementation path:**
1. Add a "checkpoint" control message to the engine core's input socket
2. On receiving it: drain scheduler, call `destroy_model_parallel()`, signal ready
3. Orchestrator runs `cuda-checkpoint --toggle` + `criu dump` on each rank
4. On restore: engine core calls `reinitialize_distributed()`, resumes event loop

| Component | Effort | Notes |
|-----------|--------|-------|
| Orchestrator coordination | Low | Coordinate ranks from llmux |
| vLLM drain hook | Medium | Pause scheduler, wait for completion |
| NCCL teardown/reinit | Low | `destroy_model_parallel` + `reinitialize_distributed` exist |
| TCPStore bootstrapping | Medium | Fresh rendezvous for NCCL reinit; rank 0 hosts new store |
| Testing | High | Race conditions, partial failures, timeout handling |

## License

MIT
