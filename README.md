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
lifecycle. Only one model is active at a time - the rest are evicted using
a configurable two-axis policy.

### Eviction policy

When a model is evicted, llmux applies two strategies in sequence:

1. **Weight strategy** — what vLLM does with model weights (via the sleep API)
2. **Process strategy** — what happens to the OS process afterward

#### Weight strategy

Applied first, via vLLM's sleep endpoint. Controls what happens to weights and KV cache:

| Strategy | Description |
|----------|-------------|
| `retain` | Nothing happens. Weights and KV cache stay on GPU. |
| `offload` | Weights copied to pinned CPU RAM. KV cache and CUDA graphs discarded. Frees most GPU memory but uses significant host RAM. |
| `discard` | Weights dropped entirely (reloaded from disk on wake). KV cache and CUDA graphs discarded. Frees most GPU memory with no CPU RAM cost. |

Both `offload` and `discard` leave a small CUDA context (~500 MiB) on the GPU.

#### Process strategy

Applied second, to the process left behind by the weight strategy:

| Strategy | Description |
|----------|-------------|
| `keep_running` | Process stays as-is. Fast, but whatever's still on GPU stays there. |
| `cuda_suspend` | Snapshots remaining VRAM to host RAM via `cuda-checkpoint`, freeing 100% of GPU memory. Process stays alive. |
| `checkpoint` | CRIU dumps the entire process (including host RAM) to disk, then kills it. Frees 100% of GPU and CPU memory. |
| `stop` | Kills the process. Everything is lost. |

#### How they interact

The weight strategy determines what's on GPU vs. CPU before the process strategy runs.
This affects speed, memory usage, and what survives a wake cycle:

| | `keep_running` | `cuda_suspend` | `checkpoint` | `stop` |
|---|---|---|---|---|
| **`retain`** | No-op (model stays loaded) | Full VRAM snapshot to CPU — weights, KV cache, CUDA graphs all preserved | Full process checkpoint to disk — large image (includes VRAM) | Everything lost |
| **`offload`** | Weights on CPU, KV lost, ~500 MiB CUDA context remains on GPU | Remaining CUDA context → CPU, weights already on CPU | Large CRIU image (weights in host RAM get written to disk) | Everything lost |
| **`discard`** | Weights gone, KV lost, ~500 MiB CUDA context remains on GPU | Remaining CUDA context → CPU, weights gone | Small CRIU image (no weights — reloaded from HF cache on wake) | Everything lost |

Common choices:

- **`offload` + `keep_running`** — Fast wake (weights already in RAM), but holds CPU memory and ~500 MiB GPU
- **`discard` + `keep_running`** — No CPU RAM cost, but slow wake (reload from disk) and ~500 MiB GPU
- **`retain` + `cuda_suspend`** — Frees 100% GPU, full state preserved, but holds all VRAM in CPU RAM
- **`discard` + `checkpoint`** — Frees 100% GPU *and* CPU, small CRIU image; wake reloads weights from disk but restores KV cache, CUDA graphs, and warmed allocator from checkpoint
- **`offload` + `checkpoint`** — Like above but CRIU image is large (includes weights); wake is faster (no disk reload) but checkpoint is slower and uses more disk

If eviction fails, llmux automatically escalates to `stop` to guarantee GPU
memory is freed.

#### CUDA suspend

Uses `cuda-checkpoint --toggle` to suspend CUDA state and copy VRAM to host
RAM. The process stays alive — no serialization, no CRIU. Wake is just another
toggle to copy state back to GPU.

Like `offload`, this holds state in CPU RAM. Unlike `offload`, it frees 100% of
GPU memory (`offload` keeps ~500 MiB for CUDA context) and preserves full state.

For TP>1, llmux coordinates NCCL teardown before checkpoint and rebuild after
restore. This requires patched vLLM with `suspend_nccl`/`resume_nccl` support
(included in the Docker image).

**Requirements:**
- `cuda-checkpoint` utility (included in Docker image)
- Root access (or passwordless `sudo` for `cuda-checkpoint`)
- For TP>1: `--enforce-eager` and `--disable-custom-all-reduce` in `extra_args`

#### CRIU checkpoint

CRIU checkpointing uses `cuda-checkpoint` and `criu` to snapshot the entire
vLLM process tree to disk, then kill it. On restore, CRIU brings the process
back with all state intact — including GPU VRAM contents, KV cache, CUDA
graphs, and the warmed memory allocator. First inference after restore is ~30ms
(no warmup needed).

**Requirements:**
- CRIU 4.x with the CUDA plugin (`libcuda_plugin.so`) (included in Docker image)
- `cuda-checkpoint` utility (included in Docker image)
- Root access (or passwordless `sudo` for `criu` and `cuda-checkpoint`)
- vLLM process must not use `io_uring` or `libuv` (set automatically)
- For TP>1: `--enforce-eager` and `--disable-custom-all-reduce` in `extra_args`

**Trade-offs vs offload:**
- Slower sleep/wake than `offload`, but no CPU RAM cost and full state preservation
- Slower wake than `offload` but faster first-inference (no warmup)

## Quickstart

Create a `config.json`:

```json
{
  "models": {
    "qwen-14b": {
      "model_path": "Qwen/Qwen3-14B",
      "port": 8001,
      "eviction": { "weights": "offload", "process": "keep_running" }
    },
    "gemma-12b": {
      "model_path": "google/gemma-3-12b-it",
      "port": 8002,
      "eviction": { "weights": "discard", "process": "keep_running" }
    }
  },
  "port": 3000
}
```

### With Docker (recommended)

The Docker image bundles vLLM v0.15.1 with patches for NCCL suspend/resume
and sleep mode fixes:

```bash
docker run --gpus all --init \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.json:/etc/llmux/config.json:ro \
  -p 3000:3000 \
  ghcr.io/doublewordai/llmux:latest
```

For `cuda_suspend` or `checkpoint` process strategies, additional flags are required:

```bash
docker run --gpus all \
  --privileged \
  --pid=host \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.json:/etc/llmux/config.json:ro \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  -p 3000:3000 \
  ghcr.io/doublewordai/llmux:latest
```

The extra flags are needed because:
- `--privileged` — CRIU requires broad namespace and ptrace access
- `--pid=host` — cuda-checkpoint needs to ptrace vLLM worker PIDs
- `--ipc=host` — NCCL uses shared memory for inter-GPU communication
- `-v /tmp/llmux-checkpoints:...` — CRIU checkpoints can be tens of GB; mount a host volume to avoid filling the container filesystem

**Important:** Do NOT use `--init` with CRIU (`checkpoint` process strategy). Docker's init process (tini)
redirects stdin to the host's `/dev/null`, whose mount ID is invisible inside the container.
CRIU dump fails with "Can't lookup mount=N for fd=0 path=/dev/null".

### From source

Requires vLLM installed and available as `vllm` on PATH:

```bash
cargo install llmux
llmux --config config.json
```

For `cuda_suspend` or `checkpoint` process strategies, you also need `cuda-checkpoint`
and `criu` (with CUDA plugin) installed and either run as root or configure passwordless sudo.

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
| `eviction` | `retain` + `stop` | Eviction policy (see below) |
| `extra_args` | `[]` | Additional vLLM CLI arguments |

The `eviction` field takes an object with `weights` and `process` keys:

```json
{
  "eviction": { "weights": "offload", "process": "keep_running" }
}
```

All vLLM-specific flags (e.g. `--gpu-memory-utilization`, `--tensor-parallel-size`,
`--dtype`) should be passed via `extra_args`:

```json
{
  "model_path": "Qwen/Qwen3-14B",
  "port": 8001,
  "extra_args": ["--gpu-memory-utilization", "0.9", "--tensor-parallel-size", "2"]
}
```

#### Tensor parallelism with cuda_suspend/checkpoint

When using `cuda_suspend` or `checkpoint` with TP>1, you **must** include `--enforce-eager`
and `--disable-custom-all-reduce` in `extra_args`:

```json
{
  "model_path": "NousResearch/Meta-Llama-3.1-8B-Instruct",
  "port": 8001,
  "eviction": { "weights": "retain", "process": "cuda_suspend" },
  "extra_args": [
    "--tensor-parallel-size", "2",
    "--enforce-eager",
    "--disable-custom-all-reduce",
    "--gpu-memory-utilization", "0.85"
  ]
}
```

- `--enforce-eager` — CUDA graphs hold stale NCCL handles and crash on resume
- `--disable-custom-all-reduce` — CustomAllReduce IPC buffers cannot survive cuda-checkpoint

llmux validates the config at startup and warns if these flags are missing.

### Top-level options

| Field | Default | Description |
|-------|---------|-------------|
| `port` | `3000` | Proxy listen port |
| `metrics_port` | `9090` | Prometheus metrics port (0 to disable) |
| `vllm_command` | `"vllm"` | vLLM binary path |

### Checkpoint config

To use `cuda_suspend` or `checkpoint` process strategies, add a `checkpoint` section:

```json
{
  "models": {
    "qwen-14b": {
      "model_path": "Qwen/Qwen3-14B",
      "port": 8001,
      "eviction": { "weights": "retain", "process": "cuda_suspend" }
    }
  },
  "checkpoint": {
    "cuda_checkpoint_path": "cuda-checkpoint"
  }
}
```

For CRIU checkpointing, the full config is:

```json
{
  "checkpoint": {
    "criu_path": "criu",
    "cuda_plugin_dir": "/usr/lib/criu/",
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
| `eviction` | `retain` + `stop` | Default eviction policy |

## Validation

llmux includes a built-in validation tool that tests sleep/wake cycles
against a running model, verifying GPU memory is freed and responses are
deterministic after wake:

```bash
llmux --config config.json --validate qwen-14b \
  --policies offload+keep_running,discard+keep_running,retain+cuda_suspend \
  --verbose
```

Output:

```
Eviction          Sleep (s)   Wake (s)   GPU Before    GPU After     GPU Wake   Response   Pass
------------------------------------------------------------------------------------------------
Offload+KeepRun        35.9        1.2      45899 MiB       1341 MiB      44033 MiB      match     OK
Discard+KeepRun         0.3        8.2      44033 MiB       1341 MiB      44033 MiB      match     OK

Result: ALL PASSED
```

## Checkpoint management

Pre-create CRIU checkpoints for fast model switching:

```bash
# Create checkpoint (start model, warm up, CRIU dump to disk)
llmux --config config.json --checkpoint qwen-14b

# Use a different weight strategy (affects CRIU image size)
llmux --config config.json --checkpoint qwen-14b --eviction retain+checkpoint

# Skip warmup inference before checkpointing
llmux --config config.json --checkpoint qwen-14b --no-warmup

# Restore from checkpoint (CRIU restore, health check, exit)
llmux --config config.json --restore qwen-14b
```

The default eviction for `--checkpoint` is `discard+checkpoint`, which produces
small CRIU images (weights are reloaded from the HF cache on restore). Use
`retain+checkpoint` or `offload+checkpoint` for larger images that restore
faster (weights already in the snapshot).

After `--restore`, the vLLM process continues running on its configured port.
The daemon can then manage it normally when started.

## Docker Compose

### Basic setup

```yaml
services:
  llmux:
    image: ghcr.io/doublewordai/llmux:latest
    init: true
    command: ["--config", "/etc/llmux/config.json"]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./config.json:/etc/llmux/config.json:ro
    ports:
      - "3000:3000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### With cuda-checkpoint/CRIU

```yaml
services:
  llmux:
    image: ghcr.io/doublewordai/llmux:latest
    init: true
    command: ["--config", "/etc/llmux/config.json"]
    pid: host
    ipc: host
    cap_add:
      - SYS_PTRACE
      - CHECKPOINT_RESTORE
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./config.json:/etc/llmux/config.json:ro
    ports:
      - "3000:3000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### With onwards (API key auth)

For production, put [onwards](https://github.com/doublewordai/onwards) in
front for API key authentication:

```yaml
services:
  llmux:
    image: ghcr.io/doublewordai/llmux:latest
    init: true
    command: ["--config", "/etc/llmux/config.json"]
    pid: host
    ipc: host
    cap_add:
      - SYS_PTRACE
      - CHECKPOINT_RESTORE
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

## Tensor parallelism

All eviction strategies work with TP>1:

| Strategy | TP>1 | Notes |
|----------|------|-------|
| `offload` | Yes | vLLM manages NCCL teardown/rebuild internally |
| `discard` | Yes | Same — vLLM handles it |
| `cuda_suspend` | Yes | llmux tears down NCCL before cuda-checkpoint, rebuilds after restore |
| `checkpoint` | Yes | Same — NCCL teardown before checkpoint, rebuild after restore |
| `stop` | Yes | Kill + cold restart always works |

For `cuda_suspend` and `checkpoint`, llmux uses vLLM's `/collective_rpc` endpoint to call
`suspend_nccl` (before cuda-checkpoint) and `resume_nccl` (after restore)
on all TP workers. This tears down NCCL IPC handles that cuda-checkpoint
cannot checkpoint, then rebuilds them after CUDA state is restored.

This requires patched vLLM with `suspend_nccl`/`resume_nccl` support. The
Docker image includes these patches. For bare-metal installs, apply
`patches/nccl-suspend-resume-v0.15.1.patch` to your vLLM installation.

## Known issues

The `--validate` flag exists specifically to catch these kinds of problems
before they hit production.

### vLLM v0.14+ sleep regression

Sleep mode (`offload`/`discard`) has a regression where weights are not discarded from GPU
memory ([vllm#32714](https://github.com/vllm-project/vllm/issues/32714)).
The Docker image includes a patch (`fix-sleep-mode-v0.15.1.patch`) that fixes
this. For bare-metal installs, apply the patch or use `cuda_suspend`/`stop` instead.

### vLLM v0.13.0

- **`openai/gpt-oss-20b` `discard` reload fails.** The MXFP4 weight loader crashes on
  wake with `default_weight_loader() got an unexpected keyword argument
  'weight_name'`. `offload` works fine (19.6s sleep, 0.6s wake). Use `offload` for this
  model.
- `offload` and `discard` both work correctly for `Qwen/Qwen3-14B` and
  `google/gemma-3-12b-it`.

### NVIDIA driver requirements

The Docker image uses vLLM v0.15.1 which requires CUDA 12.9 and
nvidia-driver-580 or later. Check your driver version with `nvidia-smi`.

## Compatibility

- **`offload`/`discard`:** Works with vLLM v0.13.x out of the box. Works with v0.15.1 with the sleep fix patch (included in Docker image).
- **`cuda_suspend`/`checkpoint`:** Works with vLLM v0.13.x+ with NCCL patches (included in Docker image). Requires `cuda-checkpoint` and CRIU (included in Docker image).
- **TP>1 with `cuda_suspend`/`checkpoint`:** Requires vLLM NCCL suspend/resume patches (included in Docker image) plus `--enforce-eager` and `--disable-custom-all-reduce` flags.

## License

MIT
