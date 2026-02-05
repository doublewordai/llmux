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

| Level | Sleep | Wake | GPU freed | CPU RAM | Use case |
|-------|-------|------|-----------|---------|----------|
| **L1** | Slow (offload to CPU) | Fast (~1s) | All | High (holds weights) | Model you expect to return to soon |
| **L2** | Fast (~1s) | Slow (reload from disk) | All | None | Model you may not need for a while |
| **L3** | Kill process | Cold start | All | None | Fallback / cleanup |

If L1/L2 sleep fails, llmux automatically escalates to L3 (kill) to guarantee
GPU memory is freed.

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
| `sleep_level` | `3` | Sleep level (1, 2, or 3) |
| `gpu_memory_utilization` | `0.9` | vLLM GPU memory fraction |
| `tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `dtype` | `"auto"` | Data type (auto, float16, bfloat16) |
| `extra_args` | `[]` | Additional vLLM CLI arguments |

### Top-level options

| Field | Default | Description |
|-------|---------|-------------|
| `port` | `3000` | Proxy listen port |
| `metrics_port` | `9090` | Prometheus metrics port (0 to disable) |
| `vllm_command` | `"vllm"` | vLLM binary path |
| `vllm_logging` | `false` | Forward vLLM stdout/stderr to logs |

### Policy options

| Field | Default | Description |
|-------|---------|-------------|
| `policy_type` | `"fifo"` | Switching policy |
| `request_timeout_secs` | `60` | Request timeout |
| `drain_before_switch` | `true` | Wait for in-flight requests before sleeping |
| `sleep_level` | `3` | Default sleep level for policy |

## Validation

llmux includes a built-in validation tool that tests sleep/wake cycles
against a running model, verifying GPU memory is freed and responses are
deterministic after wake:

```bash
llmux --config config.json --validate qwen-14b --levels 1,2 --verbose
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

Sleep mode is broken entirely — weights are not discarded from GPU memory
regardless of sleep level ([vllm#32714](https://github.com/vllm-project/vllm/issues/32714)).
Stick with v0.13.x until this is fixed upstream.

## Compatibility

Requires **vLLM v0.13.x** (see known issues above).

## License

MIT
