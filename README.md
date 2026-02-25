# llmux

[![Crates.io](https://img.shields.io/crates/v/llmux)](https://crates.io/crates/llmux)
[![GitHub](https://img.shields.io/badge/GitHub-doublewordai%2Fllmux-blue)](https://github.com/doublewordai/llmux)

LLM multiplexer. Routes OpenAI-compatible requests to model backends and
switches between them on demand using user-provided scripts.

When a request arrives for a model that isn't currently loaded, llmux drains
in-flight requests, runs your **sleep** script on the active model, then runs
your **wake** script on the requested model. The API stays up throughout —
clients just change the `model` field.

llmux doesn't manage model processes directly. You provide three shell scripts
per model (**wake**, **sleep**, **alive**) and llmux calls them at the right
time. This means it works with any backend — vLLM, SGLang, llama.cpp, Ollama,
or anything else that speaks HTTP.

## Install

```sh
cargo install llmux
```

## Quick start

Create a `config.yaml`:

```yaml
models:
  llama:
    port: 8001
    wake: ./scripts/wake-llama.sh
    sleep: ./scripts/sleep-llama.sh
    alive: curl -sf http://localhost:8001/health

  mistral:
    port: 8002
    wake: ./scripts/wake-mistral.sh
    sleep: ./scripts/sleep-mistral.sh
    alive: curl -sf http://localhost:8002/health

port: 3000
```

Run it:

```sh
llmux -c config.yaml
```

Send requests:

```sh
curl http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hi"}]}'
```

When a request comes in for `mistral`, llmux will drain active requests,
run `sleep-llama.sh`, then `wake-mistral.sh`, and proxy the request through.

## How it works

```
Client requests
     |
+---------+
|  llmux  |   port 3000 (OpenAI-compatible proxy)
+---------+
 /         \
[8001]    [8002]
 llama     mistral
(active)   (sleeping)
```

1. **Middleware** extracts the `model` field from the request JSON body
2. **Switcher** checks if that model is active. If not, triggers a switch:
   - Drains in-flight requests for the current model
   - Runs the **sleep** hook on the current model
   - Runs the **wake** hook on the target model
3. **Proxy** forwards the request to `localhost:<model_port>`
4. In-flight tracking uses RAII guards that hold through streaming responses

## Configuration

### Models

Each model needs a `port` and three hooks:

```yaml
models:
  my-model:
    port: 8001
    wake: |
      # Bring the model to a ready state (must be idempotent).
      # Exit 0 when the model is ready to serve requests.
      docker start my-model-container
      for i in $(seq 1 60); do
        curl -sf http://localhost:8001/health && exit 0
        sleep 1
      done
      exit 1
    sleep: |
      # Free resources. Exit 0 when done.
      docker stop my-model-container
    alive: |
      # Health check. Exit 0 = healthy, non-zero = unhealthy.
      curl -sf http://localhost:8001/health
```

Hooks are executed via `sh -c` with `LLMUX_MODEL` set in the environment.
They can be inline scripts (YAML `|` syntax) or paths to executables.

### Policy

```yaml
policy:
  # Max time a request waits for a switch. Omit for unlimited.
  request_timeout_secs: 300

  # Wait for in-flight requests to finish before switching. Default: true.
  drain_before_switch: true

  # Minimum seconds a model stays active before it can be switched out.
  # Prevents rapid thrashing. Default: 0.
  min_active_secs: 5
```

### Full config reference

```yaml
models:
  <model-name>:
    port: <u16>         # Where the backend listens
    wake: <string>      # Script to start/restore the model
    sleep: <string>     # Script to stop/checkpoint the model
    alive: <string>     # Health check script

policy:
  request_timeout_secs: <u64 | null>   # null = unlimited
  drain_before_switch: <bool>          # default: true
  min_active_secs: <u64>               # default: 0

port: <u16>  # Proxy listen port (default: 3000)
```

Both YAML and JSON configs are supported (detected by file extension).

## Examples

### Podman + CRIU (GPU checkpoint/restore)

The [`examples/podman-criu/`](examples/podman-criu/) directory shows how to
use CRIU to checkpoint and restore vLLM containers, achieving ~3x faster
model switches vs. cold start. See the [example README](examples/podman-criu/README.md)
for setup instructions and timings.

## Architecture

The codebase is ~1,700 lines of Rust:

| Module | What it does |
|--------|-------------|
| `main.rs` | CLI (clap), config loading, server startup |
| `config.rs` | YAML/JSON config parsing |
| `hooks.rs` | Executes wake/sleep/alive scripts via `sh -c` |
| `switcher.rs` | State machine: Idle → Switching → Active. Drain, sleep, wake. |
| `middleware.rs` | Tower layer: extract model, ensure ready, acquire in-flight guard |
| `proxy.rs` | Reverse proxy to `localhost:<port>` |
| `policy/` | Pluggable switch policy trait (currently FIFO) |

The switch policy is a trait (`SwitchPolicy`) with rich context — queue depths,
in-flight counts, timing — so future policies can make cost-aware or
batching decisions. See [`docs/scheduling-problem.md`](docs/scheduling-problem.md)
for the design space.

## License

MIT
