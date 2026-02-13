# Dynamic LoRA Demo (vLLM + llmux)

This walkthrough shows:

1. dynamic LoRA loading/unloading on vLLM directly
2. llmux LoRA mode with adapter alias switching
3. scheduler behavior (time-slice policy) under mixed adapter demand

## Prerequisites

- `vllm` installed and on `PATH`
- `llmux` built (`cargo build`) or installed
- one base model and at least two LoRA adapters available locally (or via HF cache)
- `jq` installed (used for readable output in examples)

Set demo variables:

```bash
export BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export ADAPTER_A_PATH="/models/lora/sql-assistant"
export ADAPTER_B_PATH="/models/lora/finance-assistant"
```

## 1) Optional: Verify vLLM Dynamic LoRA APIs Directly

Start vLLM manually (separate terminal):

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
vllm serve "$BASE_MODEL" \
  --port 8001 \
  --enable-lora \
  --gpu-memory-utilization 0.9
```

Load adapter A:

```bash
curl -sS -X POST http://localhost:8001/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"adapter-a\",\"lora_path\":\"$ADAPTER_A_PATH\"}"
```

Request with adapter A:

```bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"adapter-a",
    "messages":[{"role":"user","content":"Respond with: adapter A is active"}],
    "temperature":0
  }' | jq '.model, .choices[0].message.content'
```

Unload adapter A:

```bash
curl -sS -X POST http://localhost:8001/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"adapter-a"}'
```

Stop this manual vLLM before moving to llmux mode.

## 2) Start llmux in LoRA Mode

Create `demo.lora.json`:

```json
{
  "lora": {
    "base_model": {
      "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
      "port": 8001,
      "extra_args": [
        "--gpu-memory-utilization", "0.9"
      ]
    },
    "adapters": {
      "adapter-a": {
        "lora_path": "/models/lora/sql-assistant"
      },
      "adapter-b": {
        "lora_path": "/models/lora/finance-assistant"
      }
    }
  },
  "policy": {
    "policy_type": "time_slice",
    "request_timeout_secs": 300,
    "min_active_secs": 3,
    "max_wait_secs": 20,
    "tick_interval_ms": 500,
    "min_quantum_ms": 2000
  },
  "port": 3000,
  "admin_port": 3001,
  "metrics_port": 9090
}
```

Start llmux (separate terminal):

```bash
RUST_LOG=info,vllm=debug ./target/debug/llmux --config demo.lora.json
```

Notes:

- In LoRA mode, llmux eagerly starts the base model process at boot.
- llmux injects `--enable-lora` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` automatically.

## 3) Demonstrate Adapter Switching Through llmux

Request adapter A:

```bash
curl -sS http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"adapter-a",
    "messages":[{"role":"user","content":"Say: adapter A"}],
    "temperature":0
  }' | jq '.model, .choices[0].message.content'
```

Check switcher state:

```bash
curl -sS http://localhost:3001/control/status | jq '{state,active_model,mode}'
```

Switch by requesting adapter B:

```bash
curl -sS http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"adapter-b",
    "messages":[{"role":"user","content":"Say: adapter B"}],
    "temperature":0
  }' | jq '.model, .choices[0].message.content'
```

Check state again:

```bash
curl -sS http://localhost:3001/control/status | jq '{state,active_model,mode}'
```

You should see active model move from `adapter-a` to `adapter-b`.

## 4) Demonstrate Time-Slice Scheduling Behavior

The `time_slice` policy is drain-first: it avoids reactive preemption and switches when the active queue drains (or staleness bound is hit).

### 4.1 Run mixed demand workload

```bash
for i in $(seq 1 30); do
  curl -sS http://localhost:3000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"adapter-a\",\"messages\":[{\"role\":\"user\",\"content\":\"A-$i\"}],\"max_tokens\":64}" >/dev/null &
done

for i in $(seq 1 30); do
  curl -sS http://localhost:3000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"adapter-b\",\"messages\":[{\"role\":\"user\",\"content\":\"B-$i\"}],\"max_tokens\":64}" >/dev/null &
done

wait
```

### 4.2 Observe queueing and active model

```bash
curl -sS http://localhost:3001/control/status | jq .
```

Focus on:

- `.active_model`
- `.models["adapter-a"].queue_depth`
- `.models["adapter-b"].queue_depth`
- `.state` transitions like `switching:adapter-a->adapter-b`

### 4.3 Observe switch metrics

```bash
curl -sS http://localhost:9090/metrics | \
  rg 'llmux_switches_total|llmux_switch_total_seconds|llmux_request_queue_wait_seconds'
```

Expected behavior:

- requests do not cause one switch per request
- switch count grows much slower than request count
- active adapter tends to stay until its queue is drained

## 5) Optional: Force and inspect switching via control API

Force switch:

```bash
curl -sS -X POST http://localhost:3001/control/switch \
  -H "Content-Type: application/json" \
  -d '{"model":"adapter-a"}' | jq .
```

Pin/unpin manual mode:

```bash
curl -sS -X POST http://localhost:3001/control/pin \
  -H "Content-Type: application/json" \
  -d '{"model":"adapter-a"}' | jq .

curl -sS -X POST http://localhost:3001/control/unpin | jq .
```

## Troubleshooting

- `model_not_found` from llmux proxy:
  - verify alias exists under `lora.adapters` in config
- adapter load errors:
  - verify `lora_path` is valid and readable in the runtime environment
- no LoRA switching observed:
  - check `RUST_LOG=info,vllm=debug`
  - inspect `http://localhost:3001/control/status`
  - inspect metrics on `http://localhost:9090/metrics`

