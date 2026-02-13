# LoRA Throughput Experiment Results (`Qwen/Qwen3-0.6B`)

Date: 2026-02-13  
Branch: `feature/dynamic-lora-serving`  
Commit under test: `967062d`

## Goal

Measure how llmux throughput changes between:

1. single-adapter traffic (`adapter-a` only)
2. highly intertwined adapter traffic (`adapter-a`/`adapter-b` alternating)

## Environment

- GPU: `NVIDIA GeForce RTX 3060 (12 GB)`
- Driver: `560.35.03`
- `vllm`: `0.15.1`
- `torch`: `2.9.1+cu128`

## Model + Adapter Setup

- Base model: `Qwen/Qwen3-0.6B`
- Adapters:
1. `adapter-a` at `/tmp/llmux-lora/adapters/adapter-a`
2. `adapter-b` at `/tmp/llmux-lora/adapters/adapter-b`

Both adapters were generated locally with `peft` LoRA (rank 4, targets `q_proj` + `v_proj`) against `Qwen/Qwen3-0.6B`.

## llmux Config Used

```json
{
  "lora": {
    "base_model": {
      "model_path": "Qwen/Qwen3-0.6B",
      "port": 8101,
      "extra_args": [
        "--gpu-memory-utilization", "0.75",
        "--max-model-len", "1024",
        "--dtype", "float16",
        "--enforce-eager"
      ]
    },
    "adapters": {
      "adapter-a": { "lora_path": "/tmp/llmux-lora/adapters/adapter-a" },
      "adapter-b": { "lora_path": "/tmp/llmux-lora/adapters/adapter-b" }
    }
  },
  "policy": {
    "policy_type": "time_slice",
    "request_timeout_secs": 120,
    "min_active_secs": 0,
    "max_wait_secs": 2,
    "tick_interval_ms": 100,
    "min_quantum_ms": 250
  },
  "port": 8300,
  "admin_port": 8301,
  "metrics_port": 8302,
  "vllm_command": "/home/titan-5/miniconda3/bin/vllm"
}
```

## Workload Definition

Request body used for all runs:

```json
{
  "messages": [{"role":"user","content":"Reply with exactly: ok"}],
  "max_tokens": 8,
  "temperature": 0
}
```

Warmup sequence before measured runs: `adapter-a -> adapter-b -> adapter-a`.

Measured runs:

1. `single_run_1`: 10 requests, all `adapter-a`
2. `single_run_2`: 10 requests, all `adapter-a`
3. `intertwined_run_1`: 10 requests alternating `adapter-a`, `adapter-b`
4. `intertwined_run_2`: 10 requests alternating `adapter-a`, `adapter-b`

## Raw Results

| Run | Requests | Throughput (req/s) | p50 latency (ms) | p95 latency (ms) | Switches |
|---|---:|---:|---:|---:|---:|
| `single_run_1` | 10 | 3.6668 | 269.29 | 281.58 | 0 |
| `single_run_2` | 10 | 3.6708 | 269.44 | 286.06 | 0 |
| `intertwined_run_1` | 10 | 0.0537 | 20634.60 | 20928.69 | 9 |
| `intertwined_run_2` | 10 | 0.0481 | 20878.19 | 20927.99 | 10 |

## Aggregate Comparison

- Single-adapter average throughput: **3.6688 req/s**
- Intertwined average throughput: **0.0509 req/s**
- Throughput drop: **98.61%**
- Slowdown factor: **72.06x**
- Single-adapter average switches/run: **0.0**
- Intertwined average switches/run: **9.5**

## Metrics Snapshot

From `http://127.0.0.1:8302/metrics` after the full run:

- `llmux_switches_total{from="adapter-a",to="adapter-b"} = 11`
- `llmux_switches_total{from="adapter-b",to="adapter-a"} = 10`
- `llmux_switch_total_seconds_sum{from="adapter-a",to="adapter-b"} = 220.4503`
- `llmux_switch_total_seconds_sum{from="adapter-b",to="adapter-a"} = 199.7545`

Average observed switch duration from these totals is about 20s/switch.

## Observations

1. Single-adapter traffic stays on one adapter and avoids switching cost entirely.
2. Intertwined A/B traffic causes near-maximum switching frequency (about one switch per request).
3. In this branch/build, adapter switches are currently coupled with base-process stop/start behavior. `control/status` showed:
   - inactive adapter state: `sleeping:Retain+Stop`
   - active adapter state: `running`
4. That stop/start behavior dominates intertwined performance, producing very large per-request latency and major throughput collapse under alternating adapter demand.

