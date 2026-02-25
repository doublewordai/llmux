#!/bin/sh
#
# Pre-build checkpoint images for all models so the first wake is a fast
# restore (~11s) instead of a cold start (~30-100s).
#
# Usage: sudo ./warmup.sh
#
# Run once per model after pulling the vLLM image. The script cold-starts
# each container, waits for it to be healthy, checkpoints it, and leaves
# the container in a stopped+checkpointed state ready for restore.

set -eu

IMAGE="docker.io/vllm/vllm-openai:v0.8.3"
HF_CACHE="$HOME/.cache/huggingface"
MAX_WAIT=120

warmup_model() {
  local name="$1"       # container name
  local gpu="$2"        # GPU index
  local port="$3"       # host port
  local model="$4"      # HF model name
  local max_len="$5"    # max model length

  echo "=== Warming up $name ($model on GPU $gpu, port $port) ==="

  # Clean up any existing container
  podman rm -f "$name" 2>/dev/null || true

  # Cold start
  podman run -d --name "$name" \
    --privileged --device "nvidia.com/gpu=$gpu" \
    -p "$port:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE" \
    --model "$model" --max-model-len "$max_len"

  # Wait for health
  echo "  Waiting for $name to be healthy..."
  for i in $(seq 1 $MAX_WAIT); do
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
      echo "  Ready after ${i}s"
      break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
      echo "  ERROR: $name not ready after ${MAX_WAIT}s" >&2
      podman logs "$name" 2>&1 | tail -10
      exit 1
    fi
    sleep 1
  done

  # Checkpoint (stops the container)
  echo "  Checkpointing $name..."
  podman container checkpoint --tcp-established "$name"
  echo "  Done. $name is checkpointed and ready for restore."
  echo ""
}

# ── Models ──────────────────────────────────────────────────────────────
# Add/remove models here. Must match config.yaml.

warmup_model "llmux-opt-125m" 1 8001 "facebook/opt-125m" 512
warmup_model "llmux-qwen-3b"  2 8002 "Qwen/Qwen2.5-3B-Instruct" 512

echo "All models warmed up. Run llmux to start serving."
