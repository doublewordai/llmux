#!/usr/bin/env bash
# 00_restart.sh — Restart the stack and verify clean state
source "$(dirname "$0")/common.sh"

echo "=== 00: Restart ==="

echo "Restarting model-switcher..."
$SSH_CMD "cd /home/ubuntu && sudo docker compose restart model-switcher" > /dev/null 2>&1

# Wait for llmux to be listening
echo "Waiting for llmux to start..."
for i in $(seq 1 30); do
    if curl -s --max-time 2 "${BASE_URL}/v1/models" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Verify clean state
gpu_mem=$($SSH_CMD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d '[:space:]')
echo "  GPU memory used: ${gpu_mem} MiB"

if [[ "$gpu_mem" -lt 100 ]]; then
    echo "  PASS: clean restart, GPU free"
else
    echo "  WARN: GPU memory not fully free (${gpu_mem} MiB) — a vLLM process may still be running"
fi

collect_logs "00_restart" "1m"
echo ""
