#!/bin/bash
set -euo pipefail

# Test NCCL suspend/resume + sleep fix patches in Docker
# Uses llmux-vllm:0.15.1-test image with both patches applied

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
PORT=8005
TP=2
CONTAINER_NAME="vllm-patch-test"
IMAGE="${1:-llmux-vllm:0.13.0-test}"

echo "=== Docker Patch Test ==="
echo "Image: $IMAGE"
echo "Model: $MODEL (TP=$TP)"
echo ""

# Cleanup any previous test container
docker rm -f $CONTAINER_NAME 2>/dev/null || true
sleep 1

# Step 1: Launch vLLM in Docker container
echo "--- Step 1: Launching vLLM in Docker (TP=$TP) ---"
docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0,1"' \
    --pid=host \
    --ipc=host \
    --shm-size=16g \
    -p $PORT:8000 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v /home/fergus/.cache/huggingface:/root/.cache/huggingface \
    $IMAGE \
    --host 0.0.0.0 \
    --port 8000 \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --enforce-eager \
    --disable-custom-all-reduce \
    --max-model-len 512

echo "Container: $CONTAINER_NAME"

# Wait for server to be ready
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 180); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "FATAL: Container stopped during startup"
        docker logs $CONTAINER_NAME 2>&1 | tail -50
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "FATAL: vLLM not ready after 180s"
    docker logs $CONTAINER_NAME 2>&1 | tail -50
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi

# Step 2: Verify inference works
echo ""
echo "--- Step 2: Verify inference works ---"
RESPONSE=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 20,
        "temperature": 0
    }')
CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "Response: $CONTENT"
if [ -z "$CONTENT" ]; then
    echo "FATAL: No response from vLLM"
    echo "$RESPONSE"
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi

# Step 3: Test NCCL suspend via collective_rpc
echo ""
echo "--- Step 3: Suspend NCCL via collective_rpc ---"
# Note: vLLM listens on port 8000 inside container, mapped to $PORT outside
SUSPEND_RESULT=$(curl -s -w "\n%{http_code}" -X POST http://localhost:$PORT/collective_rpc \
    -H "Content-Type: application/json" \
    -d '{"method": "suspend_nccl", "args": [], "kwargs": {}}')
SUSPEND_HTTP=$(echo "$SUSPEND_RESULT" | tail -1)
SUSPEND_BODY=$(echo "$SUSPEND_RESULT" | head -n -1)
echo "HTTP $SUSPEND_HTTP: $SUSPEND_BODY"
if [ "$SUSPEND_HTTP" != "200" ]; then
    echo "FATAL: collective_rpc suspend_nccl failed"
    docker logs $CONTAINER_NAME 2>&1 | tail -20
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi
echo "NCCL suspended successfully"

# Step 4: cuda-checkpoint suspend
echo ""
echo "--- Step 4: cuda-checkpoint suspend ---"
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
echo "GPU PIDs: $GPU_PIDS"

START_T=$(date +%s%N)
CKPT_PIDS=""
for pid in $GPU_PIDS; do
    echo "  cuda-checkpoint --toggle --pid $pid (background)"
    sudo cuda-checkpoint --toggle --pid $pid &
    CKPT_PIDS="$CKPT_PIDS $!"
done
CKPT_FAIL=0
for cpid in $CKPT_PIDS; do
    wait $cpid || CKPT_FAIL=1
done
END_T=$(date +%s%N)
if [ "$CKPT_FAIL" = "1" ]; then
    echo "FATAL: cuda-checkpoint suspend failed"
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi
echo "cuda-checkpoint suspend took $(( (END_T - START_T) / 1000000 ))ms (parallel)"

# Verify GPUs are freed
echo ""
echo "--- GPU state after suspend ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null || echo "(no compute apps - GPUs freed!)"

# Step 5: cuda-checkpoint resume
echo ""
echo "--- Step 5: cuda-checkpoint resume ---"
START_T=$(date +%s%N)
CKPT_PIDS=""
for pid in $GPU_PIDS; do
    echo "  cuda-checkpoint --toggle --pid $pid (background)"
    sudo cuda-checkpoint --toggle --pid $pid &
    CKPT_PIDS="$CKPT_PIDS $!"
done
CKPT_FAIL=0
for cpid in $CKPT_PIDS; do
    wait $cpid || CKPT_FAIL=1
done
END_T=$(date +%s%N)
if [ "$CKPT_FAIL" = "1" ]; then
    echo "FATAL: cuda-checkpoint resume failed"
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi
echo "cuda-checkpoint resume took $(( (END_T - START_T) / 1000000 ))ms (parallel)"
sleep 1

# Step 6: Resume NCCL via collective_rpc
echo ""
echo "--- Step 6: Resume NCCL via collective_rpc ---"
RESUME_RESULT=$(curl -s -w "\n%{http_code}" -X POST http://localhost:$PORT/collective_rpc \
    -H "Content-Type: application/json" \
    -d '{"method": "resume_nccl", "args": [], "kwargs": {}}')
RESUME_HTTP=$(echo "$RESUME_RESULT" | tail -1)
RESUME_BODY=$(echo "$RESUME_RESULT" | head -n -1)
echo "HTTP $RESUME_HTTP: $RESUME_BODY"
if [ "$RESUME_HTTP" != "200" ]; then
    echo "FATAL: collective_rpc resume_nccl failed"
    docker logs $CONTAINER_NAME 2>&1 | tail -20
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi
echo "NCCL resumed successfully"

# Step 7: Verify inference still works after full cycle
echo ""
echo "--- Step 7: Verify inference after suspend/resume ---"
sleep 2
RESPONSE2=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 10,
        "temperature": 0
    }')
CONTENT2=$(echo "$RESPONSE2" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "Response: $CONTENT2"

if [ -z "$CONTENT2" ]; then
    echo "FAIL: No response after resume"
    echo "$RESPONSE2"
    docker logs $CONTAINER_NAME 2>&1 | tail -30
    docker rm -f $CONTAINER_NAME 2>/dev/null
    exit 1
fi

echo ""
echo "=== SUCCESS: Both patches verified in Docker ==="
echo "  - NCCL suspend/resume via collective_rpc: PASS"
echo "  - cuda-checkpoint suspend/resume at TP=$TP: PASS"
echo "  - Inference after full cycle: PASS"

# Cleanup
docker rm -f $CONTAINER_NAME 2>/dev/null || true
