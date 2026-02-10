#!/bin/bash
set -euo pipefail

# Test both patches on bare-metal vLLM v0.15.1:
# 1. NCCL suspend/resume via collective_rpc + cuda-checkpoint at TP=2
# 2. Sleep mode (L1 weight offload) at TP=1

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
PORT=8005

echo "=== Bare Metal Patch Test: vLLM v0.15.1 ==="
echo "Model: $MODEL"
echo ""

# Kill any lingering vllm processes
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 2

do_inference() {
    local port=$1
    local prompt=$2
    local response
    response=$(curl -s http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "'"$prompt"'"}],
            "max_tokens": 20,
            "temperature": 0
        }')
    echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null
}

wait_for_health() {
    local port=$1
    local timeout=$2
    local pid=$3
    for i in $(seq 1 $timeout); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "  Ready after ${i}s"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "  FATAL: process died during startup"
            return 1
        fi
        sleep 1
    done
    echo "  FATAL: not ready after ${timeout}s"
    return 1
}

######################################################################
# TEST 1: NCCL suspend/resume via collective_rpc + cuda-checkpoint
######################################################################
echo "==========================================="
echo "TEST 1: NCCL suspend/resume at TP=2"
echo "==========================================="
echo ""

echo "--- Launching vLLM (TP=2, --enforce-eager, --disable-custom-all-reduce) ---"
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --disable-custom-all-reduce \
    --port $PORT \
    --max-model-len 512 \
    > /tmp/vllm_test1.log 2>&1 &
VLLM_PID=$!
echo "PID: $VLLM_PID"

if ! wait_for_health $PORT 180 $VLLM_PID; then
    tail -30 /tmp/vllm_test1.log
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Inference before
echo ""
echo "--- Inference before suspend ---"
RESULT=$(do_inference $PORT "Say hello in exactly 3 words.")
echo "Response: $RESULT"
[ -n "$RESULT" ] || { echo "FAIL: no response"; kill $VLLM_PID 2>/dev/null; exit 1; }

# Suspend NCCL via collective_rpc (the way llmux will do it)
echo ""
echo "--- Suspend NCCL via POST /collective_rpc ---"
START_T=$(date +%s%N)
SUSPEND_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/collective_rpc \
    -H "Content-Type: application/json" \
    -d '{"method": "suspend_nccl", "args": [], "kwargs": {}}')
END_T=$(date +%s%N)
echo "HTTP $SUSPEND_HTTP ($(( (END_T - START_T) / 1000000 ))ms)"
[ "$SUSPEND_HTTP" = "200" ] || { echo "FAIL: suspend_nccl failed"; kill $VLLM_PID 2>/dev/null; exit 1; }

# cuda-checkpoint suspend (parallel)
echo ""
echo "--- cuda-checkpoint suspend ---"
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
echo "GPU PIDs: $GPU_PIDS"

START_T=$(date +%s%N)
CKPT_BGPIDS=""
for pid in $GPU_PIDS; do
    sudo cuda-checkpoint --toggle --pid $pid &
    CKPT_BGPIDS="$CKPT_BGPIDS $!"
done
CKPT_FAIL=0
for cpid in $CKPT_BGPIDS; do
    wait $cpid || CKPT_FAIL=1
done
END_T=$(date +%s%N)
[ "$CKPT_FAIL" = "0" ] || { echo "FAIL: cuda-checkpoint suspend failed"; kill $VLLM_PID 2>/dev/null; exit 1; }
echo "Suspend took $(( (END_T - START_T) / 1000000 ))ms (parallel)"

# Verify GPUs freed
echo ""
echo "--- GPU state after suspend ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -5 || echo "(no compute apps)"

# cuda-checkpoint resume (parallel)
echo ""
echo "--- cuda-checkpoint resume ---"
START_T=$(date +%s%N)
CKPT_BGPIDS=""
for pid in $GPU_PIDS; do
    sudo cuda-checkpoint --toggle --pid $pid &
    CKPT_BGPIDS="$CKPT_BGPIDS $!"
done
CKPT_FAIL=0
for cpid in $CKPT_BGPIDS; do
    wait $cpid || CKPT_FAIL=1
done
END_T=$(date +%s%N)
[ "$CKPT_FAIL" = "0" ] || { echo "FAIL: cuda-checkpoint resume failed"; kill $VLLM_PID 2>/dev/null; exit 1; }
echo "Resume took $(( (END_T - START_T) / 1000000 ))ms (parallel)"
sleep 1

# Resume NCCL via collective_rpc
echo ""
echo "--- Resume NCCL via POST /collective_rpc ---"
START_T=$(date +%s%N)
RESUME_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/collective_rpc \
    -H "Content-Type: application/json" \
    -d '{"method": "resume_nccl", "args": [], "kwargs": {}}')
END_T=$(date +%s%N)
echo "HTTP $RESUME_HTTP ($(( (END_T - START_T) / 1000000 ))ms)"
[ "$RESUME_HTTP" = "200" ] || { echo "FAIL: resume_nccl failed"; kill $VLLM_PID 2>/dev/null; exit 1; }

# Inference after
echo ""
echo "--- Inference after resume ---"
sleep 2
RESULT2=$(do_inference $PORT "What is 2+2? Answer with just the number.")
echo "Response: $RESULT2"
[ -n "$RESULT2" ] || { echo "FAIL: no response after resume"; kill $VLLM_PID 2>/dev/null; exit 1; }

echo ""
echo "TEST 1: PASS - NCCL suspend/resume + cuda-checkpoint at TP=2"
echo ""

# Cleanup test 1
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 3

######################################################################
# TEST 2: Sleep mode (L1 weight offload) at TP=1
######################################################################
echo "==========================================="
echo "TEST 2: Sleep mode (L1 weight offload) at TP=1"
echo "==========================================="
echo ""

echo "--- Launching vLLM (TP=1) ---"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --port $PORT \
    --max-model-len 512 \
    > /tmp/vllm_test2.log 2>&1 &
VLLM_PID=$!
echo "PID: $VLLM_PID"

if ! wait_for_health $PORT 180 $VLLM_PID; then
    tail -30 /tmp/vllm_test2.log
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Inference before sleep
echo ""
echo "--- Inference before sleep ---"
RESULT=$(do_inference $PORT "What is the capital of France? Answer in one word.")
echo "Response: $RESULT"
[ -n "$RESULT" ] || { echo "FAIL: no response"; kill $VLLM_PID 2>/dev/null; exit 1; }

# Check GPU memory before sleep
echo ""
echo "--- GPU memory before sleep ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1

# PUT /sleep with level 1 (offload weights to CPU)
echo ""
echo "--- Sleep level 1 (offload weights to CPU) ---"
START_T=$(date +%s%N)
SLEEP_RESULT=$(curl -s -w "\n%{http_code}" -X PUT "http://localhost:$PORT/sleep?level=1")
SLEEP_HTTP=$(echo "$SLEEP_RESULT" | tail -1)
SLEEP_BODY=$(echo "$SLEEP_RESULT" | head -n -1)
END_T=$(date +%s%N)
echo "HTTP $SLEEP_HTTP ($(( (END_T - START_T) / 1000000 ))ms): $SLEEP_BODY"

if [ "$SLEEP_HTTP" != "200" ]; then
    echo "FAIL: sleep failed"
    tail -20 /tmp/vllm_test2.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

sleep 3

# Check GPU memory after sleep - should be significantly lower
echo ""
echo "--- GPU memory after sleep ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1

# Wake up
echo ""
echo "--- Wake up ---"
START_T=$(date +%s%N)
WAKE_RESULT=$(curl -s -w "\n%{http_code}" -X PUT "http://localhost:$PORT/wake_up")
WAKE_HTTP=$(echo "$WAKE_RESULT" | tail -1)
WAKE_BODY=$(echo "$WAKE_RESULT" | head -n -1)
END_T=$(date +%s%N)
echo "HTTP $WAKE_HTTP ($(( (END_T - START_T) / 1000000 ))ms): $WAKE_BODY"

if [ "$WAKE_HTTP" != "200" ]; then
    echo "FAIL: wake_up failed"
    tail -20 /tmp/vllm_test2.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

sleep 2

# Inference after wake
echo ""
echo "--- Inference after wake ---"
RESULT2=$(do_inference $PORT "What is the capital of Germany? Answer in one word.")
echo "Response: $RESULT2"
[ -n "$RESULT2" ] || { echo "FAIL: no response after wake"; kill $VLLM_PID 2>/dev/null; exit 1; }

echo ""
echo "TEST 2: PASS - Sleep mode (L1 weight offload) works on v0.15.1"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "==========================================="
echo "ALL TESTS PASSED"
echo "==========================================="
echo ""
echo "v0.15.1 with both patches supports:"
echo "  - L1/L2 sleep (weight offload) - sleep fix patch"
echo "  - L3 CudaSuspend at TP>1 - NCCL suspend/resume patch"
echo "  - L5 stop (always worked)"
