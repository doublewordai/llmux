#!/usr/bin/env bash
# 17_vllm_overload.sh — Overwhelm vLLM with massive concurrent requests
#
# This targets vLLM's own concurrency limits. If we send 200 concurrent
# requests with high max_tokens, vLLM's scheduler might reject some or
# run into CUDA OOM, which llmux should handle gracefully (not 500).
source "$(dirname "$0")/common.sh"

echo "=== 17: vLLM Overload (200 concurrent high-token requests) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Ensure gpt-oss-20b is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Phase 1: Fire 200 concurrent requests with high max_tokens to one model
echo "Firing 200 concurrent requests to gpt-oss-20b (max_tokens=500)..."
for i in $(seq 1 200); do
    (
        code=$(send_request "gpt-oss-20b" 500)
        echo "$code" > "${tmpdir}/req_${i}"
    ) &
done

echo "Waiting for all 200 requests..."
wait

# Collect results
failures=0
non200=()
for i in $(seq 1 200); do
    code=$(cat "${tmpdir}/req_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200+=("req_${i}:${code}")
    fi
done

echo ""
echo "Results: $((200 - failures))/200 returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    echo "  Non-200 breakdown:"
    for entry in "${non200[@]}"; do
        echo "    ${entry}"
    done
    collect_logs "17_vllm_overload" "10m"
    exit 1
fi

echo "  PASS: all 200 requests returned 200"
collect_logs "17_vllm_overload" "10m"

# Phase 2: Same thing but with a switch mixed in
echo ""
echo "Phase 2: 100 high-token gpt-oss-20b + 100 gemma-12b (switch + overload)..."
for i in $(seq 1 100); do
    (
        code=$(send_request "gpt-oss-20b" 200)
        echo "$code" > "${tmpdir}/p2_gpt_${i}"
    ) &
    (
        code=$(send_request "gemma-12b" 200)
        echo "$code" > "${tmpdir}/p2_gemma_${i}"
    ) &
done

echo "Waiting for phase 2..."
wait

failures2=0
for i in $(seq 1 100); do
    code=$(cat "${tmpdir}/p2_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures2=$((failures2 + 1))
        echo "  FAIL: p2 gpt request ${i} returned ${code}"
    fi
    code=$(cat "${tmpdir}/p2_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures2=$((failures2 + 1))
        echo "  FAIL: p2 gemma request ${i} returned ${code}"
    fi
done

echo "Phase 2 Results: $((200 - failures2))/200 returned 200"

if [[ $failures2 -gt 0 ]]; then
    echo "  FAIL: ${failures2} requests did not return 200"
    collect_logs "17_vllm_overload_p2" "10m"
    exit 1
fi

echo "  PASS: all 200 phase 2 requests returned 200"
collect_logs "17_vllm_overload" "15m"
echo ""
