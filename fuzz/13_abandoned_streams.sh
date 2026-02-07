#!/usr/bin/env bash
# 13_abandoned_streams.sh — Client disconnects mid-stream, then more requests arrive
#
# Simulates: user starts streaming, closes browser, then new requests come in.
# The abandoned stream should be cleaned up without leaving the system in a bad state.
source "$(dirname "$0")/common.sh"

echo "=== 13: Abandoned Streams (client disconnect + follow-up requests) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Ensure gpt-oss-20b is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Phase 1: Start 10 streaming requests with high max_tokens, then kill them after 3s
echo "Starting 10 streaming requests (will be killed after 3s)..."
pids=()
for i in $(seq 1 10); do
    curl -s --max-time 300 \
        "${BASE_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-oss-20b\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Write a very long essay about the history of computing.\"}],
            \"max_tokens\": 2000,
            \"stream\": true
        }" > /dev/null 2>&1 &
    pids+=($!)
done

sleep 3

# Kill the streaming requests (simulates client disconnect)
echo "Killing streaming requests (simulating client disconnect)..."
for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
done
wait 2>/dev/null

sleep 1

# Phase 2: Trigger a switch to gemma-12b
echo "Triggering switch to gemma-12b..."
switch_code=$(send_request "gemma-12b" 5)
echo "  Switch request: ${switch_code}"

# Phase 3: Follow up with 20 normal requests to both models
echo "Firing 20 follow-up requests (10 per model)..."
for i in $(seq 1 10); do
    (
        code=$(send_request "gpt-oss-20b" 10)
        echo "$code" > "${tmpdir}/followup_gpt_${i}"
    ) &
    (
        code=$(send_request "gemma-12b" 10)
        echo "$code" > "${tmpdir}/followup_gemma_${i}"
    ) &
done

echo "Waiting for follow-up requests..."
wait

# Collect results
failures=0
if [[ "$switch_code" != "200" ]]; then
    failures=$((failures + 1))
    echo "  FAIL: switch request returned ${switch_code}"
fi

for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/followup_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: followup gpt request ${i} returned ${code}"
    fi
done
for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/followup_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: followup gemma request ${i} returned ${code}"
    fi
done

TOTAL=21
echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "13_abandoned_streams" "10m"
    exit 1
fi

echo "  PASS: all ${TOTAL} follow-up requests returned 200"
collect_logs "13_abandoned_streams" "10m"
echo ""
