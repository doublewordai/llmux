#!/usr/bin/env bash
# 18_context_pressure.sh — Large context + high tokens + switching
#
# Send requests with long prompts to pressure vLLM's KV cache,
# then trigger switches. If GPU memory is tight, the wake-from-L1
# might fail because there's not enough GPU memory to reload weights
# alongside existing KV cache.
source "$(dirname "$0")/common.sh"

echo "=== 18: Context Pressure (large prompts + switching) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Build a long prompt (repeat a paragraph to fill context)
LONG_CONTENT=$(python3 -c "print('Write a haiku. ' * 500)")

send_long_request() {
    local model="$1"
    local max_tokens="${2:-100}"
    local stream="${3:-false}"

    curl -s -o /dev/null -w '%{http_code}' \
        --max-time "$TIMEOUT" \
        "${BASE_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${LONG_CONTENT}\"}],
            \"max_tokens\": ${max_tokens},
            \"stream\": ${stream}
        }"
}

# Ensure gpt-oss-20b is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

TOTAL=0

# Phase 1: 50 large-context requests to gpt-oss-20b
echo "Phase 1: 50 large-context requests to gpt-oss-20b..."
for i in $(seq 1 50); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_long_request "gpt-oss-20b" 100)
        echo "$code" > "${tmpdir}/p1_${i}"
    ) &
done

# Phase 2: While those are running, trigger switch with large-context gemma requests
sleep 1
echo "Phase 2: 30 large-context requests to gemma-12b (triggers switch)..."
for i in $(seq 1 30); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_long_request "gemma-12b" 100)
        echo "$code" > "${tmpdir}/p2_${i}"
    ) &
done

# Phase 3: More gpt-oss-20b during the switch
sleep 0.5
echo "Phase 3: 20 more large-context requests to gpt-oss-20b..."
for i in $(seq 1 20); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_long_request "gpt-oss-20b" 100)
        echo "$code" > "${tmpdir}/p3_${i}"
    ) &
done

echo "Waiting for all ${TOTAL} requests..."
wait

# Collect results
failures=0
for i in $(seq 1 50); do
    code=$(cat "${tmpdir}/p1_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: p1 request ${i} returned ${code}"
    fi
done
for i in $(seq 1 30); do
    code=$(cat "${tmpdir}/p2_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: p2 request ${i} returned ${code}"
    fi
done
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/p3_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: p3 request ${i} returned ${code}"
    fi
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "18_context_pressure" "15m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "18_context_pressure" "15m"
echo ""
