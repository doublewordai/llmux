#!/usr/bin/env bash
# 07_high_intensity.sh — 100 concurrent requests per model with frequent switches
source "$(dirname "$0")/common.sh"

echo "=== 07: High Intensity (100 requests/model, frequent switches) ==="

REQUESTS_PER_MODEL=100
TOTAL=$((REQUESTS_PER_MODEL * 2))

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Warm up: make sure one model is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Fire 100 requests for each model simultaneously.
# This creates massive contention: gpt-oss-20b is active, gemma-12b requests
# trigger a switch, and both sets of 100 must complete successfully.
echo "Firing ${REQUESTS_PER_MODEL} requests for gpt-oss-20b..."
for i in $(seq 1 $REQUESTS_PER_MODEL); do
    (
        code=$(send_request "gpt-oss-20b" 10)
        echo "$code" > "${tmpdir}/gpt_${i}"
    ) &
done

echo "Firing ${REQUESTS_PER_MODEL} requests for gemma-12b..."
for i in $(seq 1 $REQUESTS_PER_MODEL); do
    (
        code=$(send_request "gemma-12b" 10)
        echo "$code" > "${tmpdir}/gemma_${i}"
    ) &
done

echo "Waiting for all ${TOTAL} requests to complete..."
wait

# Collect results
codes=()
failures=0
for i in $(seq 1 $REQUESTS_PER_MODEL); do
    code=$(cat "${tmpdir}/gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    codes+=("$code")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
    fi
done
for i in $(seq 1 $REQUESTS_PER_MODEL); do
    code=$(cat "${tmpdir}/gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    codes+=("$code")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
    fi
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    # Show breakdown of non-200 codes
    echo "  Non-200 breakdown:"
    printf '%s\n' "${codes[@]}" | grep -v '^200$' | sort | uniq -c | sort -rn
    collect_logs "07_high_intensity" "10m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "07_high_intensity" "10m"
echo ""
