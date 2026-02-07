#!/usr/bin/env bash
# 11_extreme_load.sh — Extreme concurrent load with streaming and rapid switching
#
# This test combines:
# 1. Many concurrent streaming requests on the active model
# 2. A burst of switch-triggering requests while streams are in-flight
# 3. Immediately followed by more requests to both models
#
# Goal: Find race conditions in drain/switch under very high streaming load
source "$(dirname "$0")/common.sh"

echo "=== 11: Extreme Load (streaming + switching + burst) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Phase 1: Ensure gpt-oss-20b is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Phase 2: Fire 30 streaming requests at the active model with high max_tokens
# These will be in-flight for a while
echo "Firing 30 streaming requests at gpt-oss-20b (max_tokens=500, stream=true)..."
for i in $(seq 1 30); do
    (
        code=$(send_request "gpt-oss-20b" 500 true)
        echo "$code" > "${tmpdir}/stream_${i}"
    ) &
done

# Wait just a moment for streams to start
sleep 1

# Phase 3: Fire 20 requests at gemma-12b to trigger a switch
echo "Firing 20 requests at gemma-12b (triggers switch, max_tokens=50)..."
for i in $(seq 1 20); do
    (
        code=$(send_request "gemma-12b" 50)
        echo "$code" > "${tmpdir}/switch_${i}"
    ) &
done

# Phase 4: Wait a beat, then fire 20 more requests at gpt-oss-20b
# These arrive after the switch has started - they need to wait or re-switch
sleep 0.5
echo "Firing 20 more requests at gpt-oss-20b (arrives mid-switch)..."
for i in $(seq 1 20); do
    (
        code=$(send_request "gpt-oss-20b" 10)
        echo "$code" > "${tmpdir}/late_gpt_${i}"
    ) &
done

# Phase 5: And 10 more at gemma-12b for good measure
echo "Firing 10 more requests at gemma-12b..."
for i in $(seq 1 10); do
    (
        code=$(send_request "gemma-12b" 10)
        echo "$code" > "${tmpdir}/late_gemma_${i}"
    ) &
done

TOTAL=80
echo "Waiting for all ${TOTAL} requests to complete..."
wait

# Collect results
failures=0
for i in $(seq 1 30); do
    code=$(cat "${tmpdir}/stream_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: streaming request ${i} returned ${code}"
    fi
done
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/switch_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: switch request ${i} returned ${code}"
    fi
done
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/late_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: late gpt request ${i} returned ${code}"
    fi
done
for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/late_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: late gemma request ${i} returned ${code}"
    fi
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "11_extreme_load" "15m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "11_extreme_load" "15m"
echo ""
