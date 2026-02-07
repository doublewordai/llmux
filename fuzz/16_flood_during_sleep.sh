#!/usr/bin/env bash
# 16_flood_during_sleep.sh — Flood requests during the sleep phase
#
# The nastiest race condition scenario:
# 1. Model A is active with streaming requests generating
# 2. A switch to Model B triggers, draining begins
# 3. As soon as streaming completes and sleep begins, flood with requests for Model A
# 4. These requests arrive when Model A is being put to sleep — they should
#    queue and wait for the switch to complete, then either trigger a re-switch
#    or be served after B becomes active.
#
# We simulate this by having high-token streaming + immediate flood.
source "$(dirname "$0")/common.sh"

echo "=== 16: Flood During Sleep Phase ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Ensure gpt-oss-20b is active with a streaming request
echo "Warming up gpt-oss-20b with a streaming request..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

TOTAL=0

# Phase 1: Start long streaming requests on gpt-oss-20b to keep it busy
echo "Starting 5 long streaming requests on gpt-oss-20b..."
for i in $(seq 1 5); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_request "gpt-oss-20b" 1000 true)
        echo "$code" > "${tmpdir}/stream_${i}"
    ) &
done

# Phase 2: Immediately trigger a switch
echo "Triggering switch to gemma-12b..."
TOTAL=$((TOTAL + 1))
(
    code=$(send_request "gemma-12b" 5)
    echo "$code" > "${tmpdir}/switch"
) &

# Phase 3: Wait for streams to likely complete (drain phase),
# then flood with gpt-oss-20b requests during the sleep/wake phase
sleep 2
echo "Flooding with 50 gpt-oss-20b requests (targeting sleep/wake timing)..."
for i in $(seq 1 50); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_request "gpt-oss-20b" 5)
        echo "$code" > "${tmpdir}/flood_${i}"
    ) &
done

# Phase 4: Also add some gemma-12b requests to the mix
echo "Adding 20 gemma-12b requests..."
for i in $(seq 1 20); do
    TOTAL=$((TOTAL + 1))
    (
        code=$(send_request "gemma-12b" 5)
        echo "$code" > "${tmpdir}/gemma_${i}"
    ) &
done

echo "Waiting for all ${TOTAL} requests..."
wait

# Collect results
failures=0
for i in $(seq 1 5); do
    code=$(cat "${tmpdir}/stream_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: stream ${i} returned ${code}"
    fi
done
code=$(cat "${tmpdir}/switch" 2>/dev/null || echo "TIMEOUT")
if [[ "$code" != "200" ]]; then
    failures=$((failures + 1))
    echo "  FAIL: switch request returned ${code}"
fi
for i in $(seq 1 50); do
    code=$(cat "${tmpdir}/flood_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: flood request ${i} returned ${code}"
    fi
done
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        echo "  FAIL: gemma request ${i} returned ${code}"
    fi
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "16_flood_during_sleep" "15m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "16_flood_during_sleep" "15m"
echo ""
