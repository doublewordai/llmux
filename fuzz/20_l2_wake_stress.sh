#!/usr/bin/env bash
# 20_l2_wake_stress.sh — Stress the L2 wake path specifically
#
# gemma-12b uses L2 sleep (discard weights). Waking from L2 requires:
# 1. POST /wake_up
# 2. POST /collective_rpc (reload_weights)
# 3. POST /reset_prefix_cache
#
# This is the slowest and most error-prone wake path. We stress it by:
# 1. Repeatedly switching away from gemma and back, forcing many L2 wake cycles
# 2. Sending concurrent requests during each wake cycle
source "$(dirname "$0")/common.sh"

echo "=== 20: L2 Wake Stress (repeated gemma-12b L2 wake cycles under load) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

CYCLES=5
TOTAL=0

# Start with gpt-oss-20b active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

for cycle in $(seq 1 $CYCLES); do
    echo "Cycle ${cycle}/${CYCLES}:"

    # Step 1: Switch to gemma-12b (L2 wake) with 20 concurrent requests
    echo "  Switching to gemma-12b with 20 concurrent requests..."
    for i in $(seq 1 20); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gemma-12b" 20)
            echo "$code" > "${tmpdir}/c${cycle}_gemma_${i}"
        ) &
    done

    # Also send 10 gpt-oss-20b requests to create contention
    for i in $(seq 1 10); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gpt-oss-20b" 10)
            echo "$code" > "${tmpdir}/c${cycle}_gpt_a_${i}"
        ) &
    done

    wait

    # Step 2: Switch back to gpt-oss-20b (gemma goes to L2 sleep)
    echo "  Switching back to gpt-oss-20b with 20 concurrent requests..."
    for i in $(seq 1 20); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gpt-oss-20b" 20)
            echo "$code" > "${tmpdir}/c${cycle}_gpt_b_${i}"
        ) &
    done

    # Contention from gemma requests during switch
    for i in $(seq 1 10); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gemma-12b" 10)
            echo "$code" > "${tmpdir}/c${cycle}_gemma_b_${i}"
        ) &
    done

    wait
    echo "  Cycle ${cycle} complete"
done

# Collect results
failures=0
for cycle in $(seq 1 $CYCLES); do
    for i in $(seq 1 20); do
        code=$(cat "${tmpdir}/c${cycle}_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: cycle ${cycle} gemma request ${i} returned ${code}"
        fi
    done
    for i in $(seq 1 10); do
        code=$(cat "${tmpdir}/c${cycle}_gpt_a_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: cycle ${cycle} gpt_a request ${i} returned ${code}"
        fi
    done
    for i in $(seq 1 20); do
        code=$(cat "${tmpdir}/c${cycle}_gpt_b_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: cycle ${cycle} gpt_b request ${i} returned ${code}"
        fi
    done
    for i in $(seq 1 10); do
        code=$(cat "${tmpdir}/c${cycle}_gemma_b_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: cycle ${cycle} gemma_b request ${i} returned ${code}"
        fi
    done
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "20_l2_wake_stress" "30m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "20_l2_wake_stress" "30m"
echo ""
