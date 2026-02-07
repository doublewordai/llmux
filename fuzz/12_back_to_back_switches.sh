#!/usr/bin/env bash
# 12_back_to_back_switches.sh — Force many sequential switches as fast as possible
#
# The key insight: what happens when you send a single request to model A,
# then immediately a single request to model B, repeating 20 times?
# Each request should get 200, but the switcher has to handle rapid back-to-back
# switch triggers where the previous switch may not have completed.
source "$(dirname "$0")/common.sh"

echo "=== 12: Back-to-Back Switches (20 sequential switch triggers) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

ROUNDS=20
TOTAL=0

echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Each round: fire 1 request to each model concurrently
# This forces a switch every round
models=("gpt-oss-20b" "gemma-12b")
for round in $(seq 1 $ROUNDS); do
    # Alternate which model goes first
    if (( round % 2 == 1 )); then
        m1="${models[0]}"
        m2="${models[1]}"
    else
        m1="${models[1]}"
        m2="${models[0]}"
    fi

    (
        code=$(send_request "$m1" 5)
        echo "$code" > "${tmpdir}/r${round}_a"
    ) &
    (
        code=$(send_request "$m2" 5)
        echo "$code" > "${tmpdir}/r${round}_b"
    ) &
    TOTAL=$((TOTAL + 2))

    # Don't wait between rounds — fire as fast as possible
done

echo "Fired ${TOTAL} requests across ${ROUNDS} rounds. Waiting..."
wait

# Collect results
failures=0
for round in $(seq 1 $ROUNDS); do
    for suffix in a b; do
        code=$(cat "${tmpdir}/r${round}_${suffix}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: round ${round} ${suffix} returned ${code}"
        fi
    done
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "12_back_to_back_switches" "15m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "12_back_to_back_switches" "15m"
echo ""
