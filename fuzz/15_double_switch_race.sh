#!/usr/bin/env bash
# 15_double_switch_race.sh — Try to trigger double-switch race
#
# Scenario: Requests for model A arrive, a switch to A begins.
# While the switch is in progress, requests for model B also arrive.
# We want to make sure the second batch doesn't cause a double switch
# or leave the system in a broken state.
#
# This test deliberately creates a situation where:
# 1. gemma-12b is active
# 2. A burst of gpt-oss-20b requests triggers switch to gpt-oss-20b
# 3. While that switch is happening, a burst of gemma-12b requests arrives
# 4. This should either:
#    a. Queue the gemma requests and serve them after gpt-oss-20b switch completes
#       (then potentially switch back), OR
#    b. Batch them intelligently
# Either way: no 5xx errors.
source "$(dirname "$0")/common.sh"

echo "=== 15: Double Switch Race ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Ensure gemma-12b is active
echo "Warming up gemma-12b..."
warmup_code=$(send_request "gemma-12b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

ROUNDS=5
TOTAL=0

for round in $(seq 1 $ROUNDS); do
    echo "Round ${round}/${ROUNDS}:"

    # Fire a burst at gpt-oss-20b to trigger switch
    for i in $(seq 1 10); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gpt-oss-20b" 30)
            echo "$code" > "${tmpdir}/r${round}_gpt_${i}"
        ) &
    done

    # Tiny delay, then fire a counter-burst at gemma-12b
    sleep 0.2
    for i in $(seq 1 10); do
        TOTAL=$((TOTAL + 1))
        (
            code=$(send_request "gemma-12b" 30)
            echo "$code" > "${tmpdir}/r${round}_gemma_${i}"
        ) &
    done

    # Wait for this round before starting next
    wait
    echo "  Round ${round} complete"
done

# Collect results
failures=0
for round in $(seq 1 $ROUNDS); do
    for i in $(seq 1 10); do
        code=$(cat "${tmpdir}/r${round}_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: round ${round} gpt request ${i} returned ${code}"
        fi
        code=$(cat "${tmpdir}/r${round}_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: round ${round} gemma request ${i} returned ${code}"
        fi
    done
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "15_double_switch_race" "20m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "15_double_switch_race" "20m"
echo ""
