#!/usr/bin/env bash
# 08_rapid_switch_storm.sh — Waves of requests forcing many switches
#
# Fires 10 waves of 20 requests each (10 per model), with each wave
# staggered by 3 seconds. This forces the switcher to alternate between
# models repeatedly under sustained load.
source "$(dirname "$0")/common.sh"

echo "=== 08: Rapid Switch Storm (10 waves × 20 requests) ==="

WAVES=10
PER_MODEL_PER_WAVE=10
TOTAL=$((WAVES * PER_MODEL_PER_WAVE * 2))

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Warm up
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

models=("gpt-oss-20b" "gemma-12b")

for wave in $(seq 1 $WAVES); do
    # Alternate which model gets the burst first each wave
    if (( wave % 2 == 1 )); then
        primary="${models[0]}"
        secondary="${models[1]}"
    else
        primary="${models[1]}"
        secondary="${models[0]}"
    fi

    echo "Wave ${wave}/${WAVES}: ${PER_MODEL_PER_WAVE}×${primary} + ${PER_MODEL_PER_WAVE}×${secondary}"

    for i in $(seq 1 $PER_MODEL_PER_WAVE); do
        (
            code=$(send_request "$primary" 10)
            echo "$code" > "${tmpdir}/w${wave}_a_${i}"
        ) &
    done
    for i in $(seq 1 $PER_MODEL_PER_WAVE); do
        (
            code=$(send_request "$secondary" 10)
            echo "$code" > "${tmpdir}/w${wave}_b_${i}"
        ) &
    done

    # Stagger waves by 3 seconds to create overlapping switch pressure
    if [[ $wave -lt $WAVES ]]; then
        sleep 3
    fi
done

echo "Waiting for all ${TOTAL} requests to complete..."
wait

# Collect results
failures=0
for wave in $(seq 1 $WAVES); do
    for i in $(seq 1 $PER_MODEL_PER_WAVE); do
        code=$(cat "${tmpdir}/w${wave}_a_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: wave ${wave} primary request ${i} returned ${code}"
        fi
    done
    for i in $(seq 1 $PER_MODEL_PER_WAVE); do
        code=$(cat "${tmpdir}/w${wave}_b_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            echo "  FAIL: wave ${wave} secondary request ${i} returned ${code}"
        fi
    done
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    collect_logs "08_rapid_switch_storm" "15m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "08_rapid_switch_storm" "15m"
echo ""
