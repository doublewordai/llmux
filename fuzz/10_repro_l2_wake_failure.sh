#!/usr/bin/env bash
# 10_repro_l2_wake_failure.sh — Reproduce gemma-12b L2 wake 500 error
#
# Strategy: rapidly cycle gemma-12b through L2 sleep/wake by alternating
# requests between gpt-oss-20b and gemma-12b with minimal gaps.
# The third model (qwen3-4b) is used to increase GPU memory pressure.
source "$(dirname "$0")/common.sh"

echo "=== 10: Reproduce L2 Wake Failure ==="
echo ""
echo "This test rapidly cycles gemma-12b through L2 sleep/wake to trigger"
echo "the /wake_up 500 error. Watch model-switcher logs for:"
echo "  - 'vLLM endpoint returned error' with response body"
echo "  - 'Wake step 1/3 FAILED: /wake_up returned error'"
echo "  - vllm target logs showing Python tracebacks"
echo ""

MODELS=("gpt-oss-20b" "gemma-12b")
FAILURES=0
SWITCHES=0

# Phase 1: Warm up both models
echo "--- Phase 1: Warm up ---"
for model in "${MODELS[@]}"; do
    echo -n "  Warming $model... "
    status=$(send_request "$model" 5)
    echo "$status"
done
echo ""

# Phase 2: Rapid two-model cycling (10 round-trips)
echo "--- Phase 2: Rapid cycling (10 round-trips) ---"
for i in $(seq 1 10); do
    for model in "${MODELS[@]}"; do
        echo -n "  Round $i, $model: "
        status=$(send_request "$model" 5)
        SWITCHES=$((SWITCHES + 1))
        if [[ "$status" != "200" ]]; then
            FAILURES=$((FAILURES + 1))
            echo "FAIL ($status)"
            # Grab logs immediately after failure
            collect_logs "10_failure_${FAILURES}" "30s"
        else
            echo "OK"
        fi
    done
done
echo ""
echo "  Two-model result: $FAILURES failures in $SWITCHES switches"
echo ""

# Phase 3: Add qwen3-4b to increase memory pressure
echo "--- Phase 3: Three-model cycling ---"
ALL_MODELS=("gpt-oss-20b" "gemma-12b" "qwen3-4b")
for i in $(seq 1 5); do
    for model in "${ALL_MODELS[@]}"; do
        echo -n "  Round $i, $model: "
        status=$(send_request "$model" 5)
        SWITCHES=$((SWITCHES + 1))
        if [[ "$status" != "200" ]]; then
            FAILURES=$((FAILURES + 1))
            echo "FAIL ($status)"
            collect_logs "10_failure_${FAILURES}" "30s"
        else
            echo "OK"
        fi
    done
done
echo ""

# Phase 4: Hammer gemma-12b specifically — rapid back-and-forth
echo "--- Phase 4: Targeted gemma-12b cycling ---"
for i in $(seq 1 10); do
    echo -n "  Cycle $i: gpt-oss-20b... "
    send_request "gpt-oss-20b" 1 > /dev/null
    echo -n "gemma-12b... "
    status=$(send_request "gemma-12b" 1)
    SWITCHES=$((SWITCHES + 1))
    if [[ "$status" != "200" ]]; then
        FAILURES=$((FAILURES + 1))
        echo "FAIL ($status)"
        collect_logs "10_failure_${FAILURES}" "30s"
    else
        echo "OK"
    fi
done
echo ""

collect_logs "10_repro_l2_wake" "15m"

echo "=== Summary ==="
echo "  Total switches: $SWITCHES"
echo "  Failures:       $FAILURES"
if [[ $FAILURES -gt 0 ]]; then
    echo ""
    echo "  Check logs/10_failure_*.log for vLLM error details"
    echo "  Check logs/10_repro_l2_wake.log for full timeline"
fi
echo "Done."
