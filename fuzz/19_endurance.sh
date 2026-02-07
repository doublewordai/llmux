#!/usr/bin/env bash
# 19_endurance.sh — Sustained load over many iterations with randomized patterns
#
# Runs 30 iterations of mixed load patterns, checking for any 5xx errors.
# Each iteration fires 20-40 requests with random mix of models, token counts,
# and streaming. Total: ~900 requests over several minutes of sustained activity.
source "$(dirname "$0")/common.sh"

echo "=== 19: Endurance Test (30 iterations, ~900 requests) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Ensure one model is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

ITERATIONS=30
total_requests=0
total_failures=0
models=("gpt-oss-20b" "gemma-12b")

for iter in $(seq 1 $ITERATIONS); do
    iter_dir="${tmpdir}/iter_${iter}"
    mkdir -p "$iter_dir"
    iter_count=0

    # Random pattern for this iteration
    pattern=$((RANDOM % 4))

    case $pattern in
        0)
            # Single model burst
            model="${models[$((RANDOM % 2))]}"
            count=$((20 + RANDOM % 20))
            echo "Iter ${iter}/${ITERATIONS}: burst ${count}× ${model}"
            for i in $(seq 1 $count); do
                iter_count=$((iter_count + 1))
                tokens=$((5 + RANDOM % 50))
                (
                    code=$(send_request "$model" "$tokens")
                    echo "$code" > "${iter_dir}/${i}"
                ) &
            done
            ;;
        1)
            # Mixed model requests (forces switch)
            count=$((10 + RANDOM % 15))
            echo "Iter ${iter}/${ITERATIONS}: mixed ${count}× both models"
            for i in $(seq 1 $count); do
                iter_count=$((iter_count + 1))
                model="${models[$((i % 2))]}"
                tokens=$((5 + RANDOM % 50))
                (
                    code=$(send_request "$model" "$tokens")
                    echo "$code" > "${iter_dir}/${i}"
                ) &
            done
            ;;
        2)
            # Streaming + non-streaming mix
            count=$((10 + RANDOM % 10))
            echo "Iter ${iter}/${ITERATIONS}: streaming mix ${count}× both models"
            for i in $(seq 1 $count); do
                iter_count=$((iter_count + 1))
                model="${models[$((RANDOM % 2))]}"
                stream=$(( RANDOM % 2 == 0 ))
                tokens=$((10 + RANDOM % 100))
                (
                    code=$(send_request "$model" "$tokens" "$([[ $stream -eq 1 ]] && echo true || echo false)")
                    echo "$code" > "${iter_dir}/${i}"
                ) &
            done
            ;;
        3)
            # Rapid alternation burst
            count=$((20 + RANDOM % 20))
            echo "Iter ${iter}/${ITERATIONS}: alternation ${count}×"
            for i in $(seq 1 $count); do
                iter_count=$((iter_count + 1))
                model="${models[$((i % 2))]}"
                (
                    code=$(send_request "$model" 5)
                    echo "$code" > "${iter_dir}/${i}"
                ) &
            done
            ;;
    esac

    wait

    # Check results for this iteration
    iter_failures=0
    for i in $(seq 1 $iter_count); do
        code=$(cat "${iter_dir}/${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            iter_failures=$((iter_failures + 1))
            echo "  FAIL: iter ${iter} request ${i} returned ${code}"
        fi
    done

    total_requests=$((total_requests + iter_count))
    total_failures=$((total_failures + iter_failures))

    if [[ $iter_failures -gt 0 ]]; then
        echo "  Iter ${iter}: ${iter_failures}/${iter_count} FAILED"
    fi
done

echo ""
echo "=== Endurance Results ==="
echo "Total requests: ${total_requests}"
echo "Total failures: ${total_failures}"
echo "Success rate: $(( (total_requests - total_failures) * 100 / total_requests ))%"

if [[ $total_failures -gt 0 ]]; then
    echo "  FAIL: ${total_failures} total failures"
    collect_logs "19_endurance" "30m"
    exit 1
fi

echo "  PASS: all ${total_requests} requests returned 200"
collect_logs "19_endurance" "30m"
echo ""
