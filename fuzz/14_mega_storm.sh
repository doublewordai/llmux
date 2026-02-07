#!/usr/bin/env bash
# 14_mega_storm.sh — The ultimate stress test
#
# Combines everything: high concurrency, streaming, rapid switching, mid-switch
# bursts, and sustained load over multiple phases. 500+ total requests.
source "$(dirname "$0")/common.sh"

echo "=== 14: Mega Storm (500+ requests, mixed patterns) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT
counter=0

# Ensure gpt-oss-20b is active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

# Phase 1: Sustained load on one model (50 requests)
echo "Phase 1: 50 concurrent requests to gpt-oss-20b..."
for i in $(seq 1 50); do
    counter=$((counter + 1))
    (
        code=$(send_request "gpt-oss-20b" 20)
        echo "$code" > "${tmpdir}/p1_${i}"
    ) &
done

# Phase 2: While phase 1 is running, trigger switch with 50 gemma requests
sleep 0.5
echo "Phase 2: 50 concurrent requests to gemma-12b (triggers switch)..."
for i in $(seq 1 50); do
    counter=$((counter + 1))
    (
        code=$(send_request "gemma-12b" 20)
        echo "$code" > "${tmpdir}/p2_${i}"
    ) &
done

# Phase 3: 20 streaming requests to both models simultaneously
sleep 1
echo "Phase 3: 20 streaming requests (10 per model)..."
for i in $(seq 1 10); do
    counter=$((counter + 1))
    (
        code=$(send_request "gpt-oss-20b" 100 true)
        echo "$code" > "${tmpdir}/p3_gpt_${i}"
    ) &
    counter=$((counter + 1))
    (
        code=$(send_request "gemma-12b" 100 true)
        echo "$code" > "${tmpdir}/p3_gemma_${i}"
    ) &
done

# Wait for phases 1-3 to complete
echo "Waiting for phases 1-3..."
wait

# Phase 4: Rapid-fire alternating requests (100 total, alternating models)
echo "Phase 4: 100 rapid alternating requests..."
for i in $(seq 1 50); do
    counter=$((counter + 1))
    (
        code=$(send_request "gpt-oss-20b" 5)
        echo "$code" > "${tmpdir}/p4_gpt_${i}"
    ) &
    counter=$((counter + 1))
    (
        code=$(send_request "gemma-12b" 5)
        echo "$code" > "${tmpdir}/p4_gemma_${i}"
    ) &
done

# Phase 5: While phase 4 is running, add 50 streaming requests
sleep 0.2
echo "Phase 5: 50 streaming requests (25 per model)..."
for i in $(seq 1 25); do
    counter=$((counter + 1))
    (
        code=$(send_request "gpt-oss-20b" 200 true)
        echo "$code" > "${tmpdir}/p5_gpt_${i}"
    ) &
    counter=$((counter + 1))
    (
        code=$(send_request "gemma-12b" 200 true)
        echo "$code" > "${tmpdir}/p5_gemma_${i}"
    ) &
done

# Phase 6: One more burst of 100 short requests
sleep 1
echo "Phase 6: 100 short requests (50 per model)..."
for i in $(seq 1 50); do
    counter=$((counter + 1))
    (
        code=$(send_request "gpt-oss-20b" 5)
        echo "$code" > "${tmpdir}/p6_gpt_${i}"
    ) &
    counter=$((counter + 1))
    (
        code=$(send_request "gemma-12b" 5)
        echo "$code" > "${tmpdir}/p6_gemma_${i}"
    ) &
done

TOTAL=$counter
echo "Total requests fired: ${TOTAL}"
echo "Waiting for all requests to complete..."
wait

# Collect results
failures=0
non200_codes=()

collect_phase() {
    local prefix="$1"
    local count="$2"
    for i in $(seq 1 "$count"); do
        code=$(cat "${tmpdir}/${prefix}_${i}" 2>/dev/null || echo "TIMEOUT")
        if [[ "$code" != "200" ]]; then
            failures=$((failures + 1))
            non200_codes+=("${prefix}_${i}:${code}")
        fi
    done
}

collect_phase "p1" 50
collect_phase "p2" 50
for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/p3_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p3_gpt_${i}:${code}")
    fi
    code=$(cat "${tmpdir}/p3_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p3_gemma_${i}:${code}")
    fi
done
for i in $(seq 1 50); do
    code=$(cat "${tmpdir}/p4_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p4_gpt_${i}:${code}")
    fi
    code=$(cat "${tmpdir}/p4_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p4_gemma_${i}:${code}")
    fi
done
for i in $(seq 1 25); do
    code=$(cat "${tmpdir}/p5_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p5_gpt_${i}:${code}")
    fi
    code=$(cat "${tmpdir}/p5_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p5_gemma_${i}:${code}")
    fi
done
for i in $(seq 1 50); do
    code=$(cat "${tmpdir}/p6_gpt_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p6_gpt_${i}:${code}")
    fi
    code=$(cat "${tmpdir}/p6_gemma_${i}" 2>/dev/null || echo "TIMEOUT")
    if [[ "$code" != "200" ]]; then
        failures=$((failures + 1))
        non200_codes+=("p6_gemma_${i}:${code}")
    fi
done

echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL: ${failures} requests did not return 200"
    echo "  Non-200 details:"
    for entry in "${non200_codes[@]}"; do
        echo "    ${entry}"
    done
    collect_logs "14_mega_storm" "20m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "14_mega_storm" "20m"
echo ""
