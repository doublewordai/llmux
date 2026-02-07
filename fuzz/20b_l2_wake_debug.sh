#!/usr/bin/env bash
# 20b_l2_wake_debug.sh — Debug version of L2 wake stress
#
# Runs a single cycle with better error tracking
source "$(dirname "$0")/common.sh"

echo "=== 20b: L2 Wake Debug ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Restart for clean state
echo "Restarting model-switcher..."
$SSH_CMD "cd /home/ubuntu && sudo docker compose restart model-switcher" > /dev/null 2>&1
sleep 10

# Ensure gpt-oss-20b is active (cold start)
echo "Cold starting gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
echo "  gpt-oss-20b warmup: ${warmup_code}"
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

echo ""
echo "=== Cycle 1: Switch to gemma-12b (L2 wake) with concurrent load ==="

# Fire 20 requests at gemma-12b
PIDS_A=()
for i in $(seq 1 20); do
    (
        start_time=$(date +%s)
        code=$(send_request "gemma-12b" 20)
        end_time=$(date +%s)
        echo "${code}" > "${tmpdir}/gemma_${i}"
        echo "  gemma_${i}: HTTP ${code} (took $((end_time - start_time))s)" >&2
    ) &
    PIDS_A+=($!)
done

# Fire 10 gpt-oss-20b requests for contention
for i in $(seq 1 10); do
    (
        start_time=$(date +%s)
        code=$(send_request "gpt-oss-20b" 10)
        end_time=$(date +%s)
        echo "${code}" > "${tmpdir}/gpt_a_${i}"
        echo "  gpt_a_${i}: HTTP ${code} (took $((end_time - start_time))s)" >&2
    ) &
    PIDS_A+=($!)
done

echo "Waiting for ${#PIDS_A[@]} requests..."
for pid in "${PIDS_A[@]}"; do
    if ! wait "$pid" 2>/dev/null; then
        echo "  WARNING: PID $pid exited non-zero"
    fi
done
echo "All requests from step 1 complete."

echo ""
echo "=== Cycle 1 step 2: Switch back to gpt-oss-20b ==="

PIDS_B=()
for i in $(seq 1 20); do
    (
        start_time=$(date +%s)
        code=$(send_request "gpt-oss-20b" 20)
        end_time=$(date +%s)
        echo "${code}" > "${tmpdir}/gpt_b_${i}"
        echo "  gpt_b_${i}: HTTP ${code} (took $((end_time - start_time))s)" >&2
    ) &
    PIDS_B+=($!)
done

for i in $(seq 1 10); do
    (
        start_time=$(date +%s)
        code=$(send_request "gemma-12b" 10)
        end_time=$(date +%s)
        echo "${code}" > "${tmpdir}/gemma_b_${i}"
        echo "  gemma_b_${i}: HTTP ${code} (took $((end_time - start_time))s)" >&2
    ) &
    PIDS_B+=($!)
done

echo "Waiting for ${#PIDS_B[@]} requests..."
for pid in "${PIDS_B[@]}"; do
    if ! wait "$pid" 2>/dev/null; then
        echo "  WARNING: PID $pid exited non-zero"
    fi
done
echo "All requests from step 2 complete."

echo ""
echo "=== Results ==="

failures=0
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/gemma_${i}" 2>/dev/null)
    if [[ -z "$code" ]]; then
        echo "  EMPTY: gemma_${i} (file empty or missing)"
        failures=$((failures + 1))
    elif [[ "$code" != "200" ]]; then
        echo "  FAIL: gemma_${i} returned ${code}"
        failures=$((failures + 1))
    fi
done
for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/gpt_a_${i}" 2>/dev/null)
    if [[ -z "$code" ]]; then
        echo "  EMPTY: gpt_a_${i} (file empty or missing)"
        failures=$((failures + 1))
    elif [[ "$code" != "200" ]]; then
        echo "  FAIL: gpt_a_${i} returned ${code}"
        failures=$((failures + 1))
    fi
done
for i in $(seq 1 20); do
    code=$(cat "${tmpdir}/gpt_b_${i}" 2>/dev/null)
    if [[ -z "$code" ]]; then
        echo "  EMPTY: gpt_b_${i} (file empty or missing)"
        failures=$((failures + 1))
    elif [[ "$code" != "200" ]]; then
        echo "  FAIL: gpt_b_${i} returned ${code}"
        failures=$((failures + 1))
    fi
done
for i in $(seq 1 10); do
    code=$(cat "${tmpdir}/gemma_b_${i}" 2>/dev/null)
    if [[ -z "$code" ]]; then
        echo "  EMPTY: gemma_b_${i} (file empty or missing)"
        failures=$((failures + 1))
    elif [[ "$code" != "200" ]]; then
        echo "  FAIL: gemma_b_${i} returned ${code}"
        failures=$((failures + 1))
    fi
done

TOTAL=60
echo ""
echo "Results: $((TOTAL - failures))/${TOTAL} returned 200"
collect_logs "20b_l2_wake_debug" "15m"

if [[ $failures -gt 0 ]]; then
    echo "  FAIL"
    exit 1
fi

echo "  PASS"
echo ""
