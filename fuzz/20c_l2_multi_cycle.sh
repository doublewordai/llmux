#!/usr/bin/env bash
# 20c_l2_multi_cycle.sh — Multi-cycle L2 wake stress with serial cycle execution
#
# Same as test 20 but processes one cycle at a time to avoid bash PID limits
source "$(dirname "$0")/common.sh"

echo "=== 20c: L2 Multi-Cycle Stress (5 cycles, serial) ==="

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Start with gpt-oss-20b active
echo "Warming up gpt-oss-20b..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup returned ${warmup_code}"
    exit 1
fi

CYCLES=5
total_ok=0
total_fail=0

for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "Cycle ${cycle}/${CYCLES}: Switch to gemma-12b (L2 wake) + back"
    cycle_dir="${tmpdir}/c${cycle}"
    mkdir -p "$cycle_dir"

    # Step 1: Switch to gemma-12b with concurrent load
    pids=()
    for i in $(seq 1 20); do
        (send_request "gemma-12b" 20 > "${cycle_dir}/gemma_${i}") &
        pids+=($!)
    done
    for i in $(seq 1 10); do
        (send_request "gpt-oss-20b" 10 > "${cycle_dir}/gpt_a_${i}") &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null
    done

    # Step 2: Switch back to gpt-oss-20b
    pids=()
    for i in $(seq 1 20); do
        (send_request "gpt-oss-20b" 20 > "${cycle_dir}/gpt_b_${i}") &
        pids+=($!)
    done
    for i in $(seq 1 10); do
        (send_request "gemma-12b" 10 > "${cycle_dir}/gemma_b_${i}") &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null
    done

    # Count results for this cycle
    cycle_ok=0
    cycle_fail=0
    for f in "$cycle_dir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then
            cycle_ok=$((cycle_ok + 1))
        else
            cycle_fail=$((cycle_fail + 1))
            echo "  FAIL: $(basename $f) returned '${code}'"
        fi
    done

    total_ok=$((total_ok + cycle_ok))
    total_fail=$((total_fail + cycle_fail))
    echo "  Cycle ${cycle}: ${cycle_ok}/60 OK, ${cycle_fail} failed"
done

TOTAL=$((total_ok + total_fail))
echo ""
echo "=== Total: ${total_ok}/${TOTAL} returned 200, ${total_fail} failed ==="

if [[ $total_fail -gt 0 ]]; then
    collect_logs "20c_l2_multi_cycle" "30m"
    exit 1
fi

echo "  PASS: all ${TOTAL} requests returned 200"
collect_logs "20c_l2_multi_cycle" "30m"
echo ""
