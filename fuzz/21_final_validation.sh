#!/usr/bin/env bash
# 21_final_validation.sh — Final comprehensive validation
#
# Runs multiple stress patterns back-to-back without restarts.
# This simulates sustained real-world usage over a long period.
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "  FINAL VALIDATION - Comprehensive Test"
echo "=========================================="
echo ""

total_ok=0
total_fail=0
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

run_batch() {
    local label="$1"
    local model="$2"
    local count="$3"
    local max_tokens="${4:-10}"
    local stream="${5:-false}"
    local dir="${tmpdir}/${label}"
    mkdir -p "$dir"

    local pids=()
    for i in $(seq 1 "$count"); do
        (send_request "$model" "$max_tokens" "$stream" > "${dir}/${i}") &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null
    done

    local ok=0
    local fail=0
    for i in $(seq 1 "$count"); do
        code=$(cat "${dir}/${i}" 2>/dev/null)
        if [[ "$code" == "200" ]]; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done
    total_ok=$((total_ok + ok))
    total_fail=$((total_fail + fail))
    echo "    ${label}: ${ok}/${count} OK" $(if [[ $fail -gt 0 ]]; then echo "(${fail} FAILED)"; fi)
}

# Phase 1: Warmup
echo "Phase 1: Warmup"
run_batch "warmup_gpt" "gpt-oss-20b" 1 5
run_batch "warmup_gemma" "gemma-12b" 1 5

# Phase 2: Heavy single-model load
echo "Phase 2: Single-model heavy load (100 each)"
run_batch "heavy_gpt" "gpt-oss-20b" 100 50
run_batch "heavy_gemma" "gemma-12b" 100 50

# Phase 3: Mixed concurrent load
echo "Phase 3: Mixed concurrent (50 each, simultaneous)"
mkdir -p "${tmpdir}/mixed"
pids=()
for i in $(seq 1 50); do
    (send_request "gpt-oss-20b" 20 > "${tmpdir}/mixed/gpt_${i}") &
    pids+=($!)
    (send_request "gemma-12b" 20 > "${tmpdir}/mixed/gemma_${i}") &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
done
ok=0; fail=0
for i in $(seq 1 50); do
    for prefix in gpt gemma; do
        code=$(cat "${tmpdir}/mixed/${prefix}_${i}" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
done
total_ok=$((total_ok + ok)); total_fail=$((total_fail + fail))
echo "    mixed: ${ok}/100 OK" $(if [[ $fail -gt 0 ]]; then echo "(${fail} FAILED)"; fi)

# Phase 4: Streaming stress
echo "Phase 4: Streaming (30 each, high tokens)"
run_batch "stream_gpt" "gpt-oss-20b" 30 200 true
run_batch "stream_gemma" "gemma-12b" 30 200 true

# Phase 5: Rapid alternation
echo "Phase 5: Rapid alternation (40 requests)"
mkdir -p "${tmpdir}/alt"
pids=()
for i in $(seq 1 40); do
    model=$(if (( i % 2 == 0 )); then echo "gpt-oss-20b"; else echo "gemma-12b"; fi)
    (send_request "$model" 5 > "${tmpdir}/alt/${i}") &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
done
ok=0; fail=0
for i in $(seq 1 40); do
    code=$(cat "${tmpdir}/alt/${i}" 2>/dev/null)
    if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
done
total_ok=$((total_ok + ok)); total_fail=$((total_fail + fail))
echo "    alternation: ${ok}/40 OK" $(if [[ $fail -gt 0 ]]; then echo "(${fail} FAILED)"; fi)

# Phase 6: L2 wake cycles (3 cycles)
echo "Phase 6: L2 wake cycles (3 cycles × 40 requests)"
for cycle in 1 2 3; do
    cycle_dir="${tmpdir}/l2_c${cycle}"
    mkdir -p "$cycle_dir"
    pids=()
    for i in $(seq 1 20); do
        (send_request "gemma-12b" 15 > "${cycle_dir}/gemma_${i}") &
        pids+=($!)
        (send_request "gpt-oss-20b" 15 > "${cycle_dir}/gpt_${i}") &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null
    done
    ok=0; fail=0
    for f in "$cycle_dir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    total_ok=$((total_ok + ok)); total_fail=$((total_fail + fail))
    echo "    l2_cycle_${cycle}: ${ok}/40 OK" $(if [[ $fail -gt 0 ]]; then echo "(${fail} FAILED)"; fi)
done

# Phase 7: Final burst (200 requests, mixed)
echo "Phase 7: Final burst (200 requests, mixed models and patterns)"
mkdir -p "${tmpdir}/final"
pids=()
for i in $(seq 1 200); do
    model=$(if (( RANDOM % 2 == 0 )); then echo "gpt-oss-20b"; else echo "gemma-12b"; fi)
    tokens=$((5 + RANDOM % 50))
    stream=$(if (( RANDOM % 5 == 0 )); then echo "true"; else echo "false"; fi)
    (send_request "$model" "$tokens" "$stream" > "${tmpdir}/final/${i}") &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
done
ok=0; fail=0
for i in $(seq 1 200); do
    code=$(cat "${tmpdir}/final/${i}" 2>/dev/null)
    if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
done
total_ok=$((total_ok + ok)); total_fail=$((total_fail + fail))
echo "    final_burst: ${ok}/200 OK" $(if [[ $fail -gt 0 ]]; then echo "(${fail} FAILED)"; fi)

# Summary
TOTAL=$((total_ok + total_fail))
echo ""
echo "=========================================="
echo "  TOTAL: ${total_ok}/${TOTAL} returned 200"
echo "  FAILURES: ${total_fail}"
echo "=========================================="

collect_logs "21_final_validation" "30m"

if [[ $total_fail -gt 0 ]]; then
    echo "  FAIL"
    exit 1
fi

echo "  PASS: Zero errors across all phases"
echo ""
