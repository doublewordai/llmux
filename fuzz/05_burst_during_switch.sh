#!/usr/bin/env bash
# 05_burst_during_switch.sh — Burst of mixed requests mid-switch
source "$(dirname "$0")/common.sh"

echo "=== 05: Burst During Switch ==="

# Figure out which model is currently active by sending a probe.
# (After test 04, it could be either one.)
echo "Probing active model..."
probe_body=$(send_request_body "gpt-oss-20b" 5)
echo "  gpt-oss-20b responded, it's now active"

# Now trigger a switch and immediately burst mixed requests
tmpdir=$(mktemp -d)
pids=()

echo "Firing 1 request at gemma-12b to trigger switch..."
(send_request "gemma-12b" 5 > "${tmpdir}/trigger") &
pids+=($!)

# Tiny delay — just enough that the switch is initiated
sleep 0.1

echo "Firing 10 mixed requests (5 gpt-oss-20b + 5 gemma-12b)..."
for i in $(seq 1 5); do
    (send_request "gpt-oss-20b" 5 > "${tmpdir}/a_${i}") &
    pids+=($!)
    (send_request "gemma-12b" 5 > "${tmpdir}/b_${i}") &
    pids+=($!)
done

echo "Waiting for all 11 requests..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

codes=()
codes+=("$(cat "${tmpdir}/trigger")")
for i in $(seq 1 5); do
    codes+=("$(cat "${tmpdir}/a_${i}")")
    codes+=("$(cat "${tmpdir}/b_${i}")")
done

rm -rf "$tmpdir"

assert_all_200 "burst_during_switch" "${codes[@]}"
collect_logs "05_burst_during_switch" "5m"
echo ""
