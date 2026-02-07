#!/usr/bin/env bash
# 03_switch_under_load.sh — Trigger a switch while requests are in-flight
source "$(dirname "$0")/common.sh"

echo "=== 03: Switch Under Load ==="

tmpdir=$(mktemp -d)
pids=()

# Fire 5 requests at the active model with higher max_tokens to keep them busy
echo "Firing 5 requests at gemma-12b (active, max_tokens=200)..."
for i in $(seq 1 5); do
    (send_request "gemma-12b" 200 > "${tmpdir}/a_${i}") &
    pids+=($!)
done

# Small delay so those requests are in-flight before the switch triggers
sleep 0.5

# Fire 5 requests at the other model to trigger a switch
echo "Firing 5 requests at gpt-oss-20b (triggers switch)..."
for i in $(seq 1 5); do
    (send_request "gpt-oss-20b" 5 > "${tmpdir}/b_${i}") &
    pids+=($!)
done

echo "Waiting for all 10 requests..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

codes=()
for i in $(seq 1 5); do
    codes+=("$(cat "${tmpdir}/a_${i}")")
done
for i in $(seq 1 5); do
    codes+=("$(cat "${tmpdir}/b_${i}")")
done

rm -rf "$tmpdir"

assert_all_200 "switch_under_load" "${codes[@]}"
collect_logs "03_switch_under_load" "5m"
echo ""
