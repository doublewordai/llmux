#!/usr/bin/env bash
# 02_single_model_load.sh — 20 concurrent requests to the active model
source "$(dirname "$0")/common.sh"

echo "=== 02: Single Model Load ==="

N=20
echo "Firing ${N} concurrent requests at gemma-12b..."

codes=()
pids=()
tmpdir=$(mktemp -d)

for i in $(seq 1 $N); do
    (send_request "gemma-12b" 10 > "${tmpdir}/${i}") &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

for i in $(seq 1 $N); do
    codes+=("$(cat "${tmpdir}/${i}")")
done

rm -rf "$tmpdir"

assert_all_200 "single_model_load" "${codes[@]}"
collect_logs "02_single_model_load" "2m"
echo ""
