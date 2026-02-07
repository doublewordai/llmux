#!/usr/bin/env bash
# 04_rapid_alternation.sh — Rapid model alternation
source "$(dirname "$0")/common.sh"

echo "=== 04: Rapid Alternation ==="

N=10
tmpdir=$(mktemp -d)
pids=()
models=("gpt-oss-20b" "gemma-12b")

echo "Firing ${N} requests, alternating models, as fast as possible..."
for i in $(seq 1 $N); do
    model="${models[$(( (i - 1) % 2 ))]}"
    (send_request "$model" 5 > "${tmpdir}/${i}") &
    pids+=($!)
done

echo "Waiting for all ${N} requests..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

codes=()
for i in $(seq 1 $N); do
    codes+=("$(cat "${tmpdir}/${i}")")
done

rm -rf "$tmpdir"

assert_all_200 "rapid_alternation" "${codes[@]}"
collect_logs "04_rapid_alternation" "5m"
echo ""
