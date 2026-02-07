#!/usr/bin/env bash
# benchmark.sh — Run standardized workload profiles and collect metrics
#
# Usage: ./benchmark.sh [label]
#   label: identifier for this run (e.g. "fifo", "cost_aware")
#
# Runs 5 workload profiles, collecting Prometheus metrics before and after
# each profile. Produces a report comparing GPU serving fraction, switch
# count, and P95 queue wait across profiles.
source "$(dirname "$0")/common.sh"

LABEL="${1:-unlabeled}"
PROM="http://${HOST}:9090"
REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/benchmark_results"
mkdir -p "$REPORT_DIR"
REPORT="${REPORT_DIR}/${LABEL}_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "  BENCHMARK: ${LABEL}"
echo "=========================================="
echo ""

# Helper: get a Prometheus counter value (returns 0 if no data)
prom_counter() {
    local query="$1"
    curl -s --max-time 10 "${PROM}/api/v1/query" --data-urlencode "query=${query}" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data['data']['result']
total = 0
for r in results:
    total += float(r['value'][1])
print(total)
" 2>/dev/null || echo "0"
}

# Helper: get a Prometheus histogram sum
prom_histogram_sum() {
    local query="$1"
    curl -s --max-time 10 "${PROM}/api/v1/query" --data-urlencode "query=${query}" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data['data']['result']
total = 0
for r in results:
    total += float(r['value'][1])
print(total)
" 2>/dev/null || echo "0"
}

# Snapshot all key metrics
snapshot_metrics() {
    local switches=$(prom_counter 'llmux_switches_total')
    local switch_time=$(prom_histogram_sum 'llmux_switch_total_seconds_sum')
    local requests=$(prom_counter 'llmux_requests_total')
    local queue_wait_sum=$(prom_histogram_sum 'llmux_request_queue_wait_seconds_sum')
    local queue_wait_count=$(prom_counter 'llmux_request_queue_wait_seconds_count')
    local failures=$(prom_counter 'llmux_switch_failures_total')
    echo "${switches}|${switch_time}|${requests}|${queue_wait_sum}|${queue_wait_count}|${failures}"
}

# Parse a metric snapshot
parse_snapshot() {
    echo "$1" | cut -d'|' -f"$2"
}

# Run a workload profile and measure delta
run_profile() {
    local profile_name="$1"
    local profile_fn="$2"

    echo ""
    echo "--- Profile: ${profile_name} ---"

    # Snapshot before
    local before=$(snapshot_metrics)

    # Run the workload
    local start_time=$(date +%s)
    eval "$profile_fn"
    local end_time=$(date +%s)
    local wall_time=$((end_time - start_time))

    # Wait for Prometheus to scrape (15s interval)
    echo "  Waiting for metrics scrape..."
    sleep 18

    # Snapshot after
    local after=$(snapshot_metrics)

    # Compute deltas
    local d_switches=$(python3 -c "print(float($(parse_snapshot "$after" 1)) - float($(parse_snapshot "$before" 1)))")
    local d_switch_time=$(python3 -c "print(float($(parse_snapshot "$after" 2)) - float($(parse_snapshot "$before" 2)))")
    local d_requests=$(python3 -c "print(float($(parse_snapshot "$after" 3)) - float($(parse_snapshot "$before" 3)))")
    local d_wait_sum=$(python3 -c "print(float($(parse_snapshot "$after" 4)) - float($(parse_snapshot "$before" 4)))")
    local d_wait_count=$(python3 -c "print(float($(parse_snapshot "$after" 5)) - float($(parse_snapshot "$before" 5)))")
    local d_failures=$(python3 -c "print(float($(parse_snapshot "$after" 6)) - float($(parse_snapshot "$before" 6)))")

    # Compute derived metrics
    local gpu_serving_frac=$(python3 -c "
wall = $wall_time
switch_time = $d_switch_time
if wall > 0:
    print(f'{(1 - switch_time/wall)*100:.1f}')
else:
    print('N/A')
")
    local avg_wait=$(python3 -c "
count = $d_wait_count
s = $d_wait_sum
if count > 0:
    print(f'{s/count:.2f}')
else:
    print('0.00')
")

    echo "  Wall time:    ${wall_time}s"
    echo "  Switches:     ${d_switches}"
    echo "  Switch time:  ${d_switch_time}s"
    echo "  Requests:     ${d_requests}"
    echo "  GPU serving:  ${gpu_serving_frac}%"
    echo "  Avg wait:     ${avg_wait}s"
    echo "  Failures:     ${d_failures}"

    # Append to JSON report
    python3 -c "
import json, os

entry = {
    'profile': '$profile_name',
    'wall_time_s': $wall_time,
    'switches': $d_switches,
    'switch_time_s': round($d_switch_time, 2),
    'requests': $d_requests,
    'gpu_serving_pct': '$gpu_serving_frac',
    'avg_wait_s': round(float('$avg_wait'), 2),
    'failures': $d_failures,
}

report_path = '$REPORT'
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
else:
    report = {'label': '$LABEL', 'profiles': []}

report['profiles'].append(entry)

with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
"
}

# ============================================================
# WORKLOAD PROFILES
# ============================================================

# Ensure we start with a known state
echo "Warming up both models..."
warmup_a=$(send_request "gpt-oss-20b" 5)
echo "  gpt-oss-20b: ${warmup_a}"
if [[ "$warmup_a" != "200" ]]; then
    echo "FAIL: warmup failed"
    exit 1
fi

# Wait for the switch and serve gemma too
warmup_b=$(send_request "gemma-12b" 5)
echo "  gemma-12b: ${warmup_b}"
if [[ "$warmup_b" != "200" ]]; then
    echo "FAIL: warmup failed"
    exit 1
fi

# Switch back to gpt-oss-20b as starting point
send_request "gpt-oss-20b" 5 > /dev/null

echo "  Warmup complete. Waiting for metrics to settle..."
sleep 20

# Profile 1: Balanced alternation (20 pairs = 40 requests, forced alternation)
profile_balanced() {
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN
    local total=0

    for round in $(seq 1 20); do
        # Send one request to each model, sequentially
        send_request "gpt-oss-20b" 10 > "${tmpdir}/gpt_${round}" &
        local pid_a=$!
        sleep 0.1
        send_request "gemma-12b" 10 > "${tmpdir}/gemma_${round}" &
        local pid_b=$!
        wait $pid_a $pid_b 2>/dev/null
    done

    # Count results
    local ok=0 fail=0
    for f in "$tmpdir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "  Results: ${ok}/$((ok+fail)) OK, ${fail} failed"
}

# Profile 2: Bursty (bursts of 10 for each model, 4 bursts)
profile_bursty() {
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN

    for burst in 1 2 3 4; do
        model=$(if (( burst % 2 == 1 )); then echo "gpt-oss-20b"; else echo "gemma-12b"; fi)
        local pids=()
        for i in $(seq 1 10); do
            (send_request "$model" 15 > "${tmpdir}/b${burst}_${i}") &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    done

    local ok=0 fail=0
    for f in "$tmpdir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "  Results: ${ok}/$((ok+fail)) OK, ${fail} failed"
}

# Profile 3: Dominant + occasional (80% model A, 20% model B, interleaved)
profile_dominant() {
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN
    local pids=()

    for i in $(seq 1 50); do
        if (( i % 5 == 0 )); then
            (send_request "gemma-12b" 10 > "${tmpdir}/req_${i}") &
        else
            (send_request "gpt-oss-20b" 10 > "${tmpdir}/req_${i}") &
        fi
        pids+=($!)
        # Stagger slightly
        sleep 0.05
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done

    local ok=0 fail=0
    for f in "$tmpdir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "  Results: ${ok}/$((ok+fail)) OK, ${fail} failed"
}

# Profile 4: Rapid interleave (simultaneous requests for both models)
profile_interleave() {
    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN
    local pids=()

    for i in $(seq 1 30); do
        (send_request "gpt-oss-20b" 10 > "${tmpdir}/gpt_${i}") &
        pids+=($!)
        (send_request "gemma-12b" 10 > "${tmpdir}/gemma_${i}") &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done

    local ok=0 fail=0
    for f in "$tmpdir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "  Results: ${ok}/$((ok+fail)) OK, ${fail} failed"
}

# Profile 5: Single-model sustained (no switching)
profile_single_model() {
    # Ensure gpt-oss-20b is active first
    send_request "gpt-oss-20b" 5 > /dev/null
    sleep 2

    local tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" RETURN
    local pids=()

    for i in $(seq 1 40); do
        (send_request "gpt-oss-20b" 15 > "${tmpdir}/req_${i}") &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done

    local ok=0 fail=0
    for f in "$tmpdir"/*; do
        code=$(cat "$f" 2>/dev/null)
        if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done
    echo "  Results: ${ok}/$((ok+fail)) OK, ${fail} failed"
}

# ============================================================
# RUN ALL PROFILES
# ============================================================

run_profile "single_model" "profile_single_model"
run_profile "balanced" "profile_balanced"
run_profile "bursty" "profile_bursty"
run_profile "dominant" "profile_dominant"
run_profile "interleave" "profile_interleave"

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=========================================="
echo "  BENCHMARK COMPLETE: ${LABEL}"
echo "=========================================="
echo ""

python3 -c "
import json

with open('$REPORT') as f:
    report = json.load(f)

print(f'Label: {report[\"label\"]}')
print()
print(f'{\"Profile\":<20} {\"Switches\":>10} {\"Switch Time\":>12} {\"GPU Serving\":>12} {\"Avg Wait\":>10} {\"Requests\":>10}')
print('-' * 76)

total_switches = 0
total_switch_time = 0
total_requests = 0
total_wall = 0

for p in report['profiles']:
    total_switches += p['switches']
    total_switch_time += p['switch_time_s']
    total_requests += p['requests']
    total_wall += p['wall_time_s']
    print(f'{p[\"profile\"]:<20} {p[\"switches\"]:>10.0f} {p[\"switch_time_s\"]:>11.1f}s {p[\"gpu_serving_pct\"]:>11}% {p[\"avg_wait_s\"]:>9.1f}s {p[\"requests\"]:>10.0f}')

print('-' * 76)
overall_gpu = (1 - total_switch_time/total_wall)*100 if total_wall > 0 else 0
print(f'{\"TOTAL\":<20} {total_switches:>10.0f} {total_switch_time:>11.1f}s {overall_gpu:>11.1f}% {\"\":>10} {total_requests:>10.0f}')
"

echo ""
echo "Report saved to: ${REPORT}"
collect_logs "benchmark_${LABEL}" "30m"
echo ""
