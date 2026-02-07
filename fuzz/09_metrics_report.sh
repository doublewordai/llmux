#!/usr/bin/env bash
# 09_metrics_report.sh — Query Prometheus for llmux metrics and print efficiency report
source "$(dirname "$0")/common.sh"

PROM="http://${HOST}:9090"

echo "=== 09: Metrics Report ==="
echo ""

# Helper: query a single instant value from Prometheus
prom_query() {
    local query="$1"
    curl -s --max-time 10 "${PROM}/api/v1/query" --data-urlencode "query=${query}" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data['status'] != 'success':
    print('ERROR: ' + data.get('error', 'unknown'))
    sys.exit(1)
results = data['data']['result']
if not results:
    print('(no data)')
else:
    for r in results:
        labels = ', '.join(f'{k}={v}' for k, v in r['metric'].items() if k != '__name__')
        val = r['value'][1]
        if labels:
            print(f'  {labels}: {val}')
        else:
            print(f'  {val}')
" 2>/dev/null
}

# Helper: query a single scalar value
prom_scalar() {
    local query="$1"
    curl -s --max-time 10 "${PROM}/api/v1/query" --data-urlencode "query=${query}" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data['data']['result']
if results:
    print(results[0]['value'][1])
else:
    print('')
" 2>/dev/null
}

# Check metrics endpoint is reachable
echo "Checking metrics endpoint..."
if ! curl -s --max-time 5 "${PROM}/api/v1/query?query=up" > /dev/null 2>&1; then
    echo "  FAIL: Prometheus not reachable at ${PROM}"
    exit 1
fi
echo "  OK"
echo ""

# 1. Switch counts
echo "--- Switches ---"
echo "Total switches:"
prom_query 'llmux_switches_total'
echo ""
echo "Failed switches:"
prom_query 'llmux_switch_failures_total'
echo ""

# 2. GPU efficiency (headline number)
echo "--- GPU Efficiency (5m window) ---"
efficiency=$(prom_scalar '1 - rate(llmux_switch_total_seconds_sum[5m])')
if [[ -n "$efficiency" && "$efficiency" != "" ]]; then
    pct=$(python3 -c "print(f'{float(\"${efficiency}\") * 100:.1f}%')" 2>/dev/null)
    echo "  Serving fraction: ${pct} (${efficiency})"
else
    echo "  (no data — need at least 5m of switch activity)"
fi
echo ""

# 3. Switch phase breakdown
echo "--- Switch Phase Breakdown (rate over 5m) ---"
prom_query 'rate(llmux_switch_phase_seconds_sum[5m])'
echo ""

# 4. P95 switch duration
echo "--- P95 Switch Duration (5m) ---"
prom_query 'histogram_quantile(0.95, rate(llmux_switch_total_seconds_bucket[5m]))'
echo ""

# 5. P95 request queue wait
echo "--- P95 Request Queue Wait (5m) ---"
prom_query 'histogram_quantile(0.95, rate(llmux_request_queue_wait_seconds_bucket[5m]))'
echo ""

# 6. Request counts
echo "--- Requests ---"
echo "Total requests:"
prom_query 'llmux_requests_total'
echo ""

# 7. Active model
echo "--- Active Model ---"
prom_query 'llmux_active_model_info == 1'
echo ""

# 8. In-flight
echo "--- In-Flight ---"
prom_query 'llmux_in_flight'
echo ""

# Verify key metrics exist
echo "--- Verification ---"
missing=0
for metric in llmux_switches_total llmux_switch_total_seconds_sum llmux_request_queue_wait_seconds_sum llmux_requests_total; do
    result=$(prom_scalar "${metric}")
    if [[ -z "$result" || "$result" == "(no data)" ]]; then
        echo "  WARN: ${metric} has no data (run workload tests first)"
    else
        echo "  OK: ${metric}"
    fi
done
echo ""

collect_logs "09_metrics_report" "1m"
echo "Done."
