#!/usr/bin/env bash
# 01_warmup.sh — Cold start both models, verify basic operation
source "$(dirname "$0")/common.sh"

echo "=== 01: Warmup ==="

echo "Sending request to gpt-oss-20b (cold start, may take ~60s)..."
code1=$(send_request "gpt-oss-20b" 5)
echo "  gpt-oss-20b: ${code1}"

echo "Sending request to gemma-12b (switch + cold start, may take ~120s)..."
code2=$(send_request "gemma-12b" 5)
echo "  gemma-12b: ${code2}"

assert_all_200 "warmup" "$code1" "$code2"
collect_logs "01_warmup" "5m"
echo ""
