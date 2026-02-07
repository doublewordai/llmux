#!/usr/bin/env bash
# 06_streaming_during_switch.sh — Streaming response survives a model switch
source "$(dirname "$0")/common.sh"

echo "=== 06: Streaming During Switch ==="

tmpdir=$(mktemp -d)
pids=()

# Ensure gpt-oss-20b is active first
echo "Ensuring gpt-oss-20b is active..."
warmup_code=$(send_request "gpt-oss-20b" 5)
if [[ "$warmup_code" != "200" ]]; then
    echo "  FAIL: warmup request returned ${warmup_code}"
    exit 1
fi

# Start a streaming request with lots of tokens to keep it going
echo "Starting streaming request on gpt-oss-20b (max_tokens=500)..."
(
    http_code=$(curl -s -o "${tmpdir}/stream_body" -w '%{http_code}' \
        --max-time "$TIMEOUT" \
        "${BASE_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Write a short story about a robot."}],
            "max_tokens": 500,
            "stream": true
        }')
    echo "$http_code" > "${tmpdir}/stream_code"
) &
stream_pid=$!

# Wait a moment for the stream to start, then trigger a switch
sleep 2

echo "Triggering switch to gemma-12b while stream is active..."
(send_request "gemma-12b" 5 > "${tmpdir}/switch_code") &
switch_pid=$!

echo "Waiting for both to complete..."
wait "$stream_pid"
wait "$switch_pid"

stream_code=$(cat "${tmpdir}/stream_code")
switch_code=$(cat "${tmpdir}/switch_code")

echo "  Streaming request: ${stream_code}"
echo "  Switch request:    ${switch_code}"

rm -rf "$tmpdir"

assert_all_200 "streaming_during_switch" "$stream_code" "$switch_code"
collect_logs "06_streaming_during_switch" "5m"
echo ""
