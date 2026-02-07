#!/usr/bin/env bash
# Common helpers for fuzz tests

set -euo pipefail

HOST="${LLMUX_HOST:?Set LLMUX_HOST to the server IP}"
PORT="${LLMUX_PORT:-3000}"
BASE_URL="http://${HOST}:${PORT}"
API_KEY="${LLMUX_API_KEY:?Set LLMUX_API_KEY}"
SSH_KEY="${LLMUX_SSH_KEY:?Set LLMUX_SSH_KEY to the path of the SSH private key}"
SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ubuntu@${HOST}"

# Request timeout — generous, because switches involve vLLM cold starts.
# The point is that the *server* never errors; the client controls the timeout.
TIMEOUT=300

LOGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOGS_DIR"

# Send a chat completion request. Writes HTTP status code to stdout.
# Args: model, max_tokens, [stream]
send_request() {
    local model="$1"
    local max_tokens="${2:-5}"
    local stream="${3:-false}"

    curl -s -o /dev/null -w '%{http_code}' \
        --max-time "$TIMEOUT" \
        "${BASE_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
            \"max_tokens\": ${max_tokens},
            \"stream\": ${stream}
        }"
}

# Send a request and return full body (for inspecting responses).
send_request_body() {
    local model="$1"
    local max_tokens="${2:-5}"
    local stream="${3:-false}"

    curl -s --max-time "$TIMEOUT" \
        "${BASE_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
            \"max_tokens\": ${max_tokens},
            \"stream\": ${stream}
        }"
}

# Collect logs from the model-switcher container since the given timestamp.
collect_logs() {
    local label="$1"
    local since="${2:-5m}"
    $SSH_CMD "sudo docker compose logs model-switcher --since=${since} 2>&1" \
        > "${LOGS_DIR}/${label}.log"
    echo "  Logs saved to ${LOGS_DIR}/${label}.log"
}

# Check that all status codes in an array are 200.
# Args: test_name, status_code_1, status_code_2, ...
assert_all_200() {
    local test_name="$1"
    shift
    local codes=("$@")
    local failures=0
    local total=${#codes[@]}

    for i in "${!codes[@]}"; do
        if [[ "${codes[$i]}" != "200" ]]; then
            echo "  FAIL: request $((i+1)) returned ${codes[$i]} (expected 200)"
            failures=$((failures + 1))
        fi
    done

    if [[ $failures -eq 0 ]]; then
        echo "  PASS: all ${total} requests returned 200"
    else
        echo "  FAIL: ${failures}/${total} requests did not return 200"
        return 1
    fi
}
