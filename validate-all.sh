#!/usr/bin/env bash
#
# Validate all models in config.6x4090.json
#
# Runs sleep/wake validation for each model at both L1 and L2, recording
# which levels work. Run this before launching the full llmux server.
#
# Usage:
#   ./validate-all.sh [--config CONFIG] [--binary LLMUX_BINARY]
#
# Each model is validated independently (start → baseline → sleep/wake → stop).
# Expect this to take 30-60 minutes for all 6 models due to cold starts.

set -euo pipefail

CONFIG="${1:---config}"
if [[ "$CONFIG" == "--config" ]]; then
    CONFIG="config.6x4090.json"
    shift 2>/dev/null || true
else
    shift
fi

LLMUX="${LLMUX_BINARY:-cargo run --release --}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/$CONFIG"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

# Models and the levels to validate.
# All models test L1 and L2. The validation output will show which pass.
# gpt-oss-120b L2 is expected to fail (MXFP4 weight loader bug).
MODELS=(
    "qwq-32b"
    "qwen3-32b"
    "deepseek-r1-70b"
    "llama-3.3-70b"
    "qwen2.5-72b"
    "gpt-oss-120b"
)

PASSED=()
FAILED=()

echo "============================================"
echo "  llmux validation: 6x RTX 4090 config"
echo "  Config: $CONFIG"
echo "  Models: ${#MODELS[@]}"
echo "============================================"
echo ""

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Validating: $model (L1, L2)"
    echo "============================================"
    echo ""

    if $LLMUX --config "$CONFIG_PATH" --validate "$model" --levels 1,2 --verbose; then
        PASSED+=("$model")
        echo ""
        echo ">>> $model: PASSED"
    else
        FAILED+=("$model")
        echo ""
        echo ">>> $model: FAILED (or partially failed — check output above)"
    fi

    echo ""
    echo "--------------------------------------------"
done

echo ""
echo "============================================"
echo "  VALIDATION SUMMARY"
echo "============================================"
echo ""
echo "Passed (${#PASSED[@]}/${#MODELS[@]}):"
for m in "${PASSED[@]}"; do
    echo "  OK   $m"
done
echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed (${#FAILED[@]}/${#MODELS[@]}):"
    for m in "${FAILED[@]}"; do
        echo "  FAIL $m"
    done
    echo ""
    echo "Review failures above. Common fixes:"
    echo "  - L2 fail on MXFP4 models (gpt-oss): set sleep_level to 1"
    echo "  - OOM on large models: reduce --max-model-len"
    echo "  - Timeout on cold start: model download still in progress"
    echo ""
    exit 1
else
    echo "All models passed. Ready to launch:"
    echo ""
    echo "  $LLMUX --config $CONFIG"
    echo ""
fi
