test:
    cargo test

lint:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings

fmt:
    cargo fmt

check: lint test

# Install dependencies for CRIU checkpoint/restore on a GPU machine.
# Requires: Ubuntu, NVIDIA GPU with driver 570+, sudo access.
# Use `just setup check` to verify without installing.
setup mode="install":
    #!/usr/bin/env bash
    set -euo pipefail

    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m'

    ok()   { echo -e "${GREEN}✓${NC} $1"; }
    fail() { echo -e "${RED}✗${NC} $1"; }
    warn() { echo -e "${YELLOW}!${NC} $1"; }

    CHECK={{ if mode == "check" { "true" } else { "false" } }}
    ERRORS=0

    # --- NVIDIA driver ---
    if nvidia-smi &>/dev/null; then
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        ok "NVIDIA driver $DRIVER"
    else
        fail "NVIDIA driver not found"
        ERRORS=$((ERRORS + 1))
    fi

    # --- Podman ---
    if command -v podman &>/dev/null; then
        ok "podman $(podman --version | awk '{print $NF}')"
    elif [ "$CHECK" = "true" ]; then
        fail "podman not installed"
        ERRORS=$((ERRORS + 1))
    else
        warn "Installing podman..."
        sudo apt-get update -qq && sudo apt-get install -y -qq podman
        ok "podman installed"
    fi

    # --- CRIU ---
    if command -v criu &>/dev/null; then
        ok "criu $(criu --version | head -1 | awk '{print $NF}')"
    elif [ "$CHECK" = "true" ]; then
        fail "criu not installed"
        ERRORS=$((ERRORS + 1))
    else
        warn "Installing CRIU from PPA..."
        sudo add-apt-repository -y ppa:criu/ppa 2>/dev/null
        sudo apt-get update -qq && sudo apt-get install -y -qq criu
        ok "criu installed"
    fi

    # --- CRIU default config (link-remap) ---
    if [ -f /etc/criu/default.conf ] && grep -q "link-remap" /etc/criu/default.conf; then
        ok "CRIU link-remap enabled"
    elif [ "$CHECK" = "true" ]; then
        fail "CRIU link-remap not configured (/etc/criu/default.conf)"
        ERRORS=$((ERRORS + 1))
    else
        warn "Configuring CRIU link-remap..."
        sudo mkdir -p /etc/criu
        echo "link-remap" | sudo tee /etc/criu/default.conf >/dev/null
        ok "CRIU link-remap configured"
    fi

    # --- nvidia-container-toolkit ---
    if dpkg -l nvidia-container-toolkit &>/dev/null; then
        ok "nvidia-container-toolkit installed"
    elif [ "$CHECK" = "true" ]; then
        fail "nvidia-container-toolkit not installed"
        ERRORS=$((ERRORS + 1))
    else
        warn "Installing nvidia-container-toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
        ok "nvidia-container-toolkit installed"
    fi

    # --- CDI spec ---
    if [ -f /etc/cdi/nvidia.yaml ]; then
        ok "NVIDIA CDI spec configured"
    elif [ "$CHECK" = "true" ]; then
        fail "NVIDIA CDI spec not generated (/etc/cdi/nvidia.yaml)"
        ERRORS=$((ERRORS + 1))
    else
        warn "Generating NVIDIA CDI spec..."
        sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml 2>/dev/null
        ok "NVIDIA CDI spec generated"
    fi

    # --- cuda-checkpoint ---
    if command -v cuda-checkpoint &>/dev/null; then
        ok "cuda-checkpoint found"
    else
        fail "cuda-checkpoint not found in PATH (ships with NVIDIA driver 570+)"
        ERRORS=$((ERRORS + 1))
    fi

    # --- Podman GPU test ---
    if [ "$CHECK" = "true" ] || command -v podman &>/dev/null; then
        if sudo podman run --rm --privileged --device nvidia.com/gpu=0 docker.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
            ok "podman GPU access works"
        else
            fail "podman GPU access failed"
            ERRORS=$((ERRORS + 1))
        fi
    fi

    echo ""
    if [ "$ERRORS" -gt 0 ]; then
        fail "$ERRORS issue(s) found"
        exit 1
    else
        ok "All checks passed"
    fi
