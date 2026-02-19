# Multi-stage build for llmux with vLLM
#
# This Dockerfile builds the llmux binary and packages it with vLLM,
# enabling zero-reload model switching for multiple models on shared GPU.
#
# Includes cuda-checkpoint and CRIU with CUDA plugin for sleep levels 3 and 4.

# =============================================================================
# Stage 1: Build the Rust binary
# =============================================================================
FROM rust:1.90-bookworm AS builder

WORKDIR /build

# Install dependencies for building
RUN apt-get update && apt-get install -y \
  pkg-config \
  libssl-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests

# Build the llmux binary in release mode
RUN cargo build --release --bin llmux

# =============================================================================
# Stage 2: Build CRIU with CUDA plugin
# =============================================================================
FROM ubuntu:22.04 AS criu-builder

RUN apt-get update && apt-get install -y \
  build-essential \
  pkg-config \
  libprotobuf-dev \
  libprotobuf-c-dev \
  protobuf-c-compiler \
  protobuf-compiler \
  python3-protobuf \
  libnl-3-dev \
  libcap-dev \
  libaio-dev \
  libgnutls28-dev \
  libnet1-dev \
  uuid-dev \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v4.1 https://github.com/checkpoint-restore/criu.git /criu
WORKDIR /criu
RUN make -j$(nproc) && \
    make -C plugins/cuda -j$(nproc)

# =============================================================================
# Stage 3: Create the final image with vLLM
# =============================================================================
FROM vllm/vllm-openai:v0.15.1

# Apply patches:
# 1. Fix sleep mode regression (vllm#32714): `with A and B:` -> `with A, B:`
# 2. NCCL suspend/resume for cuda-checkpoint at TP>1
# 3. Fix reload_weights for TP>1: preserve parameter subclass in replace_parameter()
COPY patches/fix-sleep-mode-v0.15.1.patch /tmp/
COPY patches/nccl-suspend-resume-v0.15.1.patch /tmp/
COPY patches/fix-reload-weights-tp-v0.15.1.patch /tmp/
RUN cd /usr/local/lib/python3.12/dist-packages && \
    patch -p1 < /tmp/fix-sleep-mode-v0.15.1.patch && \
    patch -p1 < /tmp/nccl-suspend-resume-v0.15.1.patch && \
    patch -p1 < /tmp/fix-reload-weights-tp-v0.15.1.patch && \
    rm /tmp/*.patch

# Install cuda-checkpoint from NVIDIA's repo (pre-built binary)
ADD https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint \
    /usr/local/bin/cuda-checkpoint
RUN chmod +x /usr/local/bin/cuda-checkpoint

# Install CRIU and CUDA plugin from builder stage
COPY --from=criu-builder /criu/criu/criu /usr/local/bin/criu
COPY --from=criu-builder /criu/plugins/cuda/cuda_plugin.so /usr/lib/criu/cuda_plugin.so

# CRIU runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libnl-3-200 \
  libprotobuf-c1 \
  iptables \
  libnet1 \
  && rm -rf /var/lib/apt/lists/*

# Copy the llmux binary from builder
COPY --from=builder /build/target/release/llmux /usr/local/bin/llmux

# Create config and checkpoint directories
RUN mkdir -p /etc/llmux /tmp/llmux-checkpoints

# Default config location
ENV LLMUX_CONFIG=/etc/llmux/config.json

# Expose the proxy port (default 3000) and metrics port (default 9090)
EXPOSE 3000 9090

# llmux will spawn vLLM processes internally
# It expects a config file at $LLMUX_CONFIG
ENTRYPOINT ["/usr/local/bin/llmux"]
CMD ["--config", "/etc/llmux/config.json"]
