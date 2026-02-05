# Multi-stage build for llmux with vLLM
#
# This Dockerfile builds the llmux binary and packages it with vLLM,
# enabling zero-reload model switching for multiple models on shared GPU.

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
# Stage 2: Create the final image with vLLM
# =============================================================================
# Pin to v0.13.0 - sleep mode is broken in v0.14+ (vllm#32714)
FROM vllm/vllm-openai:v0.13.0

# Copy the llmux binary from builder
COPY --from=builder /build/target/release/llmux /usr/local/bin/llmux

# Create config directory
RUN mkdir -p /etc/llmux

# Default config location
ENV LLMUX_CONFIG=/etc/llmux/config.json

# Expose the proxy port (default 3000) and metrics port (default 9090)
EXPOSE 3000 9090

# llmux will spawn vLLM processes internally
# It expects a config file at $LLMUX_CONFIG
ENTRYPOINT ["/usr/local/bin/llmux"]
CMD ["--config", "/etc/llmux/config.json"]
