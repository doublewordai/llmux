# Podman + CRIU GPU checkpoint/restore

Switch between GPU models by checkpointing and restoring vLLM containers
with CRIU. Restore is ~3x faster than cold start.

## Prerequisites

- **NVIDIA driver >= 535** with `cuda-checkpoint` binary (ships with driver)
- **CRIU >= 3.17** with CUDA plugin (check: `sudo criu check`)
- **Podman >= 4.0** (rootful — CRIU requires root)
- **vLLM container image**: `docker.io/vllm/vllm-openai:v0.8.3`

**Important**: The vLLM image's CUDA version must be compatible with your
host driver. Driver 570 (CUDA 12.8) works with vLLM v0.8.3. Newer vLLM
images (v0.15+) bundle CUDA 12.9+ and will fail with "forward compatibility"
errors.

Verify your setup:

```sh
nvidia-smi                          # driver version
sudo criu check                     # CRIU functional
which cuda-checkpoint               # checkpoint binary exists
sudo podman run --rm --device nvidia.com/gpu=0 \
  docker.io/vllm/vllm-openai:v0.8.3 \
  --help                            # container can see GPU
```

## Usage

### 1. Warmup (one-time)

Cold-start each model, checkpoint it, and leave it ready for fast restore:

```sh
sudo ./warmup.sh
```

### 2. Run llmux

```sh
cargo run --release -- -c examples/podman-criu/config.yaml -p 4000
```

### 3. Send requests

```sh
# First request restores from checkpoint (~11s)
curl http://localhost:4000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"facebook/opt-125m","prompt":"Hello","max_tokens":10}'

# Switch to another model (checkpoint + restore ~30s)
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
```

## Timings (RTX 4090)

| Operation | Time |
|---|---|
| Cold start (opt-125m) | ~34s |
| Cold start (Qwen 3B) | ~99s |
| Restore from checkpoint | ~11s |
| Full switch (checkpoint A + restore B) | ~30s |

## How it works

The `config.yaml` uses inline hook scripts:

- **wake**: Try `podman restore`, fall back to `podman run` (cold start)
- **sleep**: `podman checkpoint` (freezes process + GPU state to disk)
- **alive**: `curl` the health endpoint

CRIU's CUDA plugin calls `cuda-checkpoint` on the host to save/restore GPU
memory and execution state. The checkpoint includes the full vLLM process
tree, CUDA contexts, and TCP listening sockets (`--tcp-established`).

## Cleanup

```sh
sudo podman rm -f llmux-opt-125m llmux-qwen-3b
```
