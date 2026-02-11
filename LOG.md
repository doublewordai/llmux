# CRIU Checkpoint/Restore (L4) in Docker — Change Log

## Summary

Successfully implemented CRIU-based checkpoint/restore (sleep level 4) for vLLM processes inside Docker containers. This enables saving a model's entire process state (including CUDA GPU state) to disk and restoring it later, avoiding full model reload.

## Test Results

Full end-to-end cycle verified:
1. Start model A → serve requests
2. Swap to model B → CRIU checkpoint A to disk (~31GB), start B fresh
3. Swap back to A → Stop B, CRIU restore A from disk (~12s)
4. Swap to B again → Re-checkpoint A (using stored PID), start B
5. Swap back to A → CRIU restore A again from disk

All steps produce correct inference results after restore.

## Docker Run Flags for CRIU

```bash
docker run --gpus all \
  --privileged \
  --pid=host \
  --ipc=host \
  -v /tmp/llmux-checkpoints:/tmp/llmux-checkpoints \
  ...
```

**Do NOT use `--init`** — Docker's init (tini) redirects stdin to the host's
`/dev/null`, whose mount ID is invisible inside the container. CRIU dump fails
with `Can't lookup mount=N for fd=0 path=/dev/null`.

## Changes to `src/orchestrator.rs`

### 1. Redirect stdin to `/dev/null` for CRIU-enabled processes

CRIU cannot checkpoint file descriptors pointing to host mount namespaces.
When llmux spawns vLLM, stdin inherits from the parent. Inside Docker
(without `--init`), this is typically a pipe to the container runtime.
Redirecting stdin to the container-local `/dev/null` avoids mount ID
resolution failures during CRIU dump.

```rust
// In spawn_vllm(), for needs_criu mode:
let devnull = std::fs::File::open("/dev/null")?;
cmd.stdin(devnull);
```

### 2. Add `--link-remap` and `--enable-external-masters` to CRIU dump

- `--link-remap`: Required for POSIX named semaphores in `/dev/shm`. vLLM's
  multiprocessing creates `sem.mp-*` semaphores that CRIU can't handle without
  this flag.
- `--enable-external-masters`: Required for NVIDIA GPU proc mounts
  (`/proc/driver/nvidia/gpus/...`) which have mount sharing patterns that CRIU
  can't resolve by default.

### 3. Add `--enable-external-masters` to CRIU restore

Matches the dump flag for consistent mount handling on restore.

### 4. Clean up stale `link_remap.*` files before dump

CRIU's `--link-remap` creates temporary hardlink files named `link_remap.N` in
`/dev/shm`. These are not cleaned up after dump and cause "File exists" errors
on subsequent dumps. Added cleanup loop before each dump.

### 5. Skip pre-toggle (let CUDA plugin handle cuda-checkpoint)

On NVIDIA driver 580+, the CRIU CUDA plugin expects to find the CUDA restore
thread. If `cuda-checkpoint --toggle` is called before CRIU dump, the restore
thread is already gone and the plugin errors. Skipping the pre-toggle lets the
plugin handle cuda-checkpoint internally.

### 6. Clean up checkpoint images after successful restore

CRIU checkpoint images for even a 0.5B model are ~31GB. After a successful
restore, the images are no longer needed. Cleaning them up immediately frees
disk space for future checkpoints.

### 7. Add `parent_pid` field to `ManagedProcess`

After CRIU restore with `--restore-detached`, the tokio `Child` process handle
is gone (the process runs independently). The `parent_pid` field stores the
PID from the original spawn so it can be used for future CRIU checkpoints of
the restored process.

### 8. Fall back to stored `parent_pid` in `checkpoint_model`

The checkpoint code now tries `child.id()` first, then falls back to
`parent_pid`. This enables re-checkpointing a CRIU-restored process.

## Changes to `src/switcher.rs`

### 9. Downgrade sleep to Stop when target is checkpointed

When swapping from model B to model A (where A is already checkpointed on
disk), there's no need to CRIU-checkpoint model B — both checkpoints can't fit
on disk simultaneously (each is ~31GB). Instead, model B is simply killed
(Stop), and model A is restored from its checkpoint. Model B can be started
fresh on the next swap.

## Changes to `README.md`

### 10. Update Docker run example for CRIU

Updated the Docker run command for sleep levels 3-4 to use `--privileged`
instead of individual capabilities, added checkpoint volume mount, and added a
warning about not using `--init`.

## Known Issues

- **Disk space**: Each CRIU checkpoint is ~31GB for a 0.5B model. Larger models
  will be proportionally larger. Ensure sufficient disk space on the checkpoint
  volume.
- **PID 149830 warning**: CRIU logs a non-fatal warning "Could not find restore
  thread for process ID" for a monitoring subprocess. This doesn't affect
  functionality.

## Environment

- GPU: NVIDIA B300 (sm_103a)
- Driver: 580.95.05
- CUDA: 12.9
- CRIU: v4.1 with CUDA plugin
- vLLM: v0.15.1 with sleep mode and NCCL suspend/resume patches
- Required env vars for B300: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`,
  `VLLM_USE_FLASHINFER_MOE_FP8=0`

---

## New session: gotenks (bare metal, 6x RTX 4090)

### Key realization

The existing LOG.md documents a **working** CRIU + vLLM implementation on B300/Docker.
Critical difference from our failed attempts: **the CUDA plugin handles cuda-checkpoint
internally**. We were running cuda-checkpoint BEFORE CRIU, which conflicts with the plugin.

From item 5 above: "If cuda-checkpoint --toggle is called before CRIU dump, the restore
thread is already gone and the plugin errors. Skipping the pre-toggle lets the plugin
handle cuda-checkpoint internally."

### New approach

Instead of: sleep → cuda-checkpoint → criu dump (plugin conflicts)
Do: sleep → criu dump (plugin handles cuda-checkpoint internally)

This means the plugin is not optional — it's the mechanism that makes CRIU work with CUDA.

### External unix socket problem

The `subprocess.PIPE` approach creates unix socketpairs between the parent (benchmark
script) and the child (vLLM). One end of each socketpair is outside the dump tree,
so CRIU can't dump it. The fix from orchestrator.rs: redirect stdout/stderr to **files**
and stdin to `/dev/null`. This eliminates the external socket entirely.

### L2 wake requires reload_weights

After CRIU restore, calling `/wake_up` alone for L2 produces garbage output because
the weights were discarded during L2 sleep. The orchestrator.rs does a 3-step wake:
1. `POST /wake_up` — re-enables scheduling (0.06s)
2. `POST /collective_rpc {"method": "reload_weights"}` — reloads weights from disk (~5s)
3. `POST /reset_prefix_cache` — clears stale cache entries

Without step 2, inference returns `'!!!!!!!!!!'` (random garbage from uninitialized
GPU memory).

### CRIU benchmark results (gotenks, RTX 4090, Llama 8B)

Both L1 and L2 checkpoint/restore cycles produce **correct inference** after restore.

#### L2 + CRIU (recommended for local eviction)

| Step | Time |
|------|------|
| L2 sleep | 0.3s |
| CRIU dump | 10.1s |
| **Total sleep** | **10.4s** |
| CRIU restore | 9.4s |
| L2 wake (wake_up + reload_weights + reset_prefix_cache) | 4.9s |
| **Total wake** | **14.4s** |
| **Round-trip** | **24.8s** |
| CRIU images on disk | 3128 MB |

#### L1 + CRIU (self-contained portable images)

| Step | Time |
|------|------|
| L1 sleep (first, with pinned alloc) | 14.9-18.9s |
| CRIU dump | 33.5-35.9s |
| **Total sleep** | **~50s** |
| CRIU restore | 35.2-38.2s |
| L1 wake | 0.7s |
| **Total wake** | **~37s** |
| **Round-trip** | **~87s** |
| CRIU images on disk | 21385 MB |

L1 images are ~7x larger than L2 because they contain the model weights in pinned
CPU memory. This makes dump/restore proportionally slower. L1 images are self-contained
(portable to any node), while L2 images require the model files to be present locally.

### CRIU flags (matching orchestrator.rs)

**Dump:**
```
criu dump -t PID --images-dir DIR --shell-job --ext-unix-sk --tcp-established \
  --link-remap --enable-external-masters -L /tmp/criu/plugins/cuda -v4 --log-file dump.log
```

**Restore:**
```
criu restore --images-dir DIR --shell-job --ext-unix-sk --tcp-established \
  --enable-external-masters -L /tmp/criu/plugins/cuda --restore-detached \
  -v4 --log-file restore.log
```

### Required environment variables

```
VLLM_SERVER_DEV_MODE=1    # enables sleep/wake/collective_rpc endpoints
UV_USE_IO_URING=0         # disable io_uring in uvloop
USE_LIBUV=0               # belt-and-suspenders io_uring disable
VLLM_NO_USAGE_STATS=1     # prevent telemetry connections
DO_NOT_TRACK=1            # prevent telemetry connections
```

Plus system-wide: `echo 2 > /proc/sys/kernel/io_uring_disabled`

### CUDA plugin behavior

The plugin handles the full CUDA checkpoint/restore lifecycle:
- **Dump**: Finds CUDA processes, runs `cuda-checkpoint` to suspend them
- **Restore**: Finds restore threads, runs `cuda-checkpoint --action restore` + `--action unlock`
- Non-GPU processes (like the APIServer monitoring process) log "Could not find restore thread" — this is a non-fatal warning

The plugin reports `err 0` on both dump and restore when everything works correctly.

### TP=2 CRIU results (Qwen3-32B-FP8, 2x RTX 4090)

Cold start: ~88-120s to first inference (varies by run).

#### L2 + CRIU: FIXED with patch

`reload_weights` via `collective_rpc` was failing with `'Parameter' object has no
attribute 'load_row_parallel_weight'`. Root cause: `replace_parameter()` in
`vllm/model_executor/utils.py` creates a plain `nn.Parameter`, losing the original
subclass (e.g. `RowvLLMParameter`, `ModelWeightParameter`) and its methods.

Fix in `patches/fix-reload-weights-tp-v0.15.1.patch`:
1. `replace_parameter()` now preserves `__class__` and copies all `__dict__` attrs
   from old to new parameter, keeping subclass methods and instance attributes
2. `marlin_utils_fp4.py` uses `replace_parameter()` instead of raw `setattr()`

Related: doublewordai/vllm PR #2 (similar but only does `__dict__` copy, misses
`__class__` preservation which is needed for methods like `load_row_parallel_weight`)

#### Full comparison (cold start = 89.1s to first inference)

| Method | CRIU image | Restore | Wake | 1st infer | Total | Speedup |
|--------|-----------|---------|------|-----------|-------|---------|
| **L2+CRIU** | **6.7 GB** | **8.8s** | **14.9s** | 1.5s | **25.2s** | **3.5x** |
| L1+CRIU | 45.4 GB | 80.7s | 1.3s | 1.8s | 83.8s | 1.1x |
| NoSleep+CRIU | 50.1 GB | 71.7s | 0.3s | 1.7s | 73.7s | 1.2x |

L2+CRIU is the clear winner for TP=2: 3.5x faster than cold start with only 6.7 GB
images. The reload_weights step (14.4s) dominates the wake time but is still much
faster than cold start's full model init + weight loading (87s).

#### TP=2 wake sequence

1. `resume_nccl` — must be first (NCCL needed for any TP collective ops)
2. `wake_up` — only if vLLM sleep was used (L1/L2)
3. `reload_weights` — only for L2 (BROKEN with TP>1)
4. `reset_prefix_cache`

For NoSleep (CudaSuspend) mode, only `resume_nccl` is needed after restore.

#### TCP connection issue with TP>1

CRIU built without nftables support cannot use `--network-lock nftables`.
With `--tcp-established` and iptables (default), dump/restore works but the
PyTorch distributed store TCP connections are preserved. Using `--tcp-close`
breaks those connections, causing NCCL resume to fail with "Broken pipe".
With `--tcp-established`, the connections survive and NCCL resume works.

The TIME_WAIT issue from the first attempt (bind error on restore) was a red
herring — it was caused by stale processes from a previous run, not by CRIU.
