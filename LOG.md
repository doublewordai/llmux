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
