# Cross-container CRIU restore — progress log

## 2026-02-20

### Starting state

- In-container checkpoint/restore works
- Cross-container restore fails: CRIU restore itself fails because ephemeral
  files don't exist in the new container, and the stray GPU detector kills the
  restored process
- Constraints: no vLLM patches, no checkpoint strategy changes, fix via CRIU
  flags and filesystem orchestration only

### Step 1: Set up dev loop

- Installed Rust on VM, cloned repo
- Built inside a Docker container matching the target glibc (builder image
  committed as `llmux-builder`)
- Incremental rebuild: ~25s, no-change rebuild: ~11s
- Binary mounted into container with `-v .../target/release/llmux:/usr/local/bin/llmux:ro`
- Need `-e LD_LIBRARY_PATH=...` for CUDA driver access

### Step 2: CRIU dump analysis

Added `-v4 --log-file dump.log` to CRIU dump command. Found:

ZMQ sockets (both endpoints inside process tree):
- `/tmp/<uuid-1>` — listening socket + connected pair (APIServer → EngineCore)
- `/tmp/<uuid-2>` — listening socket + connected pair (EngineCore → APIServer)

All are SOCK_STREAM (type 1), both sides in the tree. CRIU dumps them
successfully with `--ext-unix-sk`.

### Step 3: Cross-container restore — missing files

CRIU restore fails because runtime-generated files don't exist in fresh
container:

1. `/root/.cache/flashinfer/0.6.1/90a/flashinfer_jit.log`
2. `/root/.triton/cache/.../cuda_utils.cpython-312-x86_64-linux-gnu.so`
3. `/root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch29-cuda.so`

**Fix (commit c0ed0e7):** After CRIU dump, save files from known ephemeral
directories (`/root/.cache/flashinfer/`, `/root/.cache/tvm-ffi/`,
`/root/.triton/cache/`) into `{images_dir}/rootfs/`. Before CRIU restore,
copy them back.

### Step 4: Stray GPU detector kills restored process

After fixing the missing files, CRIU restore succeeded. But the stray GPU
detector saw the restored process (which has no tokio Child handle after
`--restore-detached`) as a "stray" and killed it.

**Fix (commit 2cccbb8):** Save the parent PID to `{images_dir}/parent_pid`
during dump. Read it back during restore and set `guard.parent_pid` so the
stray detector recognizes it.

### Step 5: SUCCESS

Full cross-container test passes:

```
Container A: cold-start → inference → checkpoint via /control/sleep
Container A: docker stop && docker rm
Container B: docker run (different container, same image)
Container B: /control/wake → inference → valid response
```

The restore flow:
1. Ephemeral files restored from checkpoint
2. CRIU restore succeeded (~5s)
3. Parent PID restored from checkpoint
4. Health check passed immediately
5. reload_weights succeeded (ZMQ IPC works!)
6. Model marked active, inference succeeds

### Root cause summary

The "ZMQ IPC is broken" diagnosis from the previous session was wrong. The
actual failures were:

1. **Missing ephemeral files**: CRIU needs all open files to exist at their
   original paths during restore. vLLM creates runtime files in the
   container's OverlayFS (flashinfer JIT log, triton compiled kernels,
   tvm-ffi cache) that don't exist in a fresh container.

2. **Stray GPU detector**: After CRIU `--restore-detached`, the process runs
   independently with no tokio Child handle. The orchestrator didn't know the
   process was ours and killed it as a "stray GPU process".

Both fixes are pure orchestration — no vLLM patches, no CRIU flag changes,
no checkpoint strategy changes.
