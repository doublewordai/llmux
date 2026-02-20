use super::cuda::{find_cuda_pids, has_nvidia_mappings};
use super::{ManagedProcess, Orchestrator, OrchestratorError, ProcessState};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Strip ANSI escape sequences from a string.
pub(super) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until we hit a letter (end of escape sequence)
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Tail a log file, forwarding new lines as debug-level tracing events.
/// Used for checkpoint-enabled models where stdout/stderr go to files
/// instead of pipes (required by CRIU).
async fn tail_log_file(path: PathBuf, model: String, stream: &'static str) {
    // Wait for the file to exist
    for _ in 0..10 {
        if path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let Ok(file) = tokio::fs::File::open(&path).await else {
        warn!(model = %model, path = %path.display(), "Failed to open log file for tailing");
        return;
    };

    let mut reader = BufReader::new(file);
    let mut buf = String::new();
    loop {
        match reader.read_line(&mut buf).await {
            Ok(0) => {
                // EOF — wait and try again (file may still be written to)
                tokio::time::sleep(Duration::from_millis(200)).await;
                // Re-seek to check for new data (the file handle stays open)
            }
            Ok(_) => {
                let clean = strip_ansi(buf.trim_end());
                if !clean.is_empty() {
                    debug!(target: "vllm", model = %model, stream = stream, "{}", clean);
                }
                buf.clear();
            }
            Err(_) => break,
        }
    }
}

impl Orchestrator {
    /// Internal: start a vLLM process
    pub(super) async fn start_process_internal(
        &self,
        model: &str,
        process: &Arc<Mutex<ManagedProcess>>,
    ) -> Result<(), OrchestratorError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        info!(model = %model, port = config.port, "Starting vLLM process");

        // Mark as starting
        {
            let mut guard = process.lock().await;
            guard.state = ProcessState::Starting;
        }

        // Build vLLM command
        // Determine spawn behaviour from the model's configured eviction policy.
        // - Offload/Discard weights use vLLM's built-in sleep API
        // - CudaSuspend/Checkpoint process strategies need cuda-checkpoint or CRIU
        // - Checkpoint (CRIU) additionally needs file-based stdio and io_uring disabled
        let eviction = config.eviction;
        let needs_cuda_checkpoint = eviction.needs_cuda_checkpoint();
        let needs_criu = eviction.needs_criu();

        // vllm_args() always includes --enable-sleep-mode for consistency
        let args = config.vllm_args();
        debug!(model = %model, args = ?args, "vLLM command args");

        // Spawn process in its own process group so we can kill the entire
        // tree (vLLM spawns child processes like EngineCore that hold GPU memory).
        let mut cmd = Command::new(&self.vllm_command);
        cmd.args(&args)
            .env("NO_COLOR", "1") // Disable color codes in vLLM output
            .process_group(0);

        if needs_criu {
            // CRIU compatibility: disable io_uring and libuv (they create FDs
            // that CRIU cannot checkpoint)
            cmd.env("UV_USE_IO_URING", "0");
            cmd.env("USE_LIBUV", "0");
        }

        // Dev mode required for vLLM sleep/wake and collective_rpc API endpoints
        cmd.env("VLLM_SERVER_DEV_MODE", "1");

        // CRIU cannot checkpoint unix stream socket FDs (pipes). For
        // CRIU-enabled models, redirect stdout/stderr to files and tail
        // them for logging. For normal models (including CudaSuspend), pipe as before.
        if needs_criu {
            let ckpt_cfg = self.checkpoint_config.as_ref().unwrap();
            let log_dir = ckpt_cfg.images_dir.join(model);
            std::fs::create_dir_all(&log_dir).map_err(|e| OrchestratorError::SpawnFailed {
                model: model.to_string(),
                reason: format!("Failed to create log dir {}: {}", log_dir.display(), e),
            })?;

            let stdout_path = log_dir.join("stdout.log");
            let stderr_path = log_dir.join("stderr.log");

            // Open log files with O_APPEND so that CRIU's should_check_size()
            // skips file-size validation on restore. Without this, restoring a
            // checkpoint after a container restart fails because the log files
            // changed size during the intervening cold start.
            let stdout_file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .custom_flags(libc::O_APPEND)
                .open(&stdout_path)
                .map_err(|e| OrchestratorError::SpawnFailed {
                    model: model.to_string(),
                    reason: format!("Failed to create {}: {}", stdout_path.display(), e),
                })?;
            let stderr_file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .custom_flags(libc::O_APPEND)
                .open(&stderr_path)
                .map_err(|e| OrchestratorError::SpawnFailed {
                    model: model.to_string(),
                    reason: format!("Failed to create {}: {}", stderr_path.display(), e),
                })?;

            // CRIU requires stdin to point to a container-local /dev/null rather
            // than an inherited pipe or host fd. If stdin's mount ID belongs to
            // the host mount namespace, CRIU dump fails with
            // "Can't lookup mount=N for fd=0 path=/dev/null".
            let devnull =
                std::fs::File::open("/dev/null").map_err(|e| OrchestratorError::SpawnFailed {
                    model: model.to_string(),
                    reason: format!("Failed to open /dev/null: {}", e),
                })?;
            cmd.stdin(devnull);
            cmd.stdout(stdout_file);
            cmd.stderr(stderr_file);

            // Tail the log files for debug logging
            let model_name = model.to_string();
            let stdout_tail = stdout_path.clone();
            let stderr_tail = stderr_path.clone();
            tokio::spawn(async move {
                tail_log_file(stdout_tail, model_name.clone(), "stdout").await;
            });
            let model_name2 = model.to_string();
            tokio::spawn(async move {
                tail_log_file(stderr_tail, model_name2, "stderr").await;
            });
        } else {
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
        }

        let mut child = cmd.spawn().map_err(|e| OrchestratorError::SpawnFailed {
            model: model.to_string(),
            reason: e.to_string(),
        })?;

        // Forward vLLM stdout/stderr as debug logs under the "vllm" target,
        // filterable via RUST_LOG (e.g. RUST_LOG=info,vllm=debug).
        // (Only for non-CRIU mode; CRIU mode uses file tailing above)
        if !needs_criu {
            let model_name = model.to_string();
            if let Some(stdout) = child.stdout.take() {
                let name = model_name.clone();
                tokio::spawn(async move {
                    let reader = BufReader::new(stdout);
                    let mut lines = reader.lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let clean = strip_ansi(&line);
                        debug!(target: "vllm", model = %name, stream = "stdout", "{}", clean);
                    }
                });
            }
            if let Some(stderr) = child.stderr.take() {
                let name = model_name;
                tokio::spawn(async move {
                    let reader = BufReader::new(stderr);
                    let mut lines = reader.lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let clean = strip_ansi(&line);
                        debug!(target: "vllm", model = %name, stream = "stderr", "{}", clean);
                    }
                });
            }
        }

        // Store child process and its PID (for CRIU checkpoint after restore)
        {
            let mut guard = process.lock().await;
            guard.parent_pid = child.id();
            guard.child = Some(child);
        }

        // Wait for health check to pass
        let health_url = format!("http://localhost:{}/health", config.port);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > self.startup_timeout {
                let mut guard = process.lock().await;
                guard.state = ProcessState::Failed {
                    reason: "Startup timeout".to_string(),
                };
                // Kill the process
                if let Some(ref mut child) = guard.child {
                    let _ = child.kill().await;
                }
                return Err(OrchestratorError::StartupTimeout {
                    model: model.to_string(),
                });
            }

            // Try health check
            match self.check_health(&health_url).await {
                Ok(true) => {
                    info!(model = %model, "vLLM process is ready");
                    let mut guard = process.lock().await;
                    // Discover all PIDs with active CUDA contexts for
                    // cuda-checkpoint-enabled models (CudaSuspend and Checkpoint).
                    // This includes both the EngineCore parent (which initialises
                    // CUDA before forking) and the GPU worker(s).
                    if needs_cuda_checkpoint
                        && let Some(ref child) = guard.child
                        && let Some(pid) = child.id()
                    {
                        let cuda_path = self
                            .checkpoint_config
                            .as_ref()
                            .map(|c| c.cuda_checkpoint_path.as_str())
                            .unwrap_or("cuda-checkpoint");
                        let cuda_pids = find_cuda_pids(pid, cuda_path);
                        if cuda_pids.is_empty() {
                            warn!(model = %model, pid, "Could not find CUDA processes, will use parent PID for cuda-checkpoint");
                            guard.engine_core_pids = vec![pid];
                            guard.tp_size = 1;
                        } else {
                            // TP size = number of descendant GPU workers (with nvidia device maps).
                            // Exclude the parent process — it has nvidia maps from CUDA init
                            // but is not a TP rank.
                            let tp = cuda_pids
                                .iter()
                                .filter(|&&p| p != pid && has_nvidia_mappings(p))
                                .count()
                                .max(1);
                            info!(model = %model, pid, cuda_pids = ?cuda_pids, tp, "Found {} CUDA process(es)", cuda_pids.len());
                            guard.engine_core_pids = cuda_pids;
                            guard.tp_size = tp;
                        }
                    }
                    guard.state = ProcessState::Running { sleeping: None };
                    guard.ready_notify.notify_waiters();
                    return Ok(());
                }
                Ok(false) => {
                    debug!(model = %model, "Health check returned unhealthy, retrying...");
                }
                Err(e) => {
                    debug!(model = %model, error = %e, "Health check failed, retrying...");
                }
            }

            // Check if process died
            {
                let mut guard = process.lock().await;
                if let Some(ref mut child) = guard.child {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            let reason = format!("Process exited with status: {}", status);
                            guard.state = ProcessState::Failed {
                                reason: reason.clone(),
                            };
                            return Err(OrchestratorError::ProcessFailed {
                                model: model.to_string(),
                                reason,
                            });
                        }
                        Ok(None) => {
                            // Still running
                        }
                        Err(e) => {
                            warn!(model = %model, error = %e, "Failed to check process status");
                        }
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Check if a vLLM endpoint is healthy
    pub(super) async fn check_health(&self, url: &str) -> Result<bool, String> {
        use http_body_util::Empty;

        let client: hyper_util::client::legacy::Client<_, Empty<bytes::Bytes>> =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http();

        let uri: hyper::Uri = url.parse().map_err(|e| format!("Invalid URL: {}", e))?;

        let request = hyper::Request::builder()
            .method("GET")
            .uri(uri)
            .body(Empty::new())
            .map_err(|e| format!("Failed to build request: {}", e))?;

        let result = tokio::time::timeout(self.health_timeout, client.request(request)).await;

        match result {
            Ok(Ok(response)) => Ok(response.status().is_success()),
            Ok(Err(e)) => Err(format!("Request failed: {}", e)),
            Err(_) => Err("Health check timeout".to_string()),
        }
    }

    /// Check if a model's process is still alive. If it has exited,
    /// reset state to NotStarted so it can be restarted.
    ///
    /// This prevents the "zombie loop" where the orchestrator keeps trying
    /// to send HTTP requests to a dead process.
    pub(super) async fn check_process_alive(&self, model: &str) {
        let Some(process) = self.processes.get(model) else {
            return;
        };

        let mut guard = process.lock().await;
        if let Some(ref mut child) = guard.child {
            match child.try_wait() {
                Ok(Some(status)) => {
                    warn!(
                        model = %model,
                        status = %status,
                        "Process found dead, resetting to NotStarted"
                    );
                    guard.child = None;
                    guard.state = ProcessState::NotStarted;
                }
                Ok(None) => {
                    // Still running
                }
                Err(e) => {
                    warn!(model = %model, error = %e, "Failed to check process status");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("hello"), "hello");
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(
            strip_ansi("\x1b[1;32mgreen bold\x1b[0m text"),
            "green bold text"
        );
    }
}
