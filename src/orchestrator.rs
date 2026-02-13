//! Orchestrator - manages vLLM process lifecycle
//!
//! The orchestrator is responsible for:
//! - Lazily starting vLLM processes on first request
//! - Tracking process state (NotStarted, Starting, Running, etc.)
//! - Health checking to confirm processes are ready
//! - Coordinating with the switcher for wake/sleep operations

use crate::config::{CheckpointConfig, LoraAdapterConfig, LoraConfig, ModelConfig};
use crate::switcher::{EvictionPolicy, ProcessStrategy, WeightStrategy};
use anyhow::Result;
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify, RwLock};
use tracing::{debug, error, info, warn};

const LORA_BASE_KEY: &str = "__llmux_lora_base__";

/// Kill an entire process group by sending SIGKILL to -pgid.
#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: We're sending SIGKILL to a process group we spawned.
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
    }
}

/// State of a model's vLLM process
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    /// Process has not been started yet
    NotStarted,
    /// Process is starting up
    Starting,
    /// Process is running and ready
    Running {
        /// Whether the model is currently sleeping, and the eviction policy used
        sleeping: Option<EvictionPolicy>,
    },
    /// Process failed to start or crashed
    Failed { reason: String },
    /// Process has been checkpointed to disk via CRIU (not running, but restorable)
    Checkpointed {
        images_dir: PathBuf,
        /// The eviction policy used when checkpointing (needed for wake sequence)
        eviction: EvictionPolicy,
    },
}

/// Internal state for a managed process
struct ManagedProcess {
    #[allow(dead_code)] // Reserved for future use
    config: ModelConfig,
    state: ProcessState,
    child: Option<Child>,
    /// Notifies waiters when process becomes ready
    ready_notify: Arc<Notify>,
    /// PIDs of processes holding GPU memory (engine core + TP workers).
    /// Discovered after startup for checkpoint-enabled models.
    /// With TP=1 this is a single PID; with TP>1 one per rank.
    engine_core_pids: Vec<u32>,
    /// Parent PID of the vLLM process. Stored separately because after
    /// CRIU restore with --restore-detached, the tokio Child handle is gone
    /// but the process is running at its original PID.
    parent_pid: Option<u32>,
}

struct LoraRuntime {
    adapters: HashMap<String, LoraAdapterConfig>,
    active_adapter: Arc<RwLock<Option<String>>>,
}

/// Strip ANSI escape sequences from a string.
fn strip_ansi(s: &str) -> String {
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

/// Find all descendant processes that hold GPU memory.
///
/// vLLM spawns a child process tree. With TP=1, there's a single EngineCore
/// process. With TP>1, the EngineCore spawns worker processes (one per rank),
/// each holding a GPU. We find all descendants that have NVIDIA device
/// mappings in /proc/PID/maps.
fn find_gpu_pids(parent_pid: u32) -> Vec<u32> {
    let all_descendants = find_all_descendants(parent_pid);
    if all_descendants.is_empty() {
        return vec![];
    }

    // Filter to processes that have NVIDIA GPU mappings
    let gpu_pids: Vec<u32> = all_descendants
        .iter()
        .copied()
        .filter(|&pid| has_nvidia_mappings(pid))
        .collect();

    if !gpu_pids.is_empty() {
        return gpu_pids;
    }

    // Fallback: look for processes named EngineCore
    let engine_pids: Vec<u32> = all_descendants
        .iter()
        .copied()
        .filter(|&pid| {
            let cmdline_path = format!("/proc/{}/cmdline", pid);
            std::fs::read_to_string(&cmdline_path)
                .map(|c| c.contains("EngineCore") || c.contains("engine_core"))
                .unwrap_or(false)
        })
        .collect();

    if !engine_pids.is_empty() {
        return engine_pids;
    }

    // Last resort: return all direct children
    find_child_pids(parent_pid)
}

/// Check if a process has NVIDIA GPU device mappings.
fn has_nvidia_mappings(pid: u32) -> bool {
    let maps_path = format!("/proc/{}/maps", pid);
    std::fs::read_to_string(&maps_path)
        .map(|maps| maps.contains("/dev/nvidia"))
        .unwrap_or(false)
}

/// Get direct child PIDs of a process.
fn find_child_pids(pid: u32) -> Vec<u32> {
    let output = std::process::Command::new("pgrep")
        .args(["-P", &pid.to_string()])
        .output()
        .ok();

    match output {
        Some(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect(),
        _ => vec![],
    }
}

/// Recursively find all descendant PIDs of a process.
fn find_all_descendants(pid: u32) -> Vec<u32> {
    let mut result = Vec::new();
    let children = find_child_pids(pid);
    for &child in &children {
        result.push(child);
        result.extend(find_all_descendants(child));
    }
    result
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

/// Build a command, prefixing with `sudo` only if we're not already root.
fn maybe_sudo(program: &str) -> tokio::process::Command {
    if unsafe { libc::geteuid() } == 0 {
        tokio::process::Command::new(program)
    } else {
        let mut cmd = tokio::process::Command::new("sudo");
        cmd.arg(program);
        cmd
    }
}

/// Orchestrator manages vLLM process lifecycle
pub struct Orchestrator {
    /// Model configurations
    configs: HashMap<String, ModelConfig>,
    /// Process state for each model
    processes: DashMap<String, Arc<Mutex<ManagedProcess>>>,
    /// Runtime eviction policy overrides (takes priority over config)
    eviction_overrides: DashMap<String, EvictionPolicy>,
    /// Lock for serializing process operations
    operation_lock: Mutex<()>,
    /// Health check timeout
    health_timeout: Duration,
    /// Startup timeout
    startup_timeout: Duration,
    /// Command to use for spawning processes (e.g., "vllm" or path to mock)
    vllm_command: String,
    /// CRIU checkpoint configuration (None = checkpoint level disabled)
    checkpoint_config: Option<CheckpointConfig>,
    /// Dynamic LoRA runtime state (when enabled).
    lora_runtime: Option<LoraRuntime>,
}

impl Orchestrator {
    /// Create a new orchestrator with the given model configurations
    pub fn new(configs: HashMap<String, ModelConfig>) -> Self {
        Self::with_command(configs, "vllm".to_string())
    }

    /// Create a new orchestrator with a custom command for spawning processes
    ///
    /// This is useful for testing with mock-vllm
    pub fn with_command(configs: HashMap<String, ModelConfig>, vllm_command: String) -> Self {
        Self::with_options(configs, vllm_command, None)
    }

    /// Create a dynamic LoRA orchestrator:
    /// one base process + many adapter aliases.
    pub fn with_lora_options(
        lora: LoraConfig,
        vllm_command: String,
        checkpoint_config: Option<CheckpointConfig>,
    ) -> Self {
        let mut configs = HashMap::new();
        configs.insert(LORA_BASE_KEY.to_string(), lora.base_model.to_model_config());
        let mut orchestrator = Self::with_options(configs, vllm_command, checkpoint_config);
        orchestrator.lora_runtime = Some(LoraRuntime {
            adapters: lora.adapters,
            active_adapter: Arc::new(RwLock::new(None)),
        });
        orchestrator
    }

    /// Create a new orchestrator with full options including checkpoint config
    pub fn with_options(
        configs: HashMap<String, ModelConfig>,
        vllm_command: String,
        checkpoint_config: Option<CheckpointConfig>,
    ) -> Self {
        let processes = DashMap::new();

        for (name, config) in &configs {
            let initial_state = if let Some(ref path) = config.checkpoint_path {
                info!(
                    model = %name,
                    path = %path.display(),
                    "Model has checkpoint_path, will restore from checkpoint on first request"
                );
                ProcessState::Checkpointed {
                    images_dir: path.clone(),
                    eviction: config.eviction,
                }
            } else {
                ProcessState::NotStarted
            };

            processes.insert(
                name.clone(),
                Arc::new(Mutex::new(ManagedProcess {
                    config: config.clone(),
                    state: initial_state,
                    child: None,
                    ready_notify: Arc::new(Notify::new()),
                    engine_core_pids: vec![],
                    parent_pid: None,
                })),
            );
        }

        Self {
            configs,
            processes,
            eviction_overrides: DashMap::new(),
            operation_lock: Mutex::new(()),
            health_timeout: Duration::from_secs(5),
            startup_timeout: Duration::from_secs(900), // 15 minutes for large TP models
            vllm_command,
            checkpoint_config,
            lora_runtime: None,
        }
    }

    /// Returns true if this orchestrator is running in dynamic LoRA mode.
    pub fn is_lora_mode(&self) -> bool {
        self.lora_runtime.is_some()
    }

    fn lora_runtime(&self) -> Option<&LoraRuntime> {
        self.lora_runtime.as_ref()
    }

    fn lora_has_adapter(&self, model: &str) -> bool {
        self.lora_runtime()
            .map(|rt| rt.adapters.contains_key(model))
            .unwrap_or(false)
    }

    fn lora_adapter_config(&self, model: &str) -> Option<&LoraAdapterConfig> {
        self.lora_runtime().and_then(|rt| rt.adapters.get(model))
    }

    /// Get the current state of a model's process
    pub async fn process_state(&self, model: &str) -> Option<ProcessState> {
        if self.is_lora_mode() {
            if !self.lora_has_adapter(model) {
                return None;
            }

            let process = self.processes.get(LORA_BASE_KEY)?;
            let base_state = process.lock().await.state.clone();

            if let ProcessState::Running { sleeping: None } = base_state {
                let active = self
                    .lora_runtime()
                    .unwrap()
                    .active_adapter
                    .read()
                    .await
                    .clone();
                if active.as_deref() == Some(model) {
                    Some(ProcessState::Running { sleeping: None })
                } else {
                    Some(ProcessState::Running {
                        sleeping: Some(EvictionPolicy::STOP),
                    })
                }
            } else {
                Some(base_state)
            }
        } else {
            let process = self.processes.get(model)?;
            let guard = process.lock().await;
            Some(guard.state.clone())
        }
    }

    /// Get all registered model names
    pub fn registered_models(&self) -> Vec<String> {
        if let Some(lora) = self.lora_runtime() {
            let mut names: Vec<String> = lora.adapters.keys().cloned().collect();
            names.sort();
            names
        } else {
            self.configs.keys().cloned().collect()
        }
    }

    /// Get the effective eviction policy for a model (override > config)
    pub fn eviction_policy_for(&self, model: &str) -> Option<EvictionPolicy> {
        if self.is_lora_mode() {
            return None;
        }
        if let Some(policy) = self.eviction_overrides.get(model) {
            return Some(*policy);
        }
        self.configs.get(model).map(|c| c.eviction)
    }

    /// Check if a model is currently in the Checkpointed state
    pub fn is_checkpointed(&self, model: &str) -> bool {
        if self.is_lora_mode() {
            return false;
        }
        if let Some(process) = self.processes.get(model) {
            // try_lock to avoid blocking the switcher
            if let Ok(guard) = process.try_lock() {
                return matches!(guard.state, ProcessState::Checkpointed { .. });
            }
        }
        false
    }

    /// Override the eviction policy for a model at runtime
    pub fn set_eviction_policy(&self, model: &str, policy: EvictionPolicy) {
        if self.is_lora_mode() {
            warn!(
                model = %model,
                "Ignoring eviction override in LoRA mode (unsupported)"
            );
            return;
        }
        self.eviction_overrides.insert(model.to_string(), policy);
    }

    /// Ensure a model's process is running and ready
    ///
    /// This will:
    /// 1. Start the process if not started
    /// 2. Wait for it to become healthy
    /// 3. Return once the model is ready to serve requests
    pub async fn ensure_running(&self, model: &str) -> Result<(), OrchestratorError> {
        if self.is_lora_mode() {
            if !self.lora_has_adapter(model) {
                return Err(OrchestratorError::ModelNotFound(model.to_string()));
            }
            return self.ensure_running_legacy(LORA_BASE_KEY).await;
        }
        self.ensure_running_legacy(model).await
    }

    async fn ensure_running_legacy(&self, model: &str) -> Result<(), OrchestratorError> {
        let process = self
            .processes
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        // Check current state
        {
            let guard = process.lock().await;
            match &guard.state {
                ProcessState::Running { sleeping: None } => {
                    // Already running and awake
                    return Ok(());
                }
                ProcessState::Running { sleeping: Some(_) } => {
                    // Running but sleeping - need to wake
                    // This is handled by wake_model
                    return Ok(());
                }
                ProcessState::Starting => {
                    // Someone else is starting it, wait for ready
                    let notify = guard.ready_notify.clone();
                    drop(guard);
                    notify.notified().await;
                    return Ok(());
                }
                ProcessState::Failed { reason } => {
                    return Err(OrchestratorError::ProcessFailed {
                        model: model.to_string(),
                        reason: reason.clone(),
                    });
                }
                ProcessState::Checkpointed { .. } => {
                    // Checkpointed to disk - wake_model will handle restore
                    return Ok(());
                }
                ProcessState::NotStarted => {
                    // Need to start it
                }
            }
        }

        // Acquire operation lock to serialize starts
        let _op_guard = self.operation_lock.lock().await;

        // Double-check state after acquiring lock
        {
            let guard = process.lock().await;
            if matches!(
                guard.state,
                ProcessState::Running { .. }
                    | ProcessState::Starting
                    | ProcessState::Checkpointed { .. }
            ) {
                return Ok(());
            }
        }

        // Start the process
        self.start_process_internal(model, &process).await
    }

    /// Internal: start a vLLM process
    async fn start_process_internal(
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
        let mut args = config.vllm_args();
        let is_lora_base = self.is_lora_mode() && model == LORA_BASE_KEY;
        if is_lora_base && !args.iter().any(|a| a == "--enable-lora") {
            args.push("--enable-lora".to_string());
        }
        debug!(model = %model, args = ?args, "vLLM command args");

        // Spawn process in its own process group so we can kill the entire
        // tree (vLLM spawns child processes like EngineCore that hold GPU memory).
        let mut cmd = Command::new(&self.vllm_command);
        cmd.args(&args)
            .env("NO_COLOR", "1") // Disable color codes in vLLM output
            .process_group(0);
        if is_lora_base {
            cmd.env("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True");
        }

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

            let stdout_file = std::fs::File::create(&stdout_path).map_err(|e| {
                OrchestratorError::SpawnFailed {
                    model: model.to_string(),
                    reason: format!("Failed to create {}: {}", stdout_path.display(), e),
                }
            })?;
            let stderr_file = std::fs::File::create(&stderr_path).map_err(|e| {
                OrchestratorError::SpawnFailed {
                    model: model.to_string(),
                    reason: format!("Failed to create {}: {}", stderr_path.display(), e),
                }
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
                    // Discover GPU-holding PIDs for cuda-checkpoint-enabled models
                    // (CudaSuspend and Checkpoint). With TP=1 this is a single
                    // EngineCore process; with TP>1 one process per rank.
                    if needs_cuda_checkpoint
                        && let Some(ref child) = guard.child
                        && let Some(pid) = child.id()
                    {
                        let gpu_pids = find_gpu_pids(pid);
                        if gpu_pids.is_empty() {
                            warn!(model = %model, pid, "Could not find GPU processes, will use parent PID for cuda-checkpoint");
                            guard.engine_core_pids = vec![pid];
                        } else {
                            info!(model = %model, pid, gpu_pids = ?gpu_pids, "Found {} GPU process(es)", gpu_pids.len());
                            guard.engine_core_pids = gpu_pids;
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
    async fn check_health(&self, url: &str) -> Result<bool, String> {
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
    async fn check_process_alive(&self, model: &str) {
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
                    if model == LORA_BASE_KEY
                        && let Some(lora) = self.lora_runtime()
                    {
                        *lora.active_adapter.write().await = None;
                    }
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

    fn lora_base_port(&self) -> Result<u16, OrchestratorError> {
        self.configs
            .get(LORA_BASE_KEY)
            .map(|c| c.port)
            .ok_or_else(|| OrchestratorError::ModelNotFound(LORA_BASE_KEY.to_string()))
    }

    async fn lora_unload_active_if_any(&self) -> Result<(), OrchestratorError> {
        let Some(lora) = self.lora_runtime() else {
            return Ok(());
        };
        let active = lora.active_adapter.read().await.clone();
        if let Some(active) = active {
            self.lora_unload_adapter_by_name(&active).await?;
            *lora.active_adapter.write().await = None;
        }
        Ok(())
    }

    async fn lora_unload_adapter_by_name(&self, lora_name: &str) -> Result<(), OrchestratorError> {
        let base_url = format!("http://localhost:{}", self.lora_base_port()?);
        let body = serde_json::json!({ "lora_name": lora_name }).to_string();
        self.post_request(
            &format!("{}/v1/unload_lora_adapter", base_url),
            Some(&body),
            Duration::from_secs(120),
        )
        .await
        .map_err(|e| OrchestratorError::SleepFailed {
            model: lora_name.to_string(),
            reason: format!("Failed to unload LoRA adapter: {}", e),
        })
    }

    async fn wake_lora_model(&self, model: &str) -> Result<(), OrchestratorError> {
        if !self.lora_has_adapter(model) {
            return Err(OrchestratorError::ModelNotFound(model.to_string()));
        }

        self.check_process_alive(LORA_BASE_KEY).await;
        self.ensure_running_legacy(LORA_BASE_KEY).await?;

        let lora = self.lora_runtime().unwrap();
        if lora.active_adapter.read().await.as_deref() == Some(model) {
            return Ok(());
        }

        self.lora_unload_active_if_any().await?;

        let adapter = self
            .lora_adapter_config(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
        let mut payload = serde_json::json!({
            "lora_name": model,
            "lora_path": adapter.lora_path,
        });
        if let Some(ref base_model_name) = adapter.base_model_name {
            payload["base_model_name"] = serde_json::Value::String(base_model_name.clone());
        }
        if adapter.load_inplace {
            payload["load_inplace"] = serde_json::Value::Bool(true);
        }

        let base_url = format!("http://localhost:{}", self.lora_base_port()?);
        self.post_request(
            &format!("{}/v1/load_lora_adapter", base_url),
            Some(&payload.to_string()),
            Duration::from_secs(120),
        )
        .await
        .map_err(|e| OrchestratorError::WakeFailed {
            model: model.to_string(),
            reason: format!("Failed to load LoRA adapter: {}", e),
        })?;

        *lora.active_adapter.write().await = Some(model.to_string());
        info!(model = %model, "LoRA adapter is now active");
        Ok(())
    }

    async fn sleep_lora_model(
        &self,
        model: &str,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        if !self.lora_has_adapter(model) {
            return Err(OrchestratorError::ModelNotFound(model.to_string()));
        }

        if eviction.process == ProcessStrategy::Stop {
            let result = self
                .sleep_model_legacy(LORA_BASE_KEY, EvictionPolicy::STOP)
                .await;
            if let Some(lora) = self.lora_runtime() {
                *lora.active_adapter.write().await = None;
            }
            return result;
        }

        let Some(lora) = self.lora_runtime() else {
            return Ok(());
        };
        if lora.active_adapter.read().await.as_deref() != Some(model) {
            return Ok(());
        }

        self.lora_unload_adapter_by_name(model).await?;
        *lora.active_adapter.write().await = None;
        info!(model = %model, "LoRA adapter unloaded");
        Ok(())
    }

    /// Wake a model from sleep
    pub async fn wake_model(&self, model: &str) -> Result<(), OrchestratorError> {
        if self.is_lora_mode() {
            return self.wake_lora_model(model).await;
        }
        self.wake_model_legacy(model).await
    }

    async fn wake_model_legacy(&self, model: &str) -> Result<(), OrchestratorError> {
        // Check if the process is still alive before trying to talk to it
        self.check_process_alive(model).await;

        // Check for checkpointed state first — needs CRIU restore, not ensure_running
        {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            let guard = process.lock().await;
            if let ProcessState::Checkpointed {
                ref images_dir,
                eviction,
            } = guard.state
            {
                let images_dir = images_dir.clone();
                let eviction = eviction;
                drop(guard);
                drop(process);
                return self.restore_checkpoint(model, &images_dir, eviction).await;
            }
        }

        // First ensure the process is running
        self.ensure_running(model).await?;

        let process = self
            .processes
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        let config = self
            .configs
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        // Check if already awake, and capture the eviction policy for wake logic
        let eviction = {
            let guard = process.lock().await;
            match &guard.state {
                ProcessState::Running { sleeping: None } => return Ok(()),
                ProcessState::Running {
                    sleeping: Some(policy),
                } => *policy,
                _ => {
                    // Not running at all — ensure_running above should have started it
                    return Ok(());
                }
            }
        };

        info!(model = %model, eviction = ?eviction, "Waking model from sleep");

        // CudaSuspend: toggle CUDA back on, then rebuild NCCL for TP>1
        if eviction.process == ProcessStrategy::CudaSuspend {
            self.cuda_suspend_toggle(model, &process, false).await?;

            // For TP>1: rebuild NCCL communicators after cuda-checkpoint restore
            let gpu_count = process.lock().await.engine_core_pids.len();
            if gpu_count > 1 {
                let base_url = format!("http://localhost:{}", config.port);
                info!(model = %model, tp = gpu_count, "Resuming NCCL communicators");
                self.post_request(
                    &format!("{}/collective_rpc", base_url),
                    Some(r#"{"method": "resume_nccl"}"#),
                    Duration::from_secs(30),
                )
                .await
                .map_err(|e| OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: format!("Failed to resume NCCL: {}", e),
                })?;
            }

            let mut guard = process.lock().await;
            guard.state = ProcessState::Running { sleeping: None };
            info!(model = %model, "Model resumed from CUDA suspend");
            return Ok(());
        }

        let base_url = format!("http://localhost:{}", config.port);

        // If vLLM sleep was used (Offload or Discard weights), wake via API
        if eviction.needs_vllm_sleep() {
            // Step 1: POST /wake_up
            info!(model = %model, "Wake step 1/3: POST /wake_up");
            self.post_request(
                &format!("{}/wake_up", base_url),
                None,
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| {
                error!(model = %model, error = %e, "Wake step 1/3 FAILED: /wake_up returned error");
                OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: e,
                }
            })?;
            info!(model = %model, "Wake step 1/3: /wake_up succeeded");

            // For Discard weights, need to reload weights
            if eviction.weights == WeightStrategy::Discard {
                // Step 2: POST /collective_rpc (reload_weights)
                info!(model = %model, "Wake step 2/3: POST /collective_rpc (reload_weights)");

                if let Err(e) = self
                    .post_request(
                        &format!("{}/collective_rpc", base_url),
                        Some(r#"{"method": "reload_weights"}"#),
                        Duration::from_secs(60),
                    )
                    .await
                {
                    // Reload failed — model is partially woken, consuming GPU memory.
                    // Force it back to sleep to free GPU memory before returning error.
                    error!(model = %model, error = %e, "Wake step 2/3 FAILED: reload_weights returned error");
                    self.force_sleep(model, EvictionPolicy::STOP).await;
                    return Err(OrchestratorError::WakeFailed {
                        model: model.to_string(),
                        reason: e,
                    });
                }
                info!(model = %model, "Wake step 2/3: reload_weights succeeded");

                // Step 3: POST /reset_prefix_cache
                info!(model = %model, "Wake step 3/3: POST /reset_prefix_cache");
                self.post_request(
                    &format!("{}/reset_prefix_cache", base_url),
                    None,
                    Duration::from_secs(30),
                )
                .await
                .map_err(|e| {
                    warn!(model = %model, error = %e, "Wake step 3/3: reset_prefix_cache failed (non-fatal)");
                    // Don't fail on cache reset
                })
                .ok();
                info!(model = %model, "Wake step 3/3: reset_prefix_cache done");
            }
        }

        // Update state
        {
            let mut guard = process.lock().await;
            guard.state = ProcessState::Running { sleeping: None };
        }

        info!(model = %model, "Model is now awake");
        Ok(())
    }

    /// Put a model to sleep using the given eviction policy.
    /// Set a model's state to Checkpointed (for CLI --restore-detached and checkpoint_path config).
    ///
    /// Used to indicate a pre-existing checkpoint on disk so that
    /// `wake_model` will run CRIU restore instead of starting fresh.
    pub async fn set_checkpointed(
        &self,
        model: &str,
        images_dir: std::path::PathBuf,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        let process = self
            .processes
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
        let mut guard = process.lock().await;
        guard.state = ProcessState::Checkpointed {
            images_dir,
            eviction,
        };
        Ok(())
    }

    ///
    /// The sleep sequence is:
    /// 1. Apply weight strategy (vLLM sleep API if Offload/Discard)
    /// 2. Apply process strategy (CudaSuspend, Checkpoint, Stop, or KeepRunning)
    pub async fn sleep_model(
        &self,
        model: &str,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        if self.is_lora_mode() {
            return self.sleep_lora_model(model, eviction).await;
        }
        self.sleep_model_legacy(model, eviction).await
    }

    async fn sleep_model_legacy(
        &self,
        model: &str,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        // Check if the process is still alive before trying to talk to it
        self.check_process_alive(model).await;

        let process = self
            .processes
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        let config = self
            .configs
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        // Check if already sleeping or not running.
        // Allow Stop/Checkpoint/CudaSuspend to proceed even if already sleeping —
        // they guarantee GPU memory is freed (e.g. after a failed wake).
        {
            let guard = process.lock().await;
            match &guard.state {
                ProcessState::Running {
                    sleeping: Some(_), ..
                } if matches!(eviction.process, ProcessStrategy::KeepRunning) => {
                    return Ok(());
                }
                ProcessState::Running { .. } => {
                    // Proceed with sleep (awake, or sleeping but escalating)
                }
                ProcessState::Checkpointed { .. } if eviction.process == ProcessStrategy::Stop => {
                    // Allow Stop to clean up checkpoint images
                }
                _ => return Ok(()), // Not running, nothing to sleep
            }
        }

        info!(model = %model, eviction = ?eviction, "Putting model to sleep");

        // Process strategy: Stop — kill the process entirely
        if eviction.process == ProcessStrategy::Stop {
            let mut guard = process.lock().await;

            // If checkpointed, just clean up images dir and reset state
            if let ProcessState::Checkpointed { ref images_dir, .. } = guard.state {
                info!(model = %model, "Cleaning up checkpoint images");
                let _ = std::fs::remove_dir_all(images_dir);
                guard.state = ProcessState::NotStarted;
                guard.parent_pid = None;
                guard.engine_core_pids.clear();
                return Ok(());
            }

            if let Some(ref mut child) = guard.child {
                info!(model = %model, "Stopping vLLM process group");
                if let Some(pid) = child.id() {
                    kill_process_group(pid);
                } else {
                    let _ = child.kill().await;
                }
                let _ = child.wait().await;
            }
            guard.child = None;
            guard.parent_pid = None;
            guard.engine_core_pids.clear();
            guard.state = ProcessState::NotStarted;
            info!(model = %model, "vLLM process stopped");
            return Ok(());
        }

        let base_url = format!("http://localhost:{}", config.port);

        // Step 1: Apply weight strategy via vLLM sleep API (if Offload or Discard)
        if let Some(sleep_level) = eviction.vllm_sleep_level() {
            let url = format!("{}/sleep?level={}", base_url, sleep_level);
            self.post_request(&url, None, Duration::from_secs(120))
                .await
                .map_err(|e| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: e,
                })?;
            info!(model = %model, sleep_level, "vLLM sleep applied");
        }

        // Step 2: Apply process strategy
        match eviction.process {
            ProcessStrategy::KeepRunning => {
                // Process stays alive, just update state
            }
            ProcessStrategy::CudaSuspend => {
                let gpu_count = process.lock().await.engine_core_pids.len();

                // For TP>1: tear down NCCL communicators before cuda-checkpoint
                // (NCCL IPC handles cannot be checkpointed)
                if gpu_count > 1 {
                    info!(model = %model, tp = gpu_count, "Suspending NCCL communicators");
                    self.post_request(
                        &format!("{}/collective_rpc", base_url),
                        Some(r#"{"method": "suspend_nccl"}"#),
                        Duration::from_secs(30),
                    )
                    .await
                    .map_err(|e| OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!("Failed to suspend NCCL: {}", e),
                    })?;
                }

                self.cuda_suspend_toggle(model, &process, true).await?;
                let mut guard = process.lock().await;
                guard.state = ProcessState::Running {
                    sleeping: Some(eviction),
                };
                info!(model = %model, "Model CUDA-suspended (GPU freed, state in host RAM)");
                return Ok(());
            }
            ProcessStrategy::Checkpoint => {
                let gpu_count = process.lock().await.engine_core_pids.len();

                // For TP>1: tear down NCCL communicators before checkpoint
                if gpu_count > 1 {
                    info!(model = %model, tp = gpu_count, "Suspending NCCL communicators");
                    self.post_request(
                        &format!("{}/collective_rpc", base_url),
                        Some(r#"{"method": "suspend_nccl"}"#),
                        Duration::from_secs(30),
                    )
                    .await
                    .map_err(|e| OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!("Failed to suspend NCCL: {}", e),
                    })?;
                }

                return self.checkpoint_model(model, &process, eviction).await;
            }
            ProcessStrategy::Stop => unreachable!(), // handled above
        }

        // Update state (for KeepRunning path)
        {
            let mut guard = process.lock().await;
            guard.state = ProcessState::Running {
                sleeping: Some(eviction),
            };
        }

        info!(model = %model, "Model is now sleeping");
        Ok(())
    }

    /// Toggle CUDA suspend/resume via `cuda-checkpoint --toggle`.
    ///
    /// When `suspend` is true, CUDA state is suspended (VRAM copied to host RAM,
    /// GPU memory freed). When false, CUDA state is resumed (copied back to GPU).
    /// The process stays alive throughout.
    ///
    /// With TP>1, toggles all GPU-holding processes in parallel.
    async fn cuda_suspend_toggle(
        &self,
        model: &str,
        process: &Arc<Mutex<ManagedProcess>>,
        suspend: bool,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "CudaSuspend requires checkpoint config (for cuda_checkpoint_path)"
                        .to_string(),
                })?;

        let pids = {
            let guard = process.lock().await;
            if guard.engine_core_pids.is_empty() {
                return Err(OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "No GPU PIDs available for cuda-checkpoint".to_string(),
                });
            }
            guard.engine_core_pids.clone()
        };

        let action = if suspend { "suspend" } else { "resume" };
        info!(model = %model, pids = ?pids, action, "cuda-checkpoint --toggle ({} process(es), parallel)", pids.len());

        let mut set = tokio::task::JoinSet::new();
        for pid in &pids {
            let pid = *pid;
            let cuda_checkpoint_path = ckpt_cfg.cuda_checkpoint_path.clone();
            set.spawn(async move {
                let output = maybe_sudo(&cuda_checkpoint_path)
                    .args(["--toggle", "--pid", &pid.to_string()])
                    .output()
                    .await
                    .map_err(|e| (pid, format!("Failed to run cuda-checkpoint: {}", e)))?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err((pid, format!("cuda-checkpoint failed: {}", stderr)));
                }
                Ok(pid)
            });
        }

        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok(pid)) => {
                    debug!(model = %model, pid, action, "cuda-checkpoint toggled");
                }
                Ok(Err((pid, reason))) => {
                    return Err(OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!(
                            "cuda-checkpoint --toggle ({}) failed for PID {}: {}",
                            action, pid, reason
                        ),
                    });
                }
                Err(e) => {
                    return Err(OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!("cuda-checkpoint task panicked: {}", e),
                    });
                }
            }
        }

        info!(model = %model, action, "cuda-checkpoint toggle succeeded for all {} process(es)", pids.len());
        Ok(())
    }

    /// Checkpoint a model to disk using cuda-checkpoint + CRIU.
    ///
    /// 1. Toggle CUDA state (suspends GPU, copies VRAM to host memory)
    /// 2. Dump the process tree via CRIU (writes everything to disk, kills process)
    /// 3. Update state to Checkpointed
    async fn checkpoint_model(
        &self,
        model: &str,
        process: &Arc<Mutex<ManagedProcess>>,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "Checkpoint level requested but no checkpoint config".to_string(),
                })?;

        let parent_pid = {
            let guard = process.lock().await;
            guard
                .child
                .as_ref()
                .and_then(|c| c.id())
                .or(guard.parent_pid)
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "No child PID available for checkpoint".to_string(),
                })?
        };

        // Prepare images directory (clean up old checkpoint first)
        let images_dir = ckpt_cfg.images_dir.join(model).join("images");
        if images_dir.exists() {
            std::fs::remove_dir_all(&images_dir).map_err(|e| OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("Failed to clean old checkpoint: {}", e),
            })?;
        }
        std::fs::create_dir_all(&images_dir).map_err(|e| OrchestratorError::SleepFailed {
            model: model.to_string(),
            reason: format!("Failed to create images dir: {}", e),
        })?;

        // Step 1: Toggle CUDA checkpoint on all GPU processes.
        // This suspends CUDA state (copies VRAM to host RAM, frees GPU memory)
        // for each GPU-holding process. With TP>1, all ranks are toggled in
        // parallel. The CRIU CUDA plugin gracefully handles pre-checkpointed
        // processes: pause_devices skips the lock step, checkpoint_devices
        // skips the checkpoint step, and resume_devices_late always restores.
        //
        // NOTE: Skip pre-toggle — let the CRIU CUDA plugin handle
        // cuda-checkpoint internally. On driver 580+ the plugin expects to
        // find the restore thread, which is gone after --toggle.
        info!(model = %model, "Checkpoint step 1/2: skipping pre-toggle (CRIU plugin handles cuda-checkpoint)");

        // Clean up stale CRIU link_remap artifacts in /dev/shm before dump.
        // CRIU's --link-remap creates temporary hardlinks named link_remap.N
        // in /dev/shm for POSIX semaphores. These aren't cleaned up after
        // dump and cause "File exists" errors on subsequent dumps.
        if let Ok(entries) = std::fs::read_dir("/dev/shm") {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("link_remap.") {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        // Step 2: CRIU dump (snapshots process tree to disk, kills it)
        info!(model = %model, parent_pid, "Checkpoint step 2/2: criu dump");
        let criu_dump = maybe_sudo(&ckpt_cfg.criu_path)
            .args([
                "dump",
                "--shell-job",
                "--ext-unix-sk",
                "--tcp-established",
                "--link-remap",
                "--enable-external-masters",
                "-L",
                &ckpt_cfg.cuda_plugin_dir,
                "--images-dir",
                &images_dir.to_string_lossy(),
                "--tree",
                &parent_pid.to_string(),
            ])
            .output()
            .await
            .map_err(|e| OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("Failed to run criu dump: {}", e),
            })?;

        if !criu_dump.status.success() {
            let stderr = String::from_utf8_lossy(&criu_dump.stderr);
            error!(model = %model, "criu dump failed: {}", stderr);
            return Err(OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("criu dump failed: {}", stderr),
            });
        }

        // CRIU dump kills the process after snapshotting, so clean up our handle
        {
            let mut guard = process.lock().await;
            // The process is gone — try_wait to reap the zombie
            if let Some(ref mut child) = guard.child {
                let _ = child.try_wait();
            }
            guard.child = None;
            guard.state = ProcessState::Checkpointed {
                images_dir: images_dir.clone(),
                eviction,
            };
        }

        info!(model = %model, images_dir = %images_dir.display(), "Model checkpointed to disk");

        // Upload to S3 if object store is configured
        if let Some(ref obj_cfg) = ckpt_cfg.object_store {
            match crate::object_store::CheckpointStore::new(obj_cfg) {
                Ok(store) => {
                    info!(model = %model, "Uploading checkpoint to object store");
                    if let Err(e) = store.upload_checkpoint(model, &images_dir).await {
                        warn!(model = %model, error = %e,
                              "Failed to upload checkpoint to object store (local copy preserved)");
                    }
                }
                Err(e) => {
                    warn!(model = %model, error = %e, "Failed to initialize object store client");
                }
            }
        }

        Ok(())
    }

    /// Restore a model from a CRIU checkpoint.
    ///
    /// 1. Run criu restore (process comes back with original PID, all CUDA state intact)
    /// 2. Health-check the vLLM endpoint (should be immediately ready)
    /// 3. Update state to Running
    async fn restore_checkpoint(
        &self,
        model: &str,
        images_dir: &Path,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: "Checkpoint config missing for restore".to_string(),
                })?;

        let config = self
            .configs
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        info!(model = %model, images_dir = %images_dir.display(), "Restoring from CRIU checkpoint");

        // Download from S3 if local images are missing
        let needs_download = !images_dir.exists()
            || std::fs::read_dir(images_dir)
                .map(|mut d| d.next().is_none())
                .unwrap_or(true);

        if needs_download {
            if let Some(ref obj_cfg) = ckpt_cfg.object_store {
                let store = crate::object_store::CheckpointStore::new(obj_cfg).map_err(|e| {
                    OrchestratorError::WakeFailed {
                        model: model.to_string(),
                        reason: format!("Failed to init object store: {}", e),
                    }
                })?;

                info!(model = %model, "Local checkpoint not found, downloading from S3");
                store
                    .download_checkpoint(model, images_dir)
                    .await
                    .map_err(|e| OrchestratorError::WakeFailed {
                        model: model.to_string(),
                        reason: format!("Failed to download checkpoint from S3: {}", e),
                    })?;
            } else {
                return Err(OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: format!(
                        "Checkpoint images not found at {} and no object store configured",
                        images_dir.display()
                    ),
                });
            }
        }

        // Run criu restore
        let criu_restore = maybe_sudo(&ckpt_cfg.criu_path)
            .args([
                "restore",
                "--shell-job",
                "--ext-unix-sk",
                "--tcp-established",
                "--enable-external-masters",
                "-L",
                &ckpt_cfg.cuda_plugin_dir,
                "--restore-detached",
                "--images-dir",
                &images_dir.to_string_lossy(),
            ])
            .output()
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Failed to run criu restore: {}", e),
            })?;

        if !criu_restore.status.success() {
            let stderr = String::from_utf8_lossy(&criu_restore.stderr);
            error!(model = %model, "criu restore failed: {}", stderr);
            return Err(OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("criu restore failed: {}", stderr),
            });
        }

        info!(model = %model, "criu restore succeeded, health-checking endpoint");

        // Health-check: the restored process should be immediately ready
        // (all state including CUDA graphs is preserved)
        let health_url = format!("http://localhost:{}/health", config.port);
        let mut ready = false;
        for attempt in 0..30 {
            match self.check_health(&health_url).await {
                Ok(true) => {
                    ready = true;
                    break;
                }
                _ => {
                    debug!(model = %model, attempt, "Post-restore health check pending...");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }

        if !ready {
            return Err(OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: "Restored process failed health check".to_string(),
            });
        }

        // For TP>1: rebuild NCCL communicators after restore
        // (they were torn down before checkpoint via suspend_nccl)
        let gpu_count = {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            process.lock().await.engine_core_pids.len()
        };
        // Update state before NCCL resume — the process is already live after
        // CRIU restore, so mark it Running so cleanup paths (e.g. force_sleep)
        // correctly kill the process rather than just deleting images.
        {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            let mut guard = process.lock().await;
            guard.state = ProcessState::Running { sleeping: None };
            // Note: child handle is not available after criu restore --restore-detached.
            // The process runs independently. We track it by PID / health check.
            // Engine core PID is preserved from before checkpoint.
        }

        // For TP>1: rebuild NCCL communicators after restore
        if gpu_count > 1 {
            let base_url = format!("http://localhost:{}", config.port);
            info!(model = %model, tp = gpu_count, "Resuming NCCL communicators");
            self.post_request(
                &format!("{}/collective_rpc", base_url),
                Some(r#"{"method": "resume_nccl"}"#),
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Failed to resume NCCL: {}", e),
            })?;
        }

        // If vLLM sleep was used before checkpoint, run the wake sequence
        // (wake_up → reload_weights → reset_prefix_cache)
        if eviction.needs_vllm_sleep() {
            let base_url = format!("http://localhost:{}", config.port);

            info!(model = %model, "Post-restore: waking vLLM (POST /wake_up)");
            self.post_request(
                &format!("{}/wake_up", base_url),
                None,
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Post-restore wake_up failed: {}", e),
            })?;

            if eviction.weights == WeightStrategy::Discard {
                info!(model = %model, "Post-restore: reloading weights (POST /collective_rpc)");
                self.post_request(
                    &format!("{}/collective_rpc", base_url),
                    Some(r#"{"method": "reload_weights"}"#),
                    Duration::from_secs(60),
                )
                .await
                .map_err(|e| OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: format!("Post-restore reload_weights failed: {}", e),
                })?;

                info!(model = %model, "Post-restore: resetting prefix cache");
                self.post_request(
                    &format!("{}/reset_prefix_cache", base_url),
                    None,
                    Duration::from_secs(30),
                )
                .await
                .ok(); // non-fatal
            }
        }

        if !ckpt_cfg.keep_images {
            // Clean up checkpoint images to free disk space. For large models
            // this can be tens of GB.
            if let Err(e) = std::fs::remove_dir_all(images_dir) {
                warn!(model = %model, error = %e, "Failed to clean up checkpoint images (non-fatal)");
            } else {
                info!(model = %model, "Cleaned up checkpoint images");
            }
        }

        info!(model = %model, "Model restored from checkpoint and ready");
        Ok(())
    }

    /// Force a model to sleep, escalating to Stop if the initial policy fails.
    ///
    /// This is a guaranteed-cleanup method: it logs errors but **never returns `Err`**.
    /// Used to clean up partially-woken models that hold GPU memory.
    pub async fn force_sleep(&self, model: &str, eviction: EvictionPolicy) {
        if let Err(e) = self.sleep_model(model, eviction).await {
            if eviction.process == ProcessStrategy::Stop {
                // Already at the most drastic level, nothing to escalate to
                error!(model, error = %e, "force_sleep: Stop failed");
            } else {
                warn!(model, error = %e, "force_sleep: {:?} failed, escalating to Stop", eviction);
                if let Err(e2) = self.sleep_model(model, EvictionPolicy::STOP).await {
                    error!(model, error = %e2, "force_sleep: Stop escalation also failed");
                }
            }
        }
    }

    /// Check if a model is ready
    pub async fn is_ready(&self, model: &str) -> bool {
        if self.is_lora_mode() {
            if !self.lora_has_adapter(model) {
                return false;
            }
            let Some(process) = self.processes.get(LORA_BASE_KEY) else {
                return false;
            };
            let base_ready = {
                let guard = process.lock().await;
                matches!(guard.state, ProcessState::Running { sleeping: None })
            };
            if !base_ready {
                return false;
            }

            return self
                .lora_runtime()
                .unwrap()
                .active_adapter
                .read()
                .await
                .as_deref()
                == Some(model);
        }

        let Some(process) = self.processes.get(model) else {
            return false;
        };
        let guard = process.lock().await;
        matches!(guard.state, ProcessState::Running { sleeping: None })
    }

    /// Helper to make POST requests with retries.
    ///
    /// All vLLM control endpoints (wake_up, sleep, collective_rpc, etc.) are
    /// idempotent, so retries are safe. We retry on transient failures (connection
    /// errors, timeouts, 5xx) to avoid escalating to drastic measures (like
    /// killing the process) when the endpoint is just momentarily unresponsive.
    async fn post_request(
        &self,
        url: &str,
        body: Option<&str>,
        timeout: Duration,
    ) -> Result<(), String> {
        use http_body_util::{BodyExt, Full};
        use hyper::Request;

        let client: hyper_util::client::legacy::Client<_, Full<bytes::Bytes>> =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http();

        let uri: hyper::Uri = url.parse().map_err(|e| format!("Invalid URL: {}", e))?;
        let has_body = body.is_some();
        let body_bytes: Vec<u8> = body.map(|b| b.as_bytes().to_vec()).unwrap_or_default();

        let mut last_err = String::new();
        for attempt in 0..3u32 {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt - 1));
                warn!(url, attempt, delay = ?delay, "Retrying request");
                tokio::time::sleep(delay).await;
            }

            let request_body = Full::new(bytes::Bytes::from(body_bytes.clone()));
            let mut req_builder = Request::builder().method("POST").uri(uri.clone());
            if has_body {
                req_builder = req_builder.header("Content-Type", "application/json");
            }

            let request = match req_builder.body(request_body) {
                Ok(r) => r,
                Err(e) => return Err(format!("Failed to build request: {}", e)),
            };

            match tokio::time::timeout(timeout, client.request(request)).await {
                Ok(Ok(response)) if response.status().is_success() => return Ok(()),
                Ok(Ok(response)) => {
                    let status = response.status();
                    let body_str = match response.into_body().collect().await {
                        Ok(collected) => {
                            let bytes = collected.to_bytes();
                            String::from_utf8_lossy(&bytes).into_owned()
                        }
                        Err(e) => format!("(failed to read body: {e})"),
                    };
                    let body_preview = if body_str.len() > 2000 {
                        format!(
                            "{}...[truncated, {} total]",
                            &body_str[..2000],
                            body_str.len()
                        )
                    } else {
                        body_str
                    };
                    warn!(url, %status, body = %body_preview, "vLLM endpoint returned error");
                    last_err = format!("HTTP {status}: {body_preview}");
                }
                Ok(Err(e)) => {
                    last_err = format!("Request failed: {}", e);
                }
                Err(_) => {
                    last_err = "Request timeout".to_string();
                }
            }
        }

        Err(last_err)
    }
}

/// Errors from the orchestrator
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("failed to spawn process for {model}: {reason}")]
    SpawnFailed { model: String, reason: String },

    #[error("startup timeout for {model}")]
    StartupTimeout { model: String },

    #[error("process failed for {model}: {reason}")]
    ProcessFailed { model: String, reason: String },

    #[error("failed to wake {model}: {reason}")]
    WakeFailed { model: String, reason: String },

    #[error("failed to sleep {model}: {reason}")]
    SleepFailed { model: String, reason: String },
}

impl Drop for Orchestrator {
    fn drop(&mut self) {
        // Kill all child processes to avoid zombies
        // This is especially important for tests
        for entry in self.processes.iter() {
            if let Ok(mut guard) = entry.value().try_lock()
                && let Some(ref mut child) = guard.child
            {
                let _ = child.start_kill();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let mut configs = HashMap::new();
        configs.insert(
            "model-a".to_string(),
            ModelConfig {
                model_path: "test/model".to_string(),
                port: 8001,
                extra_args: vec![],
                eviction: EvictionPolicy::from(1),
                checkpoint_path: None,
            },
        );

        let orchestrator = Orchestrator::new(configs);
        assert_eq!(orchestrator.registered_models(), vec!["model-a"]);
    }

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
