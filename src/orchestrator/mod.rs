//! Orchestrator - manages vLLM process lifecycle
//!
//! The orchestrator is responsible for:
//! - Lazily starting vLLM processes on first request
//! - Tracking process state (NotStarted, Starting, Running, etc.)
//! - Health checking to confirm processes are ready
//! - Coordinating with the switcher for wake/sleep operations

mod criu;
mod cuda;
mod process;
mod vllm;

use crate::config::{CheckpointConfig, ModelConfig};
use crate::types::{EvictionPolicy, ProcessStrategy, WeightStrategy};
use anyhow::Result;
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Child;
use tokio::sync::{Mutex, Notify};
use tracing::{error, info, warn};

/// Kill an entire process group by sending SIGKILL to -pgid.
#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: We're sending SIGKILL to a process group we spawned.
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
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
    /// PIDs of all processes with active CUDA contexts. Includes the parent
    /// vLLM process and EngineCore worker(s). Used for cuda-checkpoint toggle.
    engine_core_pids: Vec<u32>,
    /// Number of tensor-parallel GPU workers (processes with nvidia device maps).
    /// Used to decide whether NCCL suspend/resume is needed (only for TP > 1).
    tp_size: usize,
    /// Parent PID of the vLLM process. Stored separately because after
    /// CRIU restore with --restore-detached, the tokio Child handle is gone
    /// but the process is running at its original PID.
    parent_pid: Option<u32>,
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
                    tp_size: 0,
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
            startup_timeout: Duration::from_secs(1800), // 30 minutes for large MoE models with DeepGEMM warmup
            vllm_command,
            checkpoint_config,
        }
    }

    /// Get the current state of a model's process
    pub async fn process_state(&self, model: &str) -> Option<ProcessState> {
        let process = self.processes.get(model)?;
        let guard = process.lock().await;
        Some(guard.state.clone())
    }

    /// Get all registered model names
    pub fn registered_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    /// Get the configured port for a model
    pub fn model_port(&self, model: &str) -> Option<u16> {
        self.configs.get(model).map(|c| c.port)
    }

    /// Get the configured model path (HuggingFace ID or local path) for a model
    pub fn model_path(&self, model: &str) -> Option<String> {
        self.configs.get(model).map(|c| c.model_path.clone())
    }

    /// Get the effective eviction policy for a model (override > config)
    pub fn eviction_policy_for(&self, model: &str) -> Option<EvictionPolicy> {
        if let Some(policy) = self.eviction_overrides.get(model) {
            return Some(*policy);
        }
        self.configs.get(model).map(|c| c.eviction)
    }

    /// Check if a model is currently in the Checkpointed state
    pub fn is_checkpointed(&self, model: &str) -> bool {
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
        self.eviction_overrides.insert(model.to_string(), policy);
    }

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

    /// Ensure a model's process is running and ready
    ///
    /// This will:
    /// 1. Start the process if not started
    /// 2. Wait for it to become healthy
    /// 3. Return once the model is ready to serve requests
    pub async fn ensure_running(&self, model: &str) -> Result<(), OrchestratorError> {
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

    /// Wake a model from sleep
    pub async fn wake_model(&self, model: &str) -> Result<(), OrchestratorError> {
        // Check if the process is still alive before trying to talk to it
        self.check_process_alive(model).await;

        // Kill any stray GPU processes before attempting to load a model.
        // This prevents OOM from zombie processes left by previous failures.
        #[cfg(unix)]
        self.kill_stray_gpu_processes().await;

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

        let base_url = format!("http://localhost:{}", config.port);

        // CudaSuspend: toggle CUDA back on, then rebuild NCCL for TP>1
        if eviction.process == ProcessStrategy::CudaSuspend {
            self.cuda_suspend_toggle(model, &process, false).await?;

            // For TP>1: rebuild NCCL communicators after cuda-checkpoint restore
            let gpu_count = process.lock().await.tp_size;
            if gpu_count > 1 {
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

            // If weights were retained (no vLLM sleep), CUDA restore fully
            // reconstructed GPU state — we're done.
            if !eviction.needs_vllm_sleep() {
                let mut guard = process.lock().await;
                guard.state = ProcessState::Running { sleeping: None };
                info!(model = %model, "Model resumed from CUDA suspend");
                return Ok(());
            }

            info!(model = %model, "CUDA resumed, continuing to vLLM wake sequence");
        }

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
    ///
    /// The sleep sequence is:
    /// 1. Apply weight strategy (vLLM sleep API if Offload/Discard)
    /// 2. Apply process strategy (CudaSuspend, Checkpoint, Stop, or KeepRunning)
    pub async fn sleep_model(
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
                guard.tp_size = 0;
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
            guard.tp_size = 0;
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
                let gpu_count = process.lock().await.tp_size;

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
                let gpu_count = process.lock().await.tp_size;

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
        let Some(process) = self.processes.get(model) else {
            return false;
        };

        let guard = process.lock().await;
        matches!(guard.state, ProcessState::Running { sleeping: None })
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
    use crate::config::ModelConfig;

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
}
