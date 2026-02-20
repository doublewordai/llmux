use regex::Regex;
use reqwest::StatusCode;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Notify, RwLock, oneshot};
use tracing::{debug, error, info, trace, warn};

const RELEASE_RESUME_TAGS: [&str; 2] = ["kv_cache", "weights"];
const READINESS_HTTP_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct SgLangMuxOptions {
    pub ready_timeout: Duration,
    pub request_timeout: Duration,
    pub poll_interval: Duration,
    /// Directory where per-model stdout/stderr files are written.
    pub log_dir: PathBuf,
}

impl Default for SgLangMuxOptions {
    fn default() -> Self {
        Self {
            ready_timeout: Duration::from_secs(120),
            request_timeout: Duration::from_secs(60),
            poll_interval: Duration::from_millis(100),
            log_dir: PathBuf::from("sglangmux-logs"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub model_name: String,
    pub port: u16,
    pub script_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLogPaths {
    pub stdout: PathBuf,
    pub stderr: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    NotStarted,
    Starting,
    Running { sleeping: bool },
    Failed { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitcherState {
    Idle,
    Active { model: String },
    Switching { from: Option<String>, to: String },
}

#[derive(Debug, thiserror::Error)]
pub enum SgLangMuxError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("failed to read script {path}: {source}")]
    ScriptRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("script {script} is missing required assignment: {field}=...")]
    MissingScriptMetadata {
        script: PathBuf,
        field: &'static str,
    },

    #[error("script {script} has invalid PORT value: {value}")]
    InvalidPort { script: PathBuf, value: String },

    #[error("duplicate model name in scripts: {0}")]
    DuplicateModel(String),

    #[error("duplicate port in scripts: {0}")]
    DuplicatePort(u16),

    #[error("failed to prepare log file for model {model} at {path}: {source}")]
    LogSetup {
        model: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to spawn process for {model}: {reason}")]
    SpawnFailed { model: String, reason: String },

    #[error("startup timeout for model {model}")]
    StartupTimeout { model: String },

    #[error("process failed for model {model}: {reason}")]
    ProcessFailed { model: String, reason: String },

    #[error("HTTP request failed for model {model}: {reason}")]
    Http { model: String, reason: String },

    #[error("request timed out waiting for model")]
    RequestTimeout,

    #[error("internal error: {0}")]
    Internal(String),
}

struct PendingRequest {
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    queued_at: Instant,
    ready_tx: oneshot::Sender<Result<(), SgLangMuxError>>,
}

struct ModelLockState {
    in_flight: AtomicUsize,
    in_flight_changed: Arc<Notify>,
    draining: AtomicBool,
    pending: Mutex<Vec<PendingRequest>>,
}

impl Default for ModelLockState {
    fn default() -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            in_flight_changed: Arc::new(Notify::new()),
            draining: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }
    }
}

struct ManagedProcess {
    state: ProcessState,
    child: Option<Child>,
}

impl Default for ManagedProcess {
    fn default() -> Self {
        Self {
            state: ProcessState::NotStarted,
            child: None,
        }
    }
}

struct ModelEntry {
    spec: ModelSpec,
    process: Mutex<ManagedProcess>,
    ready_notify: Arc<Notify>,
    lock_state: Arc<ModelLockState>,
}

impl ModelEntry {
    fn new(spec: ModelSpec) -> Self {
        Self {
            spec,
            process: Mutex::new(ManagedProcess::default()),
            ready_notify: Arc::new(Notify::new()),
            lock_state: Arc::new(ModelLockState::default()),
        }
    }
}

struct SgLangMuxInner {
    models: HashMap<String, Arc<ModelEntry>>,
    start_order: Vec<String>,
    options: SgLangMuxOptions,
    client: reqwest::Client,
    switch_state: RwLock<SwitcherState>,
    switch_lock: Mutex<()>,
    operation_lock: Mutex<()>,
}

pub struct SgLangMux {
    inner: Arc<SgLangMuxInner>,
}

impl Clone for SgLangMux {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl SgLangMux {
    pub fn from_scripts<I, P>(scripts: I, options: SgLangMuxOptions) -> Result<Self, SgLangMuxError>
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        debug!(
            ready_timeout_ms = options.ready_timeout.as_millis(),
            request_timeout_ms = options.request_timeout.as_millis(),
            poll_interval_ms = options.poll_interval.as_millis(),
            log_dir = %options.log_dir.display(),
            "Building SgLangMux from launch scripts"
        );
        let mut models = HashMap::new();
        let mut start_order = Vec::new();
        let mut ports = HashSet::new();

        for script in scripts {
            let script_path = script.into();
            debug!(script = %script_path.display(), "Parsing model launch script");
            let spec = parse_model_spec(script_path)?;
            debug!(
                model = %spec.model_name,
                port = spec.port,
                script = %spec.script_path.display(),
                "Parsed model spec"
            );

            if models.contains_key(&spec.model_name) {
                warn!(model = %spec.model_name, "Duplicate model name in scripts");
                return Err(SgLangMuxError::DuplicateModel(spec.model_name));
            }

            if !ports.insert(spec.port) {
                warn!(port = spec.port, "Duplicate port in scripts");
                return Err(SgLangMuxError::DuplicatePort(spec.port));
            }

            start_order.push(spec.model_name.clone());
            models.insert(spec.model_name.clone(), Arc::new(ModelEntry::new(spec)));
        }

        info!(
            model_count = start_order.len(),
            models = ?start_order,
            "SgLangMux initialized with parsed models"
        );
        Ok(Self {
            inner: Arc::new(SgLangMuxInner {
                models,
                start_order,
                options,
                client: reqwest::Client::new(),
                switch_state: RwLock::new(SwitcherState::Idle),
                switch_lock: Mutex::new(()),
                operation_lock: Mutex::new(()),
            }),
        })
    }

    pub fn model_specs(&self) -> Vec<ModelSpec> {
        self.inner
            .start_order
            .iter()
            .filter_map(|name| self.inner.models.get(name).map(|entry| entry.spec.clone()))
            .collect()
    }

    pub fn model_log_paths(&self) -> HashMap<String, ModelLogPaths> {
        self.inner
            .models
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    build_log_paths(
                        &self.inner.options.log_dir,
                        &entry.spec.model_name,
                        entry.spec.port,
                    ),
                )
            })
            .collect()
    }

    pub fn is_registered(&self, model: &str) -> bool {
        self.inner.models.contains_key(model)
    }

    pub async fn switch_state(&self) -> SwitcherState {
        self.inner.switch_state.read().await.clone()
    }

    pub async fn active_model(&self) -> Option<String> {
        match &*self.inner.switch_state.read().await {
            SwitcherState::Active { model } => Some(model.clone()),
            _ => None,
        }
    }

    pub async fn process_state(&self, model: &str) -> Option<ProcessState> {
        let entry = self.inner.models.get(model)?;
        let process = entry.process.lock().await;
        Some(process.state.clone())
    }

    pub fn in_flight_count(&self, model: &str) -> usize {
        self.inner
            .models
            .get(model)
            .map(|entry| entry.lock_state.in_flight.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    pub async fn bootstrap_sequential(&self) -> Result<(), SgLangMuxError> {
        let model_count = self.inner.start_order.len();
        info!(model_count, "Starting sequential bootstrap");

        if model_count == 0 {
            *self.inner.switch_state.write().await = SwitcherState::Idle;
            info!("No models configured; switch state set to idle");
            return Ok(());
        }

        for (idx, model) in self.inner.start_order.iter().enumerate() {
            info!(
                model = %model,
                step = idx + 1,
                total = model_count,
                "Bootstrap: ensuring model process is running"
            );
            self.ensure_running(model).await?;

            if idx + 1 < model_count {
                // Abort any background generation, then park memory before launching next model.
                debug!(model = %model, "Bootstrap: parking intermediate model (pause + sleep)");
                self.pause_generation(model).await?;
                self.sleep_model(model).await?;
            }
        }

        if let Some(last) = self.inner.start_order.last() {
            *self.inner.switch_state.write().await = SwitcherState::Active {
                model: last.clone(),
            };
            info!(model = %last, "Bootstrap complete; final model left active");
            self.notify_pending(last, Ok(())).await;
        }

        Ok(())
    }

    pub async fn ensure_model_ready(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "ensure_model_ready called");
        let model_entry = self.model_entry(model)?;

        {
            let state = self.inner.switch_state.read().await;
            if let SwitcherState::Active {
                model: active_model,
            } = &*state
                && active_model == model
            {
                // Active model can still be stale if the process exited unexpectedly.
                // Always re-validate liveness and wake state before allowing a request through.
                trace!(model = %model, "Model already active; validating liveness via wake_model");
                return self.wake_model(model).await;
            }
        }

        let (ready_tx, ready_rx) = oneshot::channel();

        {
            let mut queue = model_entry.lock_state.pending.lock().await;
            queue.push(PendingRequest {
                model: model.to_string(),
                queued_at: Instant::now(),
                ready_tx,
            });
            debug!(
                model = %model,
                queue_depth = queue.len(),
                "Queued pending request waiting for model activation"
            );
        }

        trace!(model = %model, "Evaluating whether to trigger switch");
        self.maybe_trigger_switch(model).await;

        match tokio::time::timeout(self.inner.options.request_timeout, ready_rx).await {
            Ok(Ok(result)) => {
                match &result {
                    Ok(()) => debug!(model = %model, "Pending request released: model ready"),
                    Err(error) => warn!(
                        model = %model,
                        error = %error,
                        "Pending request released with model readiness error"
                    ),
                }
                result
            }
            Ok(Err(_)) => {
                error!(model = %model, "Pending readiness channel closed unexpectedly");
                Err(SgLangMuxError::Internal(
                    "pending channel closed".to_string(),
                ))
            }
            Err(_) => {
                warn!(
                    model = %model,
                    timeout_ms = self.inner.options.request_timeout.as_millis(),
                    "Timed out waiting for model to become ready"
                );
                Err(SgLangMuxError::RequestTimeout)
            }
        }
    }

    /// Acquire the exact same llmux-style in-flight lock guard:
    /// increment first, then reject if draining started.
    pub fn acquire_in_flight(&self, model: &str) -> Option<InFlightGuard> {
        let entry = self.inner.models.get(model)?;
        let lock_state = &entry.lock_state;

        let new_in_flight = lock_state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        trace!(
            model = %model,
            in_flight = new_in_flight,
            "Acquired provisional in-flight slot"
        );

        if lock_state.draining.load(Ordering::SeqCst) {
            let restored = lock_state.in_flight.fetch_sub(1, Ordering::SeqCst) - 1;
            lock_state.in_flight_changed.notify_waiters();
            debug!(
                model = %model,
                in_flight = restored,
                "Rejected in-flight slot because model is draining"
            );
            return None;
        }

        trace!(
            model = %model,
            in_flight = new_in_flight,
            "In-flight guard granted"
        );
        Some(InFlightGuard {
            lock_state: Arc::clone(lock_state),
            model_name: model.to_string(),
        })
    }

    pub async fn shutdown_all(&self) {
        for entry in self.inner.models.values() {
            let mut process = entry.process.lock().await;
            if let Some(child) = process.child.as_mut() {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
            process.child = None;
            process.state = ProcessState::NotStarted;
            entry.ready_notify.notify_waiters();
        }

        *self.inner.switch_state.write().await = SwitcherState::Idle;
    }

    fn model_entry(&self, model: &str) -> Result<Arc<ModelEntry>, SgLangMuxError> {
        self.inner
            .models
            .get(model)
            .cloned()
            .ok_or_else(|| SgLangMuxError::ModelNotFound(model.to_string()))
    }

    async fn maybe_trigger_switch(&self, target_model: &str) {
        trace!(target_model = %target_model, "Checking whether switch should be triggered");
        {
            let state = self.inner.switch_state.read().await;
            if let SwitcherState::Switching { to, .. } = &*state
                && to == target_model
            {
                debug!(
                    target_model = %target_model,
                    "Switch already in progress toward target model; skipping trigger"
                );
                return;
            }
        }

        debug!(target_model = %target_model, "Triggering switch execution");
        self.do_switch(target_model).await;
    }

    async fn do_switch(&self, target_model: &str) {
        debug!(target_model = %target_model, "Waiting for switch lock");
        let _guard = self.inner.switch_lock.lock().await;
        debug!(target_model = %target_model, "Switch lock acquired");

        {
            let state = self.inner.switch_state.read().await;
            match &*state {
                SwitcherState::Active { model } if model == target_model => {
                    debug!(
                        target_model = %target_model,
                        "Target model already active; notifying pending requests"
                    );
                    self.notify_pending(target_model, Ok(())).await;
                    return;
                }
                SwitcherState::Switching { to, .. } if to == target_model => {
                    debug!(
                        target_model = %target_model,
                        "Another task already switching to target model; exiting"
                    );
                    return;
                }
                _ => {}
            }
        }

        let from_model = {
            let state = self.inner.switch_state.read().await;
            match &*state {
                SwitcherState::Active { model } => Some(model.clone()),
                _ => None,
            }
        };
        info!(from = ?from_model, to = %target_model, "Starting model switch");

        {
            let mut state = self.inner.switch_state.write().await;
            *state = SwitcherState::Switching {
                from: from_model.clone(),
                to: target_model.to_string(),
            };
        }

        let result = self.perform_switch(from_model.clone(), target_model).await;

        if let Some(from) = from_model.as_ref()
            && let Some(entry) = self.inner.models.get(from)
        {
            entry.lock_state.draining.store(false, Ordering::SeqCst);
            trace!(model = %from, "Cleared draining flag after switch attempt");
        }

        match result {
            Ok(()) => {
                {
                    let mut state = self.inner.switch_state.write().await;
                    *state = SwitcherState::Active {
                        model: target_model.to_string(),
                    };
                }
                info!(model = %target_model, "Switch completed successfully; model is active");
                self.notify_pending(target_model, Ok(())).await;
            }
            Err(error) => {
                {
                    let mut state = self.inner.switch_state.write().await;
                    *state = SwitcherState::Idle;
                }
                error!(
                    target_model = %target_model,
                    error = %error,
                    "Switch failed; switcher returned to idle"
                );
                self.notify_pending(target_model, Err(error)).await;
            }
        }
    }

    async fn perform_switch(
        &self,
        from_model: Option<String>,
        target_model: &str,
    ) -> Result<(), SgLangMuxError> {
        debug!(
            from = ?from_model,
            to = %target_model,
            "Entering perform_switch"
        );
        if let Some(from) = from_model.as_ref() {
            let from_state = self.refresh_process_state(from).await?;
            debug!(
                from = %from,
                state = ?from_state,
                "Evaluated source model process state before draining"
            );
            if matches!(from_state, ProcessState::Running { .. }) {
                let from_entry = self.model_entry(from)?;
                from_entry.lock_state.draining.store(true, Ordering::SeqCst);
                debug!(from = %from, "Set draining flag on source model");

                let drain_start = Instant::now();
                self.wait_for_drain(&from_entry.lock_state).await;
                debug!(
                    from = %from,
                    waited_ms = drain_start.elapsed().as_millis(),
                    "Drain completed; no in-flight requests remain"
                );

                debug!(from = %from, "Pausing generation on source model");
                self.pause_generation(from).await?;
                debug!(from = %from, "Putting source model into sleeping state");
                self.sleep_model(from).await?;
            }
        }

        debug!(target_model = %target_model, "Waking target model");
        self.wake_model(target_model).await
    }

    async fn wait_for_drain(&self, lock_state: &Arc<ModelLockState>) {
        let wait_start = Instant::now();
        let mut logged_wait = false;
        loop {
            let in_flight = lock_state.in_flight.load(Ordering::SeqCst);
            if in_flight == 0 {
                if logged_wait {
                    debug!(
                        waited_ms = wait_start.elapsed().as_millis(),
                        "Drain wait complete"
                    );
                }
                return;
            }
            if !logged_wait {
                debug!(in_flight, "Waiting for in-flight requests to drain");
                logged_wait = true;
            }

            let notify = lock_state.in_flight_changed.notified();
            if lock_state.in_flight.load(Ordering::SeqCst) == 0 {
                if logged_wait {
                    debug!(
                        waited_ms = wait_start.elapsed().as_millis(),
                        "Drain wait complete"
                    );
                }
                return;
            }
            notify.await;
        }
    }

    async fn ensure_running(&self, model: &str) -> Result<(), SgLangMuxError> {
        trace!(model = %model, "ensure_running called");
        self.refresh_process_state(model).await?;
        let entry = self.model_entry(model)?;

        loop {
            let mut should_wait = false;
            {
                let process = entry.process.lock().await;
                match &process.state {
                    ProcessState::Running { sleeping } => {
                        trace!(
                            model = %model,
                            sleeping,
                            "Model process already running"
                        );
                        return Ok(());
                    }
                    ProcessState::Failed { reason } => {
                        error!(
                            model = %model,
                            reason = %reason,
                            "Model process is in failed state"
                        );
                        return Err(SgLangMuxError::ProcessFailed {
                            model: model.to_string(),
                            reason: reason.clone(),
                        });
                    }
                    ProcessState::Starting => {
                        trace!(model = %model, "Model process currently starting; waiting");
                        should_wait = true;
                    }
                    ProcessState::NotStarted => {}
                }
            }

            if should_wait {
                entry.ready_notify.notified().await;
                trace!(model = %model, "Woke from ready notify while waiting for startup");
                continue;
            }
            break;
        }

        debug!(model = %model, "Acquiring operation lock to start model process");
        let _op_guard = self.inner.operation_lock.lock().await;

        {
            let process = entry.process.lock().await;
            match &process.state {
                ProcessState::Running { sleeping } => {
                    trace!(
                        model = %model,
                        sleeping,
                        "Model became running before start attempt"
                    );
                    return Ok(());
                }
                ProcessState::Starting => {
                    trace!(
                        model = %model,
                        "Model already starting by another task; no new start needed"
                    );
                    return Ok(());
                }
                ProcessState::Failed { reason } => {
                    error!(
                        model = %model,
                        reason = %reason,
                        "Model process failed before start attempt"
                    );
                    return Err(SgLangMuxError::ProcessFailed {
                        model: model.to_string(),
                        reason: reason.clone(),
                    });
                }
                ProcessState::NotStarted => {}
            }
        }

        info!(model = %model, "Starting model process");
        self.start_process_internal(&entry).await
    }

    async fn refresh_process_state(&self, model: &str) -> Result<ProcessState, SgLangMuxError> {
        let entry = self.model_entry(model)?;
        let mut process = entry.process.lock().await;

        if let Some(child) = process.child.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => {
                    warn!(
                        model = %model,
                        previous_state = ?process.state,
                        "Model process exited unexpectedly; marking as NotStarted"
                    );
                    process.child = None;
                    process.state = ProcessState::NotStarted;
                }
                Ok(None) => {}
                Err(error) => {
                    error!(
                        model = %model,
                        error = %error,
                        "Failed to poll model process state"
                    );
                    process.child = None;
                    process.state = ProcessState::Failed {
                        reason: format!("failed to check process state: {error}"),
                    };
                }
            }
        }

        Ok(process.state.clone())
    }

    async fn start_process_internal(&self, entry: &Arc<ModelEntry>) -> Result<(), SgLangMuxError> {
        let model = entry.spec.model_name.clone();
        info!(
            model = %model,
            port = entry.spec.port,
            script = %entry.spec.script_path.display(),
            "Launching model process"
        );

        {
            let mut process = entry.process.lock().await;
            process.state = ProcessState::Starting;
        }

        let log_paths = build_log_paths(
            &self.inner.options.log_dir,
            &entry.spec.model_name,
            entry.spec.port,
        );

        if let Err(error) = std::fs::create_dir_all(&self.inner.options.log_dir) {
            error!(
                model = %model,
                dir = %self.inner.options.log_dir.display(),
                error = %error,
                "Failed to create log directory"
            );
            let mut process = entry.process.lock().await;
            process.state = ProcessState::Failed {
                reason: error.to_string(),
            };
            entry.ready_notify.notify_waiters();
            return Err(SgLangMuxError::LogSetup {
                model,
                path: self.inner.options.log_dir.clone(),
                source: error,
            });
        }

        let stdout_file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_paths.stdout)
        {
            Ok(file) => file,
            Err(error) => {
                error!(
                    model = %model,
                    path = %log_paths.stdout.display(),
                    error = %error,
                    "Failed to open stdout log file"
                );
                let mut process = entry.process.lock().await;
                process.state = ProcessState::Failed {
                    reason: error.to_string(),
                };
                entry.ready_notify.notify_waiters();
                return Err(SgLangMuxError::LogSetup {
                    model,
                    path: log_paths.stdout,
                    source: error,
                });
            }
        };

        let stderr_file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_paths.stderr)
        {
            Ok(file) => file,
            Err(error) => {
                error!(
                    model = %model,
                    path = %log_paths.stderr.display(),
                    error = %error,
                    "Failed to open stderr log file"
                );
                let mut process = entry.process.lock().await;
                process.state = ProcessState::Failed {
                    reason: error.to_string(),
                };
                entry.ready_notify.notify_waiters();
                return Err(SgLangMuxError::LogSetup {
                    model,
                    path: log_paths.stderr,
                    source: error,
                });
            }
        };

        let mut cmd = Command::new("bash");
        cmd.arg(&entry.spec.script_path)
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file));

        debug!(
            model = %model,
            script = %entry.spec.script_path.display(),
            "Spawning model process via bash script"
        );
        let child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let reason = error.to_string();
                error!(
                    model = %model,
                    error = %reason,
                    "Failed to spawn model process"
                );
                let mut process = entry.process.lock().await;
                process.state = ProcessState::Failed {
                    reason: reason.clone(),
                };
                entry.ready_notify.notify_waiters();
                return Err(SgLangMuxError::SpawnFailed { model, reason });
            }
        };

        {
            let mut process = entry.process.lock().await;
            process.child = Some(child);
        }

        let deadline = Instant::now() + self.inner.options.ready_timeout;

        loop {
            if Instant::now() > deadline {
                error!(
                    model = %model,
                    timeout_ms = self.inner.options.ready_timeout.as_millis(),
                    "Model startup timed out"
                );
                let mut process = entry.process.lock().await;
                if let Some(child) = process.child.as_mut() {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
                process.child = None;
                process.state = ProcessState::Failed {
                    reason: "startup timeout".to_string(),
                };
                entry.ready_notify.notify_waiters();
                return Err(SgLangMuxError::StartupTimeout { model });
            }

            if self.check_health(entry.spec.port).await {
                let mut process = entry.process.lock().await;
                process.state = ProcessState::Running { sleeping: false };
                entry.ready_notify.notify_waiters();
                info!(model = %model, port = entry.spec.port, "Model process reported healthy");
                return Ok(());
            }

            {
                let mut process = entry.process.lock().await;
                if let Some(child) = process.child.as_mut() {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            let reason = format!("process exited with status {status}");
                            error!(
                                model = %model,
                                reason = %reason,
                                "Model process exited during startup"
                            );
                            process.state = ProcessState::Failed {
                                reason: reason.clone(),
                            };
                            process.child = None;
                            entry.ready_notify.notify_waiters();
                            return Err(SgLangMuxError::ProcessFailed { model, reason });
                        }
                        Ok(None) => {}
                        Err(error) => {
                            let reason = format!("failed to poll child process: {error}");
                            error!(
                                model = %model,
                                reason = %reason,
                                "Failed while polling model process during startup"
                            );
                            process.state = ProcessState::Failed {
                                reason: reason.clone(),
                            };
                            process.child = None;
                            entry.ready_notify.notify_waiters();
                            return Err(SgLangMuxError::ProcessFailed { model, reason });
                        }
                    }
                }
            }

            tokio::time::sleep(self.inner.options.poll_interval).await;
        }
    }

    async fn check_health(&self, port: u16) -> bool {
        // SGLang's /health can temporarily report 503 around sleep/resume transitions.
        // /model_info is a better "server ready" signal when available.
        let model_info_url = format!("http://127.0.0.1:{port}/model_info");
        if let Ok(Ok(response)) = tokio::time::timeout(
            READINESS_HTTP_TIMEOUT,
            self.inner.client.get(model_info_url).send(),
        )
        .await
            && response.status().is_success()
        {
            trace!(port, endpoint = "/model_info", "Readiness probe succeeded");
            return true;
        }

        let health_url = format!("http://127.0.0.1:{port}/health");
        if let Ok(Ok(response)) = tokio::time::timeout(
            READINESS_HTTP_TIMEOUT,
            self.inner.client.get(health_url).send(),
        )
        .await
        {
            let ok = response.status().is_success();
            if ok {
                trace!(port, endpoint = "/health", "Readiness probe succeeded");
            }
            return ok;
        }

        trace!(port, "Readiness probes failed");
        false
    }

    async fn sleep_model(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "sleep_model called");
        self.ensure_running(model).await?;

        let entry = self.model_entry(model)?;

        {
            let process = entry.process.lock().await;
            if matches!(process.state, ProcessState::Running { sleeping: true }) {
                trace!(model = %model, "Model already sleeping");
                return Ok(());
            }
        }

        debug!(model = %model, "Calling release_memory_occupation");
        self.post_json(
            model,
            "/release_memory_occupation",
            json!({ "tags": RELEASE_RESUME_TAGS }),
        )
        .await?;

        {
            let mut process = entry.process.lock().await;
            process.state = ProcessState::Running { sleeping: true };
        }
        info!(model = %model, "Model moved to sleeping state");

        Ok(())
    }

    async fn wake_model(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "wake_model called");
        self.ensure_running(model).await?;

        let entry = self.model_entry(model)?;

        let needs_resume = {
            let process = entry.process.lock().await;
            matches!(process.state, ProcessState::Running { sleeping: true })
        };
        debug!(model = %model, needs_resume, "wake_model evaluated model sleep state");

        if needs_resume {
            debug!(model = %model, "Calling resume_memory_occupation");
            self.post_json(
                model,
                "/resume_memory_occupation",
                json!({ "tags": RELEASE_RESUME_TAGS }),
            )
            .await?;

            // Required by local SGLang setup: reload weights after resume before serving traffic.
            debug!(model = %model, "Calling update_weights_from_disk after resume");
            self.update_weights_from_disk(model).await?;

            // Resume scheduler after pause_generation on prior switch-outs.
            debug!(model = %model, "Calling continue_generation after resume/update");
            self.continue_generation(model).await?;

            if !self.check_health(entry.spec.port).await {
                error!(
                    model = %model,
                    port = entry.spec.port,
                    "Model failed health check after resume/update"
                );
                return Err(SgLangMuxError::Http {
                    model: model.to_string(),
                    reason: "model failed health check after resume/update".to_string(),
                });
            }

            let mut process = entry.process.lock().await;
            process.state = ProcessState::Running { sleeping: false };
            info!(model = %model, "Model resumed and marked awake");
        } else {
            trace!(model = %model, "Model already awake");
        }

        Ok(())
    }

    async fn continue_generation(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "Posting /continue_generation");
        self.post_json(model, "/continue_generation", json!({}))
            .await
    }

    async fn update_weights_from_disk(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "Posting /update_weights_from_disk");
        self.post_json(
            model,
            "/update_weights_from_disk",
            json!({
                "model_path": model,
                "load_format": "auto",
                "abort_all_requests": false,
                "weight_version": model,
                "is_async": false,
                "torch_empty_cache": false,
                "keep_pause": false,
                "recapture_cuda_graph": false,
                "token_step": 0,
                "flush_cache": false
            }),
        )
        .await
    }

    async fn pause_generation(&self, model: &str) -> Result<(), SgLangMuxError> {
        debug!(model = %model, "Posting /pause_generation");
        self.post_json(model, "/pause_generation", json!({})).await
    }

    async fn post_json(
        &self,
        model: &str,
        path: &str,
        body: serde_json::Value,
    ) -> Result<(), SgLangMuxError> {
        let entry = self.model_entry(model)?;
        let url = format!("http://127.0.0.1:{}{}", entry.spec.port, path);
        trace!(
            model = %model,
            path,
            timeout_ms = self.inner.options.request_timeout.as_millis(),
            "Sending control request to model server"
        );

        let response = tokio::time::timeout(
            self.inner.options.request_timeout,
            self.inner.client.post(url).json(&body).send(),
        )
        .await
        .map_err(|_| {
            warn!(model = %model, path, "Control request timed out");
            SgLangMuxError::Http {
                model: model.to_string(),
                reason: format!("timeout calling {path}"),
            }
        })?
        .map_err(|error| {
            warn!(
                model = %model,
                path,
                error = %error,
                "Control request failed"
            );
            SgLangMuxError::Http {
                model: model.to_string(),
                reason: error.to_string(),
            }
        })?;

        if response.status() != StatusCode::OK {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable>".into());
            warn!(
                model = %model,
                path,
                status = %status,
                response_body = %body,
                "Control request returned non-200 status"
            );
            return Err(SgLangMuxError::Http {
                model: model.to_string(),
                reason: format!("{path} returned {status}: {body}"),
            });
        }

        trace!(model = %model, path, "Control request succeeded");
        Ok(())
    }

    async fn notify_pending(&self, model: &str, result: Result<(), SgLangMuxError>) {
        if let Some(entry) = self.inner.models.get(model) {
            let mut queue = entry.lock_state.pending.lock().await;
            let count = queue.len();
            debug!(
                model = %model,
                count,
                success = result.is_ok(),
                "Notifying pending requests"
            );
            for pending in queue.drain(..) {
                let send_result = match &result {
                    Ok(()) => Ok(()),
                    Err(error) => Err(SgLangMuxError::Internal(error.to_string())),
                };
                let _ = pending.ready_tx.send(send_result);
            }
        }
    }
}

impl Drop for SgLangMux {
    fn drop(&mut self) {
        // `SgLangMux` is cloneable (shared via `Arc`). Only the final owner should
        // terminate child processes; request-scoped clones must not trigger cleanup.
        if Arc::strong_count(&self.inner) != 1 {
            trace!("Dropping non-final SgLangMux handle; skipping process shutdown");
            return;
        }
        info!("Dropping final SgLangMux handle; terminating child processes");

        for entry in self.inner.models.values() {
            if let Ok(mut process) = entry.process.try_lock()
                && let Some(child) = process.child.as_mut()
            {
                debug!(model = %entry.spec.model_name, "Sending kill signal to child process");
                let _ = child.start_kill();
            }
        }
    }
}

fn sanitize_model_for_filename(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for ch in model.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn build_log_paths(log_dir: &Path, model: &str, port: u16) -> ModelLogPaths {
    let safe_name = sanitize_model_for_filename(model);
    ModelLogPaths {
        stdout: log_dir.join(format!("{safe_name}_{port}.stdout.log")),
        stderr: log_dir.join(format!("{safe_name}_{port}.stderr.log")),
    }
}

pub struct InFlightGuard {
    lock_state: Arc<ModelLockState>,
    model_name: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let previous = self.lock_state.in_flight.fetch_sub(1, Ordering::SeqCst);
        let remaining = previous.saturating_sub(1);
        trace!(
            model = %self.model_name,
            remaining_in_flight = remaining,
            "Released in-flight guard"
        );
        if previous == 1 {
            trace!(
                model = %self.model_name,
                "In-flight count reached zero; notifying drain waiters"
            );
            self.lock_state.in_flight_changed.notify_waiters();
        }
    }
}

pub fn parse_model_spec(script_path: PathBuf) -> Result<ModelSpec, SgLangMuxError> {
    let script =
        std::fs::read_to_string(&script_path).map_err(|source| SgLangMuxError::ScriptRead {
            path: script_path.clone(),
            source,
        })?;

    let model_name = parse_shell_assignment(&script, "MODEL_NAME").ok_or_else(|| {
        SgLangMuxError::MissingScriptMetadata {
            script: script_path.clone(),
            field: "MODEL_NAME",
        }
    })?;

    let port_raw = parse_shell_assignment(&script, "PORT").ok_or_else(|| {
        SgLangMuxError::MissingScriptMetadata {
            script: script_path.clone(),
            field: "PORT",
        }
    })?;

    let port = port_raw
        .parse::<u16>()
        .map_err(|_| SgLangMuxError::InvalidPort {
            script: script_path.clone(),
            value: port_raw,
        })?;

    Ok(ModelSpec {
        model_name,
        port,
        script_path,
    })
}

fn parse_shell_assignment(script: &str, key: &str) -> Option<String> {
    let pattern = format!(
        r"(?m)^\s*(?:export\s+)?{}\s*=\s*(.+?)\s*$",
        regex::escape(key)
    );
    let regex = Regex::new(&pattern).ok()?;
    let raw = regex.captures(script)?.get(1)?.as_str();

    let without_comment = raw.split('#').next().unwrap_or(raw).trim();
    if without_comment.is_empty() {
        return None;
    }

    let mut value = without_comment;
    let quoted = (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''));

    if quoted && value.len() >= 2 {
        value = &value[1..value.len() - 1];
    } else {
        value = value.split_whitespace().next().unwrap_or("");
    }

    if value.is_empty() {
        return None;
    }

    Some(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_model_spec_reads_model_and_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "#!/usr/bin/env bash\nMODEL_NAME='Qwen/Qwen3-0.6B'\nPORT=25001\n"
        )
        .unwrap();

        let spec = parse_model_spec(file.path().to_path_buf()).unwrap();

        assert_eq!(spec.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(spec.port, 25001);
    }

    #[test]
    fn parse_model_spec_requires_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "MODEL_NAME=foo").unwrap();

        let err = parse_model_spec(file.path().to_path_buf()).unwrap_err();
        assert!(matches!(
            err,
            SgLangMuxError::MissingScriptMetadata { field: "PORT", .. }
        ));
    }
}
