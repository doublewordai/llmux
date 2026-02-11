//! Model Switcher - coordinates wake/sleep between models
//!
//! The switcher tracks which model is active and coordinates transitions.

use crate::orchestrator::{Orchestrator, OrchestratorError};
use crate::policy::{PolicyContext, PolicyDecision, ScheduleContext, SwitchContext, SwitchPolicy};
use metrics::{counter, gauge, histogram};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, RwLock, oneshot};
use tracing::{debug, error, info, trace, warn};

/// Sleep level for hibernating models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepLevel {
    /// Level 1: Offload weights to CPU RAM (faster wake)
    L1,
    /// Level 2: Discard weights (slower wake, less RAM)
    L2,
    /// Level 3: CUDA suspend via cuda-checkpoint toggle (process stays alive,
    /// GPU freed, VRAM contents held in host RAM, full state preserved)
    CudaSuspend,
    /// Level 4: CRIU checkpoint (snapshot process to disk, frees all GPU/CPU memory,
    /// preserves full state including KV cache, CUDA graphs, and warmed allocator)
    Checkpoint,
    /// Level 5: Stop the vLLM process entirely (full restart on wake)
    Stop,
}

impl From<u8> for SleepLevel {
    fn from(level: u8) -> Self {
        match level {
            2 => SleepLevel::L2,
            3 => SleepLevel::CudaSuspend,
            4 => SleepLevel::Checkpoint,
            5 => SleepLevel::Stop,
            _ => SleepLevel::L1,
        }
    }
}

/// Errors from the switcher
#[derive(Debug, thiserror::Error)]
pub enum SwitchError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model not ready: {0}")]
    NotReady(String),

    #[error("request timeout")]
    Timeout,

    #[error("orchestrator error: {0}")]
    Orchestrator(#[from] OrchestratorError),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("manual mode: model {requested} not available (active: {active})")]
    ManualModeRejected { requested: String, active: String },
}

/// Switch mode controls whether model switching is automatic or manual.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SwitchMode {
    /// Policy-driven automatic switching (default behavior)
    Auto,
    /// Manual mode: no auto-switching. Only the pinned model (if set) serves requests.
    Manual {
        #[serde(skip_serializing_if = "Option::is_none")]
        pinned: Option<String>,
    },
}

/// State of the model switcher
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitcherState {
    /// No model is currently active
    Idle,
    /// A model is awake and ready
    Active { model: String },
    /// Switching from one model to another
    Switching { from: Option<String>, to: String },
}

/// A pending request waiting for a model
struct PendingRequest {
    #[allow(dead_code)] // Used for debugging/logging
    model: String,
    queued_at: Instant,
    ready_tx: oneshot::Sender<Result<(), SwitchError>>,
}

/// Per-model state tracking
struct ModelState {
    in_flight: AtomicUsize,
    pending: Mutex<Vec<PendingRequest>>,
    in_flight_changed: Arc<Notify>,
    /// Set to `true` while draining in-flight requests before sleep.
    /// When true, `acquire_in_flight` will refuse new guards so that
    /// no requests sneak in between drain completion and the actual sleep call.
    draining: AtomicBool,
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            pending: Mutex::new(Vec::new()),
            in_flight_changed: Arc::new(Notify::new()),
            draining: AtomicBool::new(false),
        }
    }
}

/// Inner state for the switcher
struct SwitcherInner {
    orchestrator: Arc<Orchestrator>,
    policy: Box<dyn SwitchPolicy>,
    state: RwLock<SwitcherState>,
    model_states: HashMap<String, Arc<ModelState>>,
    switch_lock: Mutex<()>,
    /// When the currently active model was activated (for cooldown enforcement)
    activated_at: RwLock<Option<Instant>>,
    /// When the last switch failure occurred (for backoff)
    last_switch_failure: RwLock<Option<Instant>>,
    /// Current switch mode (auto or manual)
    mode: RwLock<SwitchMode>,
}

/// The model switcher coordinates wake/sleep transitions
pub struct ModelSwitcher {
    inner: Arc<SwitcherInner>,
}

impl Clone for ModelSwitcher {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ModelSwitcher {
    /// Create a new model switcher
    pub fn new(orchestrator: Arc<Orchestrator>, policy: Box<dyn SwitchPolicy>) -> Self {
        let model_states: HashMap<String, Arc<ModelState>> = orchestrator
            .registered_models()
            .into_iter()
            .map(|model| (model, Arc::new(ModelState::default())))
            .collect();

        Self {
            inner: Arc::new(SwitcherInner {
                orchestrator,
                policy,
                state: RwLock::new(SwitcherState::Idle),
                model_states,
                switch_lock: Mutex::new(()),
                activated_at: RwLock::new(None),
                last_switch_failure: RwLock::new(None),
                mode: RwLock::new(SwitchMode::Auto),
            }),
        }
    }

    /// Get the current state
    pub async fn state(&self) -> SwitcherState {
        self.inner.state.read().await.clone()
    }

    /// Get the currently active model
    pub async fn active_model(&self) -> Option<String> {
        match &*self.inner.state.read().await {
            SwitcherState::Active { model } => Some(model.clone()),
            _ => None,
        }
    }

    /// Get the current switch mode
    pub async fn mode(&self) -> SwitchMode {
        self.inner.mode.read().await.clone()
    }

    /// Set the switch mode
    pub async fn set_mode(&self, mode: SwitchMode) {
        info!(mode = ?mode, "Switch mode changed");
        *self.inner.mode.write().await = mode;
    }

    /// Get the list of registered model names
    pub fn registered_models(&self) -> Vec<String> {
        self.inner.model_states.keys().cloned().collect()
    }

    /// Get access to the orchestrator (for process state queries)
    pub fn orchestrator(&self) -> &Arc<Orchestrator> {
        &self.inner.orchestrator
    }

    /// Get the policy's default sleep level
    pub fn policy_sleep_level(&self) -> u8 {
        self.inner.policy.sleep_level()
    }

    /// Force a switch to the given model, bypassing policy.
    ///
    /// Reuses the full `do_switch` machinery (drain, sleep, wake) so all
    /// safety guarantees still apply. Returns after the switch completes.
    pub async fn force_switch(&self, model: &str) -> Result<(), SwitchError> {
        if !self.is_registered(model) {
            return Err(SwitchError::ModelNotFound(model.to_string()));
        }

        // If already active, nothing to do
        {
            let state = self.inner.state.read().await;
            if let SwitcherState::Active { model: active } = &*state
                && active == model
            {
                return Ok(());
            }
        }

        self.do_switch(model).await;

        // Check if switch succeeded
        let state = self.inner.state.read().await;
        match &*state {
            SwitcherState::Active { model: active } if active == model => Ok(()),
            _ => Err(SwitchError::NotReady(model.to_string())),
        }
    }

    /// Check if a model is registered
    pub fn is_registered(&self, model: &str) -> bool {
        self.inner.model_states.contains_key(model)
    }

    /// Get in-flight count for a model
    pub fn in_flight_count(&self, model: &str) -> usize {
        self.inner
            .model_states
            .get(model)
            .map(|s| s.in_flight.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Ensure a model is ready for requests
    ///
    /// This will:
    /// 1. Return immediately if the model is already active
    /// 2. In manual mode: reject if model doesn't match pinned/active model
    /// 3. Queue the request and trigger a switch if needed
    /// 4. Wait for the switch to complete (up to timeout)
    pub async fn ensure_model_ready(&self, model: &str) -> Result<(), SwitchError> {
        let model_state = self
            .inner
            .model_states
            .get(model)
            .ok_or_else(|| SwitchError::ModelNotFound(model.to_string()))?;

        // Fast path: model is already active
        {
            let state = self.inner.state.read().await;
            if let SwitcherState::Active { model: active } = &*state
                && active == model
            {
                trace!(model = %model, "Model already active");
                return Ok(());
            }
        }

        // Manual mode: reject requests for non-active models
        {
            let mode = self.inner.mode.read().await;
            if let SwitchMode::Manual { ref pinned } = *mode {
                let active = self.active_model().await;
                let allowed = pinned.as_deref().or(active.as_deref());
                if allowed != Some(model) {
                    let active_name = allowed.unwrap_or("none").to_string();
                    warn!(
                        requested = %model,
                        active = %active_name,
                        "Manual mode: rejecting request for non-active model"
                    );
                    return Err(SwitchError::ManualModeRejected {
                        requested: model.to_string(),
                        active: active_name,
                    });
                }
            }
        }

        // Queue the request
        let (ready_tx, ready_rx) = oneshot::channel();
        let pending = PendingRequest {
            model: model.to_string(),
            queued_at: Instant::now(),
            ready_tx,
        };

        {
            let mut queue = model_state.pending.lock().await;
            queue.push(pending);
            debug!(model = %model, queue_depth = queue.len(), "Request queued");
        }

        // Maybe trigger switch
        self.maybe_trigger_switch(model).await;

        // Wait for ready
        let timeout = self.inner.policy.request_timeout();
        match tokio::time::timeout(timeout, ready_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(SwitchError::Internal("channel closed".to_string())),
            Err(_) => {
                warn!(model = %model, "Request timed out");
                Err(SwitchError::Timeout)
            }
        }
    }

    /// Acquire an in-flight guard.
    ///
    /// Returns `None` if the model is not registered **or** if the model is
    /// currently draining (about to be put to sleep). Uses increment-then-check
    /// to close the TOCTOU window between `notify_pending` and the drain in
    /// `do_switch`.
    pub fn acquire_in_flight(&self, model: &str) -> Option<InFlightGuard> {
        let model_state = self.inner.model_states.get(model)?;

        // Increment first so the drain sees our count if it checks concurrently.
        let new_count = model_state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;

        // If draining started, back out — the model is about to sleep.
        if model_state.draining.load(Ordering::SeqCst) {
            model_state.in_flight.fetch_sub(1, Ordering::SeqCst);
            model_state.in_flight_changed.notify_waiters();
            return None;
        }

        gauge!("llmux_in_flight", "model" => model.to_owned()).set(new_count as f64);

        Some(InFlightGuard {
            model_state: Arc::clone(model_state),
            model_name: model.to_owned(),
        })
    }

    /// Check policy and maybe trigger switch
    async fn maybe_trigger_switch(&self, target_model: &str) {
        // In manual mode, never auto-switch
        {
            let mode = self.inner.mode.read().await;
            if matches!(*mode, SwitchMode::Manual { .. }) {
                trace!(model = %target_model, "Manual mode: skipping auto-switch");
                return;
            }
        }

        let model_state = match self.inner.model_states.get(target_model) {
            Some(s) => s,
            None => return,
        };

        // Build policy context
        let ctx = {
            let state = self.inner.state.read().await;
            let queue = model_state.pending.lock().await;

            let oldest_waiting = queue
                .first()
                .map(|p| p.queued_at.elapsed())
                .unwrap_or(Duration::ZERO);

            let (active_model, active_in_flight) = match &*state {
                SwitcherState::Active { model } => {
                    (Some(model.clone()), self.in_flight_count(model))
                }
                _ => (None, 0),
            };

            let active_duration = self
                .inner
                .activated_at
                .read()
                .await
                .map(|t| t.elapsed())
                .unwrap_or(Duration::ZERO);

            PolicyContext {
                target_model: target_model.to_string(),
                active_model,
                target_queue_depth: queue.len(),
                oldest_waiting,
                active_in_flight,
                active_duration,
            }
        };

        // Already switching?
        {
            let state = self.inner.state.read().await;
            if let SwitcherState::Switching { to, .. } = &*state
                && to == target_model
            {
                return;
            }
        }

        // Ask policy
        let decision = self.inner.policy.on_pending_request(&ctx).await;

        match decision {
            PolicyDecision::SwitchNow => {
                debug!(model = %target_model, "Policy: switch now");
                self.do_switch(target_model).await;
            }
            PolicyDecision::Defer(future) => {
                debug!(model = %target_model, "Policy: defer");
                let switcher = self.clone();
                let target = target_model.to_string();
                tokio::spawn(async move {
                    future.await;
                    // Re-check mode: operator may have switched to manual while
                    // this deferred decision was waiting.
                    let mode = switcher.mode().await;
                    if matches!(mode, SwitchMode::Manual { .. }) {
                        debug!(model = %target, "Manual mode: aborting deferred auto-switch");
                        return;
                    }
                    switcher.do_switch(&target).await;
                });
            }
            PolicyDecision::Skip => {
                trace!(model = %target_model, "Policy: skip (switch already arranged)");
            }
        }
    }

    /// Perform the actual switch
    async fn do_switch(&self, target_model: &str) {
        let _guard = self.inner.switch_lock.lock().await;
        let switch_start = Instant::now();

        // Backoff: if the last switch failed recently, wait before retrying
        {
            let last_failure = self.inner.last_switch_failure.read().await;
            if let Some(failed_at) = *last_failure {
                let backoff = Duration::from_secs(2);
                let elapsed = failed_at.elapsed();
                if elapsed < backoff {
                    let remaining = backoff - elapsed;
                    info!(remaining = ?remaining, "Backing off after recent switch failure");
                    drop(last_failure);
                    tokio::time::sleep(remaining).await;
                }
            }
        }

        // Double-check state
        {
            let state = self.inner.state.read().await;
            match &*state {
                SwitcherState::Active { model } if model == target_model => {
                    self.notify_pending(target_model, Ok(())).await;
                    return;
                }
                SwitcherState::Switching { to, .. } if to == target_model => {
                    return;
                }
                _ => {}
            }
        }

        let from_model = {
            let state = self.inner.state.read().await;
            match &*state {
                SwitcherState::Active { model } => Some(model.clone()),
                _ => None,
            }
        };

        // Update state
        {
            let mut state = self.inner.state.write().await;
            *state = SwitcherState::Switching {
                from: from_model.clone(),
                to: target_model.to_string(),
            };
        }

        let from_str = from_model.as_deref().unwrap_or("");
        info!(from = ?from_model, to = %target_model, "Starting model switch");

        // Phase 1: Cooldown — ensure the old model has been active long enough
        // before sleeping. Requests can still flow to the old model during this
        // wait, which minimises the window where requests are rejected by the
        // draining flag below.
        let cooldown_start = Instant::now();
        if from_model.is_some() {
            let min_active = self.inner.policy.min_active_duration();
            let activated_at = *self.inner.activated_at.read().await;
            if let Some(activated) = activated_at {
                let elapsed = activated.elapsed();
                if elapsed < min_active {
                    let remaining = min_active - elapsed;
                    info!(
                        remaining = ?remaining,
                        "Waiting for cooldown before sleeping old model"
                    );
                    tokio::time::sleep(remaining).await;
                }
            }
        }
        let cooldown_dur = cooldown_start.elapsed();

        // Phase 2: Drain — set draining flag and wait for in-flight to complete.
        let drain_start = Instant::now();

        // Set draining flag *before* drain so that `acquire_in_flight` rejects
        // new guards. This closes the TOCTOU race where `notify_pending` sends Ok
        // but the middleware task hasn't acquired its guard yet — when the drain
        // sees in_flight == 0 it would complete instantly, then those tasks would
        // acquire guards and send requests to a sleeping model.
        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            from_state.draining.store(true, Ordering::SeqCst);
        }

        // Drain in-flight requests (policy decides whether to wait).
        // Pass the model's own `in_flight_changed` Notify so the drain wakes
        // when InFlightGuard::drop fires.
        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            let in_flight_changed = Arc::clone(&from_state.in_flight_changed);
            let from_state_clone = Arc::clone(from_state);

            let mut switch_ctx = SwitchContext::new(
                from_model.clone(),
                target_model.to_string(),
                in_flight_changed,
                Box::new(move || from_state_clone.in_flight.load(Ordering::SeqCst)),
            );

            self.inner.policy.prepare_switch(&mut switch_ctx).await;
        }
        let drain_dur = drain_start.elapsed();

        // Phase 3: Sleep old model — use per-model sleep level from config,
        // falling back to the global policy default.
        // Exception: if the target model is already checkpointed on disk,
        // downgrade the sleep to Stop (kill) to avoid needing disk space
        // for two simultaneous CRIU checkpoints.
        let sleep_start = Instant::now();
        if let Some(ref from) = from_model {
            let level_raw = self
                .inner
                .orchestrator
                .sleep_level_for(from)
                .unwrap_or_else(|| self.inner.policy.sleep_level());
            let mut sleep_level = SleepLevel::from(level_raw);
            if sleep_level == SleepLevel::Checkpoint
                && self.inner.orchestrator.is_checkpointed(target_model)
            {
                info!(
                    from = %from,
                    to = %target_model,
                    "Target is checkpointed; downgrading sleep to Stop to avoid dual checkpoint"
                );
                sleep_level = SleepLevel::Stop;
            }
            debug!(model = %from, level = ?sleep_level, "Sleeping model");
            self.inner.orchestrator.force_sleep(from, sleep_level).await;
        }

        // Clear draining flag now that sleep is complete
        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            from_state.draining.store(false, Ordering::SeqCst);
        }
        let sleep_dur = sleep_start.elapsed();

        // Phase 4: Wake new model
        let wake_start = Instant::now();
        debug!(model = %target_model, "Waking model");
        match self.inner.orchestrator.wake_model(target_model).await {
            Ok(()) => {
                // Wait for ready
                let mut ready = false;
                for attempt in 0..10 {
                    if self.inner.orchestrator.is_ready(target_model).await {
                        ready = true;
                        break;
                    }
                    debug!(model = %target_model, attempt, "Waiting for model");
                    tokio::time::sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                }

                let wake_dur = wake_start.elapsed();

                if ready {
                    info!(model = %target_model, "Model is now active");
                    {
                        let mut state = self.inner.state.write().await;
                        *state = SwitcherState::Active {
                            model: target_model.to_string(),
                        };
                    }
                    *self.inner.activated_at.write().await = Some(Instant::now());
                    *self.inner.last_switch_failure.write().await = None;

                    // Record switch metrics
                    let total_dur = switch_start.elapsed();
                    histogram!("llmux_switch_phase_seconds", "phase" => "cooldown", "from" => from_str.to_owned(), "to" => target_model.to_owned()).record(cooldown_dur.as_secs_f64());
                    histogram!("llmux_switch_phase_seconds", "phase" => "drain", "from" => from_str.to_owned(), "to" => target_model.to_owned()).record(drain_dur.as_secs_f64());
                    histogram!("llmux_switch_phase_seconds", "phase" => "sleep", "from" => from_str.to_owned(), "to" => target_model.to_owned()).record(sleep_dur.as_secs_f64());
                    histogram!("llmux_switch_phase_seconds", "phase" => "wake", "from" => from_str.to_owned(), "to" => target_model.to_owned()).record(wake_dur.as_secs_f64());
                    histogram!("llmux_switch_total_seconds", "from" => from_str.to_owned(), "to" => target_model.to_owned()).record(total_dur.as_secs_f64());
                    counter!("llmux_switches_total", "from" => from_str.to_owned(), "to" => target_model.to_owned()).increment(1);

                    // Notify policy of empirical switch timing
                    self.inner
                        .policy
                        .on_switch_complete(from_str, target_model, total_dur);

                    // Update active model gauge
                    if let Some(ref from) = from_model {
                        gauge!("llmux_active_model_info", "model" => from.clone()).set(0.0);
                    }
                    gauge!("llmux_active_model_info", "model" => target_model.to_owned()).set(1.0);

                    self.notify_pending(target_model, Ok(())).await;
                } else {
                    error!(model = %target_model, "Model failed to become ready");
                    // Clean up: force-sleep the partially-woken model to free GPU memory
                    self.inner
                        .orchestrator
                        .force_sleep(target_model, SleepLevel::Stop)
                        .await;
                    *self.inner.last_switch_failure.write().await = Some(Instant::now());
                    {
                        let mut state = self.inner.state.write().await;
                        *state = SwitcherState::Idle;
                    }
                    counter!("llmux_switch_failures_total", "to" => target_model.to_owned())
                        .increment(1);
                    self.notify_pending(
                        target_model,
                        Err(SwitchError::NotReady(target_model.to_string())),
                    )
                    .await;
                }
            }
            Err(e) => {
                error!(model = %target_model, error = %e, "Failed to wake model");
                // Clean up: force-sleep in case the model partially woke
                self.inner
                    .orchestrator
                    .force_sleep(target_model, SleepLevel::Stop)
                    .await;
                *self.inner.last_switch_failure.write().await = Some(Instant::now());
                {
                    let mut state = self.inner.state.write().await;
                    *state = SwitcherState::Idle;
                }
                counter!("llmux_switch_failures_total", "to" => target_model.to_owned())
                    .increment(1);
                self.notify_pending(target_model, Err(SwitchError::Orchestrator(e)))
                    .await;
            }
        }
    }

    /// Get the queue depth for every registered model.
    pub async fn queue_depths(&self) -> HashMap<String, usize> {
        let mut depths = HashMap::new();
        for (model, state) in &self.inner.model_states {
            let queue = state.pending.lock().await;
            depths.insert(model.clone(), queue.len());
        }
        depths
    }

    /// Build a ScheduleContext from current switcher state.
    async fn build_schedule_context(&self) -> ScheduleContext {
        let (active_model, active_in_flight) = match &*self.inner.state.read().await {
            SwitcherState::Active { model } => (Some(model.clone()), self.in_flight_count(model)),
            _ => (None, 0),
        };

        let active_duration = self
            .inner
            .activated_at
            .read()
            .await
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        let queue_depths = self.queue_depths().await;

        ScheduleContext {
            active_model,
            active_duration,
            queue_depths,
            active_in_flight,
        }
    }

    /// Spawn a background scheduler task if the policy requests one.
    ///
    /// Returns `Some(JoinHandle)` if the scheduler was spawned, `None`
    /// if the policy does not use a scheduler (i.e. `scheduler_interval`
    /// returns `None`).
    pub fn spawn_scheduler(self) -> Option<tokio::task::JoinHandle<()>> {
        let interval = self.inner.policy.scheduler_interval()?;

        info!(
            interval_ms = interval.as_millis(),
            "Spawning background scheduler"
        );

        Some(tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tick.tick().await;
                // Skip scheduler ticks in manual mode
                if matches!(*self.inner.mode.read().await, SwitchMode::Manual { .. }) {
                    continue;
                }
                let ctx = self.build_schedule_context().await;
                if let Some(target) = self.inner.policy.schedule_tick(&ctx) {
                    debug!(target = %target, "Scheduler: triggering switch");
                    self.do_switch(&target).await;
                }
            }
        }))
    }

    /// Notify pending requests
    async fn notify_pending(&self, model: &str, result: Result<(), SwitchError>) {
        if let Some(model_state) = self.inner.model_states.get(model) {
            let mut queue = model_state.pending.lock().await;
            let count = queue.len();

            for pending in queue.drain(..) {
                let r = match &result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(SwitchError::Internal(e.to_string())),
                };
                let _ = pending.ready_tx.send(r);
            }

            debug!(model = %model, count, "Notified pending requests");
        }
    }
}

/// Guard that tracks in-flight requests
pub struct InFlightGuard {
    model_state: Arc<ModelState>,
    model_name: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let prev = self.model_state.in_flight.fetch_sub(1, Ordering::SeqCst);
        gauge!("llmux_in_flight", "model" => self.model_name.clone()).set((prev - 1) as f64);
        if prev == 1 {
            self.model_state.in_flight_changed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::policy::FifoPolicy;
    use std::collections::HashMap;

    fn make_test_orchestrator() -> Arc<Orchestrator> {
        let mut configs = HashMap::new();
        configs.insert(
            "model-a".to_string(),
            ModelConfig {
                model_path: "test".to_string(),
                port: 8001,
                extra_args: vec![],
                sleep_level: 1,
            },
        );
        configs.insert(
            "model-b".to_string(),
            ModelConfig {
                model_path: "test".to_string(),
                port: 8002,
                extra_args: vec![],
                sleep_level: 1,
            },
        );
        Arc::new(Orchestrator::new(configs))
    }

    #[test]
    fn test_switcher_creation() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        assert!(switcher.is_registered("model-a"));
        assert!(switcher.is_registered("model-b"));
        assert!(!switcher.is_registered("model-c"));
    }

    #[tokio::test]
    async fn test_in_flight_tracking() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        assert_eq!(switcher.in_flight_count("model-a"), 0);

        {
            let _guard = switcher.acquire_in_flight("model-a");
            assert_eq!(switcher.in_flight_count("model-a"), 1);
        }

        assert_eq!(switcher.in_flight_count("model-a"), 0);
    }

    #[test]
    fn test_acquire_in_flight_rejected_while_draining() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        // Acquire should succeed when not draining
        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_some());
        assert_eq!(switcher.in_flight_count("model-a"), 1);
        drop(guard);

        // Set draining flag
        let model_state = switcher.inner.model_states.get("model-a").unwrap();
        model_state.draining.store(true, Ordering::SeqCst);

        // Acquire should now return None
        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_none());
        // In-flight count should remain 0 (increment was backed out)
        assert_eq!(switcher.in_flight_count("model-a"), 0);

        // Clear draining flag
        model_state.draining.store(false, Ordering::SeqCst);

        // Acquire should succeed again
        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_some());
        assert_eq!(switcher.in_flight_count("model-a"), 1);
        drop(guard);
    }

    #[tokio::test]
    async fn test_default_mode_is_auto() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        assert_eq!(switcher.mode().await, SwitchMode::Auto);
    }

    #[tokio::test]
    async fn test_set_mode_manual() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        switcher
            .set_mode(SwitchMode::Manual {
                pinned: Some("model-a".to_string()),
            })
            .await;

        assert_eq!(
            switcher.mode().await,
            SwitchMode::Manual {
                pinned: Some("model-a".to_string())
            }
        );
    }

    #[tokio::test]
    async fn test_manual_mode_rejects_non_active_model() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        // Set active state to model-a (simulate it being active)
        {
            let mut state = switcher.inner.state.write().await;
            *state = SwitcherState::Active {
                model: "model-a".to_string(),
            };
        }

        // Enter manual mode with model-a pinned
        switcher
            .set_mode(SwitchMode::Manual {
                pinned: Some("model-a".to_string()),
            })
            .await;

        // Request for model-a should succeed (fast path: already active)
        let result = switcher.ensure_model_ready("model-a").await;
        assert!(result.is_ok());

        // Request for model-b should be rejected
        let result = switcher.ensure_model_ready("model-b").await;
        assert!(matches!(result, Err(SwitchError::ManualModeRejected { .. })));
    }

    #[tokio::test]
    async fn test_manual_mode_no_pin_uses_active() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        // Set active state to model-a
        {
            let mut state = switcher.inner.state.write().await;
            *state = SwitcherState::Active {
                model: "model-a".to_string(),
            };
        }

        // Enter manual mode without pinning a specific model
        switcher
            .set_mode(SwitchMode::Manual { pinned: None })
            .await;

        // Request for active model should succeed
        let result = switcher.ensure_model_ready("model-a").await;
        assert!(result.is_ok());

        // Request for other model should be rejected
        let result = switcher.ensure_model_ready("model-b").await;
        assert!(matches!(result, Err(SwitchError::ManualModeRejected { .. })));
    }

    #[tokio::test]
    async fn test_force_switch_unknown_model() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        let result = switcher.force_switch("nonexistent").await;
        assert!(matches!(result, Err(SwitchError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn test_force_switch_already_active() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        // Set active state to model-a
        {
            let mut state = switcher.inner.state.write().await;
            *state = SwitcherState::Active {
                model: "model-a".to_string(),
            };
        }

        // Force switch to already-active model should be a no-op
        let result = switcher.force_switch("model-a").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_registered_models() {
        let orchestrator = make_test_orchestrator();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(orchestrator, policy);

        let mut models = switcher.registered_models();
        models.sort();
        assert_eq!(models, vec!["model-a", "model-b"]);
    }

    #[tokio::test]
    async fn test_switch_mode_serde() {
        let auto = SwitchMode::Auto;
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(json, r#"{"mode":"auto"}"#);

        let manual = SwitchMode::Manual {
            pinned: Some("llama".to_string()),
        };
        let json = serde_json::to_string(&manual).unwrap();
        assert!(json.contains(r#""mode":"manual""#));
        assert!(json.contains(r#""pinned":"llama""#));

        let manual_no_pin = SwitchMode::Manual { pinned: None };
        let json = serde_json::to_string(&manual_no_pin).unwrap();
        assert!(json.contains(r#""mode":"manual""#));
        assert!(!json.contains("pinned"));

        // Deserialize
        let parsed: SwitchMode = serde_json::from_str(r#"{"mode":"auto"}"#).unwrap();
        assert_eq!(parsed, SwitchMode::Auto);

        let parsed: SwitchMode =
            serde_json::from_str(r#"{"mode":"manual","pinned":"test"}"#).unwrap();
        assert_eq!(
            parsed,
            SwitchMode::Manual {
                pinned: Some("test".to_string())
            }
        );
    }
}
