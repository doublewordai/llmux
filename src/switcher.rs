//! Model Switcher - coordinates wake/sleep between models
//!
//! The switcher tracks which model is active and coordinates transitions.

use crate::orchestrator::{Orchestrator, OrchestratorError};
use crate::policy::{PolicyContext, PolicyDecision, SwitchContext, SwitchPolicy};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Level 3: Stop the vLLM process entirely (slowest wake, frees all GPU memory)
    Stop,
}

impl From<u8> for SleepLevel {
    fn from(level: u8) -> Self {
        match level {
            2 => SleepLevel::L2,
            3 => SleepLevel::Stop,
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
    in_flight_changed: Notify,
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            pending: Mutex::new(Vec::new()),
            in_flight_changed: Notify::new(),
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
    /// 2. Queue the request and trigger a switch if needed
    /// 3. Wait for the switch to complete (up to timeout)
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

    /// Acquire an in-flight guard
    pub fn acquire_in_flight(&self, model: &str) -> Option<InFlightGuard> {
        let model_state = self.inner.model_states.get(model)?;
        model_state.in_flight.fetch_add(1, Ordering::SeqCst);
        Some(InFlightGuard {
            model_state: Arc::clone(model_state),
        })
    }

    /// Check policy and maybe trigger switch
    async fn maybe_trigger_switch(&self, target_model: &str) {
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

            PolicyContext {
                target_model: target_model.to_string(),
                active_model,
                target_queue_depth: queue.len(),
                oldest_waiting,
                active_in_flight,
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
                    switcher.do_switch(&target).await;
                });
            }
        }
    }

    /// Perform the actual switch
    async fn do_switch(&self, target_model: &str) {
        let _guard = self.inner.switch_lock.lock().await;

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

        info!(from = ?from_model, to = %target_model, "Starting model switch");

        // Prepare switch (drain in-flight if needed)
        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            let in_flight_drained = Arc::new(Notify::new());
            let from_state_clone = Arc::clone(from_state);

            let mut switch_ctx = SwitchContext::new(
                from_model.clone(),
                target_model.to_string(),
                Arc::clone(&in_flight_drained),
                Box::new(move || from_state_clone.in_flight.load(Ordering::SeqCst)),
            );

            self.inner.policy.prepare_switch(&mut switch_ctx).await;
        }

        // Cooldown: ensure the old model has been active long enough before sleeping
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

        // Sleep old model — use per-model sleep level from config, falling back
        // to the global policy default if the model is somehow not in the config.
        if let Some(ref from) = from_model {
            let level_raw = self
                .inner
                .orchestrator
                .sleep_level_for(from)
                .unwrap_or_else(|| self.inner.policy.sleep_level());
            let sleep_level = SleepLevel::from(level_raw);
            debug!(model = %from, level = ?sleep_level, "Sleeping model");
            self.inner.orchestrator.force_sleep(from, sleep_level).await;
        }

        // Wake new model
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
                self.notify_pending(target_model, Err(SwitchError::Orchestrator(e)))
                    .await;
            }
        }
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
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let prev = self.model_state.in_flight.fetch_sub(1, Ordering::SeqCst);
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
}
