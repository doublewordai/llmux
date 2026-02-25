//! Model Switcher — coordinates wake/sleep transitions between models.
//!
//! The switcher tracks which model is active, manages in-flight request
//! counting, drains requests before switching, and delegates all lifecycle
//! operations to the [`HookRunner`].

use crate::cost::SwitchCostTracker;
use crate::hooks::HookRunner;
use crate::policy::{PolicyContext, PolicyDecision, ScheduleContext, SwitchContext, SwitchPolicy};
use crate::types::{SwitchError, SwitcherState};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify, RwLock, mpsc, oneshot};
use tracing::{debug, error, info, trace, warn};

/// Signal sent through the oneshot channel when a model becomes active.
///
/// The receiver must call [`ReadySignal::settle`] after acquiring its
/// in-flight guard. `notify_pending` blocks on these settle signals so
/// that the switch lock is held until all notified requests are actively
/// being processed — preventing the next switch from draining the model
/// before any request has started.
pub(crate) struct ReadySignal {
    settle_tx: mpsc::Sender<()>,
}

impl ReadySignal {
    pub(crate) async fn settle(self) {
        let _ = self.settle_tx.send(()).await;
    }
}

/// A pending request waiting for a model to become active
struct PendingRequest {
    #[allow(dead_code)]
    model: String,
    queued_at: Instant,
    ready_tx: oneshot::Sender<Result<ReadySignal, SwitchError>>,
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

struct SwitcherInner {
    hooks: Arc<HookRunner>,
    policy: Box<dyn SwitchPolicy>,
    state: RwLock<SwitcherState>,
    model_states: HashMap<String, Arc<ModelState>>,
    switch_lock: Mutex<()>,
    /// When the currently active model was activated (for cooldown enforcement)
    activated_at: RwLock<Option<Instant>>,
    /// When the last switch failure occurred (for backoff)
    last_switch_failure: RwLock<Option<Instant>>,
    /// Empirical switch cost tracking (EMA of observed durations)
    cost_tracker: SwitchCostTracker,
}

/// The model switcher coordinates wake/sleep transitions.
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
    pub fn new(hooks: Arc<HookRunner>, policy: Box<dyn SwitchPolicy>) -> Self {
        let model_states: HashMap<String, Arc<ModelState>> = hooks
            .registered_models()
            .into_iter()
            .map(|model| (model, Arc::new(ModelState::default())))
            .collect();

        Self {
            inner: Arc::new(SwitcherInner {
                hooks,
                policy,
                state: RwLock::new(SwitcherState::Idle),
                model_states,
                switch_lock: Mutex::new(()),
                activated_at: RwLock::new(None),
                last_switch_failure: RwLock::new(None),
                cost_tracker: SwitchCostTracker::new(0.3),
            }),
        }
    }

    pub async fn state(&self) -> SwitcherState {
        self.inner.state.read().await.clone()
    }

    pub async fn active_model(&self) -> Option<String> {
        match &*self.inner.state.read().await {
            SwitcherState::Active { model } => Some(model.clone()),
            _ => None,
        }
    }

    pub fn registered_models(&self) -> Vec<String> {
        self.inner.model_states.keys().cloned().collect()
    }

    pub fn hooks(&self) -> &Arc<HookRunner> {
        &self.inner.hooks
    }

    pub fn is_registered(&self, model: &str) -> bool {
        self.inner.model_states.contains_key(model)
    }

    pub fn model_port(&self, model: &str) -> Option<u16> {
        self.inner.hooks.model_port(model)
    }

    pub fn in_flight_count(&self, model: &str) -> usize {
        self.inner
            .model_states
            .get(model)
            .map(|s| s.in_flight.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Estimated cost of switching between two models, based on observed durations.
    /// `from` is `None` for cold starts from Idle.
    pub fn estimated_switch_cost(&self, from: Option<&str>, to: &str) -> Option<Duration> {
        self.inner.cost_tracker.estimate(from, to)
    }

    /// Force a switch to the given model, bypassing policy.
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

        let state = self.inner.state.read().await;
        match &*state {
            SwitcherState::Active { model: active } if active == model => Ok(()),
            _ => Err(SwitchError::NotReady(model.to_string())),
        }
    }

    /// Ensure a model is ready for requests.
    ///
    /// Returns `Ok(None)` immediately if the model is already active (fast
    /// path). Otherwise queues the request, triggers a switch if needed, and
    /// waits up to the policy timeout. Returns `Ok(Some(signal))` — the
    /// caller **must** call [`ReadySignal::settle`] after acquiring its
    /// in-flight guard so that `notify_pending` knows the request is actively
    /// being processed.
    pub(crate) async fn ensure_model_ready(
        &self,
        model: &str,
    ) -> Result<Option<ReadySignal>, SwitchError> {
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
                return Ok(None);
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
            let depth = queue.len();
            debug!(model = %model, queue_depth = depth, "Request queued");
            metrics::gauge!("llmux_request_queue_depth", "model" => model.to_string())
                .set(depth as f64);
        }

        self.maybe_trigger_switch(model).await;

        match self.inner.policy.request_timeout() {
            Some(timeout) => match tokio::time::timeout(timeout, ready_rx).await {
                Ok(Ok(result)) => result.map(Some),
                Ok(Err(_)) => Err(SwitchError::Internal("channel closed".to_string())),
                Err(_) => {
                    warn!(
                        event = "request_timeout",
                        model = %model,
                        timeout_secs = timeout.as_secs_f64(),
                        "Request timed out waiting for model"
                    );
                    Err(SwitchError::Timeout)
                }
            },
            None => match ready_rx.await {
                Ok(result) => result.map(Some),
                Err(_) => Err(SwitchError::Internal("channel closed".to_string())),
            },
        }
    }

    /// Acquire an in-flight guard.
    ///
    /// Returns `None` if the model is not registered or if it is currently
    /// draining. Uses increment-then-check to close the TOCTOU window.
    pub fn acquire_in_flight(&self, model: &str) -> Option<InFlightGuard> {
        let model_state = self.inner.model_states.get(model)?;

        let new_count = model_state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;

        if model_state.draining.load(Ordering::SeqCst) {
            model_state.in_flight.fetch_sub(1, Ordering::SeqCst);
            model_state.in_flight_changed.notify_waiters();
            return None;
        }

        metrics::gauge!("llmux_model_in_flight", "model" => model.to_string())
            .set(new_count as f64);

        Some(InFlightGuard {
            model_state: Arc::clone(model_state),
            model: model.to_string(),
        })
    }

    /// Get queue depths for every registered model.
    pub async fn queue_depths(&self) -> HashMap<String, usize> {
        let mut depths = HashMap::new();
        for (model, state) in &self.inner.model_states {
            let queue = state.pending.lock().await;
            depths.insert(model.clone(), queue.len());
        }
        depths
    }

    /// Spawn a background scheduler task if the policy requests one.
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
                let ctx = self.build_schedule_context().await;
                if let Some(target) = self.inner.policy.schedule_tick(&ctx) {
                    debug!(target = %target, "Scheduler: triggering switch");
                    self.do_switch(&target).await;
                }
            }
        }))
    }

    // -----------------------------------------------------------------------
    // Private
    // -----------------------------------------------------------------------

    async fn maybe_trigger_switch(&self, target_model: &str) {
        let model_state = match self.inner.model_states.get(target_model) {
            Some(s) => s,
            None => return,
        };

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

            let estimated_switch_cost = self
                .inner
                .cost_tracker
                .estimate(active_model.as_deref(), target_model);

            PolicyContext {
                target_model: target_model.to_string(),
                active_model,
                target_queue_depth: queue.len(),
                oldest_waiting,
                active_in_flight,
                active_duration,
                estimated_switch_cost,
            }
        };

        // Already switching to this model?
        {
            let state = self.inner.state.read().await;
            if let SwitcherState::Switching { to, .. } = &*state
                && to == target_model
            {
                return;
            }
        }

        let decision = self.inner.policy.on_pending_request(&ctx).await;

        match decision {
            PolicyDecision::SwitchNow => {
                debug!(model = %target_model, "Policy: switch now");
                let switcher = self.clone();
                let target = target_model.to_string();
                tokio::spawn(async move {
                    switcher.do_switch(&target).await;
                });
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
            PolicyDecision::Skip => {
                trace!(model = %target_model, "Policy: skip");
            }
        }
    }

    async fn do_switch(&self, target_model: &str) {
        let _guard = self.inner.switch_lock.lock().await;
        let switch_start = Instant::now();

        // Backoff after recent failure
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

        // Skip if there are no pending requests for the target model.
        // This happens when a stale do_switch task grabs the lock after
        // the target model's requests were already served by an earlier
        // activation.
        if let Some(target_state) = self.inner.model_states.get(target_model) {
            let queue = target_state.pending.lock().await;
            if queue.is_empty() {
                debug!(
                    model = %target_model,
                    "No pending requests, skipping stale switch"
                );
                return;
            }
        }

        let from_model = {
            let state = self.inner.state.read().await;
            match &*state {
                SwitcherState::Active { model } => Some(model.clone()),
                _ => None,
            }
        };

        let from_label = from_model.as_deref().unwrap_or("idle").to_string();

        // Record how long the outgoing model was active
        if from_model.is_some() {
            let active_dur = self
                .inner
                .activated_at
                .read()
                .await
                .map(|t| t.elapsed())
                .unwrap_or(Duration::ZERO);
            metrics::histogram!(
                "llmux_model_active_duration_seconds",
                "model" => from_label.clone()
            )
            .record(active_dur.as_secs_f64());
        }

        // Update state to Switching
        {
            let mut state = self.inner.state.write().await;
            *state = SwitcherState::Switching {
                from: from_model.clone(),
                to: target_model.to_string(),
            };
        }

        info!(
            event = "switch_started",
            from = %from_label,
            to = %target_model,
            "Starting model switch"
        );

        // Phase 1: Cooldown — ensure the old model has been active long enough
        if from_model.is_some() {
            let min_active = self.inner.policy.min_active_duration();
            let activated_at = *self.inner.activated_at.read().await;
            if let Some(activated) = activated_at {
                let elapsed = activated.elapsed();
                if elapsed < min_active {
                    let remaining = min_active - elapsed;
                    info!(remaining = ?remaining, "Waiting for cooldown");
                    tokio::time::sleep(remaining).await;
                }
            }
        }

        // Phase 2: Drain — set draining flag and wait for in-flight to complete
        let drain_start = Instant::now();

        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            from_state.draining.store(true, Ordering::SeqCst);
        }

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

        if from_model.is_some() {
            metrics::histogram!(
                "llmux_switch_drain_duration_seconds",
                "from" => from_label.clone(),
                "to" => target_model.to_string()
            )
            .record(drain_start.elapsed().as_secs_f64());
        }

        // Phase 3: Sleep old model via hook
        if let Some(ref from) = from_model {
            debug!(model = %from, "Running sleep hook");
            if let Err(e) = self.inner.hooks.run_sleep(from).await {
                error!(
                    event = "sleep_hook_failed",
                    model = %from,
                    to = %target_model,
                    error = %e,
                    "Sleep hook failed, continuing with wake (idempotent)"
                );
            }
        }

        // Clear draining flag
        if let Some(ref from) = from_model
            && let Some(from_state) = self.inner.model_states.get(from)
        {
            from_state.draining.store(false, Ordering::SeqCst);
        }

        // Phase 4: Wake new model via hook
        debug!(model = %target_model, "Running wake hook");
        match self.inner.hooks.run_wake(target_model).await {
            Ok(()) => {
                let total_dur = switch_start.elapsed();

                {
                    let mut state = self.inner.state.write().await;
                    *state = SwitcherState::Active {
                        model: target_model.to_string(),
                    };
                }
                *self.inner.activated_at.write().await = Some(Instant::now());
                *self.inner.last_switch_failure.write().await = None;

                self.inner
                    .cost_tracker
                    .record(from_model.as_deref(), target_model, total_dur);

                // Structured log event for timeline reconstruction
                info!(
                    event = "model_activated",
                    model = %target_model,
                    from = %from_label,
                    duration_secs = total_dur.as_secs_f64(),
                    "Model is now active"
                );

                // Metrics
                metrics::counter!(
                    "llmux_switch_total",
                    "from" => from_label.clone(),
                    "to" => target_model.to_string(),
                    "result" => "success"
                )
                .increment(1);
                metrics::histogram!(
                    "llmux_switch_duration_seconds",
                    "from" => from_label.clone(),
                    "to" => target_model.to_string()
                )
                .record(total_dur.as_secs_f64());

                if let Some(ema) = self
                    .inner
                    .cost_tracker
                    .estimate(from_model.as_deref(), target_model)
                {
                    metrics::gauge!(
                        "llmux_switch_cost_ema_seconds",
                        "from" => from_label,
                        "to" => target_model.to_string()
                    )
                    .set(ema.as_secs_f64());
                }

                let from_str = from_model.as_deref().unwrap_or("");
                self.inner
                    .policy
                    .on_switch_complete(from_str, target_model, total_dur);

                self.notify_pending(target_model, Ok(())).await;
            }
            Err(e) => {
                // Structured log event for timeline reconstruction
                info!(
                    event = "switch_failed",
                    model = %target_model,
                    from = %from_label,
                    error = %e,
                    "Switch failed, returning to idle"
                );

                metrics::counter!(
                    "llmux_switch_total",
                    "from" => from_label,
                    "to" => target_model.to_string(),
                    "result" => "failure"
                )
                .increment(1);

                // Try to clean up the partially-woken model
                let _ = self.inner.hooks.run_sleep(target_model).await;

                *self.inner.last_switch_failure.write().await = Some(Instant::now());
                {
                    let mut state = self.inner.state.write().await;
                    *state = SwitcherState::Idle;
                }

                self.notify_pending(
                    target_model,
                    Err(SwitchError::HookFailed {
                        model: target_model.to_string(),
                        detail: e.to_string(),
                    }),
                )
                .await;
            }
        }
    }

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

        let switch_costs = self
            .inner
            .cost_tracker
            .estimates_from(active_model.as_deref());

        ScheduleContext {
            active_model,
            active_duration,
            queue_depths,
            active_in_flight,
            switch_costs,
        }
    }

    /// Notify pending requests and — on success — wait for them to settle.
    ///
    /// When the result is `Ok`, each notified request receives a
    /// [`ReadySignal`]. The request must call `settle()` after acquiring
    /// its in-flight guard. This method blocks until all delivered signals
    /// have settled (or their receivers are dropped), ensuring the switch
    /// lock is held while requests transition from "notified" to "in-flight."
    ///
    /// When the result is `Err`, requests are notified of the failure and
    /// no settle wait is needed.
    async fn notify_pending(&self, model: &str, result: Result<(), SwitchError>) {
        let Some(model_state) = self.inner.model_states.get(model) else {
            return;
        };

        let mut queue = model_state.pending.lock().await;
        let count = queue.len();
        if count == 0 {
            return;
        }

        let mut delivered = 0;

        // For successful activations, create a settle channel so we can
        // wait for notified requests to acquire their in-flight guards.
        let settle_tx = if result.is_ok() {
            // +1 capacity: one per request, non-blocking sends
            Some(mpsc::channel::<()>(count))
        } else {
            None
        };

        for pending in queue.drain(..) {
            let r = match (&result, &settle_tx) {
                (Ok(()), Some((tx, _))) => Ok(ReadySignal {
                    settle_tx: tx.clone(),
                }),
                (Err(e), _) => Err(SwitchError::Internal(e.to_string())),
                _ => unreachable!(),
            };
            if pending.ready_tx.send(r).is_ok() {
                delivered += 1;
            }
        }

        metrics::gauge!("llmux_request_queue_depth", "model" => model.to_string()).set(0.0);
        drop(queue); // release pending lock before the settle wait

        if count > 0 {
            let expired = count - delivered;
            if expired > 0 {
                warn!(model = %model, count, delivered, expired,
                    "Notified pending requests ({expired} already timed out)");
            } else {
                debug!(model = %model, count, "Notified pending requests");
            }
        }

        // Wait for all delivered requests to acquire in-flight guards.
        // Each request calls ReadySignal::settle() after acquiring its
        // guard, which sends () on the channel. If a request drops its
        // signal without settling (e.g. cancelled), its sender clone is
        // dropped; once all senders are gone the channel closes and recv
        // returns None.
        if let Some((tx, mut rx)) = settle_tx {
            // Drop the original sender — only the clones sent to requests
            // should keep the channel alive.
            drop(tx);

            if delivered > 0 {
                let settle_wait = async {
                    for _ in 0..delivered {
                        if rx.recv().await.is_none() {
                            break; // all senders dropped
                        }
                    }
                };

                if tokio::time::timeout(Duration::from_secs(5), settle_wait)
                    .await
                    .is_err()
                {
                    warn!(
                        model = %model,
                        delivered,
                        "Settle timeout — proceeding with switch lock release"
                    );
                }
            }
        }
    }
}

/// Guard that tracks in-flight requests. When dropped, decrements the count
/// and notifies the drain waiter.
pub struct InFlightGuard {
    model_state: Arc<ModelState>,
    model: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let prev = self.model_state.in_flight.fetch_sub(1, Ordering::SeqCst);
        metrics::gauge!("llmux_model_in_flight", "model" => self.model.clone())
            .set((prev - 1) as f64);
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

    fn make_test_hooks() -> Arc<HookRunner> {
        let mut configs = HashMap::new();
        configs.insert(
            "model-a".to_string(),
            ModelConfig {
                port: 8001,
                wake: "true".to_string(),
                sleep: "true".to_string(),
                alive: "true".to_string(),
            },
        );
        configs.insert(
            "model-b".to_string(),
            ModelConfig {
                port: 8002,
                wake: "true".to_string(),
                sleep: "true".to_string(),
                alive: "true".to_string(),
            },
        );
        Arc::new(HookRunner::new(configs))
    }

    #[test]
    fn test_switcher_creation() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        assert!(switcher.is_registered("model-a"));
        assert!(switcher.is_registered("model-b"));
        assert!(!switcher.is_registered("model-c"));
    }

    #[tokio::test]
    async fn test_in_flight_tracking() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        assert_eq!(switcher.in_flight_count("model-a"), 0);

        {
            let _guard = switcher.acquire_in_flight("model-a");
            assert_eq!(switcher.in_flight_count("model-a"), 1);
        }

        assert_eq!(switcher.in_flight_count("model-a"), 0);
    }

    #[test]
    fn test_acquire_in_flight_rejected_while_draining() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_some());
        assert_eq!(switcher.in_flight_count("model-a"), 1);
        drop(guard);

        // Set draining flag
        let model_state = switcher.inner.model_states.get("model-a").unwrap();
        model_state.draining.store(true, Ordering::SeqCst);

        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_none());
        assert_eq!(switcher.in_flight_count("model-a"), 0);

        model_state.draining.store(false, Ordering::SeqCst);

        let guard = switcher.acquire_in_flight("model-a");
        assert!(guard.is_some());
        assert_eq!(switcher.in_flight_count("model-a"), 1);
        drop(guard);
    }

    #[tokio::test]
    async fn test_model_port() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        assert_eq!(switcher.model_port("model-a"), Some(8001));
        assert_eq!(switcher.model_port("model-b"), Some(8002));
        assert_eq!(switcher.model_port("model-c"), None);
    }

    #[tokio::test]
    async fn test_force_switch_unknown_model() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        let result = switcher.force_switch("nonexistent").await;
        assert!(matches!(result, Err(SwitchError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn test_force_switch_already_active() {
        let hooks = make_test_hooks();
        let policy = Box::new(FifoPolicy::default());
        let switcher = ModelSwitcher::new(hooks, policy);

        {
            let mut state = switcher.inner.state.write().await;
            *state = SwitcherState::Active {
                model: "model-a".to_string(),
            };
        }

        let result = switcher.force_switch("model-a").await;
        assert!(result.is_ok());
    }
}
