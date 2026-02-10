//! Switch policies for model switching decisions.
//!
//! Three policies are available:
//!
//! - [`FifoPolicy`]: Switches immediately on the first request for a non-active model.
//!   Simple and predictable, but causes excessive switching under mixed workloads.
//!
//! - [`CostAwarePolicy`]: Uses empirical switch timing (EMA), a coalescing window,
//!   a serving window, and a cost threshold to reduce unnecessary switches. The
//!   serving window ensures a model serves long enough to amortize its wake cost
//!   before yielding. See `reports/switching-efficiency.md` Report 4 for benchmarks.
//!
//! - [`TimeSlicePolicy`]: Drain-first policy with a proactive background scheduler.
//!   Never preempts a serving model — instead, the scheduler switches to the model
//!   with the most demand only after the active model's queue fully drains. This
//!   minimizes switches while a staleness bound caps maximum wait time.

use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Notify;
use tracing::debug;

/// Context provided to policies when making switch decisions
#[derive(Debug, Clone)]
pub struct PolicyContext {
    /// The model that the pending request is for
    pub target_model: String,
    /// Currently active/awake model (if any)
    pub active_model: Option<String>,
    /// Number of requests queued for the target model
    pub target_queue_depth: usize,
    /// How long the oldest request has been waiting
    pub oldest_waiting: Duration,
    /// Number of in-flight requests for the active model
    pub active_in_flight: usize,
    /// How long the active model has been active (since last wake completed)
    pub active_duration: Duration,
}

/// Context provided to the background scheduler on each tick
#[derive(Debug, Clone)]
pub struct ScheduleContext {
    /// Currently active/awake model (if any)
    pub active_model: Option<String>,
    /// How long the active model has been awake
    pub active_duration: Duration,
    /// Number of pending requests per model
    pub queue_depths: HashMap<String, usize>,
    /// Number of in-flight requests for the active model
    pub active_in_flight: usize,
}

/// Context for preparing a switch
pub struct SwitchContext {
    pub from_model: Option<String>,
    pub to_model: String,
    in_flight_drained: Arc<Notify>,
    get_in_flight: Box<dyn Fn() -> usize + Send + Sync>,
}

impl SwitchContext {
    pub fn new(
        from_model: Option<String>,
        to_model: String,
        in_flight_drained: Arc<Notify>,
        get_in_flight: Box<dyn Fn() -> usize + Send + Sync>,
    ) -> Self {
        Self {
            from_model,
            to_model,
            in_flight_drained,
            get_in_flight,
        }
    }

    /// Wait for all in-flight requests to complete
    pub async fn wait_for_in_flight(&self) {
        while (self.get_in_flight)() > 0 {
            self.in_flight_drained.notified().await;
        }
    }

    pub fn in_flight_count(&self) -> usize {
        (self.get_in_flight)()
    }
}

/// Decision returned by policy
pub enum PolicyDecision {
    /// Switch immediately
    SwitchNow,
    /// Defer - wait for the future to complete, then switch
    Defer(Pin<Box<dyn Future<Output = ()> + Send + 'static>>),
    /// Skip - a switch is already being arranged; do nothing
    Skip,
}

/// Policy trait for controlling model switching behavior
#[async_trait]
pub trait SwitchPolicy: Send + Sync {
    /// Called when a request arrives for an inactive model
    async fn on_pending_request(&self, ctx: &PolicyContext) -> PolicyDecision;

    /// Called before switching. Can wait for in-flight drain.
    async fn prepare_switch(&self, ctx: &mut SwitchContext);

    /// Called after a switch completes successfully with the measured duration.
    /// Policies can use this to track empirical switch costs.
    fn on_switch_complete(&self, _from: &str, _to: &str, _duration: Duration) {}

    /// Sleep level to use (1 or 2)
    fn sleep_level(&self) -> u8;

    /// Request timeout
    fn request_timeout(&self) -> Duration;

    /// Minimum time a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing that can cause GPU page faults.
    fn min_active_duration(&self) -> Duration;

    /// If `Some(interval)`, the switcher will spawn a background scheduler
    /// that calls [`schedule_tick`] every `interval`.
    fn scheduler_interval(&self) -> Option<Duration> {
        None
    }

    /// Called periodically by the background scheduler. Returns the model
    /// name to switch to, or `None` to stay on the current model.
    fn schedule_tick(&self, _ctx: &ScheduleContext) -> Option<String> {
        None
    }
}

/// FIFO policy - switch immediately on first request
pub struct FifoPolicy {
    sleep_level: u8,
    request_timeout: Duration,
    drain_before_switch: bool,
    min_active_duration: Duration,
}

impl FifoPolicy {
    pub fn new(
        sleep_level: u8,
        request_timeout: Duration,
        drain_before_switch: bool,
        min_active_duration: Duration,
    ) -> Self {
        Self {
            sleep_level,
            request_timeout,
            drain_before_switch,
            min_active_duration,
        }
    }
}

impl Default for FifoPolicy {
    fn default() -> Self {
        Self::new(1, Duration::from_secs(300), true, Duration::from_secs(5))
    }
}

#[async_trait]
impl SwitchPolicy for FifoPolicy {
    async fn on_pending_request(&self, _ctx: &PolicyContext) -> PolicyDecision {
        PolicyDecision::SwitchNow
    }

    async fn prepare_switch(&self, ctx: &mut SwitchContext) {
        if self.drain_before_switch {
            ctx.wait_for_in_flight().await;
        }
    }

    fn sleep_level(&self) -> u8 {
        self.sleep_level
    }

    fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn min_active_duration(&self) -> Duration {
        self.min_active_duration
    }
}

/// Tracks observed switch durations for a specific direction (from → to).
/// Uses an exponential moving average to adapt to changing conditions.
struct SwitchTiming {
    /// EMA of switch duration in milliseconds
    ema_ms: AtomicU64,
    /// Number of observations (capped for EMA smoothing)
    count: AtomicU64,
}

impl SwitchTiming {
    fn new(initial_estimate_ms: u64) -> Self {
        Self {
            ema_ms: AtomicU64::new(initial_estimate_ms),
            count: AtomicU64::new(0),
        }
    }

    fn record(&self, duration: Duration) {
        let ms = duration.as_millis() as u64;
        // Cap individual observations at 60s to avoid cold-start contamination.
        // A warm switch should never take more than 60s; longer durations indicate
        // a cold start or failure recovery that shouldn't inform future estimates.
        let ms = ms.min(60_000);
        let n = self.count.fetch_add(1, Ordering::Relaxed);
        // EMA with alpha = 0.3 for quick adaptation, but use the raw value
        // for the first observation
        if n == 0 {
            self.ema_ms.store(ms, Ordering::Relaxed);
        } else {
            let old = self.ema_ms.load(Ordering::Relaxed);
            // new_ema = alpha * sample + (1 - alpha) * old
            // = 0.3 * ms + 0.7 * old
            let new_ema = (3 * ms + 7 * old) / 10;
            self.ema_ms.store(new_ema, Ordering::Relaxed);
        }
    }

    fn estimated_ms(&self) -> u64 {
        self.ema_ms.load(Ordering::Relaxed)
    }
}

/// Per-model coalescing state
struct CoalesceState {
    /// Whether a coalescing defer is already in flight for this model
    defer_pending: AtomicBool,
}

impl Default for CoalesceState {
    fn default() -> Self {
        Self {
            defer_pending: AtomicBool::new(false),
        }
    }
}

/// Cost-aware coalescing policy.
///
/// Reduces wasted GPU time by:
/// 1. **Serving window**: Keeps a model active for at least as long as its
///    wake cost, ensuring the switch is amortized through serving time
/// 2. **Coalescing**: Waits a short window before switching to collect demand
/// 3. **Cost-awareness**: Only switches when accumulated queue depth justifies
///    the empirically-measured switch cost
/// 4. **Staleness bound**: Always switches if any request has waited too long
///
/// The policy tracks switch durations per direction (from_model → to_model)
/// using an exponential moving average. It uses these to estimate both the
/// cost threshold and the serving window duration.
pub struct CostAwarePolicy {
    sleep_level: u8,
    request_timeout: Duration,
    min_active_duration: Duration,

    /// How long to wait on the first request before deciding to switch.
    /// During this window, more requests may arrive, building a larger batch
    /// that better justifies the switch cost.
    coalesce_window: Duration,

    /// Minimum queue depth required to justify a switch, as a fraction of
    /// estimated switch cost in seconds. E.g. with amortization_factor=2.0
    /// and switch_cost=10s, we need at least 2*10=20 pending requests
    /// (assuming ~1s service time each) to make the switch worthwhile.
    ///
    /// Lower values = more willing to switch. 0.0 = always switch (like FIFO).
    amortization_factor: f64,

    /// Maximum time any request is allowed to wait before we force a switch,
    /// regardless of cost calculations.
    max_wait: Duration,

    /// Empirical switch timing per direction. Key is "from→to".
    timings: HashMap<String, SwitchTiming>,

    /// Default estimate for directions we haven't observed yet (ms)
    default_estimate_ms: u64,

    /// Per-model coalescing state
    coalesce_states: HashMap<String, CoalesceState>,
}

impl CostAwarePolicy {
    pub fn new(
        sleep_level: u8,
        request_timeout: Duration,
        min_active_duration: Duration,
        coalesce_window: Duration,
        amortization_factor: f64,
        max_wait: Duration,
        model_names: Vec<String>,
    ) -> Self {
        let mut timings = HashMap::new();
        let mut coalesce_states = HashMap::new();

        // Pre-populate timing entries for all model pairs
        for from in &model_names {
            for to in &model_names {
                if from != to {
                    let key = format!("{}→{}", from, to);
                    // Initial estimate: 10s. This is deliberately moderate —
                    // L1 wakes are ~2-4s, L2 wakes ~8-15s. The EMA will quickly
                    // adapt to actual observed values.
                    timings.insert(key, SwitchTiming::new(10_000));
                }
            }
            coalesce_states.insert(from.clone(), CoalesceState::default());
        }

        Self {
            sleep_level,
            request_timeout,
            min_active_duration,
            coalesce_window,
            amortization_factor,
            max_wait,
            timings,
            default_estimate_ms: 10_000,
            coalesce_states,
        }
    }

    /// Record an observed switch duration for a specific direction
    fn record_switch(&self, from: &str, to: &str, duration: Duration) {
        let key = format!("{}→{}", from, to);
        if let Some(timing) = self.timings.get(&key) {
            timing.record(duration);
            debug!(
                direction = %key,
                observed_ms = duration.as_millis(),
                new_ema_ms = timing.estimated_ms(),
                "Recorded switch timing"
            );
        }
    }

    /// Get estimated switch cost for a direction in seconds
    fn estimated_switch_cost(&self, from: Option<&str>, to: &str) -> f64 {
        match from {
            None => {
                // Cold start — use default estimate
                self.default_estimate_ms as f64 / 1000.0
            }
            Some(from) => {
                let key = format!("{}→{}", from, to);
                self.timings
                    .get(&key)
                    .map(|t| t.estimated_ms() as f64 / 1000.0)
                    .unwrap_or(self.default_estimate_ms as f64 / 1000.0)
            }
        }
    }

    /// Minimum queue depth to justify a switch, given the estimated cost
    fn min_queue_depth(&self, switch_cost_secs: f64) -> usize {
        // queue_depth >= amortization_factor * switch_cost
        // This means each queued request "amortizes" the switch cost.
        // With factor=0.5 and cost=10s, we need 5 requests.
        let min = (self.amortization_factor * switch_cost_secs).ceil() as usize;
        // Always at least 1 — we never reject a switch entirely
        min.max(1)
    }
}

#[async_trait]
impl SwitchPolicy for CostAwarePolicy {
    async fn on_pending_request(&self, ctx: &PolicyContext) -> PolicyDecision {
        let switch_cost =
            self.estimated_switch_cost(ctx.active_model.as_deref(), &ctx.target_model);
        let required_depth = self.min_queue_depth(switch_cost);

        debug!(
            target_model = %ctx.target_model,
            queue_depth = ctx.target_queue_depth,
            oldest_waiting_ms = ctx.oldest_waiting.as_millis(),
            switch_cost_s = format!("{:.1}", switch_cost),
            required_depth,
            active_in_flight = ctx.active_in_flight,
            "CostAware: evaluating switch"
        );

        // Staleness override: if any request has waited too long, switch immediately
        if ctx.oldest_waiting >= self.max_wait {
            debug!("CostAware: staleness override — switching now");
            return PolicyDecision::SwitchNow;
        }

        // If there's no active model (idle state), switch immediately
        if ctx.active_model.is_none() {
            return PolicyDecision::SwitchNow;
        }

        // Serving window: the active model should serve for at least as long as
        // the next switch would cost. This ensures we amortize the wake cost of
        // the current model before paying for another switch. Without this, serial
        // alternating traffic (A, B, A, B...) triggers a switch on every request.
        //
        // The cost threshold check comes after the serving window because even
        // a high queue depth should not preempt a model that hasn't served long
        // enough to justify its own wake cost. The staleness bound above still
        // provides a hard cap on wait time.
        let serving_window = Duration::from_secs_f64(switch_cost);
        if ctx.active_duration < serving_window {
            let remaining = serving_window - ctx.active_duration;
            debug!(
                active_duration_ms = ctx.active_duration.as_millis(),
                serving_window_ms = serving_window.as_millis(),
                remaining_ms = remaining.as_millis(),
                "CostAware: within serving window, deferring"
            );

            let coalesce_state = self.coalesce_states.get(&ctx.target_model);
            if let Some(state) = coalesce_state {
                if state.defer_pending.load(Ordering::SeqCst) {
                    return PolicyDecision::Skip;
                }
                state.defer_pending.store(true, Ordering::SeqCst);
            }

            let target = ctx.target_model.clone();
            return PolicyDecision::Defer(Box::pin(async move {
                tokio::time::sleep(remaining).await;
                debug!(model = %target, "CostAware: serving window expired");
            }));
        }

        // If queue depth meets the cost threshold, switch immediately
        if ctx.target_queue_depth >= required_depth {
            debug!(
                "CostAware: queue depth {} >= required {}, switching now",
                ctx.target_queue_depth, required_depth
            );
            return PolicyDecision::SwitchNow;
        }

        // Queue depth is below threshold and serving window has passed — coalesce
        let coalesce_state = self.coalesce_states.get(&ctx.target_model);

        // If a defer is already pending, don't spawn another one
        if let Some(state) = coalesce_state
            && state.defer_pending.load(Ordering::SeqCst)
        {
            return PolicyDecision::Skip;
        }

        // Start a coalescing window
        if let Some(state) = coalesce_state {
            state.defer_pending.store(true, Ordering::SeqCst);
        }

        let window = self.coalesce_window;
        let target = ctx.target_model.clone();
        PolicyDecision::Defer(Box::pin(async move {
            tokio::time::sleep(window).await;
            debug!(model = %target, "CostAware: coalescing window expired");
        }))
    }

    async fn prepare_switch(&self, ctx: &mut SwitchContext) {
        // Clear coalescing state for the target model
        if let Some(state) = self.coalesce_states.get(&ctx.to_model) {
            state.defer_pending.store(false, Ordering::SeqCst);
        }

        // Always drain in-flight requests before switching
        ctx.wait_for_in_flight().await;
    }

    fn on_switch_complete(&self, from: &str, to: &str, duration: Duration) {
        self.record_switch(from, to, duration);
    }

    fn sleep_level(&self) -> u8 {
        self.sleep_level
    }

    fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn min_active_duration(&self) -> Duration {
        self.min_active_duration
    }
}

/// Drain-first scheduling policy with a proactive background scheduler.
///
/// This policy minimizes GPU time wasted on model switches by following two
/// principles:
///
/// 1. **Never preempt a serving model.** When a request arrives for a non-active
///    model, the policy defers to the background scheduler rather than switching
///    reactively. The only exception is the staleness bound, which forces a
///    switch if any request has waited longer than `max_wait`.
///
/// 2. **Switch when idle.** The background scheduler periodically checks all
///    models' queue depths. When the active model has completely drained its
///    queue (no pending requests, no in-flight), the scheduler switches to the
///    model with the most waiting requests.
///
/// This is equivalent to "serve everything from the active model's queue, then
/// switch to whoever has the most demand." The scheduler's global visibility
/// into all queue depths prevents the pathological back-and-forth switching
/// that reactive policies cause under interleaved or dominant workloads.
///
/// In simulation across 12 workload profiles at switch costs from 2s to 20s,
/// this policy achieves 61-94% GPU serving time vs CostAware's 40-81% and
/// FIFO's 33-79%, while also delivering 2-6x lower maximum wait times.
pub struct TimeSlicePolicy {
    sleep_level: u8,
    request_timeout: Duration,
    min_active_duration: Duration,
    max_wait: Duration,

    /// How often the scheduler ticks
    tick_interval: Duration,
}

impl TimeSlicePolicy {
    pub fn new(
        sleep_level: u8,
        request_timeout: Duration,
        min_active_duration: Duration,
        max_wait: Duration,
        _min_quantum: Duration,
        tick_interval: Duration,
        _model_names: Vec<String>,
    ) -> Self {
        Self {
            sleep_level,
            request_timeout,
            min_active_duration,
            max_wait,
            tick_interval,
        }
    }
}

#[async_trait]
impl SwitchPolicy for TimeSlicePolicy {
    async fn on_pending_request(&self, ctx: &PolicyContext) -> PolicyDecision {
        // Staleness override: if any request has waited too long, switch immediately.
        // This is the only reactive switch trigger — everything else defers to the
        // scheduler, which has global visibility into all models' queue depths.
        if ctx.oldest_waiting >= self.max_wait {
            debug!(
                target_model = %ctx.target_model,
                oldest_waiting_ms = ctx.oldest_waiting.as_millis(),
                "TimeSlice: staleness override — switching now"
            );
            return PolicyDecision::SwitchNow;
        }

        // Cold start: no active model, switch immediately
        if ctx.active_model.is_none() {
            return PolicyDecision::SwitchNow;
        }

        // Defer to the background scheduler. It will switch when the active
        // model's queue drains naturally.
        PolicyDecision::Skip
    }

    async fn prepare_switch(&self, ctx: &mut SwitchContext) {
        // Always drain in-flight requests before switching
        ctx.wait_for_in_flight().await;
    }

    fn sleep_level(&self) -> u8 {
        self.sleep_level
    }

    fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn min_active_duration(&self) -> Duration {
        self.min_active_duration
    }

    fn scheduler_interval(&self) -> Option<Duration> {
        Some(self.tick_interval)
    }

    fn schedule_tick(&self, ctx: &ScheduleContext) -> Option<String> {
        let active = ctx.active_model.as_deref()?;

        // Respect min_active_duration in the scheduler so that the cooldown
        // check in do_switch is always a no-op. Without this, the scheduler
        // eagerly triggers switches as soon as the queue drains (e.g. 0.5s
        // after activation), and do_switch then waits 4.5s in Phase 1 for
        // the cooldown — wasting GPU time that shows up in the metrics.
        if ctx.active_duration < self.min_active_duration {
            return None;
        }

        // Only switch when the active model has completely drained: no pending
        // requests and no in-flight requests. This ensures we never waste GPU
        // time by switching away from a model that still has work to do.
        let active_depth = ctx.queue_depths.get(active).copied().unwrap_or(0);
        if active_depth > 0 || ctx.active_in_flight > 0 {
            return None;
        }

        // Pick the model with the most waiting requests
        let best_target = ctx
            .queue_depths
            .iter()
            .filter(|(model, _)| model.as_str() != active)
            .filter(|(_, depth)| **depth > 0)
            .max_by_key(|(_, depth)| **depth);

        let (target_model, _) = best_target?;

        debug!(
            from = %active,
            to = %target_model,
            "TimeSlice: active model idle, switching to model with most demand"
        );
        Some(target_model.clone())
    }
}
