use super::{PolicyContext, PolicyDecision, ScheduleContext, SwitchContext, SwitchPolicy};
use crate::types::EvictionPolicy;
use async_trait::async_trait;
use std::time::Duration;
use tracing::debug;

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
    eviction: EvictionPolicy,
    request_timeout: Duration,
    min_active_duration: Duration,
    max_wait: Duration,

    /// How often the scheduler ticks
    tick_interval: Duration,
}

impl TimeSlicePolicy {
    pub fn new(
        eviction: EvictionPolicy,
        request_timeout: Duration,
        min_active_duration: Duration,
        max_wait: Duration,
        _min_quantum: Duration,
        tick_interval: Duration,
        _model_names: Vec<String>,
    ) -> Self {
        Self {
            eviction,
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

    fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction
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
