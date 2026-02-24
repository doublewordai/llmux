use super::{PolicyContext, PolicyDecision, ScheduleContext, SwitchContext, SwitchPolicy};
use async_trait::async_trait;
use std::time::Duration;

/// FIFO policy — switch immediately on first request for a non-active model.
///
/// Uses a background scheduler to pick up stranded pending requests (e.g.
/// requests that arrived while a switch was already in progress for a
/// different model).
pub struct FifoPolicy {
    request_timeout: Duration,
    drain_before_switch: bool,
    min_active_duration: Duration,
}

impl FifoPolicy {
    pub fn new(
        request_timeout: Duration,
        drain_before_switch: bool,
        min_active_duration: Duration,
    ) -> Self {
        Self {
            request_timeout,
            drain_before_switch,
            min_active_duration,
        }
    }
}

impl Default for FifoPolicy {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), true, Duration::from_secs(5))
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

    fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn min_active_duration(&self) -> Duration {
        self.min_active_duration
    }

    fn scheduler_interval(&self) -> Option<Duration> {
        Some(Duration::from_millis(100))
    }

    fn schedule_tick(&self, ctx: &ScheduleContext) -> Option<String> {
        // Find the first model with pending requests that isn't already active
        ctx.queue_depths
            .iter()
            .find(|(model, depth)| **depth > 0 && ctx.active_model.as_ref() != Some(model))
            .map(|(model, _)| model.clone())
    }
}
