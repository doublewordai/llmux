use super::{PolicyContext, PolicyDecision, SwitchContext, SwitchPolicy};
use async_trait::async_trait;
use std::time::Duration;

/// FIFO policy — switch immediately on first request for a non-active model.
///
/// No background scheduler needed: every request spawns its own switch attempt
/// via `maybe_trigger_switch`, and the switch lock serializes them.
pub struct FifoPolicy {
    // TODO: Is this the concern of the policy
    request_timeout: Duration,
    // TODO: is this the concern of the policy?
    drain_before_switch: bool,
    // TODO: is this the concern of the policy?
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
}
