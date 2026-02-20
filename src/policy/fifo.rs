use super::{PolicyContext, PolicyDecision, SwitchContext, SwitchPolicy};
use crate::types::EvictionPolicy;
use async_trait::async_trait;
use std::time::Duration;

/// FIFO policy - switch immediately on first request
pub struct FifoPolicy {
    eviction: EvictionPolicy,
    request_timeout: Duration,
    drain_before_switch: bool,
    min_active_duration: Duration,
}

impl FifoPolicy {
    pub fn new(
        eviction: EvictionPolicy,
        request_timeout: Duration,
        drain_before_switch: bool,
        min_active_duration: Duration,
    ) -> Self {
        Self {
            eviction,
            request_timeout,
            drain_before_switch,
            min_active_duration,
        }
    }
}

impl Default for FifoPolicy {
    fn default() -> Self {
        Self::new(
            EvictionPolicy::from(1),
            Duration::from_secs(300),
            true,
            Duration::from_secs(5),
        )
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

    fn eviction_policy(&self) -> EvictionPolicy {
        self.eviction
    }

    fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    fn min_active_duration(&self) -> Duration {
        self.min_active_duration
    }
}
