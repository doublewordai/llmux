//! Switch policies for model switching decisions

use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

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
}

/// Policy trait for controlling model switching behavior
#[async_trait]
pub trait SwitchPolicy: Send + Sync {
    /// Called when a request arrives for an inactive model
    async fn on_pending_request(&self, ctx: &PolicyContext) -> PolicyDecision;

    /// Called before switching. Can wait for in-flight drain.
    async fn prepare_switch(&self, ctx: &mut SwitchContext);

    /// Sleep level to use (1 or 2)
    fn sleep_level(&self) -> u8;

    /// Request timeout
    fn request_timeout(&self) -> Duration;

    /// Minimum time a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing that can cause GPU page faults.
    fn min_active_duration(&self) -> Duration;
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
        Self::new(1, Duration::from_secs(60), true, Duration::from_secs(5))
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
