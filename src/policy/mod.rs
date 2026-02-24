//! Switch policies for model switching decisions.
//!
//! The policy trait is intentionally rich — it receives detailed context about
//! queue depths, in-flight counts, and timing so that future policies can make
//! sophisticated decisions. The initial [`FifoPolicy`] uses only a fraction
//! of this data.

mod fifo;

pub use fifo::FifoPolicy;

use async_trait::async_trait;
use std::collections::HashMap;
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

/// Context for preparing a switch (drain phase)
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
    /// Defer — wait for the future to complete, then switch
    Defer(Pin<Box<dyn Future<Output = ()> + Send + 'static>>),
    /// Skip — a switch is already being arranged; do nothing
    Skip,
}

/// Policy trait for controlling model switching behavior.
///
/// Policies receive rich context about the current state of the system and
/// return decisions about when and how to switch models. The trait is
/// intentionally broad to support future policies (batching, priority-based,
/// cost-aware, etc.) without API changes.
#[async_trait]
pub trait SwitchPolicy: Send + Sync {
    /// Called when a request arrives for an inactive model
    async fn on_pending_request(&self, ctx: &PolicyContext) -> PolicyDecision;

    /// Called before switching. Can wait for in-flight drain.
    async fn prepare_switch(&self, ctx: &mut SwitchContext);

    /// Called after a switch completes successfully with the measured duration.
    /// Policies can use this to track empirical switch costs.
    fn on_switch_complete(&self, _from: &str, _to: &str, _duration: Duration) {}

    /// Request timeout
    fn request_timeout(&self) -> Duration;

    /// Minimum time a model must stay active before it can be put to sleep.
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
