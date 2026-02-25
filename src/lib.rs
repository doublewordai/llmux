//! # llmux v2
//!
//! Hook-driven LLM model multiplexer with pluggable switch policy.
//!
//! All model lifecycle management (start, stop, health check) is delegated to
//! user-provided scripts. llmux handles request routing, in-flight tracking,
//! draining, and policy-driven switching.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                        llmux                            │
//! │  ┌───────────────────────────────────────────────────┐  │
//! │  │ Middleware (Tower Layer)                           │  │
//! │  │ - Extracts model from request                     │  │
//! │  │ - Ensures model ready (triggers switch if needed) │  │
//! │  │ - Acquires in-flight guard                        │  │
//! │  │ - Wraps response in GuardedBody for streaming     │  │
//! │  └───────────────────────────────────────────────────┘  │
//! │                          │                              │
//! │  ┌───────────────────────────────────────────────────┐  │
//! │  │ Switcher                                          │  │
//! │  │ - Drain → sleep hook → wake hook                  │  │
//! │  │ - Policy decides when to switch                   │  │
//! │  └───────────────────────────────────────────────────┘  │
//! │                          │                              │
//! │  ┌───────────────────────────────────────────────────┐  │
//! │  │ Reverse Proxy                                     │  │
//! │  │ - Forwards to localhost:model_port                │  │
//! │  └───────────────────────────────────────────────────┘  │
//! │                          │                              │
//! │      ┌───────────────────┼───────────────────┐         │
//! │      ▼                   ▼                   ▼         │
//! │  [backend:8001]     [backend:8002]      [backend:8003] │
//! │   wake.sh / sleep.sh / alive.sh                        │
//! └─────────────────────────────────────────────────────────┘
//! ```

mod config;
mod cost;
mod hooks;
mod middleware;
pub mod policy;
mod proxy;
mod switcher;
pub(crate) mod types;

pub use config::{Config, ModelConfig, PolicyConfig};
pub use cost::SwitchCostTracker;
pub use hooks::HookRunner;
pub use middleware::{ModelSwitcherLayer, ModelSwitcherService};
pub use policy::{FifoPolicy, PolicyContext, PolicyDecision, ScheduleContext, SwitchPolicy};
pub use proxy::{ProxyState, proxy_handler};
pub use switcher::{InFlightGuard, ModelSwitcher};
pub use types::{SwitchError, SwitcherState};

use anyhow::Result;
use axum::routing::get;
use axum::{Json, Router};
use std::sync::Arc;
use tracing::info;

/// Build the complete llmux stack.
///
/// Returns:
/// - The main Axum router (proxy + middleware)
/// - The model switcher (for external use)
pub async fn build_app(config: Config) -> Result<(Router, ModelSwitcher)> {
    info!("Building llmux with {} models", config.models.len());

    let hooks = Arc::new(HookRunner::new(config.models.clone()));
    let policy = config.policy.build_policy();
    let switcher = ModelSwitcher::new(hooks, policy);

    // Spawn background scheduler if the policy uses one
    let _scheduler_handle = switcher.clone().spawn_scheduler();

    // Build proxy
    let proxy_state = ProxyState::new();

    // Pre-compute /v1/models response from config
    let models_response = {
        let mut data: Vec<_> = config
            .models
            .keys()
            .map(|id| {
                serde_json::json!({
                    "id": id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llmux"
                })
            })
            .collect::<Vec<_>>();
        data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        serde_json::json!({
            "object": "list",
            "data": data
        })
    };

    // Main app: proxy with model switcher middleware
    let app = Router::new()
        .route(
            "/v1/models",
            get(move || {
                let resp = models_response.clone();
                async move { Json(resp) }
            }),
        )
        .fallback(proxy_handler)
        .with_state(proxy_state)
        .layer(ModelSwitcherLayer::new(switcher.clone()));

    Ok((app, switcher))
}
