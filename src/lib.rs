//! # llmux
//!
//! Zero-reload model switching for vLLM - manages multiple models on shared GPU.
//!
//! This crate provides:
//! - **Orchestrator**: Lazily starts vLLM processes on first request
//! - **Switcher**: Coordinates wake/sleep between models
//! - **Middleware**: Axum layer that integrates with onwards proxy
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     llmux                          │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │ Orchestrator                                         │   │
//! │  │ - Spawns vLLM processes lazily                       │   │
//! │  │ - Tracks: NotStarted | Starting | Running | Sleeping │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                          │                                  │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │ Middleware Layer                                     │   │
//! │  │ - Extracts model from request                        │   │
//! │  │ - Ensures model ready before forwarding              │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                          │                                  │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │ Onwards Proxy                                        │   │
//! │  │ - Routes to vLLM by model name                       │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                          │                                  │
//! │      ┌───────────────────┼───────────────────┐             │
//! │      ▼                   ▼                   ▼             │
//! │  [vLLM:8001]        [vLLM:8002]         [vLLM:8003]        │
//! │   (llama)           (mistral)           (qwen)            │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod config;
mod middleware;
mod orchestrator;
mod policy;
mod switcher;
pub mod validate;

pub use config::{Config, ModelConfig, PolicyConfig};
pub use middleware::{ModelSwitcherLayer, ModelSwitcherService};
pub use orchestrator::{Orchestrator, OrchestratorError, ProcessState};
pub use policy::{FifoPolicy, PolicyContext, PolicyDecision, SwitchPolicy};
pub use switcher::{ModelSwitcher, SleepLevel, SwitchError, SwitcherState};

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

/// Build the complete llmux stack
///
/// Returns an Axum router with:
/// - Model switching middleware
/// - Onwards proxy configured for all models
pub async fn build_app(config: Config) -> Result<axum::Router> {
    info!("Building llmux with {} models", config.models.len());

    // Create orchestrator with configured command
    let orchestrator = Arc::new(Orchestrator::with_command(
        config.models.clone(),
        config.vllm_command.clone(),
    ));

    // Create policy
    let policy = config.policy.build_policy();

    // Create switcher
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    // Build onwards targets from model configs
    let targets = config.build_onwards_targets()?;

    // Create onwards app state
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);

    // Wrap with model switcher middleware
    let app = onwards_router.layer(ModelSwitcherLayer::new(switcher));

    Ok(app)
}
