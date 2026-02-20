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
pub mod control;
mod middleware;
pub mod object_store;
mod orchestrator;
mod policy;
mod switcher;
pub mod validate;

pub use config::{CheckpointConfig, Config, ModelConfig, ObjectStoreConfig, PolicyConfig};
pub use middleware::{ModelSwitcherLayer, ModelSwitcherService};
pub use orchestrator::{Orchestrator, OrchestratorError, ProcessState};
pub use policy::{
    CostAwarePolicy, FifoPolicy, PolicyContext, PolicyDecision, ScheduleContext, SwitchPolicy,
    TimeSlicePolicy,
};
pub use switcher::{
    EvictionPolicy, ModelSwitcher, ProcessStrategy, SwitchError, SwitchMode, SwitcherState,
    WeightStrategy,
};

use anyhow::Result;
use std::sync::Arc;
use tracing::info;

/// Build the complete llmux stack
///
/// Returns:
/// - The main Axum router (proxy + middleware)
/// - An optional metrics router (when `config.metrics_port > 0`)
/// - The control API router (for the admin port)
/// - The model switcher (for driving warmup before serving)
pub async fn build_app(
    config: Config,
) -> Result<(
    axum::Router,
    Option<axum::Router>,
    axum::Router,
    ModelSwitcher,
)> {
    info!("Building llmux with {} models", config.models.len());

    // Create orchestrator with configured command
    let orchestrator = Arc::new(Orchestrator::with_options(
        config.models.clone(),
        config.vllm_command.clone(),
        config.checkpoint.clone(),
    ));

    // Create policy
    let model_names: Vec<String> = config.models.keys().cloned().collect();
    let policy = config.policy.build_policy(&model_names);

    // Create switcher
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    // Spawn background scheduler if the policy uses one
    let _scheduler_handle = switcher.clone().spawn_scheduler();

    // Build control API router (served on separate admin port)
    let control = control::control_router(switcher.clone());

    // Build onwards targets from model configs
    let targets = config.build_onwards_targets()?;

    // Create onwards app state
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);

    // Wrap with model switcher middleware
    let mut app = onwards_router.layer(ModelSwitcherLayer::new(switcher.clone()));

    // Install metrics layer and build metrics router if enabled
    let metrics_router = if config.metrics_port > 0 {
        let (prometheus_layer, handle) = onwards::build_metrics_layer_and_handle("llmux");
        app = app.layer(prometheus_layer);
        Some(onwards::build_metrics_router(handle))
    } else {
        None
    };

    Ok((app, metrics_router, control, switcher))
}

/// Run the warmup phase: start each model, run one inference, then sleep it.
///
/// Iterates all models sequentially. Each model is cold-started, warmed with
/// a single inference request (to compile CUDA graphs and warm the allocator),
/// then put to sleep using its configured eviction policy. After warmup,
/// every model is in its warm sleeping state so the first real request
/// triggers a fast wake rather than a cold start.
pub async fn run_warmup(switcher: &ModelSwitcher) -> Result<()> {
    let orchestrator = switcher.orchestrator();
    let models = orchestrator.registered_models();

    info!(count = models.len(), "Starting warmup phase");

    for model in &models {
        let eviction = orchestrator
            .eviction_policy_for(model)
            .expect("model registered but no eviction policy");

        let port = switcher
            .orchestrator()
            .model_port(model)
            .expect("model registered but no port");

        let model_path = switcher
            .orchestrator()
            .model_path(model)
            .expect("model registered but no model_path");

        info!(model = %model, "Warmup: starting");
        orchestrator
            .ensure_running(model)
            .await
            .map_err(|e| anyhow::anyhow!("warmup: failed to start {}: {}", model, e))?;

        info!(model = %model, "Warmup: running inference");
        validate::run_warmup_inference(port, &model_path).await?;

        info!(model = %model, ?eviction, "Warmup: sleeping");
        orchestrator
            .sleep_model(model, eviction)
            .await
            .map_err(|e| anyhow::anyhow!("warmup: failed to sleep {}: {}", model, e))?;

        info!(model = %model, "Warmup: complete");
    }

    info!("Warmup phase complete — all models warmed and sleeping");
    Ok(())
}
