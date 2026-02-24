//! Script-based lifecycle hooks for model management.
//!
//! All model lifecycle operations (wake, sleep, liveness) are delegated to
//! user-provided scripts. llmux does not know or care how models are started,
//! stopped, or health-checked — that's entirely up to the scripts.
//!
//! Hooks are executed via `sh -c`, so they can be either a path to an
//! executable or an inline shell script.

use crate::config::ModelConfig;
use std::collections::HashMap;
use tokio::process::Command;
use tracing::{debug, warn};

/// Errors from hook script execution
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("hook {hook} failed for model {model} (exit code {code}): {stderr}")]
    Failed {
        model: String,
        hook: String,
        code: i32,
        stderr: String,
    },

    #[error("hook execution error: {0}")]
    Io(#[from] std::io::Error),

    #[error("model not found: {0}")]
    ModelNotFound(String),
}

/// Runs lifecycle scripts for models.
pub struct HookRunner {
    configs: HashMap<String, ModelConfig>,
}

impl HookRunner {
    pub fn new(configs: HashMap<String, ModelConfig>) -> Self {
        Self { configs }
    }

    pub fn registered_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    pub fn model_port(&self, model: &str) -> Option<u16> {
        self.configs.get(model).map(|c| c.port)
    }

    pub fn is_registered(&self, model: &str) -> bool {
        self.configs.contains_key(model)
    }

    /// Run the wake script for a model. Returns Ok(()) when the model is ready.
    ///
    /// The wake script must be idempotent — it should bring a model from any
    /// state (stopped, sleeping, already running) to a running state.
    pub async fn run_wake(&self, model: &str) -> Result<(), HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        run_hook(&config.wake, model, "wake").await
    }

    /// Run the sleep script for a model. Returns Ok(()) when the model is asleep.
    pub async fn run_sleep(&self, model: &str) -> Result<(), HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        run_hook(&config.sleep, model, "sleep").await
    }

    /// Run the alive script for a model. Returns true if healthy, false if not.
    ///
    /// Only returns Err for execution failures (script not found, permission
    /// denied, etc.), not for a non-zero exit code (which means "unhealthy").
    pub async fn run_alive(&self, model: &str) -> Result<bool, HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        match run_hook(&config.alive, model, "alive").await {
            Ok(()) => Ok(true),
            Err(HookError::Failed { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

async fn run_hook(script: &str, model: &str, hook_name: &str) -> Result<(), HookError> {
    debug!(model = %model, hook = %hook_name, "Running hook");

    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .env("LLMUX_MODEL", model)
        .output()
        .await
        .map_err(HookError::Io)?;

    if !output.stdout.is_empty() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!(model = %model, hook = %hook_name, stdout = %stdout.trim_end(), "Hook stdout");
    }

    if !output.stderr.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() {
            debug!(model = %model, hook = %hook_name, stderr = %stderr.trim_end(), "Hook stderr");
        } else {
            warn!(model = %model, hook = %hook_name, stderr = %stderr.trim_end(), "Hook failed");
        }
    }

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(HookError::Failed {
            model: model.to_string(),
            hook: hook_name.to_string(),
            code,
            stderr,
        });
    }

    debug!(model = %model, hook = %hook_name, "Hook completed successfully");
    Ok(())
}
