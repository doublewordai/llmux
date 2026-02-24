//! Configuration for llmux v2.
//!
//! All model lifecycle management is delegated to user-provided scripts.
//! llmux only handles request routing, draining, and policy decisions.
//! TODO: switch to yaml for better readability and support for multiline strings (for inline
//! scripts).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Top-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Models to manage
    pub models: HashMap<String, ModelConfig>,

    /// Switch policy configuration
    #[serde(default)]
    pub policy: PolicyConfig,

    /// Proxy port
    #[serde(default = "default_port")]
    pub port: u16,
}

/// Configuration for a single model.
///
/// Each model is defined by a target port (where the inference server listens)
/// and three lifecycle scripts: wake, sleep, alive.
///
/// ```json
/// {
///   "port": 8001,
///   "wake": "./scripts/wake.sh",
///   "sleep": "./scripts/sleep.sh",
///   "alive": "./scripts/alive.sh"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Port where the model's inference server listens
    pub port: u16,

    /// Script to bring the model to a running state (idempotent).
    /// Called with LLMUX_MODEL env var set to the model name.
    /// Must exit 0 when the model is ready to serve requests.
    /// TODO: allow inline scripts here: switch config to yaml and use gha style multiline strings
    /// for convenience.
    pub wake: PathBuf,

    /// Script to put the model to sleep / free resources.
    /// Called with LLMUX_MODEL env var set to the model name.
    /// Must exit 0 when the model is fully stopped/sleeping.
    /// TODO: allow inline scripts here: switch config to yaml and use gha style multiline strings
    /// for convenience.
    pub sleep: PathBuf,

    /// Script to check if the model is alive and healthy.
    /// Called with LLMUX_MODEL env var set to the model name.
    /// Exit 0 = healthy, non-zero = unhealthy.
    /// TODO: allow inline scripts here: switch config to yaml and use gha style multiline strings
    /// for convenience.
    pub alive: PathBuf,
}

fn default_port() -> u16 {
    3000
}

impl Config {
    /// Load configuration from a JSON file
    pub async fn from_file(path: &Path) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }
}

/// Policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Request timeout in seconds
    /// TODO: default to unlimited (None) and let the user set it explicitly if they want a
    /// timeout.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Whether to drain in-flight requests before switching
    #[serde(default = "default_drain_before_switch")]
    pub drain_before_switch: bool,

    /// Minimum seconds a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing.
    /// TODO: Default to 0 (no minimum) and let the user set it if they want a minimum active
    /// duration.
    #[serde(default = "default_min_active_secs")]
    pub min_active_secs: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_request_timeout(),
            drain_before_switch: default_drain_before_switch(),
            min_active_secs: default_min_active_secs(),
        }
    }
}

fn default_request_timeout() -> u64 {
    300
}

fn default_drain_before_switch() -> bool {
    true
}

fn default_min_active_secs() -> u64 {
    5
}

impl PolicyConfig {
    pub fn build_policy(&self) -> Box<dyn crate::policy::SwitchPolicy> {
        Box::new(crate::policy::FifoPolicy::new(
            Duration::from_secs(self.request_timeout_secs),
            self.drain_before_switch,
            Duration::from_secs(self.min_active_secs),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let json = r#"{
            "models": {
                "llama": {
                    "port": 8001,
                    "wake": "./scripts/wake-llama.sh",
                    "sleep": "./scripts/sleep-llama.sh",
                    "alive": "./scripts/alive-llama.sh"
                },
                "mistral": {
                    "port": 8002,
                    "wake": "./scripts/wake-mistral.sh",
                    "sleep": "./scripts/sleep-mistral.sh",
                    "alive": "./scripts/alive-mistral.sh"
                }
            },
            "policy": {
                "request_timeout_secs": 30
            },
            "port": 3000
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models["llama"].port, 8001);
        assert_eq!(config.policy.request_timeout_secs, 30);
    }

    #[test]
    fn test_defaults() {
        let json = r#"{
            "models": {
                "llama": {
                    "port": 8001,
                    "wake": "./wake.sh",
                    "sleep": "./sleep.sh",
                    "alive": "./alive.sh"
                }
            }
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.port, 3000);
        assert_eq!(config.policy.request_timeout_secs, 300);
        assert!(config.policy.drain_before_switch);
        assert_eq!(config.policy.min_active_secs, 5);
    }
}
