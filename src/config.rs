//! Configuration for llmux

use crate::policy::{FifoPolicy, SwitchPolicy};
use anyhow::{Context, Result};
use onwards::target::Targets;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
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

    /// Metrics port (0 to disable)
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Command to use for spawning vLLM processes (default: "vllm")
    /// Can be overridden for testing with mock-vllm
    #[serde(default = "default_vllm_command")]
    pub vllm_command: String,
}

fn default_vllm_command() -> String {
    "vllm".to_string()
}

fn default_port() -> u16 {
    3000
}

fn default_metrics_port() -> u16 {
    9090
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

    /// Build onwards Targets from model configs
    pub fn build_onwards_targets(&self) -> Result<Targets> {
        use dashmap::DashMap;
        use onwards::load_balancer::ProviderPool;
        use onwards::target::Target;
        use std::sync::Arc;

        let targets_map: DashMap<String, ProviderPool> = DashMap::new();

        for (name, model_config) in &self.models {
            let url = format!("http://localhost:{}", model_config.port)
                .parse()
                .with_context(|| format!("Invalid URL for model {}", name))?;

            let target = Target {
                url,
                keys: None,
                onwards_key: None,
                onwards_model: Some(model_config.model_path.clone()),
                limiter: None,
                concurrency_limiter: None,
                upstream_auth_header_name: None,
                upstream_auth_header_prefix: None,
                response_headers: None,
                sanitize_response: false,
            };

            let pool = target.into_pool();
            targets_map.insert(name.clone(), pool);
        }

        Ok(Targets {
            targets: Arc::new(targets_map),
            key_rate_limiters: Arc::new(DashMap::new()),
            key_concurrency_limiters: Arc::new(DashMap::new()),
        })
    }
}

/// Configuration for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model (HuggingFace model ID or local path)
    pub model_path: String,

    /// Port for this model's vLLM instance
    pub port: u16,

    /// Additional vLLM CLI arguments (e.g. `["--gpu-memory-utilization", "0.8"]`)
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Sleep level when hibernating:
    /// - 1: Offload weights to CPU RAM (fastest wake)
    /// - 2: Discard weights (slower wake, less RAM)
    /// - 3: Stop the vLLM process entirely (slowest wake, frees all GPU memory)
    #[serde(default = "default_sleep_level")]
    pub sleep_level: u8,
}

fn default_sleep_level() -> u8 {
    3
}

impl ModelConfig {
    /// Build vLLM command line arguments
    pub fn vllm_args(&self) -> Vec<String> {
        let mut args = vec![
            "serve".to_string(),
            self.model_path.clone(),
            "--port".to_string(),
            self.port.to_string(),
            "--enable-sleep-mode".to_string(),
        ];

        args.extend(self.extra_args.clone());
        args
    }
}

/// Policy configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PolicyConfig {
    /// Policy type: "fifo" (default) or "batching"
    #[serde(default = "default_policy_type")]
    pub policy_type: String,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Whether to drain in-flight requests before switching
    #[serde(default = "default_drain_before_switch")]
    pub drain_before_switch: bool,

    /// Default sleep level (1, 2, or 3)
    #[serde(default = "default_sleep_level")]
    pub sleep_level: u8,

    /// Minimum seconds a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing that can cause GPU page faults.
    #[serde(default = "default_min_active_secs")]
    pub min_active_secs: u64,
}

fn default_policy_type() -> String {
    "fifo".to_string()
}

fn default_request_timeout() -> u64 {
    60
}

fn default_drain_before_switch() -> bool {
    true
}

fn default_min_active_secs() -> u64 {
    5
}

impl PolicyConfig {
    /// Build a SwitchPolicy from this config
    pub fn build_policy(&self) -> Box<dyn SwitchPolicy> {
        // Currently only FIFO policy is implemented
        // Future: add other policy types like "batching"
        Box::new(FifoPolicy::new(
            self.sleep_level,
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
                    "model_path": "meta-llama/Llama-2-7b",
                    "port": 8001
                },
                "mistral": {
                    "model_path": "mistralai/Mistral-7B-v0.1",
                    "port": 8002,
                    "extra_args": ["--gpu-memory-utilization", "0.8"]
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
        assert_eq!(config.models["mistral"].extra_args.len(), 2);
        assert_eq!(config.policy.request_timeout_secs, 30);
    }

    #[test]
    fn test_vllm_args() {
        let config = ModelConfig {
            model_path: "meta-llama/Llama-2-7b".to_string(),
            port: 8001,
            extra_args: vec![
                "--tensor-parallel-size".to_string(),
                "2".to_string(),
                "--max-model-len".to_string(),
                "4096".to_string(),
            ],
            sleep_level: 1,
        };

        let args = config.vllm_args();
        assert!(args.contains(&"--enable-sleep-mode".to_string()));
        assert!(args.contains(&"--tensor-parallel-size".to_string()));
        assert!(args.contains(&"2".to_string()));
    }
}
