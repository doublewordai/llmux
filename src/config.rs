//! Configuration for llmux

use crate::policy::{CostAwarePolicy, FifoPolicy, SwitchPolicy, TimeSlicePolicy};
use crate::switcher::EvictionPolicy;
use anyhow::{Context, Result};
use onwards::target::Targets;
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

    /// Metrics port (0 to disable)
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Admin/control API port (None to disable)
    #[serde(default)]
    pub admin_port: Option<u16>,

    /// Command to use for spawning vLLM processes (default: "vllm")
    /// Can be overridden for testing with mock-vllm
    #[serde(default = "default_vllm_command")]
    pub vllm_command: String,

    /// CRIU/CUDA checkpoint configuration.
    /// Required when any model uses CudaSuspend or Checkpoint process strategy.
    #[serde(default)]
    pub checkpoint: Option<CheckpointConfig>,
}

/// Configuration for CUDA/CRIU-based checkpointing.
///
/// Required when any model uses `ProcessStrategy::CudaSuspend` or
/// `ProcessStrategy::Checkpoint`. Provides paths to `cuda-checkpoint` and
/// `criu` binaries, and the CUDA plugin directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Path to the criu binary (default: "criu" on PATH)
    #[serde(default = "default_criu_path")]
    pub criu_path: String,

    /// Directory containing the CUDA checkpoint plugin (libcuda_plugin.so)
    #[serde(default = "default_cuda_plugin_dir")]
    pub cuda_plugin_dir: String,

    /// Base directory for checkpoint images (per-model subdirectories are created)
    #[serde(default = "default_images_dir")]
    pub images_dir: PathBuf,

    /// Path to cuda-checkpoint binary (default: "cuda-checkpoint" on PATH)
    #[serde(default = "default_cuda_checkpoint_path")]
    pub cuda_checkpoint_path: String,

    /// Keep checkpoint images on disk after restore (default: true).
    /// When true, images are preserved so the same checkpoint can be
    /// restored multiple times without re-checkpointing.
    #[serde(default = "default_keep_images")]
    pub keep_images: bool,

    /// Optional S3-compatible object store for checkpoint persistence.
    /// When configured, checkpoints are uploaded after creation and
    /// downloaded on demand before restore.
    #[serde(default)]
    pub object_store: Option<ObjectStoreConfig>,
}

/// S3-compatible object store configuration for checkpoint persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreConfig {
    /// S3-compatible endpoint URL (e.g. "http://localhost:9200")
    pub endpoint: String,
    /// Bucket name
    pub bucket: String,
    /// Access key (AWS_ACCESS_KEY_ID equivalent)
    pub access_key: String,
    /// Secret key (AWS_SECRET_ACCESS_KEY equivalent)
    pub secret_key: String,
    /// Region (default: "us-east-1")
    #[serde(default = "default_region")]
    pub region: String,
}

fn default_vllm_command() -> String {
    "vllm".to_string()
}

fn default_criu_path() -> String {
    "criu".to_string()
}

fn default_cuda_plugin_dir() -> String {
    "/usr/lib/criu/".to_string()
}

fn default_images_dir() -> PathBuf {
    PathBuf::from("/tmp/llmux-checkpoints")
}

fn default_cuda_checkpoint_path() -> String {
    "cuda-checkpoint".to_string()
}

fn default_keep_images() -> bool {
    true
}

fn default_region() -> String {
    "us-east-1".to_string()
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

    /// Validate configuration, warning about common misconfigurations.
    ///
    /// Checks that models using CudaSuspend or Checkpoint process strategies
    /// with tensor parallelism have the required vLLM flags.
    pub fn validate(&self) {
        use crate::switcher::ProcessStrategy;
        use tracing::warn;

        for (name, model) in &self.models {
            if !model.eviction.needs_cuda_checkpoint() {
                continue;
            }

            // Parse --tensor-parallel-size from extra_args
            let tp = model.tensor_parallel_size();
            if tp <= 1 {
                continue;
            }

            let strategy_name = match model.eviction.process {
                ProcessStrategy::CudaSuspend => "CudaSuspend",
                ProcessStrategy::Checkpoint => "Checkpoint",
                _ => continue,
            };

            if !model.extra_args.contains(&"--enforce-eager".to_string()) {
                warn!(
                    model = %name,
                    "Model uses {} at TP={} but is missing --enforce-eager. \
                     CUDA graphs hold stale NCCL handles and will crash on resume.",
                    strategy_name, tp
                );
            }

            if !model
                .extra_args
                .contains(&"--disable-custom-all-reduce".to_string())
            {
                warn!(
                    model = %name,
                    "Model uses {} at TP={} but is missing --disable-custom-all-reduce. \
                     CustomAllReduce IPC buffers cannot survive cuda-checkpoint.",
                    strategy_name, tp
                );
            }
        }

        // Validate checkpoint_path entries
        let has_object_store = self
            .checkpoint
            .as_ref()
            .and_then(|c| c.object_store.as_ref())
            .is_some();

        for (name, model) in &self.models {
            if let Some(ref checkpoint_path) = model.checkpoint_path {
                if !checkpoint_path.exists() && !has_object_store {
                    tracing::error!(
                        model = %name,
                        path = %checkpoint_path.display(),
                        "checkpoint_path does not exist. Remove it from config or \
                         create the checkpoint with --checkpoint first."
                    );
                    std::process::exit(1);
                } else if !checkpoint_path.exists() {
                    tracing::info!(
                        model = %name,
                        path = %checkpoint_path.display(),
                        "checkpoint_path does not exist locally; will download from object store on first request"
                    );
                }
                if self.checkpoint.is_none() {
                    tracing::error!(
                        model = %name,
                        "checkpoint_path is set but no top-level 'checkpoint' config section. \
                         Add a 'checkpoint' section with criu_path, cuda_plugin_dir, etc."
                    );
                    std::process::exit(1);
                }
            }
        }
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

/// Configuration for a single model.
///
/// ```json
/// {
///   "model_path": "...",
///   "port": 8001,
///   "eviction": { "weights": "discard", "process": "checkpoint" }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model (HuggingFace model ID or local path)
    pub model_path: String,

    /// Port for this model's vLLM instance
    pub port: u16,

    /// Additional vLLM CLI arguments (e.g. `["--gpu-memory-utilization", "0.8"]`)
    #[serde(default)]
    pub extra_args: Vec<String>,

    /// Eviction policy: controls what happens to weights and to the process
    /// when this model is put to sleep.
    #[serde(default)]
    pub eviction: EvictionPolicy,

    /// Path to existing CRIU checkpoint images. When set, the daemon restores
    /// from checkpoint on first request instead of cold-starting vLLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_path: Option<PathBuf>,
}

impl ModelConfig {
    /// Parse tensor parallel size from extra_args (default: 1)
    pub fn tensor_parallel_size(&self) -> usize {
        self.extra_args
            .windows(2)
            .find(|w| w[0] == "--tensor-parallel-size")
            .and_then(|w| w[1].parse().ok())
            .unwrap_or(1)
    }

    /// Build vLLM command line arguments.
    ///
    /// Always includes `--enable-sleep-mode` for consistency across all
    /// eviction strategies. The dev mode endpoints (sleep/wake/collective_rpc)
    /// are available regardless.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Policy type: "fifo" (default) or "cost_aware"
    #[serde(default = "default_policy_type")]
    pub policy_type: String,

    /// Request timeout in seconds
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Whether to drain in-flight requests before switching
    #[serde(default = "default_drain_before_switch")]
    pub drain_before_switch: bool,

    /// Default eviction policy for models that don't specify one
    #[serde(default)]
    pub eviction: EvictionPolicy,

    /// Minimum seconds a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing that can cause GPU page faults.
    #[serde(default = "default_min_active_secs")]
    pub min_active_secs: u64,

    /// (cost_aware) Coalescing window in milliseconds. On the first request
    /// for a non-active model, wait this long to collect more demand.
    #[serde(default = "default_coalesce_window_ms")]
    pub coalesce_window_ms: u64,

    /// (cost_aware) Amortization factor. Higher values require more pending
    /// requests before switching. 0.0 = always switch immediately.
    #[serde(default = "default_amortization_factor")]
    pub amortization_factor: f64,

    /// (cost_aware) Maximum wait in seconds before forcing a switch,
    /// regardless of cost calculations.
    #[serde(default = "default_max_wait_secs")]
    pub max_wait_secs: u64,

    /// (time_slice) Scheduler tick interval in milliseconds. Controls how
    /// often the background scheduler evaluates queue depths.
    #[serde(default = "default_tick_interval_ms")]
    pub tick_interval_ms: u64,

    /// (time_slice) Minimum quantum in milliseconds. Each model gets at
    /// least this much GPU time before the scheduler considers switching.
    #[serde(default = "default_min_quantum_ms")]
    pub min_quantum_ms: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            policy_type: default_policy_type(),
            request_timeout_secs: default_request_timeout(),
            drain_before_switch: default_drain_before_switch(),
            eviction: EvictionPolicy::default(),
            min_active_secs: default_min_active_secs(),
            coalesce_window_ms: default_coalesce_window_ms(),
            amortization_factor: default_amortization_factor(),
            max_wait_secs: default_max_wait_secs(),
            tick_interval_ms: default_tick_interval_ms(),
            min_quantum_ms: default_min_quantum_ms(),
        }
    }
}

fn default_policy_type() -> String {
    "fifo".to_string()
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

fn default_coalesce_window_ms() -> u64 {
    2000
}

fn default_amortization_factor() -> f64 {
    0.5
}

fn default_max_wait_secs() -> u64 {
    15
}

fn default_tick_interval_ms() -> u64 {
    500
}

fn default_min_quantum_ms() -> u64 {
    2000
}

impl PolicyConfig {
    /// Build a SwitchPolicy from this config.
    /// `model_names` is needed for cost_aware policy to pre-populate timing tables.
    pub fn build_policy(&self, model_names: &[String]) -> Box<dyn SwitchPolicy> {
        match self.policy_type.as_str() {
            "cost_aware" => Box::new(CostAwarePolicy::new(
                self.eviction,
                Duration::from_secs(self.request_timeout_secs),
                Duration::from_secs(self.min_active_secs),
                Duration::from_millis(self.coalesce_window_ms),
                self.amortization_factor,
                Duration::from_secs(self.max_wait_secs),
                model_names.to_vec(),
            )),
            "time_slice" => Box::new(TimeSlicePolicy::new(
                self.eviction,
                Duration::from_secs(self.request_timeout_secs),
                Duration::from_secs(self.min_active_secs),
                Duration::from_secs(self.max_wait_secs),
                Duration::from_millis(self.min_quantum_ms),
                Duration::from_millis(self.tick_interval_ms),
                model_names.to_vec(),
            )),
            _ => Box::new(FifoPolicy::new(
                self.eviction,
                Duration::from_secs(self.request_timeout_secs),
                self.drain_before_switch,
                Duration::from_secs(self.min_active_secs),
            )),
        }
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
            eviction: EvictionPolicy::from(1),
            checkpoint_path: None,
        };

        let args = config.vllm_args();
        assert!(args.contains(&"--enable-sleep-mode".to_string()));
        assert!(args.contains(&"--tensor-parallel-size".to_string()));
        assert!(args.contains(&"2".to_string()));
    }

    #[test]
    fn test_eviction_policy_deserialize() {
        let json = r#"{
            "models": {
                "llama": {
                    "model_path": "meta-llama/Llama-2-7b",
                    "port": 8001,
                    "eviction": { "weights": "discard", "process": "checkpoint" }
                }
            },
            "port": 3000
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        let model = &config.models["llama"];
        assert_eq!(model.eviction.weights, crate::switcher::WeightStrategy::Discard);
        assert_eq!(model.eviction.process, crate::switcher::ProcessStrategy::Checkpoint);
    }
}
