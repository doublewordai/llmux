//! Shared domain types used across the codebase.
//!
//! These types define the eviction policy (weight strategy x process strategy),
//! switcher state machine, switch mode, and error types.

use crate::orchestrator::OrchestratorError;

/// What to do with model weights when freeing GPU memory.
///
/// This is one axis of the eviction policy. The other is [`ProcessStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightStrategy {
    /// Keep weights on GPU (no vLLM sleep). Only meaningful with CudaSuspend
    /// or Checkpoint — the CUDA plugin/cuda-checkpoint handles GPU state directly.
    Retain,
    /// vLLM L1: offload to pinned CPU RAM. Fast wake (~1s), but consumes
    /// host RAM and produces large CRIU images.
    Offload,
    /// vLLM L2: discard weights. Slow wake (reload from disk ~15s), but
    /// frees host RAM and produces small CRIU images.
    Discard,
}

/// What to do with the OS process after weight strategy is applied.
///
/// This is one axis of the eviction policy. The other is [`WeightStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStrategy {
    /// Process stays alive, CUDA context active.
    KeepRunning,
    /// cuda-checkpoint: suspend CUDA context, free remaining VRAM.
    /// Process stays alive in host RAM.
    CudaSuspend,
    /// CRIU: snapshot to disk, process killed. Frees everything.
    Checkpoint,
    /// Kill process. Cold start on next use.
    Stop,
}

/// Two-axis eviction policy: weight management x process management.
///
/// Combinations and their tradeoffs:
///
/// | Weights × Process | Images | Wake cost | GPU freed | Host RAM |
/// |---|---|---|---|---|
/// | Discard + Checkpoint | tiny (~7GB) | reload weights (~15s) | 100% | 100% |
/// | Offload + Checkpoint | large (~45GB) | instant (~1s) | 100% | 100% |
/// | Offload + CudaSuspend | none (in RAM) | instant (~3s) | 100% | weights stay |
/// | Offload + KeepRunning | none | instant (~1s) | ~90% | weights stay |
/// | Discard + KeepRunning | none | reload weights (~15s) | ~90% | freed |
/// | Retain + CudaSuspend | none (in RAM) | instant (~3s) | 100% | VRAM copy |
/// | Retain + Stop | none | cold start | 100% | 100% |
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvictionPolicy {
    pub weights: WeightStrategy,
    pub process: ProcessStrategy,
}

impl EvictionPolicy {
    /// Convenience constant: kill the process (no checkpoint, no sleep).
    pub const STOP: Self = Self {
        weights: WeightStrategy::Retain,
        process: ProcessStrategy::Stop,
    };

    /// Whether this policy uses vLLM's sleep API (L1 or L2).
    pub fn needs_vllm_sleep(&self) -> bool {
        matches!(
            self.weights,
            WeightStrategy::Offload | WeightStrategy::Discard
        )
    }

    /// Whether this policy needs cuda-checkpoint or CRIU (which includes
    /// the CUDA plugin that handles cuda-checkpoint internally).
    pub fn needs_cuda_checkpoint(&self) -> bool {
        matches!(
            self.process,
            ProcessStrategy::CudaSuspend | ProcessStrategy::Checkpoint
        )
    }

    /// Whether this policy uses CRIU for process snapshotting.
    pub fn needs_criu(&self) -> bool {
        matches!(self.process, ProcessStrategy::Checkpoint)
    }

    /// The vLLM sleep level number (1 or 2), or None if no vLLM sleep.
    pub fn vllm_sleep_level(&self) -> Option<u8> {
        match self.weights {
            WeightStrategy::Offload => Some(1),
            WeightStrategy::Discard => Some(2),
            WeightStrategy::Retain => None,
        }
    }
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            weights: WeightStrategy::Retain,
            process: ProcessStrategy::Stop,
        }
    }
}

/// Backward-compatible conversion from the old numeric sleep levels.
///
/// - 1 → Offload + KeepRunning (old L1)
/// - 2 → Discard + KeepRunning (old L2)
/// - 3 → Retain + CudaSuspend (old CudaSuspend)
/// - 4 → Discard + Checkpoint (old Checkpoint — now uses the optimal combo)
/// - 5 → Retain + Stop (old Stop)
impl From<u8> for EvictionPolicy {
    fn from(level: u8) -> Self {
        match level {
            1 => EvictionPolicy {
                weights: WeightStrategy::Offload,
                process: ProcessStrategy::KeepRunning,
            },
            2 => EvictionPolicy {
                weights: WeightStrategy::Discard,
                process: ProcessStrategy::KeepRunning,
            },
            3 => EvictionPolicy {
                weights: WeightStrategy::Retain,
                process: ProcessStrategy::CudaSuspend,
            },
            4 => EvictionPolicy {
                weights: WeightStrategy::Discard,
                process: ProcessStrategy::Checkpoint,
            },
            5 | _ => EvictionPolicy {
                weights: WeightStrategy::Retain,
                process: ProcessStrategy::Stop,
            },
        }
    }
}

/// Errors from the switcher
#[derive(Debug, thiserror::Error)]
pub enum SwitchError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model not ready: {0}")]
    NotReady(String),

    #[error("request timeout")]
    Timeout,

    #[error("orchestrator error: {0}")]
    Orchestrator(#[from] OrchestratorError),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("manual mode: model {requested} not available (active: {active})")]
    ManualModeRejected { requested: String, active: String },
}

/// Switch mode controls whether model switching is automatic or manual.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SwitchMode {
    /// Policy-driven automatic switching (default behavior)
    Auto,
    /// Manual mode: no auto-switching. Only the pinned model (if set) serves requests.
    Manual {
        #[serde(skip_serializing_if = "Option::is_none")]
        pinned: Option<String>,
    },
}

/// State of the model switcher
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitcherState {
    /// No model is currently active
    Idle,
    /// A model is awake and ready
    Active { model: String },
    /// Switching from one model to another
    Switching { from: Option<String>, to: String },
}
