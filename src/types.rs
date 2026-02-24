//! Shared domain types used across the codebase.

/// Errors from the switcher
#[derive(Debug, thiserror::Error)]
pub enum SwitchError {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model not ready: {0}")]
    NotReady(String),

    #[error("request timeout")]
    Timeout,

    #[error("hook failed for {model}: {detail}")]
    HookFailed { model: String, detail: String },

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
