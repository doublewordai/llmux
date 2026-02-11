//! Control API for manual model management.
//!
//! Provides HTTP endpoints for inspecting and controlling the model switcher
//! outside of the normal request-driven switching flow. Intended to run on
//! a separate admin port.
//!
//! ## Endpoints
//!
//! | Method | Path                    | Description                              |
//! |--------|-------------------------|------------------------------------------|
//! | GET    | `/control/status`       | Current state, mode, in-flight, queues   |
//! | POST   | `/control/mode`         | Set switch mode (auto/manual)            |
//! | POST   | `/control/pin`          | Pin a model (enters manual mode + switch)|
//! | POST   | `/control/unpin`        | Unpin and return to auto mode            |
//! | POST   | `/control/switch`       | Force switch to a model                  |
//! | GET    | `/control/eviction`     | Current eviction policy per model         |
//! | POST   | `/control/eviction`     | Override a model's eviction policy        |
//! | POST   | `/control/sleep`        | Put a model to sleep                      |
//! | POST   | `/control/wake`         | Wake a sleeping model                     |

use crate::orchestrator::ProcessState;
use crate::switcher::{EvictionPolicy, ModelSwitcher, SwitchMode, SwitcherState};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::{get, post}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Build the control API router.
pub fn control_router(switcher: ModelSwitcher) -> Router {
    Router::new()
        .route("/control/status", get(get_status))
        .route("/control/mode", post(set_mode))
        .route("/control/pin", post(pin_model))
        .route("/control/unpin", post(unpin_model))
        .route("/control/switch", post(force_switch))
        .route("/control/eviction", get(get_eviction_policies).post(set_eviction_policy))
        .route("/control/sleep", post(sleep_model))
        .route("/control/wake", post(wake_model))
        .with_state(switcher)
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StatusResponse {
    state: String,
    active_model: Option<String>,
    mode: SwitchMode,
    models: HashMap<String, ModelStatus>,
}

#[derive(Serialize)]
struct ModelStatus {
    in_flight: usize,
    queue_depth: usize,
    process_state: String,
}

#[derive(Deserialize)]
struct SetModeRequest {
    mode: String,
}

#[derive(Deserialize)]
struct PinRequest {
    model: String,
}

#[derive(Deserialize)]
struct SwitchRequest {
    model: String,
}

#[derive(Deserialize)]
struct SetEvictionRequest {
    model: String,
    eviction: EvictionPolicy,
}

#[derive(Deserialize)]
struct SleepRequest {
    model: String,
    eviction: Option<EvictionPolicy>,
}

#[derive(Deserialize)]
struct WakeRequest {
    model: String,
}

#[derive(Serialize)]
struct EvictionPoliciesResponse {
    models: HashMap<String, EvictionPolicyInfo>,
}

#[derive(Serialize)]
struct EvictionPolicyInfo {
    eviction: EvictionPolicy,
    process_state: String,
}

#[derive(Serialize)]
struct MessageResponse {
    message: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn get_status(State(switcher): State<ModelSwitcher>) -> impl IntoResponse {
    let state = switcher.state().await;
    let mode = switcher.mode().await;
    let queue_depths = switcher.queue_depths().await;

    let active_model = match &state {
        SwitcherState::Active { model } => Some(model.clone()),
        _ => None,
    };

    let state_str = match &state {
        SwitcherState::Idle => "idle".to_string(),
        SwitcherState::Active { model } => format!("active:{}", model),
        SwitcherState::Switching { from, to } => {
            format!(
                "switching:{}->{}",
                from.as_deref().unwrap_or("none"),
                to
            )
        }
    };

    let mut models = HashMap::new();
    for model_name in switcher.registered_models() {
        let in_flight = switcher.in_flight_count(&model_name);
        let queue_depth = queue_depths.get(&model_name).copied().unwrap_or(0);
        let process_state = switcher
            .orchestrator()
            .process_state(&model_name)
            .await
            .map(format_process_state)
            .unwrap_or_else(|| "unknown".to_string());

        models.insert(
            model_name,
            ModelStatus {
                in_flight,
                queue_depth,
                process_state,
            },
        );
    }

    Json(StatusResponse {
        state: state_str,
        active_model,
        mode,
        models,
    })
}

async fn set_mode(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<SetModeRequest>,
) -> impl IntoResponse {
    match body.mode.as_str() {
        "auto" => {
            switcher.set_mode(SwitchMode::Auto).await;
            (
                StatusCode::OK,
                Json(MessageResponse {
                    message: "Switched to auto mode".to_string(),
                }),
            )
        }
        "manual" => {
            switcher
                .set_mode(SwitchMode::Manual { pinned: None })
                .await;
            (
                StatusCode::OK,
                Json(MessageResponse {
                    message: "Switched to manual mode".to_string(),
                }),
            )
        }
        other => (
            StatusCode::BAD_REQUEST,
            Json(MessageResponse {
                message: format!("Unknown mode: {}. Use 'auto' or 'manual'.", other),
            }),
        ),
    }
}

async fn pin_model(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<PinRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    // Remember the previous mode so we can rollback on failure
    let prev_mode = switcher.mode().await;

    // Enter manual mode with the pinned model
    switcher
        .set_mode(SwitchMode::Manual {
            pinned: Some(body.model.clone()),
        })
        .await;

    // If the pinned model isn't already active, force-switch to it
    let active = switcher.active_model().await;
    if active.as_deref() != Some(&body.model) {
        if let Err(e) = switcher.force_switch(&body.model).await {
            // Rollback to previous mode so the switcher isn't stuck in a
            // broken manual state with a pin that was never activated.
            switcher.set_mode(prev_mode).await;
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to switch to pinned model: {}", e),
                }),
            ));
        }
    }

    Ok(Json(MessageResponse {
        message: format!("Pinned to model: {}", body.model),
    }))
}

async fn unpin_model(State(switcher): State<ModelSwitcher>) -> impl IntoResponse {
    switcher.set_mode(SwitchMode::Auto).await;
    Json(MessageResponse {
        message: "Unpinned. Switched to auto mode.".to_string(),
    })
}

async fn force_switch(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<SwitchRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    // If in manual mode, update the pin to the new model
    let mode = switcher.mode().await;
    if let SwitchMode::Manual { .. } = mode {
        switcher
            .set_mode(SwitchMode::Manual {
                pinned: Some(body.model.clone()),
            })
            .await;
    }

    if let Err(e) = switcher.force_switch(&body.model).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Switch failed: {}", e),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: format!("Switched to model: {}", body.model),
    }))
}

async fn get_eviction_policies(State(switcher): State<ModelSwitcher>) -> impl IntoResponse {
    let orch = switcher.orchestrator();
    let policy_default = switcher.policy_eviction();
    let mut models = HashMap::new();

    for model_name in switcher.registered_models() {
        let eviction = orch
            .eviction_policy_for(&model_name)
            .unwrap_or(policy_default);
        let process_state = orch
            .process_state(&model_name)
            .await
            .map(format_process_state)
            .unwrap_or_else(|| "unknown".to_string());

        models.insert(
            model_name,
            EvictionPolicyInfo {
                eviction,
                process_state,
            },
        );
    }

    Json(EvictionPoliciesResponse { models })
}

async fn set_eviction_policy(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<SetEvictionRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    switcher
        .orchestrator()
        .set_eviction_policy(&body.model, body.eviction);

    Ok(Json(MessageResponse {
        message: format!(
            "Eviction policy for {} set to {:?}",
            body.model, body.eviction
        ),
    }))
}

async fn sleep_model(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<SleepRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    let eviction = body.eviction.unwrap_or_else(|| {
        switcher
            .orchestrator()
            .eviction_policy_for(&body.model)
            .unwrap_or(switcher.policy_eviction())
    });

    let orch = switcher.orchestrator();

    if let Err(e) = orch.sleep_model(&body.model, eviction).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to sleep model: {}", e),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: format!("Model {} sleeping with policy {:?}", body.model, eviction),
    }))
}

async fn wake_model(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<WakeRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    let orch = switcher.orchestrator();

    if let Err(e) = orch.wake_model(&body.model).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Failed to wake model: {}", e),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: format!("Model {} woken", body.model),
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_process_state(state: ProcessState) -> String {
    match state {
        ProcessState::NotStarted => "not_started".to_string(),
        ProcessState::Starting => "starting".to_string(),
        ProcessState::Running { sleeping: None } => "running".to_string(),
        ProcessState::Running {
            sleeping: Some(eviction),
        } => format!(
            "sleeping:{:?}+{:?}",
            eviction.weights, eviction.process
        ),
        ProcessState::Failed { reason } => format!("failed:{}", reason),
        ProcessState::Checkpointed { .. } => "checkpointed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::orchestrator::Orchestrator;
    use crate::policy::FifoPolicy;
    use axum::body::Body;
    use http::Request;
    use tower::ServiceExt;

    fn make_test_switcher() -> ModelSwitcher {
        use crate::switcher::EvictionPolicy;
        let mut configs = std::collections::HashMap::new();
        configs.insert(
            "model-a".to_string(),
            ModelConfig {
                model_path: "test".to_string(),
                port: 8001,
                extra_args: vec![],
                eviction: EvictionPolicy::from(1),
            },
        );
        configs.insert(
            "model-b".to_string(),
            ModelConfig {
                model_path: "test".to_string(),
                port: 8002,
                extra_args: vec![],
                eviction: EvictionPolicy::from(1),
            },
        );
        let orchestrator = std::sync::Arc::new(Orchestrator::new(configs));
        let policy = Box::new(FifoPolicy::default());
        ModelSwitcher::new(orchestrator, policy)
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/control/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["state"], "idle");
        assert_eq!(json["mode"]["mode"], "auto");
        assert!(json["models"].is_object());
    }

    #[tokio::test]
    async fn test_set_mode_auto() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/mode")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"mode":"auto"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_set_mode_manual() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/mode")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"mode":"manual"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_set_mode_invalid() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/mode")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"mode":"invalid"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_pin_unknown_model() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/pin")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model":"nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_unpin_returns_auto() {
        let switcher = make_test_switcher();

        // Set manual mode first
        switcher
            .set_mode(SwitchMode::Manual {
                pinned: Some("model-a".to_string()),
            })
            .await;

        let app = control_router(switcher.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/unpin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(switcher.mode().await, SwitchMode::Auto);
    }

    #[tokio::test]
    async fn test_switch_unknown_model() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/switch")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model":"nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_eviction_policies() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/control/eviction")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(json["models"]["model-a"]["eviction"].is_object());
        assert!(json["models"]["model-b"]["eviction"].is_object());
    }

    #[tokio::test]
    async fn test_set_eviction_policy() {
        let switcher = make_test_switcher();
        let app = control_router(switcher.clone());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/eviction")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"model":"model-a","eviction":{"weights":"retain","process":"cuda_suspend"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Verify the override took effect
        let policy = switcher
            .orchestrator()
            .eviction_policy_for("model-a")
            .unwrap();
        assert_eq!(policy, EvictionPolicy::from(3)); // Retain + CudaSuspend
    }

    #[tokio::test]
    async fn test_set_eviction_policy_unknown_model() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/eviction")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"model":"nonexistent","eviction":{"weights":"discard","process":"stop"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_sleep_unknown_model() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/sleep")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model":"nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_wake_unknown_model() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/control/wake")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model":"nonexistent"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
