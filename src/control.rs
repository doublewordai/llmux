//! Control API for model management.
//!
//! Provides HTTP endpoints for inspecting and controlling the model switcher.
//! Intended to run on a separate admin port.
//!
//! ## Endpoints
//!
//! | Method | Path                  | Description                              |
//! |--------|-----------------------|------------------------------------------|
//! | GET    | `/control/status`     | Current state, in-flight, queues         |
//! | POST   | `/control/switch`     | Force switch to a model                  |
//! | POST   | `/control/sleep`      | Run sleep hook for a model               |
//! | POST   | `/control/wake`       | Run wake hook for a model                |
//! | GET    | `/control/alive`      | Run alive hook for a model               |

use crate::switcher::ModelSwitcher;
use crate::types::SwitcherState;
use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub fn control_router(switcher: ModelSwitcher) -> Router {
    Router::new()
        .route("/control/status", get(get_status))
        .route("/control/switch", post(force_switch))
        .route("/control/sleep", post(sleep_model))
        .route("/control/wake", post(wake_model))
        .route("/control/alive", get(check_alive))
        .with_state(switcher)
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StatusResponse {
    state: String,
    active_model: Option<String>,
    models: HashMap<String, ModelStatus>,
}

#[derive(Serialize)]
struct ModelStatus {
    in_flight: usize,
    queue_depth: usize,
}

#[derive(Deserialize)]
struct ModelRequest {
    model: String,
}

#[derive(Deserialize)]
struct AliveQuery {
    model: String,
}

#[derive(Serialize)]
struct MessageResponse {
    message: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct AliveResponse {
    model: String,
    alive: bool,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn get_status(State(switcher): State<ModelSwitcher>) -> impl IntoResponse {
    let state = switcher.state().await;
    let queue_depths = switcher.queue_depths().await;

    let active_model = match &state {
        SwitcherState::Active { model } => Some(model.clone()),
        _ => None,
    };

    let state_str = match &state {
        SwitcherState::Idle => "idle".to_string(),
        SwitcherState::Active { model } => format!("active:{}", model),
        SwitcherState::Switching { from, to } => {
            format!("switching:{}->{}", from.as_deref().unwrap_or("none"), to)
        }
    };

    let mut models = HashMap::new();
    for model_name in switcher.registered_models() {
        let in_flight = switcher.in_flight_count(&model_name);
        let queue_depth = queue_depths.get(&model_name).copied().unwrap_or(0);

        models.insert(
            model_name,
            ModelStatus {
                in_flight,
                queue_depth,
            },
        );
    }

    Json(StatusResponse {
        state: state_str,
        active_model,
        models,
    })
}

async fn force_switch(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<ModelRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
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

async fn sleep_model(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<ModelRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    if let Err(e) = switcher.hooks().run_sleep(&body.model).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Sleep hook failed: {}", e),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: format!("Model {} sleeping", body.model),
    }))
}

async fn wake_model(
    State(switcher): State<ModelSwitcher>,
    Json(body): Json<ModelRequest>,
) -> Result<Json<MessageResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&body.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", body.model),
            }),
        ));
    }

    if let Err(e) = switcher.hooks().run_wake(&body.model).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Wake hook failed: {}", e),
            }),
        ));
    }

    Ok(Json(MessageResponse {
        message: format!("Model {} woken", body.model),
    }))
}

async fn check_alive(
    State(switcher): State<ModelSwitcher>,
    Query(query): Query<AliveQuery>,
) -> Result<Json<AliveResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !switcher.is_registered(&query.model) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model not found: {}", query.model),
            }),
        ));
    }

    match switcher.hooks().run_alive(&query.model).await {
        Ok(alive) => Ok(Json(AliveResponse {
            model: query.model,
            alive,
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Alive hook error: {}", e),
            }),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::hooks::HookRunner;
    use crate::policy::FifoPolicy;
    use axum::body::Body;
    use http::Request;
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn true_path() -> PathBuf {
        ["/usr/bin/true", "/bin/true"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .expect("cannot find `true` binary")
    }

    fn make_test_switcher() -> ModelSwitcher {
        let t = true_path();
        let mut configs = std::collections::HashMap::new();
        configs.insert(
            "model-a".to_string(),
            ModelConfig {
                port: 8001,
                wake: t.clone(),
                sleep: t.clone(),
                alive: t.clone(),
            },
        );
        configs.insert(
            "model-b".to_string(),
            ModelConfig {
                port: 8002,
                wake: t.clone(),
                sleep: t.clone(),
                alive: t.clone(),
            },
        );
        let hooks = std::sync::Arc::new(HookRunner::new(configs));
        let policy = Box::new(FifoPolicy::default());
        ModelSwitcher::new(hooks, policy)
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
        assert!(json["models"].is_object());
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
    async fn test_alive_check() {
        let switcher = make_test_switcher();
        let app = control_router(switcher);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/control/alive?model=model-a")
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

        assert_eq!(json["model"], "model-a");
        assert_eq!(json["alive"], true);
    }
}
