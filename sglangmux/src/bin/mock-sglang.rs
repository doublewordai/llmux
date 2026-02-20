use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[derive(Clone)]
struct MockState {
    model: String,
    sleeping: Arc<RwLock<bool>>,
    weights_loaded: Arc<RwLock<bool>>,
    pause_count: Arc<RwLock<u64>>,
    continue_count: Arc<RwLock<u64>>,
    release_count: Arc<RwLock<u64>>,
    resume_count: Arc<RwLock<u64>>,
    update_count: Arc<RwLock<u64>>,
    request_count: Arc<RwLock<u64>>,
    startup_unix_ms: u128,
    last_release_unix_ms: Arc<RwLock<u128>>,
}

#[derive(Debug)]
struct Args {
    model: String,
    port: u16,
    startup_delay_ms: u64,
}

#[derive(Debug, Deserialize)]
struct ActionRequest {
    #[serde(default)]
    tags: Vec<String>,
    sleep_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    #[allow(dead_code)]
    messages: Vec<ChatMessage>,
    sleep_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    #[allow(dead_code)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    #[allow(dead_code)]
    text: String,
    sleep_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct UpdateWeightsRequest {
    model_path: String,
    sleep_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    model: String,
    sleeping: bool,
    weights_loaded: bool,
    pause_count: u64,
    continue_count: u64,
    release_count: u64,
    resume_count: u64,
    update_count: u64,
    request_count: u64,
    startup_unix_ms: u128,
    last_release_unix_ms: u128,
}

fn parse_args() -> Args {
    let mut model: Option<String> = None;
    let mut port: Option<u16> = None;
    let mut startup_delay_ms = 0_u64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = args.next();
            }
            "--port" => {
                port = args.next().and_then(|p| p.parse::<u16>().ok());
            }
            "--startup-delay-ms" => {
                startup_delay_ms = args.next().and_then(|d| d.parse::<u64>().ok()).unwrap_or(0);
            }
            _ => {}
        }
    }

    Args {
        model: model.unwrap_or_else(|| "mock-model".to_string()),
        port: port.unwrap_or(8000),
        startup_delay_ms,
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

async fn maybe_delay(delay_ms: Option<u64>) {
    if let Some(delay) = delay_ms
        && delay > 0
    {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn model_info(State(state): State<MockState>) -> impl IntoResponse {
    Json(json!({
        "model_path": state.model,
        "is_generation": true
    }))
}

async fn pause_generation(
    State(state): State<MockState>,
    Json(body): Json<ActionRequest>,
) -> impl IntoResponse {
    maybe_delay(body.sleep_ms).await;

    let mut pause_count = state.pause_count.write().await;
    *pause_count += 1;

    Json(json!({"ok": true, "paused": *pause_count}))
}

async fn continue_generation(
    State(state): State<MockState>,
    Json(body): Json<ActionRequest>,
) -> impl IntoResponse {
    maybe_delay(body.sleep_ms).await;

    let mut continue_count = state.continue_count.write().await;
    *continue_count += 1;

    Json(json!({"ok": true, "continued": *continue_count}))
}

fn tags_valid(tags: &[String]) -> bool {
    tags.contains(&"kv_cache".to_string()) && tags.contains(&"weights".to_string())
}

async fn release_memory_occupation(
    State(state): State<MockState>,
    Json(body): Json<ActionRequest>,
) -> impl IntoResponse {
    if !tags_valid(&body.tags) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "tags must include kv_cache and weights"})),
        )
            .into_response();
    }

    maybe_delay(body.sleep_ms).await;

    {
        let mut sleeping = state.sleeping.write().await;
        *sleeping = true;
    }

    {
        let mut loaded = state.weights_loaded.write().await;
        *loaded = false;
    }

    {
        let mut release_count = state.release_count.write().await;
        *release_count += 1;
    }

    {
        let mut ts = state.last_release_unix_ms.write().await;
        *ts = now_unix_ms();
    }

    (StatusCode::OK, Json(json!({"ok": true}))).into_response()
}

async fn resume_memory_occupation(
    State(state): State<MockState>,
    Json(body): Json<ActionRequest>,
) -> impl IntoResponse {
    if !tags_valid(&body.tags) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "tags must include kv_cache and weights"})),
        )
            .into_response();
    }

    maybe_delay(body.sleep_ms).await;

    {
        let mut sleeping = state.sleeping.write().await;
        *sleeping = false;
    }

    {
        let mut resume_count = state.resume_count.write().await;
        *resume_count += 1;
    }

    (StatusCode::OK, Json(json!({"ok": true}))).into_response()
}

async fn update_weights_from_disk(
    State(state): State<MockState>,
    Json(body): Json<UpdateWeightsRequest>,
) -> impl IntoResponse {
    if body.model_path != state.model {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "model path mismatch"})),
        )
            .into_response();
    }

    if *state.sleeping.read().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model still sleeping"})),
        )
            .into_response();
    }

    maybe_delay(body.sleep_ms).await;

    {
        let mut loaded = state.weights_loaded.write().await;
        *loaded = true;
    }

    {
        let mut update_count = state.update_count.write().await;
        *update_count += 1;
    }

    (StatusCode::OK, Json(json!({"ok": true}))).into_response()
}

async fn chat_completions(
    State(state): State<MockState>,
    Query(query): Query<HashMap<String, String>>,
    Json(body): Json<ChatRequest>,
) -> impl IntoResponse {
    if body.model != state.model {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "model mismatch"})),
        )
            .into_response();
    }

    if *state.sleeping.read().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model sleeping"})),
        )
            .into_response();
    }

    if !*state.weights_loaded.read().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "weights not loaded"})),
        )
            .into_response();
    }

    let query_delay = query.get("sleep_ms").and_then(|v| v.parse::<u64>().ok());
    maybe_delay(body.sleep_ms.or(query_delay)).await;

    let mut request_count = state.request_count.write().await;
    *request_count += 1;

    (
        StatusCode::OK,
        Json(json!({
            "id": "cmpl-mock",
            "object": "chat.completion",
            "model": state.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]
        })),
    )
        .into_response()
}

async fn stats(State(state): State<MockState>) -> impl IntoResponse {
    let response = StatsResponse {
        model: state.model.clone(),
        sleeping: *state.sleeping.read().await,
        weights_loaded: *state.weights_loaded.read().await,
        pause_count: *state.pause_count.read().await,
        continue_count: *state.continue_count.read().await,
        release_count: *state.release_count.read().await,
        resume_count: *state.resume_count.read().await,
        update_count: *state.update_count.read().await,
        request_count: *state.request_count.read().await,
        startup_unix_ms: state.startup_unix_ms,
        last_release_unix_ms: *state.last_release_unix_ms.read().await,
    };

    Json(response)
}

async fn generate(
    State(state): State<MockState>,
    Json(body): Json<GenerateRequest>,
) -> impl IntoResponse {
    if *state.sleeping.read().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "model sleeping"})),
        )
            .into_response();
    }

    if !*state.weights_loaded.read().await {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "weights not loaded"})),
        )
            .into_response();
    }

    maybe_delay(body.sleep_ms).await;

    let mut request_count = state.request_count.write().await;
    *request_count += 1;

    (StatusCode::OK, Json(json!({"text": "ok"}))).into_response()
}

#[tokio::main]
async fn main() {
    let args = parse_args();

    if args.startup_delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(args.startup_delay_ms)).await;
    }

    let state = MockState {
        model: args.model,
        sleeping: Arc::new(RwLock::new(false)),
        weights_loaded: Arc::new(RwLock::new(true)),
        pause_count: Arc::new(RwLock::new(0)),
        continue_count: Arc::new(RwLock::new(0)),
        release_count: Arc::new(RwLock::new(0)),
        resume_count: Arc::new(RwLock::new(0)),
        update_count: Arc::new(RwLock::new(0)),
        request_count: Arc::new(RwLock::new(0)),
        startup_unix_ms: now_unix_ms(),
        last_release_unix_ms: Arc::new(RwLock::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/model_info", get(model_info))
        .route("/pause_generation", post(pause_generation))
        .route("/continue_generation", post(continue_generation))
        .route(
            "/release_memory_occupation",
            post(release_memory_occupation),
        )
        .route("/resume_memory_occupation", post(resume_memory_occupation))
        .route("/update_weights_from_disk", post(update_weights_from_disk))
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/stats", get(stats))
        .with_state(state);

    let addr: SocketAddr = format!("127.0.0.1:{}", args.port)
        .parse()
        .expect("invalid listen address");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind listener");

    println!("READY {}", addr.port());

    axum::serve(listener, app)
        .await
        .expect("mock sglang server crashed");
}
