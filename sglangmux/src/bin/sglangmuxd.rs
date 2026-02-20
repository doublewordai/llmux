use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde_json::{Value, json};
use sglangmux::{SgLangMux, SgLangMuxOptions};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct AppState {
    mux: SgLangMux,
    model_ports: HashMap<String, u16>,
    client: Client,
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("sglangmux=debug,sglangmuxd=debug,warn"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .try_init();
}

fn usage() -> ! {
    eprintln!(
        "Usage: sglangmuxd [--listen-port PORT] [--log-dir DIR] <script1.sh> <script2.sh> ..."
    );
    std::process::exit(2);
}

fn parse_args() -> (u16, PathBuf, Vec<PathBuf>) {
    let mut listen_port: u16 = 30100;
    let mut log_dir = PathBuf::from("sglangmux-logs");
    let mut scripts = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen-port" => {
                let Some(value) = args.next() else {
                    usage();
                };
                let Ok(port) = value.parse::<u16>() else {
                    usage();
                };
                listen_port = port;
            }
            "--log-dir" => {
                let Some(value) = args.next() else {
                    usage();
                };
                log_dir = PathBuf::from(value);
            }
            "-h" | "--help" => usage(),
            _ => scripts.push(PathBuf::from(arg)),
        }
    }

    if scripts.is_empty() {
        usage();
    }

    (listen_port, log_dir, scripts)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let active_model = state.mux.active_model().await;
    debug!(active_model = ?active_model, "Health check served");
    Json(json!({
        "status": "ok",
        "active_model": active_model
    }))
}

async fn proxy_generation_request(
    state: &AppState,
    body: Value,
    upstream_path: &str,
) -> axum::response::Response {
    let request_id = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);

    let Some(model) = body.get("model").and_then(Value::as_str) else {
        warn!(
            request_id,
            upstream_path, "Rejecting request without required string field `model`"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing string field: model"})),
        )
            .into_response();
    };

    let message_count = body
        .get("messages")
        .and_then(Value::as_array)
        .map(std::vec::Vec::len);
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    info!(
        request_id,
        model = %model,
        upstream_path,
        message_count = ?message_count,
        stream,
        "Incoming generation request"
    );

    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        debug!(
            request_id,
            model = %model,
            attempt,
            "Ensuring requested model is ready"
        );
        if let Err(error) = state.mux.ensure_model_ready(model).await {
            error!(
                request_id,
                model = %model,
                attempt,
                error = %error,
                "Failed to ensure requested model is ready"
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": format!("model not ready: {}", error)})),
            )
                .into_response();
        }

        let Some(_guard) = state.mux.acquire_in_flight(model) else {
            // Active model started draining between readiness check and guard acquire.
            // Re-enter readiness flow, same pattern used by llmux middleware.
            debug!(
                request_id,
                model = %model,
                attempt,
                "Model started draining between readiness and lock acquisition; retrying"
            );
            continue;
        };
        debug!(
            request_id,
            model = %model,
            attempt,
            "Acquired in-flight lock for request"
        );

        let Some(port) = state.model_ports.get(model).copied() else {
            error!(
                request_id,
                model = %model,
                "Model was not found in daemon port map"
            );
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("unknown model: {}", model)})),
            )
                .into_response();
        };

        let url = format!("http://127.0.0.1:{port}{upstream_path}");
        debug!(
            request_id,
            model = %model,
            attempt,
            port,
            url,
            "Forwarding request to upstream model server"
        );
        let upstream_start = Instant::now();
        let upstream = tokio::time::timeout(
            Duration::from_secs(120),
            state.client.post(&url).json(&body).send(),
        )
        .await;

        return match upstream {
            Ok(Ok(response)) => {
                let status = response.status();
                debug!(
                    request_id,
                    model = %model,
                    attempt,
                    status = %status,
                    latency_ms = upstream_start.elapsed().as_millis(),
                    "Upstream responded"
                );
                match response.json::<Value>().await {
                    Ok(payload) => {
                        info!(
                            request_id,
                            model = %model,
                            attempt,
                            status = %status,
                            "Request completed successfully through mux"
                        );
                        (status, Json(payload)).into_response()
                    }
                    Err(error) => {
                        error!(
                            request_id,
                            model = %model,
                            attempt,
                            error = %error,
                            "Upstream returned non-JSON payload"
                        );
                        (
                            StatusCode::BAD_GATEWAY,
                            Json(json!({"error": format!("invalid upstream response: {}", error)})),
                        )
                            .into_response()
                    }
                }
            }
            Ok(Err(error)) => {
                error!(
                    request_id,
                    model = %model,
                    attempt,
                    error = %error,
                    latency_ms = upstream_start.elapsed().as_millis(),
                    "Upstream request failed"
                );
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("upstream request failed: {}", error)})),
                )
                    .into_response()
            }
            Err(_) => {
                error!(
                    request_id,
                    model = %model,
                    attempt,
                    timeout_s = 120,
                    "Upstream request timed out"
                );
                (
                    StatusCode::GATEWAY_TIMEOUT,
                    Json(json!({"error": "upstream request timed out"})),
                )
                    .into_response()
            }
        };
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    debug!("Received /v1/chat/completions request");
    proxy_generation_request(&state, body, "/v1/chat/completions").await
}

async fn completions(State(state): State<AppState>, Json(body): Json<Value>) -> impl IntoResponse {
    debug!("Received /v1/completions request");
    proxy_generation_request(&state, body, "/v1/completions").await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let (listen_port, log_dir, scripts) = parse_args();
    info!(
        listen_port,
        log_dir = %log_dir.display(),
        script_count = scripts.len(),
        "Starting sglangmuxd"
    );

    let options = SgLangMuxOptions {
        log_dir,
        ..SgLangMuxOptions::default()
    };

    let mux = SgLangMux::from_scripts(scripts, options)?;
    debug!(models = ?mux.model_specs(), "Parsed model specs");
    mux.bootstrap_sequential().await?;
    info!(active_model = ?mux.active_model().await, "Bootstrap complete");

    let model_ports: HashMap<String, u16> = mux
        .model_specs()
        .into_iter()
        .map(|spec| (spec.model_name, spec.port))
        .collect();
    info!(model_ports = ?model_ports, "Built model port map");

    let state = AppState {
        mux: mux.clone(),
        model_ports,
        client: Client::new(),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", listen_port)).await?;
    info!("sglangmuxd listening on http://127.0.0.1:{listen_port}");
    axum::serve(listener, app).await?;

    Ok(())
}
