//! Mock vLLM server for testing llmux
//!
//! Supports two modes:
//! 1. Direct: `mock-vllm --port 8001 --model test-model`
//! 2. vLLM-compatible: `mock-vllm serve model-name --port 8001 --gpu-memory-utilization 0.9 ...`
//!
//! Simulates vLLM sleep mode API endpoints for integration testing.

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "mock-vllm")]
#[command(about = "Mock vLLM server for testing")]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Port to listen on (direct mode)
    #[arg(short, long, default_value = "8001", global = true)]
    port: u16,

    /// Model name to serve (direct mode)
    #[arg(short, long, default_value = "test-model")]
    model: String,

    /// Artificial latency for responses (ms)
    #[arg(long, default_value = "50", global = true)]
    latency_ms: u64,

    /// Artificial startup delay (ms)
    #[arg(long, default_value = "0", global = true)]
    startup_delay_ms: u64,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// vLLM-compatible serve mode.
    /// Accepts and ignores vLLM-specific flags so it can stand in for real vLLM.
    Serve {
        /// Model to serve (positional, like vllm)
        model: String,

        /// Port (vLLM-style)
        #[arg(long)]
        port: Option<u16>,

        /// GPU memory utilization (ignored in mock, accepted for compatibility)
        #[arg(long, default_value = "0.9")]
        gpu_memory_utilization: f32,

        /// Tensor parallel size (ignored in mock)
        #[arg(long, default_value = "1")]
        tensor_parallel_size: usize,

        /// Data type (ignored in mock)
        #[arg(long, default_value = "auto")]
        dtype: String,

        /// Enable sleep mode (ignored in mock, always enabled)
        #[arg(long)]
        enable_sleep_mode: bool,

        /// Enable LoRA mode (ignored in mock, accepted for compatibility)
        #[arg(long)]
        enable_lora: bool,

        /// Max model length (ignored in mock)
        #[arg(long)]
        max_model_len: Option<usize>,
    },
}

/// Server state
#[derive(Debug)]
struct MockState {
    model: String,
    sleeping: RwLock<bool>,
    sleep_level: RwLock<u8>,
    latency: RwLock<Duration>,
    request_count: RwLock<u64>,
    /// When true, /sleep returns 500 (for testing L3 fallback)
    fail_sleep: RwLock<bool>,
    /// When true, /wake_up returns 500 (for testing wake failure cleanup)
    fail_wake: RwLock<bool>,
    /// Artificial sleep delay in milliseconds (for testing timeouts)
    sleep_delay_ms: RwLock<u64>,
    /// Loaded LoRA adapters keyed by adapter name.
    loaded_loras: RwLock<HashMap<String, String>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("mock_vllm=debug,tower_http=debug")
        .init();

    let args = Args::parse();

    // Determine model and port based on mode
    let (model, port) = match args.command {
        Some(Commands::Serve {
            model,
            port: serve_port,
            ..
        }) => {
            // vLLM-compatible mode: use serve subcommand's model and port
            let port = serve_port.unwrap_or(args.port);
            (model, port)
        }
        None => {
            // Direct mode: use top-level args
            (args.model, args.port)
        }
    };

    // Simulate startup delay
    if args.startup_delay_ms > 0 {
        info!(delay_ms = args.startup_delay_ms, "Simulating startup delay");
        tokio::time::sleep(Duration::from_millis(args.startup_delay_ms)).await;
    }

    let state = Arc::new(MockState {
        model: model.clone(),
        sleeping: RwLock::new(false),
        sleep_level: RwLock::new(0),
        latency: RwLock::new(Duration::from_millis(args.latency_ms)),
        request_count: RwLock::new(0),
        fail_sleep: RwLock::new(false),
        fail_wake: RwLock::new(false),
        sleep_delay_ms: RwLock::new(0),
        loaded_loras: RwLock::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/sleep", post(sleep))
        .route("/wake_up", post(wake_up))
        .route("/collective_rpc", post(collective_rpc))
        .route("/reset_prefix_cache", post(reset_prefix_cache))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/load_lora_adapter", post(load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(unload_lora_adapter))
        .route("/v1/models", get(list_models))
        .route("/stats", get(stats))
        .route("/control/fail-sleep", post(control_fail_sleep))
        .route("/control/fail-wake", post(control_fail_wake))
        .route("/control/sleep-delay", post(control_sleep_delay))
        .route("/control/latency", post(control_latency))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;

    // Get the actual port (important when port=0 for dynamic allocation)
    let actual_port = listener.local_addr()?.port();

    info!(
        model = %model,
        port = actual_port,
        "Mock vLLM server listening"
    );

    // Signal readiness to stdout for test harness
    // Format: "READY <port>" on its own line
    println!("READY {}", actual_port);

    axum::serve(listener, app).await?;
    Ok(())
}

/// Health check endpoint
async fn health(State(state): State<Arc<MockState>>) -> impl IntoResponse {
    let sleeping = *state.sleeping.read().await;
    if sleeping {
        info!("Health check: sleeping");
        // vLLM still returns healthy when sleeping
    }
    StatusCode::OK
}

#[derive(Deserialize)]
struct SleepQuery {
    level: Option<u8>,
}

/// Sleep endpoint - PUT model to sleep
async fn sleep(
    State(state): State<Arc<MockState>>,
    Query(query): Query<SleepQuery>,
) -> impl IntoResponse {
    let level = query.level.unwrap_or(1);
    info!(level = level, "Putting model to sleep");

    // Check if sleep should fail (for testing L3 fallback)
    if *state.fail_sleep.read().await {
        warn!("Sleep forced to fail via /control/fail-sleep");
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    // Apply artificial sleep delay (for testing timeouts)
    let delay = *state.sleep_delay_ms.read().await;
    if delay > 0 {
        info!(delay_ms = delay, "Applying artificial sleep delay");
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }

    *state.sleeping.write().await = true;
    *state.sleep_level.write().await = level;

    StatusCode::OK
}

/// Wake up endpoint
async fn wake_up(State(state): State<Arc<MockState>>) -> impl IntoResponse {
    // Check if wake should fail (for testing wake failure cleanup)
    if *state.fail_wake.read().await {
        warn!("Wake forced to fail via /control/fail-wake");
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    info!("Waking up model");
    *state.sleeping.write().await = false;
    StatusCode::OK
}

#[derive(Deserialize)]
struct CollectiveRpcRequest {
    method: String,
}

/// Collective RPC endpoint (for weight reloading)
async fn collective_rpc(
    State(_state): State<Arc<MockState>>,
    Json(request): Json<CollectiveRpcRequest>,
) -> impl IntoResponse {
    info!(method = %request.method, "Collective RPC call");

    if request.method == "reload_weights" {
        // Simulate weight reload time
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    StatusCode::OK
}

/// Reset prefix cache endpoint
async fn reset_prefix_cache() -> impl IntoResponse {
    info!("Resetting prefix cache");
    StatusCode::OK
}

#[derive(Deserialize)]
struct LoadLoraAdapterRequest {
    lora_name: String,
    lora_path: String,
    #[serde(default)]
    #[allow(dead_code)]
    base_model_name: Option<String>,
    #[serde(default)]
    load_inplace: bool,
}

async fn load_lora_adapter(
    State(state): State<Arc<MockState>>,
    Json(request): Json<LoadLoraAdapterRequest>,
) -> impl IntoResponse {
    if request.lora_path.starts_with("fail:") {
        return (StatusCode::INTERNAL_SERVER_ERROR, "forced load failure").into_response();
    }

    let mut loaded = state.loaded_loras.write().await;
    if loaded.contains_key(&request.lora_name) && !request.load_inplace {
        return (
            StatusCode::BAD_REQUEST,
            "adapter already loaded (set load_inplace=true to replace)",
        )
            .into_response();
    }
    loaded.insert(request.lora_name.clone(), request.lora_path.clone());
    StatusCode::OK.into_response()
}

#[derive(Deserialize)]
struct UnloadLoraAdapterRequest {
    lora_name: String,
}

async fn unload_lora_adapter(
    State(state): State<Arc<MockState>>,
    Json(request): Json<UnloadLoraAdapterRequest>,
) -> impl IntoResponse {
    let mut loaded = state.loaded_loras.write().await;
    if loaded.remove(&request.lora_name).is_none() {
        return (StatusCode::NOT_FOUND, "adapter not loaded").into_response();
    }
    StatusCode::OK.into_response()
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_max_tokens")]
    #[allow(dead_code)] // Parsed but not used in mock response
    max_tokens: u32,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
}

fn default_max_tokens() -> u32 {
    100
}

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: Message,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Chat completions endpoint
async fn chat_completions(
    State(state): State<Arc<MockState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Check if sleeping
    if *state.sleeping.read().await {
        warn!(model = %request.model, "Request received while model is sleeping");
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Model is sleeping".to_string(),
        ));
    }

    // Accept either base model name or a loaded LoRA adapter alias.
    if request.model != state.model {
        let loaded = state.loaded_loras.read().await;
        if !loaded.contains_key(&request.model) {
            return Err((
                StatusCode::NOT_FOUND,
                format!("LoRA adapter '{}' is not loaded", request.model),
            ));
        }
    }

    // Simulate latency
    tokio::time::sleep(*state.latency.read().await).await;

    // Increment request count
    {
        let mut count = state.request_count.write().await;
        *count += 1;
    }

    let count = *state.request_count.read().await;

    info!(
        model = %request.model,
        messages = request.messages.len(),
        stream = request.stream,
        request_num = count,
        "Processing chat completion"
    );

    if request.stream {
        // For now, return non-streaming response even if stream requested
        // A full implementation would return SSE
        warn!("Streaming requested but returning non-streaming response");
    }

    // Generate mock response
    // Deterministic mode: when temperature == 0.0 and seed is set, always return "4"
    let deterministic = request.temperature == Some(0.0) && request.seed.is_some();
    let response_content = if deterministic {
        "4".to_string()
    } else {
        format!(
            "Mock response from {} (request #{}): You said \"{}\"",
            state.model,
            count,
            request
                .messages
                .last()
                .map(|m| m.content.as_str())
                .unwrap_or("")
        )
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-mock-{}", count),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: request.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    };

    Ok(Json(response))
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

/// List models endpoint
async fn list_models(State(state): State<Arc<MockState>>) -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model.clone(),
            object: "model".to_string(),
            owned_by: "mock-vllm".to_string(),
        }],
    };

    Json(response)
}

#[derive(Serialize)]
struct StatsResponse {
    model: String,
    sleeping: bool,
    sleep_level: u8,
    request_count: u64,
    loaded_loras: Vec<String>,
}

/// Stats endpoint for testing inspection
async fn stats(State(state): State<Arc<MockState>>) -> impl IntoResponse {
    let mut loaded_loras: Vec<String> = state.loaded_loras.read().await.keys().cloned().collect();
    loaded_loras.sort();
    let response = StatsResponse {
        model: state.model.clone(),
        sleeping: *state.sleeping.read().await,
        sleep_level: *state.sleep_level.read().await,
        request_count: *state.request_count.read().await,
        loaded_loras,
    };

    Json(response)
}

#[derive(Deserialize)]
struct ControlFailSleep {
    enabled: bool,
}

/// Control endpoint: make /sleep return 500
async fn control_fail_sleep(
    State(state): State<Arc<MockState>>,
    Json(body): Json<ControlFailSleep>,
) -> impl IntoResponse {
    info!(enabled = body.enabled, "Setting fail_sleep");
    *state.fail_sleep.write().await = body.enabled;
    StatusCode::OK
}

#[derive(Deserialize)]
struct ControlSleepDelay {
    delay_ms: u64,
}

#[derive(Deserialize)]
struct ControlFailWake {
    enabled: bool,
}

/// Control endpoint: make /wake_up return 500
async fn control_fail_wake(
    State(state): State<Arc<MockState>>,
    Json(body): Json<ControlFailWake>,
) -> impl IntoResponse {
    info!(enabled = body.enabled, "Setting fail_wake");
    *state.fail_wake.write().await = body.enabled;
    StatusCode::OK
}

/// Control endpoint: set artificial sleep delay
async fn control_sleep_delay(
    State(state): State<Arc<MockState>>,
    Json(body): Json<ControlSleepDelay>,
) -> impl IntoResponse {
    info!(delay_ms = body.delay_ms, "Setting sleep_delay_ms");
    *state.sleep_delay_ms.write().await = body.delay_ms;
    StatusCode::OK
}

#[derive(Deserialize)]
struct ControlLatency {
    latency_ms: u64,
}

/// Control endpoint: set request latency
async fn control_latency(
    State(state): State<Arc<MockState>>,
    Json(body): Json<ControlLatency>,
) -> impl IntoResponse {
    info!(latency_ms = body.latency_ms, "Setting latency");
    *state.latency.write().await = Duration::from_millis(body.latency_ms);
    StatusCode::OK
}
