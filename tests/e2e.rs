//! End-to-end tests for llmux.
//!
//! Spins up mock backends (simple axum echo servers), writes tiny hook scripts,
//! and drives requests through the full stack: middleware → switcher → hooks → proxy.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use llmux::{Config, ModelConfig, PolicyConfig};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tower::ServiceExt;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Create a temp directory with executable hook scripts.
/// wake/sleep scripts sleep for `wake_ms`/`sleep_ms` then exit 0.
/// alive script always exits 0.
struct MockHooks {
    #[allow(dead_code)] // TempDir must be held to keep the directory alive
    dir: tempfile::TempDir,
    wake_path: PathBuf,
    sleep_path: PathBuf,
    alive_path: PathBuf,
}

impl MockHooks {
    fn new(wake_ms: u64, sleep_ms: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();

        let wake_path = dir.path().join("wake.sh");
        let sleep_path = dir.path().join("sleep.sh");
        let alive_path = dir.path().join("alive.sh");

        std::fs::write(
            &wake_path,
            format!("#!/bin/sh\nsleep {}\n", wake_ms as f64 / 1000.0),
        )
        .unwrap();
        std::fs::write(
            &sleep_path,
            format!("#!/bin/sh\nsleep {}\n", sleep_ms as f64 / 1000.0),
        )
        .unwrap();
        std::fs::write(&alive_path, "#!/bin/sh\nexit 0\n").unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            std::fs::set_permissions(&wake_path, perms.clone()).unwrap();
            std::fs::set_permissions(&sleep_path, perms.clone()).unwrap();
            std::fs::set_permissions(&alive_path, perms).unwrap();
        }

        Self {
            dir,
            wake_path,
            sleep_path,
            alive_path,
        }
    }
}

/// Spawn a mock backend that echoes the model name and a counter.
async fn spawn_mock_backend(port: u16) -> (SocketAddr, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let c = counter_clone.fetch_add(1, Ordering::SeqCst);
            let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            async move {
                Json(json!({
                    "model": model,
                    "request_number": c,
                    "choices": [{"message": {"content": "hello"}}]
                }))
            }
        }),
    );

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Wait for the server to be ready
    tokio::time::sleep(Duration::from_millis(10)).await;

    (addr, counter)
}

/// Build a test config with two models.
fn test_config(
    model_a_port: u16,
    model_b_port: u16,
    hooks_a: &MockHooks,
    hooks_b: &MockHooks,
) -> Config {
    let mut models = HashMap::new();
    models.insert(
        "model-a".to_string(),
        ModelConfig {
            port: model_a_port,
            wake: hooks_a.wake_path.clone(),
            sleep: hooks_a.sleep_path.clone(),
            alive: hooks_a.alive_path.clone(),
        },
    );
    models.insert(
        "model-b".to_string(),
        ModelConfig {
            port: model_b_port,
            wake: hooks_b.wake_path.clone(),
            sleep: hooks_b.sleep_path.clone(),
            alive: hooks_b.alive_path.clone(),
        },
    );

    Config {
        models,
        policy: PolicyConfig {
            request_timeout_secs: 30,
            drain_before_switch: true,
            min_active_secs: 0,
        },
        port: 0,
    }
}

/// Send a chat completion request through the app and return the response body.
async fn chat_request(app: &Router, model: &str) -> (StatusCode, Value) {
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status();
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body_bytes).unwrap_or(json!({"raw": String::from_utf8_lossy(&body_bytes).to_string()}));

    (status, json)
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Basic: request for model-a goes to model-a's backend.
#[tokio::test]
async fn test_single_model_request() {
    let hooks_a = MockHooks::new(0, 0); // instant wake/sleep
    let hooks_b = MockHooks::new(0, 0);

    let (addr_a, counter_a) = spawn_mock_backend(0).await;
    let (addr_b, _counter_b) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _switcher) = llmux::build_app(config).await.unwrap();

    let (status, body) = chat_request(&app, "model-a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "model-a");
    assert_eq!(counter_a.load(Ordering::SeqCst), 1);
}

/// Two sequential requests to different models triggers a switch.
#[tokio::test]
async fn test_model_switch() {
    let hooks_a = MockHooks::new(10, 10); // 10ms wake/sleep
    let hooks_b = MockHooks::new(10, 10);

    let (addr_a, counter_a) = spawn_mock_backend(0).await;
    let (addr_b, counter_b) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _switcher) = llmux::build_app(config).await.unwrap();

    // First request: model-a (cold start)
    let (status, body) = chat_request(&app, "model-a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "model-a");

    // Second request: model-b (switch from a → b)
    let (status, body) = chat_request(&app, "model-b").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "model-b");

    assert_eq!(counter_a.load(Ordering::SeqCst), 1);
    assert_eq!(counter_b.load(Ordering::SeqCst), 1);
}

/// Multiple requests to the same model don't trigger a switch.
#[tokio::test]
async fn test_same_model_no_switch() {
    let hooks_a = MockHooks::new(10, 10);
    let hooks_b = MockHooks::new(10, 10);

    let (addr_a, counter_a) = spawn_mock_backend(0).await;
    let (addr_b, counter_b) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _switcher) = llmux::build_app(config).await.unwrap();

    for _ in 0..5 {
        let (status, body) = chat_request(&app, "model-a").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "model-a");
    }

    assert_eq!(counter_a.load(Ordering::SeqCst), 5);
    assert_eq!(counter_b.load(Ordering::SeqCst), 0);
}

/// Unknown model returns 404.
#[tokio::test]
async fn test_unknown_model() {
    let hooks_a = MockHooks::new(0, 0);
    let hooks_b = MockHooks::new(0, 0);

    let (addr_a, _) = spawn_mock_backend(0).await;
    let (addr_b, _) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _) = llmux::build_app(config).await.unwrap();

    let (status, body) = chat_request(&app, "nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"]["message"].as_str().unwrap().contains("not found"));
}

/// Switch cost is real wall-clock time from the hook scripts.
#[tokio::test]
async fn test_switch_timing() {
    let hooks_a = MockHooks::new(100, 50); // 100ms wake, 50ms sleep
    let hooks_b = MockHooks::new(100, 50);

    let (addr_a, _) = spawn_mock_backend(0).await;
    let (addr_b, _) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _switcher) = llmux::build_app(config).await.unwrap();

    // First request: cold start wake for model-a (~100ms)
    let t0 = Instant::now();
    let (status, _) = chat_request(&app, "model-a").await;
    let cold_start = t0.elapsed();
    assert_eq!(status, StatusCode::OK);
    assert!(cold_start >= Duration::from_millis(80), "cold start took {:?}", cold_start);

    // Second request: switch a→b (sleep a ~50ms + wake b ~100ms = ~150ms)
    let t1 = Instant::now();
    let (status, _) = chat_request(&app, "model-b").await;
    let switch_time = t1.elapsed();
    assert_eq!(status, StatusCode::OK);
    assert!(
        switch_time >= Duration::from_millis(120),
        "switch took {:?}, expected >= 120ms (sleep + wake)",
        switch_time
    );
}

/// Concurrent requests for the same model all get served.
#[tokio::test]
async fn test_concurrent_same_model() {
    let hooks_a = MockHooks::new(50, 10);
    let hooks_b = MockHooks::new(50, 10);

    let (addr_a, counter_a) = spawn_mock_backend(0).await;
    let (addr_b, _) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _) = llmux::build_app(config).await.unwrap();

    // Send 10 concurrent requests for model-a
    let mut handles = Vec::new();
    for _ in 0..10 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            chat_request(&app, "model-a").await
        }));
    }

    for handle in handles {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"], "model-a");
    }

    assert_eq!(counter_a.load(Ordering::SeqCst), 10);
}

/// Concurrent requests for different models: all eventually served via FIFO switching.
#[tokio::test]
async fn test_concurrent_different_models() {
    let hooks_a = MockHooks::new(50, 10);
    let hooks_b = MockHooks::new(50, 10);

    let (addr_a, counter_a) = spawn_mock_backend(0).await;
    let (addr_b, counter_b) = spawn_mock_backend(0).await;

    let config = test_config(addr_a.port(), addr_b.port(), &hooks_a, &hooks_b);
    let (app, _) = llmux::build_app(config).await.unwrap();

    // Send requests for both models concurrently
    let mut handles = Vec::new();
    for i in 0..6 {
        let app = app.clone();
        let model = if i % 2 == 0 { "model-a" } else { "model-b" };
        handles.push(tokio::spawn(async move {
            chat_request(&app, model).await
        }));
    }

    let mut statuses = Vec::new();
    for handle in handles {
        let (status, _) = handle.await.unwrap();
        statuses.push(status);
    }

    // All should succeed (FIFO will switch back and forth)
    assert!(
        statuses.iter().all(|s| *s == StatusCode::OK),
        "Some requests failed: {:?}",
        statuses
    );

    let total = counter_a.load(Ordering::SeqCst) + counter_b.load(Ordering::SeqCst);
    assert_eq!(total, 6);
}

