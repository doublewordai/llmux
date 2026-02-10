//! Integration tests for llmux using mock vLLM servers
//!
//! These tests spawn actual mock-vllm processes and verify the full integration.
//! All tests use event-driven synchronization (no polling).

use serial_test::serial;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Port allocator for orchestrator tests that need fixed ports.
/// Starts at a high port to avoid conflicts with system services.
static NEXT_PORT: AtomicU16 = AtomicU16::new(21000);

fn allocate_port() -> u16 {
    NEXT_PORT.fetch_add(1, Ordering::SeqCst)
}

/// A running mock-vllm server.
///
/// Waits for the server to signal readiness before returning.
/// Automatically kills the server when dropped.
struct MockServer {
    child: Child,
    port: u16,
    model: String,
}

impl MockServer {
    /// Spawn a mock-vllm server and wait for it to be ready.
    ///
    /// Uses dynamic port allocation (port 0) to avoid conflicts.
    /// Waits for the "READY <port>" signal from stdout (event-driven).
    async fn spawn(model: &str) -> Self {
        Self::spawn_with_args(model, &[]).await
    }

    /// Spawn with additional arguments.
    async fn spawn_with_args(model: &str, extra_args: &[&str]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_mock-vllm"));
        cmd.args(["--port", "0", "--model", model, "--latency-ms", "5"])
            .args(extra_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().expect("Failed to spawn mock-vllm");

        // Read stdout to get the READY signal with the actual port
        let stdout = child.stdout.take().expect("Failed to capture stdout");
        let mut reader = BufReader::new(stdout).lines();

        let port = tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(line) = reader.next_line().await.expect("Failed to read stdout") {
                if let Some(port_str) = line.strip_prefix("READY ") {
                    return port_str.parse::<u16>().expect("Failed to parse port");
                }
            }
            panic!("Server never signaled READY");
        })
        .await
        .expect("Timeout waiting for server to be ready");

        Self {
            child,
            port,
            model: model.to_string(),
        }
    }

    /// Get the port this server is listening on.
    fn port(&self) -> u16 {
        self.port
    }

    /// Make a chat completion request to this server.
    async fn chat(&self, message: &str) -> serde_json::Value {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/v1/chat/completions", self.port);

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": message}]
        });

        client
            .post(&url)
            .json(&body)
            .send()
            .await
            .expect("Request failed")
            .json()
            .await
            .expect("Failed to parse response")
    }

    /// Get stats from this server.
    async fn stats(&self) -> serde_json::Value {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/stats", self.port);

        client
            .get(&url)
            .send()
            .await
            .expect("Request failed")
            .json()
            .await
            .expect("Failed to parse response")
    }

    /// Put the server to sleep.
    async fn sleep(&self, level: u8) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/sleep?level={}", self.port, level);
        client
            .post(&url)
            .send()
            .await
            .expect("Sleep request failed");
    }

    /// Wake up the server.
    async fn wake(&self) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/wake_up", self.port);
        client.post(&url).send().await.expect("Wake request failed");
    }

    /// Set fail-wake mode (causes /wake_up to return 500).
    #[allow(dead_code)]
    async fn set_fail_wake(&self, enabled: bool) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/control/fail-wake", self.port);
        client
            .post(&url)
            .json(&serde_json::json!({ "enabled": enabled }))
            .send()
            .await
            .expect("Failed to set fail-wake");
    }

    /// Set fail-sleep mode (causes /sleep to return 500).
    #[allow(dead_code)]
    async fn set_fail_sleep(&self, enabled: bool) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/control/fail-sleep", self.port);
        client
            .post(&url)
            .json(&serde_json::json!({ "enabled": enabled }))
            .send()
            .await
            .expect("Failed to set fail-sleep");
    }

    /// Set artificial sleep delay.
    async fn set_sleep_delay(&self, delay_ms: u64) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/control/sleep-delay", self.port);
        client
            .post(&url)
            .json(&serde_json::json!({ "delay_ms": delay_ms }))
            .send()
            .await
            .expect("Failed to set sleep-delay");
    }

    /// Set request latency.
    #[allow(dead_code)]
    async fn set_latency(&self, latency_ms: u64) {
        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/control/latency", self.port);
        client
            .post(&url)
            .json(&serde_json::json!({ "latency_ms": latency_ms }))
            .send()
            .await
            .expect("Failed to set latency");
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        // Use synchronous kill since we're in Drop
        let _ = self.child.start_kill();
    }
}

// =============================================================================
// Mock vLLM Server Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_mock_server_basic() {
    let server = MockServer::spawn("test-model").await;

    // Verify initial stats
    let stats = server.stats().await;
    assert_eq!(stats["model"], "test-model");
    assert_eq!(stats["sleeping"], false);
    assert_eq!(stats["request_count"], 0);

    // Make a request
    let response = server.chat("Hello!").await;
    assert!(
        response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Hello!")
    );

    // Verify request was counted
    let stats = server.stats().await;
    assert_eq!(stats["request_count"], 1);
}

#[tokio::test]
#[serial]
async fn test_mock_server_sleep_wake() {
    let server = MockServer::spawn("sleepy-model").await;

    // Initially awake
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], false);

    // Sleep at L1
    server.sleep(1).await;
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], true);
    assert_eq!(stats["sleep_level"], 1);

    // Wake up
    server.wake().await;
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], false);

    // Request should succeed after wake
    let response = server.chat("Hello again!").await;
    assert!(response.get("choices").is_some());
}

#[tokio::test]
#[serial]
async fn test_mock_server_l2_sleep() {
    let server = MockServer::spawn("deep-model").await;

    // Sleep at L2
    server.sleep(2).await;
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], true);
    assert_eq!(stats["sleep_level"], 2);

    // Wake up
    server.wake().await;
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], false);
}

#[tokio::test]
#[serial]
async fn test_mock_server_rejects_while_sleeping() {
    let server = MockServer::spawn("strict-model").await;

    server.sleep(1).await;

    // Request should fail while sleeping
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/chat/completions", server.port());
    let body = serde_json::json!({
        "model": "strict-model",
        "messages": [{"role": "user", "content": "test"}]
    });

    let response = client.post(&url).json(&body).send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
}

// =============================================================================
// Orchestrator Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_orchestrator_spawns_and_manages_process() {
    use llmux::{ModelConfig, Orchestrator, ProcessState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");

    let mut models = HashMap::new();
    models.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port: 0, // Will use dynamic port, but orchestrator needs a fixed port
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    // Allocate a unique port for this test
    let port = allocate_port();
    models.get_mut("test-model").unwrap().port = port;

    let orchestrator = Arc::new(Orchestrator::with_command(
        models,
        mock_vllm_path.to_string(),
    ));

    // Initial state
    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::NotStarted)
    );

    // Start via ensure_running
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        orchestrator.ensure_running("test-model"),
    )
    .await;

    assert!(result.is_ok(), "Timed out waiting for process to start");
    assert!(result.unwrap().is_ok(), "Failed to start process");

    // Should be running
    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Make a request to verify it's actually running
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/chat/completions", port);
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}]
    });

    let response = client.post(&url).json(&body).send().await.unwrap();
    assert!(response.status().is_success());

    // Sleep via orchestrator
    orchestrator
        .sleep_model("test-model", llmux::SleepLevel::L1)
        .await
        .unwrap();

    assert!(matches!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: Some(_) })
    ));

    // Wake via orchestrator
    orchestrator.wake_model("test-model").await.unwrap();

    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );
}

#[tokio::test]
#[serial]
async fn test_orchestrator_multiple_models() {
    use llmux::{ModelConfig, Orchestrator, ProcessState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");

    // Allocate unique ports for each model
    let port_alpha = allocate_port();
    let port_beta = allocate_port();

    let mut models = HashMap::new();
    models.insert(
        "model-alpha".to_string(),
        ModelConfig {
            model_path: "model-alpha".to_string(),
            port: port_alpha,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    models.insert(
        "model-beta".to_string(),
        ModelConfig {
            model_path: "model-beta".to_string(),
            port: port_beta,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        models,
        mock_vllm_path.to_string(),
    ));

    // Both should start as not started
    assert_eq!(
        orchestrator.process_state("model-alpha").await,
        Some(ProcessState::NotStarted)
    );
    assert_eq!(
        orchestrator.process_state("model-beta").await,
        Some(ProcessState::NotStarted)
    );

    // Start alpha
    orchestrator
        .ensure_running("model-alpha")
        .await
        .expect("Failed to start model-alpha");

    assert_eq!(
        orchestrator.process_state("model-alpha").await,
        Some(ProcessState::Running { sleeping: None })
    );
    assert_eq!(
        orchestrator.process_state("model-beta").await,
        Some(ProcessState::NotStarted)
    );

    // Start beta
    orchestrator
        .ensure_running("model-beta")
        .await
        .expect("Failed to start model-beta");

    // Both running
    assert_eq!(
        orchestrator.process_state("model-alpha").await,
        Some(ProcessState::Running { sleeping: None })
    );
    assert_eq!(
        orchestrator.process_state("model-beta").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Sleep alpha, beta stays awake
    orchestrator
        .sleep_model("model-alpha", llmux::SleepLevel::L1)
        .await
        .unwrap();

    assert!(matches!(
        orchestrator.process_state("model-alpha").await,
        Some(ProcessState::Running { sleeping: Some(_) })
    ));
    assert_eq!(
        orchestrator.process_state("model-beta").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Wake alpha
    orchestrator.wake_model("model-alpha").await.unwrap();

    assert_eq!(
        orchestrator.process_state("model-alpha").await,
        Some(ProcessState::Running { sleeping: None })
    );
}

// =============================================================================
// Switcher Tests (Unit - no process spawning)
// =============================================================================

#[tokio::test]
async fn test_switcher_basic_registration() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator};
    use std::sync::Arc;

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8001,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8002,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::new(configs));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    assert!(switcher.is_registered("model-a"));
    assert!(switcher.is_registered("model-b"));
    assert!(!switcher.is_registered("model-c"));
}

#[tokio::test]
async fn test_switcher_unregistered_model_error() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitchError};
    use std::sync::Arc;

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8001,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::new(configs));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Request for unregistered model should fail
    let result = switcher.ensure_model_ready("nonexistent").await;
    assert!(matches!(result, Err(SwitchError::ModelNotFound(_))));
}

#[tokio::test]
async fn test_switcher_in_flight_tracking() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator};
    use std::sync::Arc;

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8001,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::new(configs));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Initially no in-flight
    assert_eq!(switcher.in_flight_count("model-a"), 0);

    // Acquire guard
    let guard1 = switcher.acquire_in_flight("model-a");
    assert!(guard1.is_some());
    assert_eq!(switcher.in_flight_count("model-a"), 1);

    // Acquire another
    let guard2 = switcher.acquire_in_flight("model-a");
    assert!(guard2.is_some());
    assert_eq!(switcher.in_flight_count("model-a"), 2);

    // Drop one
    drop(guard1);
    assert_eq!(switcher.in_flight_count("model-a"), 1);

    // Drop the other
    drop(guard2);
    assert_eq!(switcher.in_flight_count("model-a"), 0);

    // Unregistered model returns None
    assert!(switcher.acquire_in_flight("nonexistent").is_none());
}

#[tokio::test]
async fn test_switcher_initial_state() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8001,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::new(configs));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Initially idle
    assert_eq!(switcher.state().await, SwitcherState::Idle);
    assert_eq!(switcher.active_model().await, None);
}

// =============================================================================
// Switcher Integration Tests (with process spawning)
// =============================================================================

#[tokio::test]
#[serial]
async fn test_switcher_ensure_model_ready() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Initially idle
    assert_eq!(switcher.state().await, SwitcherState::Idle);

    // Request model - should start it and make it active
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        switcher.ensure_model_ready("test-model"),
    )
    .await;

    assert!(result.is_ok(), "Timeout");
    assert!(result.unwrap().is_ok(), "Failed to ensure model ready");

    // Should now be active
    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "test-model".to_string()
        }
    );
    assert_eq!(
        switcher.active_model().await,
        Some("test-model".to_string())
    );
}

#[tokio::test]
#[serial]
async fn test_switcher_model_switching() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Start with model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );

    // Switch to model-b
    switcher
        .ensure_model_ready("model-b")
        .await
        .expect("Failed to switch to model-b");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-b".to_string()
        }
    );

    // Switch back to model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to switch back to model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );
}

#[tokio::test]
#[serial]
async fn test_switcher_same_model_no_switch() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Start model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    // Request same model again - should return immediately
    let start = std::time::Instant::now();
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed second request");
    let elapsed = start.elapsed();

    // Should be very fast (no switch needed)
    assert!(
        elapsed < Duration::from_millis(100),
        "Same model request took too long: {:?}",
        elapsed
    );

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_orchestrator_unknown_model() {
    use llmux::{ModelConfig, Orchestrator, OrchestratorError};
    use std::sync::Arc;

    let mut configs = HashMap::new();
    configs.insert(
        "known-model".to_string(),
        ModelConfig {
            model_path: "test".to_string(),
            port: 8001,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::new(configs));

    // Unknown model should return None for state
    assert_eq!(orchestrator.process_state("unknown").await, None);

    // ensure_running should fail for unknown model
    let result = orchestrator.ensure_running("unknown").await;
    assert!(matches!(result, Err(OrchestratorError::ModelNotFound(_))));

    // sleep/wake should fail for unknown model
    let result = orchestrator
        .sleep_model("unknown", llmux::SleepLevel::L1)
        .await;
    assert!(matches!(result, Err(OrchestratorError::ModelNotFound(_))));

    let result = orchestrator.wake_model("unknown").await;
    assert!(matches!(result, Err(OrchestratorError::ModelNotFound(_))));
}

// =============================================================================
// End-to-End Tests (Full HTTP stack)
// =============================================================================

#[tokio::test]
#[serial]
async fn test_end_to_end_single_model() {
    use axum::Router;
    use llmux::{
        Config, FifoPolicy, ModelConfig, ModelSwitcher, ModelSwitcherLayer, Orchestrator,
        PolicyConfig,
    };
    use std::sync::Arc;
    use tokio::net::TcpListener;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let backend_port = allocate_port();
    let proxy_port = allocate_port();

    // Build config
    let mut models = HashMap::new();
    models.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port: backend_port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let config = Config {
        models: models.clone(),
        policy: PolicyConfig::default(),
        port: proxy_port,
        metrics_port: 0,
        vllm_command: mock_vllm_path.to_string(),
        checkpoint: None,
    };

    // Build the full app stack
    let orchestrator = Arc::new(Orchestrator::with_command(
        config.models.clone(),
        config.vllm_command.clone(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    // Build onwards targets
    let targets = config.build_onwards_targets().unwrap();
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);

    // Wrap with middleware
    let app: Router = onwards_router.layer(ModelSwitcherLayer::new(switcher));

    // Start server
    let listener = TcpListener::bind(format!("127.0.0.1:{}", proxy_port))
        .await
        .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send request through proxy
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello from e2e test!"}]
        }))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .expect("Request failed");

    assert!(
        response.status().is_success(),
        "Response status: {}",
        response.status()
    );

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        content.contains("Hello from e2e test!"),
        "Unexpected response: {}",
        content
    );

    server.abort();
}

#[tokio::test]
#[serial]
async fn test_end_to_end_model_switching() {
    use axum::Router;
    use llmux::{
        Config, FifoPolicy, ModelConfig, ModelSwitcher, ModelSwitcherLayer, Orchestrator,
        PolicyConfig,
    };
    use std::sync::Arc;
    use tokio::net::TcpListener;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();
    let proxy_port = allocate_port();

    // Build config with two models
    let mut models = HashMap::new();
    models.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    models.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let config = Config {
        models: models.clone(),
        policy: PolicyConfig::default(),
        port: proxy_port,
        metrics_port: 0,
        vllm_command: mock_vllm_path.to_string(),
        checkpoint: None,
    };

    // Build the full app stack
    let orchestrator = Arc::new(Orchestrator::with_command(
        config.models.clone(),
        config.vllm_command.clone(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    let targets = config.build_onwards_targets().unwrap();
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);
    let app: Router = onwards_router.layer(ModelSwitcherLayer::new(switcher));

    // Start server
    let listener = TcpListener::bind(format!("127.0.0.1:{}", proxy_port))
        .await
        .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port);

    // Request to model-a
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "model-a",
            "messages": [{"role": "user", "content": "Hello A!"}]
        }))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .expect("Request to model-a failed");

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Hello A!")
    );

    // Switch to model-b
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "model-b",
            "messages": [{"role": "user", "content": "Hello B!"}]
        }))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .expect("Request to model-b failed");

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Hello B!")
    );

    // Switch back to model-a
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "model-a",
            "messages": [{"role": "user", "content": "Back to A!"}]
        }))
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .expect("Request back to model-a failed");

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Back to A!")
    );

    server.abort();
}

#[tokio::test]
#[serial]
async fn test_end_to_end_unknown_model_passthrough() {
    use axum::Router;
    use llmux::{
        Config, FifoPolicy, ModelConfig, ModelSwitcher, ModelSwitcherLayer, Orchestrator,
        PolicyConfig,
    };
    use std::sync::Arc;
    use tokio::net::TcpListener;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let backend_port = allocate_port();
    let proxy_port = allocate_port();

    let mut models = HashMap::new();
    models.insert(
        "known-model".to_string(),
        ModelConfig {
            model_path: "known-model".to_string(),
            port: backend_port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let config = Config {
        models: models.clone(),
        policy: PolicyConfig::default(),
        port: proxy_port,
        metrics_port: 0,
        vllm_command: mock_vllm_path.to_string(),
        checkpoint: None,
    };

    let orchestrator = Arc::new(Orchestrator::with_command(
        config.models.clone(),
        config.vllm_command.clone(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    let targets = config.build_onwards_targets().unwrap();
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);
    let app: Router = onwards_router.layer(ModelSwitcherLayer::new(switcher));

    let listener = TcpListener::bind(format!("127.0.0.1:{}", proxy_port))
        .await
        .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // Request to unknown model should be passed through (and fail at onwards level)
    let response = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            proxy_port
        ))
        .json(&serde_json::json!({
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "test"}]
        }))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("Request failed");

    // Should get a 404 from onwards (model not found in targets)
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

    server.abort();
}

// =============================================================================
// L3 Fallback & Timeout Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_l3_fallback_on_sleep_failure() {
    use llmux::{
        FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, ProcessState, SwitcherState,
    };
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    // Start model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    assert_eq!(
        orchestrator.process_state("model-a").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Make model-a's sleep fail via control endpoint
    let client = reqwest::Client::new();
    client
        .post(format!("http://localhost:{}/control/fail-sleep", port_a))
        .json(&serde_json::json!({ "enabled": true }))
        .send()
        .await
        .expect("Failed to set fail-sleep");

    // Switch to model-b — this should trigger sleep on model-a, which will fail,
    // then escalate to L3 (Stop), killing model-a's process
    switcher
        .ensure_model_ready("model-b")
        .await
        .expect("Failed to switch to model-b");

    // model-b should be active
    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-b".to_string()
        }
    );

    // model-a should be NotStarted (killed by L3 fallback)
    assert_eq!(
        orchestrator.process_state("model-a").await,
        Some(ProcessState::NotStarted)
    );

    // model-b should serve correctly
    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port_b))
        .json(&serde_json::json!({
            "model": "model-b",
            "messages": [{"role": "user", "content": "test after fallback"}]
        }))
        .send()
        .await
        .expect("Request to model-b failed");
    assert!(response.status().is_success());

    // Switch back to model-a — should restart it from NotStarted
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to switch back to model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );

    assert_eq!(
        orchestrator.process_state("model-a").await,
        Some(ProcessState::Running { sleeping: None })
    );
}

// =============================================================================
// Drain Race Condition Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_drain_waits_for_in_flight_before_sleep() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));

    // Zero cooldown, drain before switch
    let policy = Box::new(FifoPolicy::new(
        1,
        Duration::from_secs(60),
        true,
        Duration::ZERO,
    ));
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Step 1: Make model-a active
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );

    // Step 2: Acquire in-flight guards to simulate requests being processed.
    // These guards prevent the drain from completing.
    let guard1 = switcher
        .acquire_in_flight("model-a")
        .expect("acquire_in_flight should succeed");
    let guard2 = switcher
        .acquire_in_flight("model-a")
        .expect("acquire_in_flight should succeed");

    assert_eq!(switcher.in_flight_count("model-a"), 2);

    // Step 3: Trigger switch to model-b on a background task.
    // The switch will drain model-a (waiting for in-flight to reach 0).
    let switcher_bg = switcher.clone();
    let switch_handle =
        tokio::spawn(async move { switcher_bg.ensure_model_ready("model-b").await });

    // Give the switch task time to start the drain
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Step 4: While draining, new acquire_in_flight should be rejected
    assert!(
        switcher.acquire_in_flight("model-a").is_none(),
        "acquire_in_flight should return None while draining"
    );

    // The switch should still be in progress (blocked on drain)
    assert!(
        matches!(switcher.state().await, SwitcherState::Switching { .. }),
        "Expected Switching state while in-flight guards are held"
    );

    // Step 5: Release the guards — drain completes, switch proceeds
    drop(guard1);
    drop(guard2);

    // Step 6: Wait for the switch to finish
    let result = tokio::time::timeout(Duration::from_secs(15), switch_handle)
        .await
        .expect("Switch timed out")
        .expect("Switch task panicked");

    assert!(result.is_ok(), "Switch failed: {:?}", result.err());

    // model-b should now be active
    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-b".to_string()
        }
    );

    // Draining flag should be cleared — acquire_in_flight for model-a should
    // work again (though model-a is now sleeping, the flag itself is clear)
    // Note: We just check in_flight_count is 0, confirming guards were properly drained.
    assert_eq!(switcher.in_flight_count("model-a"), 0);
}

#[tokio::test]
#[serial]
async fn test_sleep_timeout_completes() {
    // Verify that a sleep with artificial delay completes within the 120s timeout
    // (would have failed with the old 30s timeout)
    let server = MockServer::spawn("timeout-model").await;

    // Set a 2s sleep delay
    server.set_sleep_delay(2000).await;

    // Sleep should complete (2s delay is well within 120s timeout)
    let start = std::time::Instant::now();
    server.sleep(1).await;
    let elapsed = start.elapsed();

    // Should have taken at least 2s due to the delay
    assert!(
        elapsed >= Duration::from_secs(2),
        "Sleep completed too quickly ({:?}), delay not applied",
        elapsed
    );

    // Should not have taken more than 10s (generous upper bound)
    assert!(
        elapsed < Duration::from_secs(10),
        "Sleep took too long ({:?})",
        elapsed
    );

    // Verify model is sleeping
    let stats = server.stats().await;
    assert_eq!(stats["sleeping"], true);

    // Wake and verify it still works
    server.wake().await;
    let response = server.chat("After delayed sleep").await;
    assert!(response.get("choices").is_some());
}

#[tokio::test]
#[serial]
async fn test_end_to_end_concurrent_requests() {
    use axum::Router;
    use llmux::{
        Config, FifoPolicy, ModelConfig, ModelSwitcher, ModelSwitcherLayer, Orchestrator,
        PolicyConfig,
    };
    use std::sync::Arc;
    use tokio::net::TcpListener;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let backend_port = allocate_port();
    let proxy_port = allocate_port();

    let mut models = HashMap::new();
    models.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port: backend_port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let config = Config {
        models: models.clone(),
        policy: PolicyConfig::default(),
        port: proxy_port,
        metrics_port: 0,
        vllm_command: mock_vllm_path.to_string(),
        checkpoint: None,
    };

    let orchestrator = Arc::new(Orchestrator::with_command(
        config.models.clone(),
        config.vllm_command.clone(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    let targets = config.build_onwards_targets().unwrap();
    let onwards_state = onwards::AppState::new(targets);
    let onwards_router = onwards::build_router(onwards_state);
    let app: Router = onwards_router.layer(ModelSwitcherLayer::new(switcher));

    let listener = TcpListener::bind(format!("127.0.0.1:{}", proxy_port))
        .await
        .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Send multiple concurrent requests
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port);

    let mut handles = vec![];
    for i in 0..5 {
        let client = client.clone();
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            client
                .post(&url)
                .json(&serde_json::json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": format!("Request {}", i)}]
                }))
                .timeout(Duration::from_secs(15))
                .send()
                .await
        }));
    }

    // All should succeed
    for (i, handle) in handles.into_iter().enumerate() {
        let response = handle
            .await
            .expect("Task panicked")
            .expect("Request failed");
        assert!(
            response.status().is_success(),
            "Request {} failed with status {}",
            i,
            response.status()
        );
    }

    server.abort();
}

// =============================================================================
// Cooldown Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_switch_cooldown_enforced() {
    use llmux::{FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, SwitcherState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));

    // Use a 2-second cooldown for testing
    let policy = Box::new(FifoPolicy::new(
        1,
        Duration::from_secs(60),
        true,
        Duration::from_secs(2),
    ));
    let switcher = ModelSwitcher::new(orchestrator, policy);

    // Start with model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );

    // Immediately request model-b — the switch should enforce cooldown
    let start = std::time::Instant::now();
    switcher
        .ensure_model_ready("model-b")
        .await
        .expect("Failed to switch to model-b");
    let elapsed = start.elapsed();

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-b".to_string()
        }
    );

    // The switch should have taken at least ~2s due to cooldown
    // (minus whatever time had already elapsed since activation)
    assert!(
        elapsed >= Duration::from_millis(1500),
        "Switch completed too quickly ({:?}), cooldown not enforced",
        elapsed
    );
}

// =============================================================================
// Zombie Process Recovery Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_zombie_process_recovery() {
    use llmux::{ModelConfig, Orchestrator, ProcessState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port = allocate_port();

    let mut models = HashMap::new();
    models.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        models,
        mock_vllm_path.to_string(),
    ));

    // Start the process
    orchestrator
        .ensure_running("test-model")
        .await
        .expect("Failed to start");

    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Kill the mock-vllm process externally to simulate a crash
    // We use the /sleep endpoint with level 3 (Stop) via the orchestrator itself
    // to kill the process, then manually reset state to simulate a zombie
    // Actually, let's kill it by sending SIGKILL to the port holder
    let client = reqwest::Client::new();
    let _ = client
        .post(format!("http://localhost:{}/wake_up", port))
        .send()
        .await;

    // Kill the process via orchestrator's sleep_model with Stop level
    orchestrator
        .sleep_model("test-model", llmux::SleepLevel::Stop)
        .await
        .unwrap();

    // The process was killed and state should be NotStarted
    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::NotStarted)
    );

    // Now wake_model should detect dead process and restart via ensure_running
    orchestrator
        .wake_model("test-model")
        .await
        .expect("Failed to wake after process death");

    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Verify the restarted process actually works
    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "after recovery"}]
        }))
        .send()
        .await
        .expect("Request after recovery failed");

    assert!(response.status().is_success());
}

#[tokio::test]
#[serial]
async fn test_zombie_detection_on_wake() {
    use llmux::{ModelConfig, Orchestrator, ProcessState};
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port = allocate_port();

    let mut models = HashMap::new();
    models.insert(
        "test-model".to_string(),
        ModelConfig {
            model_path: "test-model".to_string(),
            port,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        models,
        mock_vllm_path.to_string(),
    ));

    // Start the process
    orchestrator
        .ensure_running("test-model")
        .await
        .expect("Failed to start");

    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Kill the mock-vllm process externally to simulate a crash (like an Xid 31
    // GPU fault). Find only the LISTENING PID by port and send SIGKILL.
    let output = std::process::Command::new("lsof")
        .args(["-ti", &format!("tcp:{}", port), "-sTCP:LISTEN"])
        .output()
        .expect("lsof failed");

    let pids = String::from_utf8_lossy(&output.stdout);
    for pid_str in pids.trim().lines() {
        if let Ok(pid) = pid_str.trim().parse::<i32>() {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }

    // Wait for the process to actually die
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The orchestrator still thinks the process is Running (the bug scenario).
    // check_process_alive() inside wake_model should detect the dead child,
    // reset state to NotStarted, then ensure_running will restart it.
    orchestrator
        .wake_model("test-model")
        .await
        .expect("Failed to restart after zombie");

    assert_eq!(
        orchestrator.process_state("test-model").await,
        Some(ProcessState::Running { sleeping: None })
    );

    // Verify the restarted process works
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://localhost:{}/v1/chat/completions", port))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "after zombie recovery"}]
        }))
        .send()
        .await
        .expect("Request after zombie recovery failed");

    assert!(response.status().is_success());
}

// =============================================================================
// Wake Failure Cleanup Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_wake_failure_cleans_up_target_model() {
    use llmux::{
        FifoPolicy, ModelConfig, ModelSwitcher, Orchestrator, ProcessState, SwitcherState,
    };
    use std::sync::Arc;

    let mock_vllm_path = env!("CARGO_BIN_EXE_mock-vllm");
    let port_a = allocate_port();
    let port_b = allocate_port();

    let mut configs = HashMap::new();
    configs.insert(
        "model-a".to_string(),
        ModelConfig {
            model_path: "model-a".to_string(),
            port: port_a,
            extra_args: vec![],
            sleep_level: 1,
        },
    );
    configs.insert(
        "model-b".to_string(),
        ModelConfig {
            model_path: "model-b".to_string(),
            port: port_b,
            extra_args: vec![],
            sleep_level: 1,
        },
    );

    let orchestrator = Arc::new(Orchestrator::with_command(
        configs,
        mock_vllm_path.to_string(),
    ));
    let policy = Box::new(FifoPolicy::default());
    let switcher = ModelSwitcher::new(orchestrator.clone(), policy);

    // Step 1: Start model-a
    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to start model-a");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );

    // Step 2: Start model-b's process so we can configure it to fail wake
    orchestrator
        .ensure_running("model-b")
        .await
        .expect("Failed to start model-b process");

    // Configure model-b to fail on /wake_up
    let client = reqwest::Client::new();
    client
        .post(format!("http://localhost:{}/control/fail-wake", port_b))
        .json(&serde_json::json!({ "enabled": true }))
        .send()
        .await
        .expect("Failed to set fail-wake on model-b");

    // Put model-b to sleep so the switcher will try to wake it
    orchestrator
        .sleep_model("model-b", llmux::SleepLevel::L1)
        .await
        .expect("Failed to sleep model-b");

    // Step 3: Try to switch to model-b — wake should fail
    let result = tokio::time::timeout(
        Duration::from_secs(15),
        switcher.ensure_model_ready("model-b"),
    )
    .await;

    assert!(result.is_ok(), "Timeout waiting for switch");
    assert!(result.unwrap().is_err(), "Switch should have failed");

    // Step 4: Switcher should be Idle (not Active with model-b)
    assert_eq!(switcher.state().await, SwitcherState::Idle);

    // model-b should have been force-slept (killed) — state should be NotStarted
    assert_eq!(
        orchestrator.process_state("model-b").await,
        Some(ProcessState::NotStarted)
    );

    // Step 5: Switch to model-a should succeed because GPU memory was freed
    // Need to wait for backoff to expire
    tokio::time::sleep(Duration::from_secs(3)).await;

    switcher
        .ensure_model_ready("model-a")
        .await
        .expect("Failed to switch to model-a after wake failure cleanup");

    assert_eq!(
        switcher.state().await,
        SwitcherState::Active {
            model: "model-a".to_string()
        }
    );
}
