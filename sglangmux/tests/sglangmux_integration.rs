use serial_test::serial;
use sglangmux::{SgLangMux, SgLangMuxOptions};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn allocate_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("failed to reserve ephemeral port")
        .local_addr()
        .expect("failed to read local addr")
        .port()
}

fn write_launch_script(
    dir: &Path,
    script_name: &str,
    model_name: &str,
    port: u16,
    startup_delay_ms: u64,
) -> PathBuf {
    let script_path = dir.join(script_name);
    let mut file = std::fs::File::create(&script_path).expect("failed to create script");

    let mock_bin = env!("CARGO_BIN_EXE_mock-sglang");

    writeln!(file, "#!/usr/bin/env bash").unwrap();
    writeln!(file, "set -euo pipefail").unwrap();
    writeln!(file, "MODEL_NAME='{}'", model_name).unwrap();
    writeln!(file, "PORT={}", port).unwrap();
    writeln!(
        file,
        "exec \"{}\" --model \"$MODEL_NAME\" --port \"$PORT\" --startup-delay-ms {}",
        mock_bin, startup_delay_ms
    )
    .unwrap();

    let mut perms = file.metadata().unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script_path, perms).expect("failed to set executable bit");

    script_path
}

async fn fetch_stats(port: u16) -> serde_json::Value {
    let client = reqwest::Client::new();
    client
        .get(format!("http://127.0.0.1:{port}/stats"))
        .send()
        .await
        .expect("stats request failed")
        .json::<serde_json::Value>()
        .await
        .expect("invalid stats payload")
}

async fn chat_once(port: u16, model: &str) -> reqwest::StatusCode {
    let client = reqwest::Client::new();
    client
        .post(format!(
            "http://127.0.0.1:{port}/v1/chat/completions?sleep_ms=100"
        ))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .expect("chat request failed")
        .status()
}

fn default_options(log_dir: PathBuf) -> SgLangMuxOptions {
    SgLangMuxOptions {
        ready_timeout: Duration::from_secs(20),
        request_timeout: Duration::from_secs(20),
        poll_interval: Duration::from_millis(50),
        log_dir,
    }
}

#[tokio::test]
#[serial]
async fn models_bootstrap_one_at_a_time() {
    let tempdir = tempfile::tempdir().expect("failed to create tempdir");

    let model_a = "Qwen/Qwen3-0.6B";
    let model_b = "HuggingFaceTB/SmolLM-360M";
    let port_a = allocate_port();
    let port_b = allocate_port();

    let script_a = write_launch_script(tempdir.path(), "qwen.sh", model_a, port_a, 350);
    let script_b = write_launch_script(tempdir.path(), "smol.sh", model_b, port_b, 350);

    let mux = SgLangMux::from_scripts(
        vec![script_a, script_b],
        default_options(tempdir.path().join("logs")),
    )
    .expect("failed to build mux");

    mux.bootstrap_sequential()
        .await
        .expect("bootstrap should succeed");

    assert_eq!(mux.active_model().await, Some(model_b.to_string()));

    let stats_a = fetch_stats(port_a).await;
    let stats_b = fetch_stats(port_b).await;

    assert_eq!(stats_a["model"], model_a);
    assert_eq!(stats_b["model"], model_b);
    assert_eq!(stats_a["sleeping"], true);
    assert_eq!(stats_a["weights_loaded"], false);
    assert_eq!(stats_b["sleeping"], false);
    assert_eq!(stats_b["weights_loaded"], true);
    assert_eq!(stats_a["release_count"], 1);
    assert_eq!(stats_a["update_count"], 0);
    assert_eq!(stats_b["update_count"], 0);

    let log_paths = mux.model_log_paths();
    let model_a_logs = log_paths.get(model_a).expect("missing model-a logs");
    let model_b_logs = log_paths.get(model_b).expect("missing model-b logs");

    assert!(model_a_logs.stdout.exists());
    assert!(model_a_logs.stderr.exists());
    assert!(model_b_logs.stdout.exists());
    assert!(model_b_logs.stderr.exists());

    let model_a_stdout = std::fs::read_to_string(&model_a_logs.stdout).expect("read model-a logs");
    let model_b_stdout = std::fs::read_to_string(&model_b_logs.stdout).expect("read model-b logs");
    assert!(
        model_a_stdout.contains("READY"),
        "model-a stdout should contain startup logs"
    );
    assert!(
        model_b_stdout.contains("READY"),
        "model-b stdout should contain startup logs"
    );

    let last_release = stats_a["last_release_unix_ms"].as_u64().unwrap_or(0);
    let second_start = stats_b["startup_unix_ms"].as_u64().unwrap_or(0);

    assert!(
        second_start >= last_release,
        "expected model-b startup ({second_start}) to happen after model-a release ({last_release})"
    );

    mux.shutdown_all().await;
}

#[tokio::test]
#[serial]
async fn requests_wait_when_they_do_not_hold_the_lock() {
    let tempdir = tempfile::tempdir().expect("failed to create tempdir");

    let model_a = "Qwen/Qwen3-0.6B";
    let model_b = "HuggingFaceTB/SmolLM-360M";
    let port_a = allocate_port();
    let port_b = allocate_port();

    let script_a = write_launch_script(tempdir.path(), "qwen.sh", model_a, port_a, 10);
    let script_b = write_launch_script(tempdir.path(), "smol.sh", model_b, port_b, 10);

    let mux = SgLangMux::from_scripts(
        vec![script_a, script_b],
        default_options(tempdir.path().join("logs")),
    )
    .expect("failed to build mux");

    mux.bootstrap_sequential().await.expect("bootstrap failed");

    let held_guard = mux
        .acquire_in_flight(model_b)
        .expect("expected lock guard on active model");

    let mux_bg = mux.clone();
    let switch_waiter = tokio::spawn(async move { mux_bg.ensure_model_ready(model_a).await });

    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(
        !switch_waiter.is_finished(),
        "switch should block while active model still has in-flight holders"
    );

    drop(held_guard);

    let switch_result = tokio::time::timeout(Duration::from_secs(10), switch_waiter)
        .await
        .expect("switch task timed out")
        .expect("switch task panicked");

    assert!(
        switch_result.is_ok(),
        "switch should succeed after lock release"
    );
    assert_eq!(mux.active_model().await, Some(model_a.to_string()));
    let stats_a = fetch_stats(port_a).await;
    assert_eq!(stats_a["update_count"], 1);
    assert_eq!(stats_a["weights_loaded"], true);

    mux.shutdown_all().await;
}

#[tokio::test]
#[serial]
async fn requests_pass_when_their_model_holds_the_lock() {
    let tempdir = tempfile::tempdir().expect("failed to create tempdir");

    let model_a = "Qwen/Qwen3-0.6B";
    let model_b = "HuggingFaceTB/SmolLM-360M";
    let port_a = allocate_port();
    let port_b = allocate_port();

    let script_a = write_launch_script(tempdir.path(), "qwen.sh", model_a, port_a, 10);
    let script_b = write_launch_script(tempdir.path(), "smol.sh", model_b, port_b, 10);

    let mux = SgLangMux::from_scripts(
        vec![script_a, script_b],
        default_options(tempdir.path().join("logs")),
    )
    .expect("failed to build mux");

    mux.bootstrap_sequential().await.expect("bootstrap failed");

    let held_guard = mux
        .acquire_in_flight(model_b)
        .expect("expected guard for active model");

    let mux_bg = mux.clone();
    let switching_task = tokio::spawn(async move { mux_bg.ensure_model_ready(model_a).await });

    tokio::time::sleep(Duration::from_millis(200)).await;

    let status = chat_once(port_b, model_b).await;
    assert!(
        status.is_success(),
        "request should pass while the active model still holds the lock"
    );

    assert!(
        mux.acquire_in_flight(model_b).is_none(),
        "new lock acquisition should be rejected once draining starts"
    );

    assert!(
        !switching_task.is_finished(),
        "switch should still be waiting for existing in-flight guard"
    );

    drop(held_guard);

    let switch_result = tokio::time::timeout(Duration::from_secs(10), switching_task)
        .await
        .expect("switch task timed out")
        .expect("switch task panicked");

    assert!(switch_result.is_ok());
    assert_eq!(mux.active_model().await, Some(model_a.to_string()));
    let stats_a = fetch_stats(port_a).await;
    assert_eq!(stats_a["update_count"], 1);
    assert_eq!(stats_a["weights_loaded"], true);

    mux.shutdown_all().await;
}
