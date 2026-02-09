//! Validation tool for sleep/wake cycles
//!
//! Tests that models can sleep and wake correctly at each level,
//! producing deterministic output after wake.

use crate::config::Config;
use crate::orchestrator::Orchestrator;
use crate::switcher::SleepLevel;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Result of testing a single sleep level
struct LevelResult {
    level: SleepLevel,
    sleep_secs: f64,
    wake_secs: f64,
    gpu_before_sleep: u64,
    gpu_after_sleep: u64,
    gpu_after_wake: u64,
    response_matches: bool,
    pass: bool,
}

/// Run a full validation cycle for the given model.
///
/// If `levels` is `Some`, only test those levels (1=L1, 2=L2).
/// If `None`, test all levels (L1, L2).
///
/// Returns `true` if all tested levels pass, `false` if any fail.
pub async fn run_validation(
    config: &Config,
    model_name: &str,
    levels: Option<&[u8]>,
) -> Result<bool> {
    let model_config = config.models.get(model_name).with_context(|| {
        let available: Vec<_> = config.models.keys().collect();
        format!(
            "Model '{}' not found in config. Available: {:?}",
            model_name, available
        )
    })?;

    let port = model_config.port;
    let model_path = model_config.model_path.clone();

    info!(model = %model_name, port, "Starting validation");

    // Create orchestrator with just this model
    let mut models = HashMap::new();
    models.insert(model_name.to_string(), model_config.clone());

    let orchestrator = Arc::new(Orchestrator::with_options(
        models,
        config.vllm_command.clone(),
        config.checkpoint.clone(),
    ));

    // Start the model
    println!("Starting model '{}'...", model_name);
    orchestrator
        .ensure_running(model_name)
        .await
        .context("Failed to start model")?;
    println!("Model is running.");

    // Run baseline inference
    println!("Running baseline inference...");
    let baseline = run_deterministic_request(port, &model_path).await?;
    let baseline_gpu = query_gpu_memory();
    println!("Baseline response: {:?}", baseline);
    println!("Baseline GPU memory: {} MiB", baseline_gpu);

    // Test each sleep level
    let test_levels: Vec<SleepLevel> = match levels {
        Some(nums) => nums.iter().map(|&n| SleepLevel::from(n)).collect(),
        None => vec![SleepLevel::L1, SleepLevel::L2],
    };
    let mut results = Vec::new();

    for level in &test_levels {
        println!("\n--- Testing {:?} ---", level);
        let result = test_sleep_level(
            &orchestrator,
            model_name,
            port,
            &model_path,
            *level,
            &baseline,
            baseline_gpu,
        )
        .await;

        match result {
            Ok(r) => results.push(r),
            Err(e) => {
                println!("ERROR testing {:?}: {}", level, e);
                results.push(LevelResult {
                    level: *level,
                    sleep_secs: 0.0,
                    wake_secs: 0.0,
                    gpu_before_sleep: baseline_gpu,
                    gpu_after_sleep: 0,
                    gpu_after_wake: 0,
                    response_matches: false,
                    pass: false,
                });
            }
        }
    }

    // Cleanup: stop the model
    println!("\nStopping model (L5)...");
    let _ = orchestrator.sleep_model(model_name, SleepLevel::Stop).await;

    // Print results table
    print_results(&results);

    let all_pass = results.iter().all(|r| r.pass);
    if all_pass {
        println!("\nResult: ALL PASSED");
    } else {
        println!("\nResult: SOME FAILED");
    }

    Ok(all_pass)
}

/// Test a single sleep level: sleep → measure → wake → measure → verify response
async fn test_sleep_level(
    orchestrator: &Arc<Orchestrator>,
    model: &str,
    port: u16,
    model_path: &str,
    level: SleepLevel,
    baseline: &str,
    _baseline_gpu: u64,
) -> Result<LevelResult> {
    let gpu_before_sleep = query_gpu_memory();

    // Sleep
    let sleep_start = Instant::now();
    orchestrator
        .sleep_model(model, level)
        .await
        .context("sleep_model failed")?;
    let sleep_secs = sleep_start.elapsed().as_secs_f64();
    println!("  Sleep took {:.1}s", sleep_secs);

    // Check GPU after sleep
    // Give a moment for memory to be released
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let gpu_after_sleep = query_gpu_memory();
    println!(
        "  GPU memory: {} MiB → {} MiB (freed {} MiB)",
        gpu_before_sleep,
        gpu_after_sleep,
        gpu_before_sleep.saturating_sub(gpu_after_sleep)
    );

    // Wake
    let wake_start = Instant::now();
    orchestrator
        .wake_model(model)
        .await
        .context("wake_model failed")?;
    let wake_secs = wake_start.elapsed().as_secs_f64();
    println!("  Wake took {:.1}s", wake_secs);

    let gpu_after_wake = query_gpu_memory();

    // Run inference and compare
    let response = run_deterministic_request(port, model_path).await?;
    let response_matches = response == baseline;
    println!(
        "  Response: {:?} ({})",
        response,
        if response_matches {
            "matches baseline"
        } else {
            "MISMATCH"
        }
    );

    let pass = response_matches;

    Ok(LevelResult {
        level,
        sleep_secs,
        wake_secs,
        gpu_before_sleep,
        gpu_after_sleep,
        gpu_after_wake,
        response_matches,
        pass,
    })
}

/// Run a deterministic inference request against a vLLM endpoint
async fn run_deterministic_request(port: u16, model_path: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/chat/completions", port);

    let body = serde_json::json!({
        "model": model_path,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 10,
    });

    let response = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .context("inference request failed")?;

    if !response.status().is_success() {
        bail!("inference request returned status {}", response.status());
    }

    let json: serde_json::Value = response.json().await.context("failed to parse response")?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(content)
}

/// Query total GPU memory usage via nvidia-smi.
///
/// Returns 0 if nvidia-smi is not available (e.g., no GPU or in CI).
fn query_gpu_memory() -> u64 {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            // Sum across all GPUs (one line per GPU)
            text.lines()
                .filter_map(|line| line.trim().parse::<u64>().ok())
                .sum()
        }
        Ok(out) => {
            warn!(
                "nvidia-smi failed with status {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
            0
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!("nvidia-smi not found, GPU memory tracking disabled");
            0
        }
        Err(e) => {
            warn!("Failed to run nvidia-smi: {}", e);
            0
        }
    }
}

fn print_results(results: &[LevelResult]) {
    println!();
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12} {:>12} {:>10} {:>6}",
        "Level", "Sleep (s)", "Wake (s)", "GPU Before", "GPU After", "GPU Wake", "Response", "Pass"
    );
    println!("{}", "-".repeat(88));

    for r in results {
        let level_str = match r.level {
            SleepLevel::L1 => "L1",
            SleepLevel::L2 => "L2",
            SleepLevel::CudaSuspend => "CudaSus",
            SleepLevel::Checkpoint => "CRIU",
            SleepLevel::Stop => "Stop",
        };

        println!(
            "{:<8} {:>10.1} {:>10.1} {:>10} MiB {:>10} MiB {:>10} MiB {:>10} {:>6}",
            level_str,
            r.sleep_secs,
            r.wake_secs,
            r.gpu_before_sleep,
            r.gpu_after_sleep,
            r.gpu_after_wake,
            if r.response_matches {
                "match"
            } else {
                "MISMATCH"
            },
            if r.pass { "OK" } else { "FAIL" },
        );
    }
}
