//! Validation tool for sleep/wake cycles
//!
//! Tests that models can sleep and wake correctly at each level,
//! producing deterministic output after wake.

use crate::config::Config;
use crate::orchestrator::Orchestrator;
use crate::switcher::{EvictionPolicy, ProcessStrategy, WeightStrategy};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// Result of testing a single eviction policy
struct LevelResult {
    eviction: EvictionPolicy,
    sleep_secs: f64,
    wake_secs: f64,
    gpu_before_sleep: u64,
    gpu_after_sleep: u64,
    gpu_after_wake: u64,
    response_matches: bool,
    pass: bool,
}

/// Parse a policy string like "offload+keep_running" into an EvictionPolicy.
fn parse_policy(s: &str) -> Result<EvictionPolicy> {
    let parts: Vec<&str> = s.split('+').collect();
    if parts.len() != 2 {
        bail!("Invalid policy '{}': expected format 'weights+process' (e.g. offload+keep_running)", s);
    }

    let weights = match parts[0] {
        "retain" => WeightStrategy::Retain,
        "offload" => WeightStrategy::Offload,
        "discard" => WeightStrategy::Discard,
        other => bail!("Unknown weight strategy '{}': expected retain, offload, or discard", other),
    };

    let process = match parts[1] {
        "keep_running" => ProcessStrategy::KeepRunning,
        "cuda_suspend" => ProcessStrategy::CudaSuspend,
        "checkpoint" => ProcessStrategy::Checkpoint,
        "stop" => ProcessStrategy::Stop,
        other => bail!("Unknown process strategy '{}': expected keep_running, cuda_suspend, checkpoint, or stop", other),
    };

    Ok(EvictionPolicy { weights, process })
}

/// Run a full validation cycle for the given model.
///
/// If `policies` is `Some`, test those eviction policies.
/// If `None`, test offload+keep_running and discard+keep_running.
///
/// Returns `true` if all tested policies pass, `false` if any fail.
pub async fn run_validation(
    config: &Config,
    model_name: &str,
    policies: Option<&[String]>,
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

    // Test each eviction policy
    let test_policies: Vec<EvictionPolicy> = match policies {
        Some(strs) => strs
            .iter()
            .map(|s| parse_policy(s))
            .collect::<Result<Vec<_>>>()?,
        None => vec![
            EvictionPolicy { weights: WeightStrategy::Offload, process: ProcessStrategy::KeepRunning },
            EvictionPolicy { weights: WeightStrategy::Discard, process: ProcessStrategy::KeepRunning },
        ],
    };
    let mut results = Vec::new();

    for eviction in &test_policies {
        println!("\n--- Testing {:?} ---", eviction);
        let result = test_sleep_level(
            &orchestrator,
            model_name,
            port,
            &model_path,
            *eviction,
            &baseline,
            baseline_gpu,
        )
        .await;

        match result {
            Ok(r) => results.push(r),
            Err(e) => {
                println!("ERROR testing {:?}: {}", eviction, e);
                results.push(LevelResult {
                    eviction: *eviction,
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
    println!("\nStopping model...");
    let _ = orchestrator.sleep_model(model_name, EvictionPolicy::STOP).await;

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

/// Test a single eviction policy: sleep → measure → wake → measure → verify response
async fn test_sleep_level(
    orchestrator: &Arc<Orchestrator>,
    model: &str,
    port: u16,
    model_path: &str,
    eviction: EvictionPolicy,
    baseline: &str,
    _baseline_gpu: u64,
) -> Result<LevelResult> {
    let gpu_before_sleep = query_gpu_memory();

    // Sleep
    let sleep_start = Instant::now();
    orchestrator
        .sleep_model(model, eviction)
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
        eviction,
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

/// Create a CRIU checkpoint for the given model.
///
/// Starts the model, optionally warms it up with one inference request
/// (to compile CUDA graphs and warm the allocator), then checkpoints to disk.
///
/// Returns `true` on success.
pub async fn run_checkpoint(
    config: &Config,
    model_name: &str,
    eviction_str: &str,
    warmup: bool,
) -> Result<bool> {
    let eviction = parse_policy(eviction_str)?;
    if eviction.process != ProcessStrategy::Checkpoint {
        bail!(
            "Process strategy must be 'checkpoint' for --checkpoint, got '{:?}'. \
             Did you mean '{:?}+checkpoint'?",
            eviction.process,
            eviction.weights,
        );
    }

    let model_config = config.models.get(model_name).with_context(|| {
        let available: Vec<_> = config.models.keys().collect();
        format!(
            "Model '{}' not found in config. Available: {:?}",
            model_name, available
        )
    })?;

    if config.checkpoint.is_none() {
        bail!("Checkpoint config required in config.json for --checkpoint");
    }

    let port = model_config.port;
    let model_path = model_config.model_path.clone();

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
    let start = Instant::now();
    orchestrator
        .ensure_running(model_name)
        .await
        .context("Failed to start model")?;
    println!("Model started in {:.1}s", start.elapsed().as_secs_f64());

    // Warmup: run one inference to compile CUDA graphs and warm allocator
    if warmup {
        println!("Running warmup inference...");
        let response = run_deterministic_request(port, &model_path).await?;
        println!("Warmup response: {:?}", response);
    }

    let gpu_before = query_gpu_memory();
    println!("GPU memory before checkpoint: {} MiB", gpu_before);

    // Checkpoint
    println!("Checkpointing with {:?}+{:?}...", eviction.weights, eviction.process);
    let ckpt_start = Instant::now();
    orchestrator
        .sleep_model(model_name, eviction)
        .await
        .context("Checkpoint failed")?;
    let ckpt_secs = ckpt_start.elapsed().as_secs_f64();

    let gpu_after = query_gpu_memory();

    // Report
    let images_dir = config
        .checkpoint
        .as_ref()
        .unwrap()
        .images_dir
        .join(model_name)
        .join("images");
    let image_size = dir_size(&images_dir);

    println!();
    println!("Checkpoint complete:");
    println!("  Time:      {:.1}s", ckpt_secs);
    println!("  Location:  {}", images_dir.display());
    println!("  Size:      {:.1} GB", image_size as f64 / 1_073_741_824.0);
    println!(
        "  GPU freed: {} MiB → {} MiB",
        gpu_before, gpu_after
    );

    Ok(true)
}

/// Restore a model from a CRIU checkpoint on disk.
///
/// Verifies the checkpoint exists, runs CRIU restore, health-checks
/// the endpoint, and runs one inference to verify correctness.
/// The restored vLLM process continues running after llmux exits.
///
/// Returns `true` on success.
pub async fn run_restore(config: &Config, model_name: &str) -> Result<bool> {
    let model_config = config.models.get(model_name).with_context(|| {
        let available: Vec<_> = config.models.keys().collect();
        format!(
            "Model '{}' not found in config. Available: {:?}",
            model_name, available
        )
    })?;

    let ckpt_cfg = config
        .checkpoint
        .as_ref()
        .context("Checkpoint config required in config.json for --restore")?;

    let images_dir = ckpt_cfg.images_dir.join(model_name).join("images");
    if !images_dir.exists() {
        bail!(
            "No checkpoint found at {}. Run --checkpoint {} first.",
            images_dir.display(),
            model_name,
        );
    }

    let port = model_config.port;
    let model_path = model_config.model_path.clone();

    // Create orchestrator with just this model
    let mut models = HashMap::new();
    models.insert(model_name.to_string(), model_config.clone());

    let orchestrator = Arc::new(Orchestrator::with_options(
        models,
        config.vllm_command.clone(),
        config.checkpoint.clone(),
    ));

    // Set initial state to Checkpointed so wake_model runs CRIU restore
    orchestrator
        .set_checkpointed(model_name, images_dir.clone())
        .await
        .context("Failed to set checkpointed state")?;

    // Restore
    println!("Restoring '{}' from {}...", model_name, images_dir.display());
    let start = Instant::now();
    orchestrator
        .wake_model(model_name)
        .await
        .context("Restore failed")?;
    let restore_secs = start.elapsed().as_secs_f64();

    let gpu_after = query_gpu_memory();

    // Verify with inference
    println!("Running verification inference...");
    let response = run_deterministic_request(port, &model_path).await?;

    println!();
    println!("Restore complete:");
    println!("  Time:     {:.1}s", restore_secs);
    println!("  Port:     {}", port);
    println!("  GPU:      {} MiB", gpu_after);
    println!("  Response: {:?}", response);
    println!();
    println!("Model is running on port {}. Kill with: kill $(lsof -ti tcp:{})", port, port);

    Ok(true)
}

/// Calculate total size of a directory in bytes.
fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = entry.metadata();
            if let Ok(m) = meta {
                if m.is_file() {
                    total += m.len();
                } else if m.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn print_results(results: &[LevelResult]) {
    println!();
    println!(
        "{:<20} {:>10} {:>10} {:>12} {:>12} {:>12} {:>10} {:>6}",
        "Policy", "Sleep (s)", "Wake (s)", "GPU Before", "GPU After", "GPU Wake", "Response", "Pass"
    );
    println!("{}", "-".repeat(100));

    for r in results {
        let policy_str = format!("{:?}+{:?}", r.eviction.weights, r.eviction.process);

        println!(
            "{:<20} {:>10.1} {:>10.1} {:>10} MiB {:>10} MiB {:>10} MiB {:>10} {:>6}",
            policy_str,
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
