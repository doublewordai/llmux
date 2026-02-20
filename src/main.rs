//! llmux - Zero-reload model switching for vLLM
//!
//! This binary manages multiple vLLM models on shared GPU(s), starting them
//! lazily and coordinating wake/sleep to allow seamless model switching.

use anyhow::{Context, Result};
use clap::Parser;
use llmux::Config;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "llmux")]
#[command(about = "Zero-reload model switching for vLLM")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.json")]
    config: PathBuf,

    /// Port to listen on (overrides config)
    #[arg(short, long)]
    port: Option<u16>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Run a sleep/wake validation cycle for the given model and exit
    #[arg(long, value_name = "MODEL", conflicts_with_all = ["checkpoint", "restore_detached"])]
    validate: Option<String>,

    /// Eviction policies to validate (default: offload+keep_running,discard+keep_running).
    /// Format: weights+process, comma-separated.
    /// Example: offload+keep_running,retain+cuda_suspend
    #[arg(
        long,
        value_name = "POLICIES",
        value_delimiter = ',',
        requires = "validate"
    )]
    policies: Vec<String>,

    /// Create a CRIU checkpoint for the given model and exit.
    /// Starts the model, warms it up, then checkpoints to disk.
    #[arg(long, value_name = "MODEL", conflicts_with_all = ["validate", "restore_detached"])]
    checkpoint: Option<String>,

    /// Restore a model from a CRIU checkpoint, verify health, and exit.
    /// The restored vLLM process keeps running after llmux exits.
    #[arg(long, value_name = "MODEL", conflicts_with_all = ["validate", "checkpoint"])]
    restore_detached: Option<String>,

    /// Eviction policy for --checkpoint (default: discard+checkpoint).
    /// Only the weight strategy matters — process is always checkpoint.
    #[arg(long, value_name = "POLICY", requires = "checkpoint")]
    eviction: Option<String>,

    /// Skip warmup inference before checkpointing
    #[arg(long, requires = "checkpoint")]
    no_warmup: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging.
    // vLLM process output is logged under the "vllm" target at debug level,
    // so it can be enabled with e.g. RUST_LOG=info,vllm=debug.
    let filter = if args.verbose {
        EnvFilter::new("llmux=debug,onwards=debug,tower_http=debug,vllm=debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();

    info!("Starting llmux");

    // Load configuration
    let mut config = Config::from_file(&args.config)
        .await
        .with_context(|| format!("Failed to load config from {}", args.config.display()))?;

    // Override port if specified
    if let Some(port) = args.port {
        config.port = port;
    }

    // Validate configuration (warns about misconfigurations)
    config.validate();

    info!(
        models = ?config.models.keys().collect::<Vec<_>>(),
        port = config.port,
        "Configuration loaded"
    );

    // Run validation if --validate is specified
    if let Some(model_name) = args.validate {
        let policies = if args.policies.is_empty() {
            None
        } else {
            Some(args.policies)
        };
        let success =
            llmux::validate::run_validation(&config, &model_name, policies.as_deref()).await?;
        std::process::exit(if success { 0 } else { 1 });
    }

    // Create checkpoint if --checkpoint is specified
    if let Some(model_name) = args.checkpoint {
        let eviction = args.eviction.as_deref().unwrap_or("discard+checkpoint");
        let success =
            llmux::validate::run_checkpoint(&config, &model_name, eviction, !args.no_warmup)
                .await?;
        std::process::exit(if success { 0 } else { 1 });
    }

    // Restore from checkpoint if --restore-detached is specified
    if let Some(model_name) = args.restore_detached {
        let success = llmux::validate::run_restore(&config, &model_name).await?;
        std::process::exit(if success { 0 } else { 1 });
    }

    // Build the application
    let (app, metrics_router, control_router, switcher) = llmux::build_app(config.clone())
        .await
        .context("Failed to build application")?;

    // Run warmup phase if enabled
    if config.warmup {
        llmux::run_warmup(&switcher)
            .await
            .context("Warmup phase failed")?;
    }

    // Spawn metrics server if enabled
    if let Some(metrics_router) = metrics_router {
        let metrics_addr = format!("0.0.0.0:{}", config.metrics_port);
        let metrics_listener = TcpListener::bind(&metrics_addr)
            .await
            .with_context(|| format!("Failed to bind metrics to {}", metrics_addr))?;
        info!(addr = %metrics_addr, "Serving metrics");
        tokio::spawn(async move {
            if let Err(e) = axum::serve(metrics_listener, metrics_router).await {
                tracing::error!(error = %e, "Metrics server error");
            }
        });
    }

    // Spawn admin/control API server if enabled
    if let Some(admin_port) = config.admin_port {
        let admin_addr = format!("0.0.0.0:{}", admin_port);
        let admin_listener = TcpListener::bind(&admin_addr)
            .await
            .with_context(|| format!("Failed to bind admin API to {}", admin_addr))?;
        info!(addr = %admin_addr, "Serving control API");
        tokio::spawn(async move {
            if let Err(e) = axum::serve(admin_listener, control_router).await {
                tracing::error!(error = %e, "Admin server error");
            }
        });
    }

    // Start server
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    info!(addr = %addr, "Listening for requests");

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}
