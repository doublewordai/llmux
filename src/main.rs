//! llmux — Hook-driven LLM model multiplexer
//!
//! Routes requests to model backends and coordinates wake/sleep transitions
//! using user-provided scripts. All lifecycle management is external.

use anyhow::{Context, Result};
use clap::Parser;
use llmux::Config;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "llmux")]
#[command(about = "Hook-driven LLM model multiplexer")]
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let filter = if args.verbose {
        EnvFilter::new("llmux=debug,tower_http=debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();

    info!("Starting llmux");

    let mut config = Config::from_file(&args.config)
        .await
        .with_context(|| format!("Failed to load config from {}", args.config.display()))?;

    if let Some(port) = args.port {
        config.port = port;
    }

    info!(
        models = ?config.models.keys().collect::<Vec<_>>(),
        port = config.port,
        "Configuration loaded"
    );

    let (app, _switcher) = llmux::build_app(config.clone())
        .await
        .context("Failed to build application")?;

    let addr = format!("0.0.0.0:{}", config.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    info!(addr = %addr, "Listening for requests");

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}
