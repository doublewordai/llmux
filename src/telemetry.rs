//! Prometheus metrics setup and descriptions.
//!
//! Metrics are recorded throughout the codebase using the `metrics` crate's
//! macros. This module installs the Prometheus exporter and registers
//! human-readable descriptions for each metric.

use metrics::{describe_counter, describe_gauge, describe_histogram};
use metrics_exporter_prometheus::PrometheusHandle;

/// Install the Prometheus recorder and register metric descriptions.
///
/// Returns `None` if a recorder is already installed (e.g. in tests where
/// multiple `build_app` calls share a process). Metric recording still works
/// — the `metrics` macros route to whichever recorder was installed first.
pub fn install() -> Option<PrometheusHandle> {
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .ok()?;
    describe();
    Some(handle)
}

fn describe() {
    // -- Switching strategy metrics --
    describe_counter!("llmux_switch_total", "Total model switches attempted");
    describe_histogram!(
        "llmux_switch_duration_seconds",
        "Wall-clock duration of model switches (drain + sleep + wake)"
    );
    describe_histogram!(
        "llmux_switch_drain_duration_seconds",
        "Time spent draining in-flight requests before a switch"
    );
    describe_histogram!(
        "llmux_model_active_duration_seconds",
        "How long a model stayed active before being switched out"
    );
    describe_gauge!(
        "llmux_switch_cost_ema_seconds",
        "EMA-smoothed estimated switch cost per (from, to) pair"
    );
    describe_gauge!(
        "llmux_request_queue_depth",
        "Number of requests waiting for a model to become active"
    );
    describe_histogram!(
        "llmux_request_queue_wait_seconds",
        "Time a request spent queued waiting for model readiness"
    );
    describe_gauge!(
        "llmux_model_in_flight",
        "Current in-flight requests per model"
    );

    // -- Standard observability metrics (RED) --
    describe_counter!("llmux_requests_total", "Total requests processed");
    describe_histogram!(
        "llmux_request_duration_seconds",
        "End-to-end request duration (time to first byte)"
    );

    // -- Hook metrics --
    describe_histogram!("llmux_hook_duration_seconds", "Hook script execution time");
    describe_counter!("llmux_hook_failures_total", "Total hook script failures");
}
