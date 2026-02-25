# llmux

Hook-driven LLM model multiplexer in Rust. Routes OpenAI-compatible requests to model backends, switching between them via user-provided shell scripts.

## Commands

```bash
just test    # cargo test
just lint    # cargo fmt --check + cargo clippy -D warnings
just fmt     # cargo fmt
just check   # lint + test
```

## Architecture

~1,700 lines. Axum + Tower middleware.

```
Request → Middleware (extract model from JSON body)
        → Switcher (drain → sleep hook → wake hook)
        → Reverse Proxy (forward to localhost:<port>)
```

Key modules:

- `src/main.rs` — CLI entry point (clap), config loading, server startup
- `src/config.rs` — YAML/JSON config parsing
- `src/hooks.rs` — Executes wake/sleep/alive scripts via `sh -c`
- `src/switcher.rs` — State machine (Idle → Switching → Active), drain logic, in-flight tracking
- `src/middleware.rs` — Tower layer: extract model, ensure ready, acquire in-flight guard
- `src/proxy.rs` — Reverse proxy to `localhost:<port>`
- `src/policy/` — Pluggable switch policy trait (currently FIFO)
- `src/telemetry.rs` — Prometheus metrics setup and descriptions
- `tests/e2e.rs` — End-to-end tests with mock backends

## Config format

YAML or JSON (detected by extension). Each model has a `port` and three hooks (`wake`, `sleep`, `alive`). Hooks run via `sh -c` with `LLMUX_MODEL` env var.

## Telemetry

Prometheus metrics at `GET /metrics` via `metrics` + `metrics-exporter-prometheus` crates. Structured log events via `tracing` for lossless timeline reconstruction.

### Metrics

Switching strategy:

| Metric | Type | Labels |
|--------|------|--------|
| `llmux_switch_total` | counter | `from`, `to`, `result` |
| `llmux_switch_duration_seconds` | histogram | `from`, `to` |
| `llmux_switch_drain_duration_seconds` | histogram | `from`, `to` |
| `llmux_model_active_duration_seconds` | histogram | `model` |
| `llmux_switch_cost_ema_seconds` | gauge | `from`, `to` |
| `llmux_request_queue_depth` | gauge | `model` |
| `llmux_request_queue_wait_seconds` | histogram | `model` |
| `llmux_model_in_flight` | gauge | `model` |

Standard observability (RED):

| Metric | Type | Labels |
|--------|------|--------|
| `llmux_requests_total` | counter | `model`, `status` |
| `llmux_request_duration_seconds` | histogram | `model`, `status` |
| `llmux_hook_duration_seconds` | histogram | `model`, `hook` |
| `llmux_hook_failures_total` | counter | `model`, `hook` |

### Structured log events

All events have an `event` field for Loki queries (`| json | event = "model_activated"`):

- `switch_started` — `from`, `to`
- `model_activated` — `model`, `from`, `duration_secs`
- `switch_failed` — `model`, `from`, `error`
- `sleep_hook_failed` — `model`, `to`, `error`
- `request_timeout` — `model`, `timeout_secs`
- `hook_completed` — `model`, `hook`, `duration_secs`
- `hook_failed` — `model`, `hook`, `exit_code`, `duration_secs`, `stderr`

## Testing

25 tests total: 16 unit + 9 e2e. All mock external boundaries. Fast (~0.5s).

## CI

- `.github/workflows/ci.yaml` — fmt + clippy + test
- `.github/workflows/release-plz.yaml` — crates.io publishing on push to main

## Releasing

Uses [release-plz](https://github.com/release-plz/release-plz). On every push to main, the `release` job publishes any unpublished version to crates.io and creates a GitHub release.

To release a new version:

1. Bump `version` in `Cargo.toml`
2. Run `cargo check` to update `Cargo.lock`
3. Commit **both** `Cargo.toml` and `Cargo.lock`
4. Push to main — release-plz handles the rest

Important: always commit `Cargo.lock` with version bumps. `cargo publish` fails on a dirty lockfile.
