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
- `tests/e2e.rs` — End-to-end tests with mock backends

## Config format

YAML or JSON (detected by extension). Each model has a `port` and three hooks (`wake`, `sleep`, `alive`). Hooks run via `sh -c` with `LLMUX_MODEL` env var.

## Testing

18 tests total: 11 unit + 7 e2e. All mock external boundaries. Fast (~0.3s).

## CI

- `.github/workflows/ci.yaml` — fmt + clippy + test
- `.github/workflows/release-plz.yaml` — crates.io publishing on push to main
