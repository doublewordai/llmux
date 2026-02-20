use super::Orchestrator;
use std::time::Duration;
use tracing::warn;

impl Orchestrator {
    /// Helper to make POST requests with retries.
    ///
    /// All vLLM control endpoints (wake_up, sleep, collective_rpc, etc.) are
    /// idempotent, so retries are safe. We retry on transient failures (connection
    /// errors, timeouts, 5xx) to avoid escalating to drastic measures (like
    /// killing the process) when the endpoint is just momentarily unresponsive.
    pub(super) async fn post_request(
        &self,
        url: &str,
        body: Option<&str>,
        timeout: Duration,
    ) -> Result<(), String> {
        use http_body_util::{BodyExt, Full};
        use hyper::Request;

        let client: hyper_util::client::legacy::Client<_, Full<bytes::Bytes>> =
            hyper_util::client::legacy::Client::builder(hyper_util::rt::TokioExecutor::new())
                .build_http();

        let uri: hyper::Uri = url.parse().map_err(|e| format!("Invalid URL: {}", e))?;
        let has_body = body.is_some();
        let body_bytes: Vec<u8> = body.map(|b| b.as_bytes().to_vec()).unwrap_or_default();

        let mut last_err = String::new();
        for attempt in 0..3u32 {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * 2u64.pow(attempt - 1));
                warn!(url, attempt, delay = ?delay, "Retrying request");
                tokio::time::sleep(delay).await;
            }

            let request_body = Full::new(bytes::Bytes::from(body_bytes.clone()));
            let mut req_builder = Request::builder().method("POST").uri(uri.clone());
            if has_body {
                req_builder = req_builder.header("Content-Type", "application/json");
            }

            let request = match req_builder.body(request_body) {
                Ok(r) => r,
                Err(e) => return Err(format!("Failed to build request: {}", e)),
            };

            match tokio::time::timeout(timeout, client.request(request)).await {
                Ok(Ok(response)) if response.status().is_success() => return Ok(()),
                Ok(Ok(response)) => {
                    let status = response.status();
                    let body_str = match response.into_body().collect().await {
                        Ok(collected) => {
                            let bytes = collected.to_bytes();
                            String::from_utf8_lossy(&bytes).into_owned()
                        }
                        Err(e) => format!("(failed to read body: {e})"),
                    };
                    let body_preview = if body_str.len() > 2000 {
                        format!(
                            "{}...[truncated, {} total]",
                            &body_str[..2000],
                            body_str.len()
                        )
                    } else {
                        body_str
                    };
                    warn!(url, %status, body = %body_preview, "vLLM endpoint returned error");
                    last_err = format!("HTTP {status}: {body_preview}");
                }
                Ok(Err(e)) => {
                    last_err = format!("Request failed: {}", e);
                }
                Err(_) => {
                    last_err = "Request timeout".to_string();
                }
            }
        }

        Err(last_err)
    }
}
