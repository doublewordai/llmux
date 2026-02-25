//! Axum middleware layer for model switching.
//!
//! Intercepts requests, extracts the model name, ensures the model is ready
//! (triggering a switch if needed), acquires an in-flight guard, and wraps
//! the response body so the guard is held until streaming completes.

use crate::proxy::ProxyTarget;
use crate::switcher::{InFlightGuard, ModelSwitcher};
use crate::types::SwitchError;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use bytes::Bytes;
use futures_util::future::BoxFuture;
use http_body::Frame;
use http_body_util::BodyExt;
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};
use tracing::{debug, error, trace, warn};

/// Layer that adds model switching to a service.
#[derive(Clone)]
pub struct ModelSwitcherLayer {
    switcher: ModelSwitcher,
}

impl ModelSwitcherLayer {
    pub fn new(switcher: ModelSwitcher) -> Self {
        Self { switcher }
    }
}

impl<S> Layer<S> for ModelSwitcherLayer {
    type Service = ModelSwitcherService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ModelSwitcherService {
            switcher: self.switcher.clone(),
            inner,
        }
    }
}

/// Service that wraps requests with model switching.
#[derive(Clone)]
pub struct ModelSwitcherService<S> {
    switcher: ModelSwitcher,
    inner: S,
}

impl<S> Service<Request<Body>> for ModelSwitcherService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let switcher = self.switcher.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let (parts, body) = req.into_parts();

            // Collect body bytes so we can inspect the model field
            let body_bytes = match body.collect().await {
                Ok(collected) => collected.to_bytes(),
                Err(e) => {
                    error!(error = %e, "Failed to read request body");
                    return Ok(error_response(
                        StatusCode::BAD_REQUEST,
                        "Failed to read request body",
                    ));
                }
            };

            let model = extract_model(&body_bytes);

            let Some(model) = model else {
                // No model specified — pass through (health checks, etc.)
                trace!("No model in request, passing through");
                let req = Request::from_parts(parts, Body::from(body_bytes));
                return inner.call(req).await;
            };

            debug!(model = %model, "Extracted model from request");

            if !switcher.is_registered(&model) {
                warn!(model = %model, "Model not registered");
                return Ok(error_response(
                    StatusCode::NOT_FOUND,
                    &format!("Model not found: {}", model),
                ));
            }

            // Ensure model is ready and acquire in-flight guard.
            // If the model starts draining between the ready check and the
            // guard acquisition, loop back so the request waits for the next
            // switch instead of getting a 503.
            let guard = loop {
                if let Err(e) = switcher.ensure_model_ready(&model).await {
                    error!(model = %model, error = %e, "Failed to ensure model ready");
                    return Ok(switch_error_response(e));
                }

                match switcher.acquire_in_flight(&model) {
                    Some(guard) => break guard,
                    None => {
                        debug!(model = %model, "Model draining, re-entering ensure_model_ready");
                        continue;
                    }
                }
            };

            // Set proxy target so the proxy handler knows where to forward
            let port = switcher.model_port(&model).unwrap();
            let mut parts = parts;
            parts.extensions.insert(ProxyTarget { port });

            let req = Request::from_parts(parts, Body::from(body_bytes));

            // Forward to inner service, then wrap the response body so the
            // in-flight guard is held until the full response (including
            // streamed body) is consumed.
            let response = inner.call(req).await?;
            let (resp_parts, body) = response.into_parts();
            let guarded = GuardedBody {
                inner: body,
                _guard: Some(guard),
            };
            Ok(Response::from_parts(resp_parts, Body::new(guarded)))
        })
    }
}

/// Extract model name from the JSON request body.
fn extract_model(body: &Bytes) -> Option<String> {
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(body)
        && let Some(model) = json.get("model").and_then(|v| v.as_str())
    {
        return Some(model.to_string());
    }

    None
}

fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "llmux_error"
        }
    });

    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn switch_error_response(error: SwitchError) -> Response<Body> {
    let (status, message) = match &error {
        SwitchError::ModelNotFound(m) => (StatusCode::NOT_FOUND, format!("Model not found: {}", m)),
        SwitchError::NotReady(m) => (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Model not ready: {}", m),
        ),
        SwitchError::Timeout => (
            StatusCode::GATEWAY_TIMEOUT,
            "Request timed out waiting for model".to_string(),
        ),
        SwitchError::HookFailed { model, detail } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Hook failed for {}: {}", model, detail),
        ),
        SwitchError::Internal(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Internal error: {}", e),
        ),
    };

    error_response(status, &message)
}

/// Response body wrapper that holds an [`InFlightGuard`] until the body is
/// fully consumed. For streaming responses (SSE), this ensures the in-flight
/// count stays accurate until the backend finishes generating.
struct GuardedBody {
    inner: Body,
    _guard: Option<InFlightGuard>,
}

impl http_body::Body for GuardedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_from_body() {
        let body = Bytes::from(r#"{"model": "mistral", "messages": []}"#);
        assert_eq!(extract_model(&body), Some("mistral".to_string()));
    }

    #[test]
    fn test_extract_model_none() {
        let body = Bytes::from(r#"{"messages": []}"#);
        assert_eq!(extract_model(&body), None);
    }
}
