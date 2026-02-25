//! Reverse proxy for forwarding requests to model backends.

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use tracing::error;

/// Target port for a proxied request, set by the middleware as a request extension.
#[derive(Clone, Copy)]
pub struct ProxyTarget {
    pub port: u16,
}

/// Shared state for the proxy handler.
#[derive(Clone)]
pub struct ProxyState {
    client: Client<HttpConnector, Body>,
}

impl Default for ProxyState {
    fn default() -> Self {
        let client = Client::builder(TokioExecutor::new()).build_http();
        Self { client }
    }
}

impl ProxyState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Axum fallback handler that forwards requests to model backends.
///
/// Reads the [`ProxyTarget`] extension (set by the model switcher middleware)
/// to determine which localhost port to forward to.
pub async fn proxy_handler(State(state): State<ProxyState>, req: Request<Body>) -> Response<Body> {
    let target = req.extensions().get::<ProxyTarget>().copied();

    match target {
        Some(ProxyTarget { port }) => match forward(state.client, req, port).await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "Proxy error");
                error_response(StatusCode::BAD_GATEWAY, &format!("Backend error: {}", e))
            }
        },
        None => error_response(StatusCode::NOT_FOUND, "No model specified in request"),
    }
}

async fn forward(
    client: Client<HttpConnector, Body>,
    mut req: Request<Body>,
    port: u16,
) -> Result<Response<Body>, hyper_util::client::legacy::Error> {
    // Rewrite URI to target backend
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.to_string())
        .unwrap_or_else(|| "/".to_string());

    let uri: Uri = format!("http://127.0.0.1:{}{}", port, path_and_query)
        .parse()
        .expect("valid proxy URI");

    *req.uri_mut() = uri;
    req.headers_mut().remove("host");

    let resp = client.request(req).await?;
    let (parts, body) = resp.into_parts();
    Ok(Response::from_parts(parts, Body::new(body)))
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
