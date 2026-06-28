mod engine;

use axum::{
    Json, Router,
    extract::State,
    response::sse::{Event, Sse},
    routing::post,
};
use engine::{EdgeEngine, InferenceRequest};
use futures_util::stream::Stream;
use serde::Deserialize;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Deserialize)]
struct ChatRequest {
    prompt: String,
}

struct AppState {
    engine_tx: mpsc::Sender<InferenceRequest>,
}

#[tokio::main]
async fn main() {
    let model_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let cpu_threads = 4;

    let engine = EdgeEngine::new(model_path, cpu_threads);
    let engine_tx = engine.spawn_worker();
    let shared_state = Arc::new(AppState { engine_tx });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = mpsc::channel(100);

    let request = InferenceRequest {
        prompt: payload.prompt,
        response_tx: tx,
    };

    // Forward the work payload to the dedicated C++ execution loop thread
    let _ = state.engine_tx.send(request).await;

    // Convert the receiver channel into an async stream for Server-Sent Events (SSE)
    let stream = async_stream::stream! {
        while let Some(token) = rx.recv().await {
            yield Ok(Event::default().data(token));
        }
    };

    Sse::new(stream)
}
