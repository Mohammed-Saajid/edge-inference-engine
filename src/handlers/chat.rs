use crate::engine::InferenceRequest;
use crate::state::AppState;
use axum::{
    Json,
    extract::State,
    response::sse::{Event, Sse},
};
use futures_util::stream::Stream;
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

pub async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = mpsc::channel(100);

    let request = InferenceRequest {
        prompt: payload.prompt,
        response_tx: tx,
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        max_tokens: payload.max_tokens.unwrap_or(128),
    };

    let _ = state.engine_tx.send(request).await;

    let stream = async_stream::stream! {
        while let Some(token) = rx.recv().await {
            yield Ok(Event::default().data(token));
        }
    };

    Sse::new(stream)
}
