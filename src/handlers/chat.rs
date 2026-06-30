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
    prompt: String,
}

pub async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = mpsc::channel(100);

    let request = InferenceRequest {
        prompt: payload.prompt,
        response_tx: tx,
    };

    let _ = state.engine_tx.send(request).await;

    let stream = async_stream::stream! {
        while let Some(token) = rx.recv().await {
            yield Ok(Event::default().data(token));
        }
    };

    Sse::new(stream)
}