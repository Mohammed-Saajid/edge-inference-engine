mod config;
mod engine;
mod handlers;
mod state;

use axum::{Router, routing::post};
use config::AppConfig;
use engine::EdgeEngine;
use state::AppState;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let config = AppConfig::default();

    // Initialize Compute Subsystem
    let engine = EdgeEngine::new(config.model_path.clone(), config.cpu_threads);
    let engine_tx = engine.spawn_worker(config.channel_capacity);

    // Encapsulate Central Application State Context
    let shared_state = Arc::new(AppState {
        config: config,
        engine_tx,
    });

    // Bind HTTP Architecture Layers
    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::handle_chat))
        .with_state(shared_state.clone());

    let listener = tokio::net::TcpListener::bind(shared_state.config.server_address)
        .await
        .unwrap();

    println!(
        "Server operating dynamically at http://{}",
        shared_state.config.server_address
    );
    axum::serve(listener, app).await.unwrap();
}
