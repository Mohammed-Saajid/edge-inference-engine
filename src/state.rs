use crate::config::AppConfig;
use crate::engine::InferenceRequest;
use tokio::sync::mpsc;

pub struct AppState {
    pub config: AppConfig,
    pub engine_tx: mpsc::Sender<InferenceRequest>,
}
