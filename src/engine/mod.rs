mod pipeline;
mod worker;
use std::{path::PathBuf, sync::Arc};

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
};
use tokio::sync::mpsc;
pub struct InferenceRequest {
    pub prompt: String,
    pub response_tx: mpsc::Sender<String>,
}

pub struct EdgeEngine {
    model: Arc<LlamaModel>,
    num_threads: u32,
    backend: LlamaBackend,
}

impl EdgeEngine {
    pub fn new(model_path: PathBuf, num_threads: u32) -> Self {
        let backend = LlamaBackend::init().expect("Failed to Initialize Backend");
        let model_params = LlamaModelParams::default();

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .expect("Failed to Initialize Model");

        Self {
            model: Arc::new(model),
            num_threads: num_threads,
            backend: backend,
        }
    }
}
