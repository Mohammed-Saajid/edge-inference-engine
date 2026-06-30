use tokio::sync::mpsc;

use crate::engine::{EdgeEngine, InferenceRequest};

impl EdgeEngine {
    pub fn spawn_worker(self, capacity: usize) -> mpsc::Sender<InferenceRequest> {
        let (tx, mut rx) = mpsc::channel::<InferenceRequest>(capacity);
        let model = self.model.clone();
        let threads = self.num_threads;

        std::thread::spawn(move || {
            while let Some(request) = rx.blocking_recv() {
                if let Err(e) =
                    super::pipeline::execute_inference(&self.backend, &model, request, threads)
                {
                    eprintln!("Inference generation error loop step failed: {:?}", e);
                }
            }
        });
        tx
    }
}
