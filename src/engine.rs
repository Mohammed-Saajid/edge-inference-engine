use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct InferenceRequest {
    pub prompt: String,
    pub response_tx: mpsc::Sender<String>,
}

pub struct EdgeEngine {
    model: Arc<LlamaModel>,
    num_threads: u32,
    backend: llama_backend::LlamaBackend,
}

impl EdgeEngine {
    pub fn new(model_path: PathBuf, num_threads: u32) -> Self {
        let backend =
            llama_backend::LlamaBackend::init().expect("Failed to initialize llama.cpp backend");

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .expect("Failed to load GGUF model file");

        Self {
            model: Arc::new(model),
            num_threads,
            backend,
        }
    }

    pub fn spawn_worker(self) -> mpsc::Sender<InferenceRequest> {
        let (tx, mut rx) = mpsc::channel::<InferenceRequest>(32);
        let model = self.model.clone();
        let threads = self.num_threads;

        std::thread::spawn(move || {
            while let Some(request) = rx.blocking_recv() {
                let ctx_params = LlamaContextParams::default().with_n_threads(threads as i32);

                let mut ctx = model
                    .new_context(&self.backend, ctx_params)
                    .expect("Failed to create inference context");

                // Initialize a stateful UTF-8 decoder for partial byte streams
                let mut decoder = UTF_8.new_decoder();

                // Updated Tokenization API
                let prompt_tokens = model
                    .str_to_token(&request.prompt, AddBos::Always)
                    .expect("Failed to convert string to tokens");

                // Initialize a batch to hold the tokens for evaluation
                let mut batch = LlamaBatch::new(512, 1);
                let last_index = prompt_tokens.len() - 1;

                // Add all prompt tokens to the batch
                for (i, &token) in prompt_tokens.iter().enumerate() {
                    let is_last = i == last_index;
                    batch
                        .add(token, i as i32, &[0], is_last)
                        .expect("Failed to add to batch");
                }
                // Prefill phase: Evaluate the entire prompt batch to populate the KV cache
                ctx.decode(&mut batch).expect("Failed to decode prompt");

                let mut decoder = UTF_8.new_decoder();
                let mut current_token_count = prompt_tokens.len();
                let max_tokens = 128;

                // Initialize a simple sampler chain using greedy decoding
                let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

                // Decode phase: Autoregressive generation loop
                while current_token_count < max_tokens {
                    // Sample the next token directly using the new API.
                    // This automatically evaluates the logits of the last token in the batch
                    // and updates the sampler's internal state.
                    let new_token_id = sampler.sample(&ctx, batch.n_tokens() - 1);

                    // Break if the model outputs an End-Of-Generation/Sequence token
                    if model.is_eog_token(new_token_id) {
                        break;
                    }

                    // Decode the raw token ID into a string piece safely
                    let token_str = model
                        .token_to_piece(new_token_id, &mut decoder, true, None)
                        .unwrap_or_default();

                    // Stream the generated token back to the async router
                    if request.response_tx.blocking_send(token_str).is_err() {
                        break; // Connection closed by client
                    }

                    // Prepare the batch for the next iteration containing only the newly generated token
                    batch.clear();
                    batch
                        .add(new_token_id, current_token_count as i32, &[0], true)
                        .expect("Failed to batch next token");

                    // Evaluate the single new token to update the KV cache and generate the next logits
                    ctx.decode(&mut batch).expect("Failed to decode next token");

                    current_token_count += 1;
                }
            }
        });

        tx
    }
}
