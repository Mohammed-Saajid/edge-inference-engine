use encoding_rs::UTF_8;
use llama_cpp_2::{
    context::{params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
};

use crate::engine::InferenceRequest;

pub fn execute_inference(
    backend: &LlamaBackend,
    model: &LlamaModel,
    request: InferenceRequest,
    threads: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx_params = LlamaContextParams::default().with_n_threads(threads as i32);

    let mut ctx = model.new_context(backend, ctx_params)?;

    let mut decoder = UTF_8.new_decoder();
    let prompt_tokens = model.str_to_token(&request.prompt, AddBos::Always)?;

    let mut batch = LlamaBatch::new(512, 1);

    let last_index = prompt_tokens.len() - 1;

    for (i, &token) in prompt_tokens.iter().enumerate() {
        let is_last = i == last_index;
        batch.add(token, i as i32, &[0], is_last)?;
    }

    // prefill phase:
    ctx.decode(&mut batch)?;

    let mut current_token = prompt_tokens.len();
    let max_tokens = 128;
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    while current_token < max_tokens {
        let new_token_id = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(new_token_id) {
            break;
        }
        let token_str = model
            .token_to_piece(new_token_id, &mut decoder, true, None)
            .unwrap_or_default();

        if request.response_tx.blocking_send(token_str).is_err() {
            break;
        }
        batch.clear();
        batch.add(new_token_id, current_token as i32, &[0], true)?;

        ctx.decode(&mut batch)?;

        current_token += 1;
    }

    Ok(())
}
