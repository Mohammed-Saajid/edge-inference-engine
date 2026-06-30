use std::path::PathBuf;

#[derive(Clone)]
pub struct AppConfig {
    pub server_address: &'static str,
    pub model_path: PathBuf,
    pub cpu_threads: u32,
    pub channel_capacity: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server_address: "127.0.0.1:3000",
            model_path: PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            cpu_threads: 4,
            channel_capacity: 32,
        }
    }
}
