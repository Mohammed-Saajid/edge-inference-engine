# Edge LLM Engine

A lightweight, local inference server for running GGUF language models at the edge with Rust, `llama-cpp-2`, and Axum.

The service exposes a single streaming endpoint compatible with a chat-style request body and returns generated tokens as Server-Sent Events (SSE).

---

## Features

* **Local Inference:** Fully offline text generation using native `llama.cpp` bindings (`llama-cpp-2`).
* **Low Latency:** Real-time token streaming over HTTP Server-Sent Events (SSE).
* **Actor-Based Architecture:** Dedicated CPU inference worker thread decoupled from the async HTTP request handling pool to prevent thread starvation.
* **Minimal API Surface:** Drop-in simplicity for edge devices and local integrations.

## Tech Stack

* **Language:** Rust (Edition 2024)
* **Web Framework:** Axum + Tokio Asynchronous Runtime
* **Inference Engine:** llama-cpp-2 (C++ bindings)
* **Generators:** async-stream

## Project Structure

```text
.
├── Cargo.toml
├── README.md
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
└── src/
    ├── main.rs      # HTTP server topology & SSE streaming routes
    └── engine.rs    # Model lifecycle management & C++ worker loop

```

---

## Requirements

* **Rust Toolchain:** Stable channel (v1.80+ recommended).
* **GGUF Model File:** The engine targets a specific quantized model structure out of the box. Ensure your model file is placed exactly at:
```text
models/qwen2.5-0.5b-instruct-q4_k_m.gguf

```



---

## Getting Started

1. Clone the repository and navigate to the project root.
2. Ensure the required `.gguf` file is present in the `models/` directory.
3. Compile and launch the release server:
```bash
cargo run --release

```



The server binds to the local loopback interface: `http://127.0.0.1:3000`

---

## Architecture & How It Works

The engine relies on a strict separation of concerns between I/O bound tasks and CPU-bound tensor math:

1. **Initialization:** `main.rs` initializes the global `llama.cpp` backend context, creates a cross-thread message pipeline via an `mpsc` channel, and mounts the Axum router.
2. **Work Offloading:** Incoming HTTP `POST` requests are intercepted by Tokio worker green-threads. The payload is validated, packaged with a private response channel, and dropped into the global channel buffer before yielding control back to the network engine.
3. **Execution Loop:** `engine.rs` operates a dedicated native OS background thread running a synchronous processing loop. It pulls requests out of the channel, tokenizes the prompt, populates the KV cache via a prefill pass, and enters an autoregressive greedy decoding loop.
4. **Reactive Streaming:** Generated raw string pieces are immediately sent back to the awaiting Axum task via the private response channel. The handler wraps these tokens in an SSE abstraction, flushing chunks over the active TCP connection as they generate.

---

## API Specification

### POST /v1/chat/completions

**Headers:**

* `Content-Type: application/json`

**Request Body:**

```json
{
    "prompt": "Write a short haiku about edge AI.",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 128
}
```

**Request Parameters:**

| Parameter | Type | Optional | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | No | — | The input text to complete |
| `temperature` | float | Yes | 0.7 | Sampling temperature (0.0 = deterministic, higher = more random) |
| `top_p` | float | Yes | 0.9 | Nucleus sampling parameter |
| `max_tokens` | integer | Yes | 128 | Maximum tokens to generate |

**Response Type:** `text/event-stream`

### Example Request & Response Tracing

Execute a raw streaming request using `curl` (the `-N` flag disables output buffering):

```bash
curl -N -X POST http://127.0.0.1:3000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "Explain edge inference in 3 words.",
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 256
    }'
```

**Expected Stream Output:**

```text
data: Local
data: compute
data: power

```

---

## Current Runtime Configuration

The following parameters are configured during initialization:

* **Model Path:** `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
* **Network Interface:** `127.0.0.1:3000`
* **Matrix Math Allocation:** `4` CPU threads via `LlamaContextParams`
* **Channel Capacity:** `32` pending inference requests

**Request Generation Defaults:**

* **Temperature:** `0.7`
* **Top-p:** `0.9`
* **Max Tokens:** `128`

## System Limitations

* **Single Worker Thread:** Concurrently submitted prompts are processed sequentially via the global pipeline queue.
* **Zero Authentication:** No built-in bearer tokens or API key validation middleware.
* **Static Configurations:** Any updates to threads, model paths, or targets require recompilation.
* **Minimal Observability:** Lacks explicit structured logging (`tracing`) and performance telemetry metrics.

---

## Roadmap

* [ ] Implement full OpenAI-compatible request/response schemas (`/v1/chat/completions` compliant fields).
* [ ] Move system constants to environment variables or command-line flags (`clap`).
* [ ] Add dedicated health checking (`/healthz`) and system telemetry tracking.
* [ ] Integrate asynchronous structured logging and tracing spans.