# loci

> _A memory-based context proxy for LLMs — semantic memory storage, session-aware retrieval,
> and transparent prompt enrichment._

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## What is loci?

**loci** (inspired by the [method of loci](https://en.wikipedia.org/wiki/Method_of_loci)
mnemonic technique) is a Rust library and CLI tool that acts as an intelligent middleware
layer between any client — a user, a script, an agent — and a target LLM.

Instead of every conversation starting from a blank slate, loci:

1. **Associates each conversation with a session** identified by a session ID, looked up
   from a pluggable session store.
2. **Retrieves relevant memories** from a semantic vector store and injects them into the
   prompt as context before forwarding to the LLM.
3. **Extracts new memories asynchronously** from each prompt/response pair so the system
   continuously learns from interactions.
4. **Applies eviction strategies** to keep the memory store focused and within bounds.

The proxy itself is stateless — all session and memory state lives in external stores,
making it easy to scale horizontally or swap backends.

```
Client / REPL
     │  prompt + session_id
     ▼
┌──────────────────────────────────────┐
│           loci Proxy                 │
│  1. load session from SessionStore   │
│  2. query MemoryStore (semantic)     │
│  3. inject [MEMORY CONTEXT] block    │
│  4. forward enriched prompt ───────► │  Target LLM
│  ◄────────────────────────────────── │  (Ollama / any backend)
│  5. stream / return response         │
│  6. extract memories  (async)        │
│  7. apply eviction strategies        │
│  8. persist updated session          │
└──────────────────────────────────────┘
     │  response
     ▼
  Client / REPL
```

---

## Current State

The following is fully implemented and working today.

### Workspace

| Crate | Path | Purpose |
|---|---|---|
| `loci-core` | `crates/loci-core` | Traits, domain types, `Contextualizer` |
| `loci-memory-store-qdrant` | `crates/loci-memory-store-qdrant` | Qdrant-backed `MemoryStore` with deduplication |
| `loci-backend-ollama` | `crates/loci-backend-ollama` | Ollama embedding + text generation backend |
| `loci-cli` | `crates/loci-cli` | `loci` CLI binary for CRUD + prompt enhancement |

### Core Abstractions (`loci-core`)

| Trait | Purpose |
|---|---|
| `MemoryStore` | Save, query, update, delete, clear memories |
| `TextEmbedder` | Embed text into a vector |
| `EmbeddingBackend` | Raw embedding backend (HTTP, model name) |
| `TextGenerationBackend` | Raw text generation backend |

Key domain types: `Memory`, `MemoryEntry`, `MemoryInput`, `MemoryQuery`, `Score`, `Embedding`.

The `Contextualizer` retrieves relevant memories and prepends a `[MEMORY CONTEXT]` block to
the user's prompt before calling the LLM.

### Storage (`loci-storage-qdrant`)

`QdrantMemoryStore` uses [Qdrant](https://qdrant.tech/) for cosine-similarity vector search.

Features:
- Configurable deduplication (similarity threshold — reuses existing memory ID instead of
  creating a duplicate)
- Metadata filtering (AND semantics, exact match)
- Min/max score thresholds and max result limits

### Backends (`loci-backend-ollama`)

`OllamaBackend` implements both `EmbeddingBackend` and `TextGenerationBackend` against a
local [Ollama](https://ollama.com/) instance.

Default models used by the CLI:
- Embedding: `nomic-embed-text` (768 dimensions)
- Text generation: `qwen3:0.6b`

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (2024 edition)
- [Docker](https://www.docker.com/) (for Qdrant and Ollama)

### Start infrastructure

```bash
docker compose up -d
```

This starts:
- **Qdrant** on `http://localhost:6334` (gRPC) / `http://localhost:6333` (HTTP)
- **Ollama** on `http://localhost:11434`

Pull the required Ollama models once:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:0.6b
```

### Build

```bash
cargo build --release
# Binary at: target/release/loci
```

Or run directly:

```bash
cargo run --bin loci -- <subcommand>
```

---

## CLI Reference

### Global options

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--qdrant-url` | `QDRANT_URL` | `http://localhost:6334` | Qdrant gRPC endpoint |
| `--ollama-url` | `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `--collection` | `COLLECTION_NAME` | `memories` | Qdrant collection name |
| `--similarity-threshold` | `SIMILARITY_THRESHOLD` | _(none)_ | Deduplication threshold [0.0, 1.0] |

### `loci save`

Store a new memory.

```bash
loci save --content "The project uses Qdrant for vector storage"
loci save --content "Deployment target is Kubernetes" --meta env=production --meta team=platform
```

| Flag | Description |
|---|---|
| `--content <text>` | Memory text (required) |
| `--meta KEY=VALUE` | Metadata key-value pair (repeatable) |

### `loci query`

Retrieve semantically similar memories.

```bash
loci query --topic "vector database"
loci query --topic "deployment" --max-results 3 --min-score 0.7 --filter env=production
```

| Flag | Default | Description |
|---|---|---|
| `--topic <text>` | _(required)_ | Query topic |
| `--max-results <n>` | `10` | Maximum number of results |
| `--min-score <f64>` | `0.0` | Minimum similarity score [0.0, 1.0] |
| `--filter KEY=VALUE` | _(none)_ | Metadata filter (repeatable, AND semantics) |

### `loci update`

Update the content or metadata of an existing memory by UUID.

```bash
loci update --id <uuid> --content "Updated content" --meta key=value
```

### `loci delete`

Remove a memory by UUID.

```bash
loci delete --id <uuid>
```

### `loci clear`

Remove **all** memories from the collection.

```bash
loci clear
```

### `loci prompt`

Enhance a prompt with relevant memories and send it to the LLM.

```bash
loci prompt "What storage backend do we use?"
loci prompt "Summarise our deployment setup" --llm-model llama3.2 --max-memories 8
```

| Flag | Env var | Default | Description |
|---|---|---|---|
| `<prompt>` | — | _(required)_ | Prompt text (positional) |
| `--llm-model <name>` | `LLM_MODEL` | `qwen3:0.6b` | Ollama model for generation |
| `--max-memories <n>` | — | `5` | Max memories to inject as context |
| `--min-score <f64>` | — | `0.0` | Minimum similarity score for context memories |

---

## Development

```bash
cargo check          # type-check workspace
cargo test           # run all unit tests
cargo clippy         # lint
cargo fmt            # format

# Integration tests (requires Docker)
cargo test -p loci-memory-store-qdrant -- --ignored --test-threads=1
```

---

## Roadmap

The items below are **planned** — they are not yet implemented.

### Session-Aware Memory Proxy

A `SessionStore` trait (pluggable: in-process HashMap, SQLite, Redis, …) keyed by a
session ID. Each session carries configuration (filters, model preferences, context window
size) and a lightweight interaction history. The proxy itself remains stateless — no
per-session state is held in process memory beyond the current request.

### Memory Extraction Strategies

After each LLM turn, the prompt/response pair is processed asynchronously by a
`MemoryExtractionStrategy`. Two built-in strategies are planned:

- **`LlmSummarizationStrategy`** — sends the pair to the LLM with a system prompt that
  extracts factual statements as new memories.
- **`KeywordEntityStrategy`** — lightweight keyword and entity extraction that does not
  require an additional LLM call.

The trait is open for extension; custom strategies can be plugged in.

### Memory Eviction Strategies

An `EvictionStrategy` trait applied after extraction to keep the memory store healthy:

| Strategy | Description |
|---|---|
| **TTL** | Expire memories older than a configurable duration |
| **Max-Count / LRU** | Keep only the _N_ most recently accessed memories |
| **Score-based** | Evict memories whose relevance score falls below a threshold |

Strategies are composable — multiple strategies can be applied in sequence.

### Enhanced REPL CLI

A chat-mode REPL for interactive sessions:

- Multi-turn conversation with persistent session state
- Line editing, input history, and auto-complete (`rustyline` or similar)
- Formatted memory context display with relevance scores
- Session ID management directly from the REPL prompt

### Protocol Layer

The long-term vision is to expose loci as a network proxy that any client can target
without modification. The specific protocol is undecided; an **OpenAI-compatible API** is
the leading candidate for maximum client interoperability.

---

## Contributing

Contributions are welcome! Please open an issue to discuss significant changes before
submitting a pull request.

```bash
# Fork, clone, create a feature branch
git checkout -b feat/my-feature

# Make changes, ensure tests pass
cargo test && cargo clippy

# Open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE).

Copyright © 2025 Daniel Götten
