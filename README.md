![loci](.github/assets/logo.svg)

> _A memory-based context proxy for LLMs — semantic memory storage, session-aware retrieval,
> and transparent prompt enrichment._

[![CI](https://github.com/dangoe/loci/actions/workflows/ci.yml/badge.svg)](https://github.com/dangoe/loci/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

---

## What is loci?

**loci** (inspired by the [method of loci](https://en.wikipedia.org/wiki/Method_of_loci)
mnemonic technique) is a Rust library and CLI tool that acts as an intelligent middleware
layer between any client — a user, a script, an agent — and a target LLM.

Instead of every conversation starting from a blank slate, loci:

1. **Retrieves relevant memory entries** from a semantic vector store and injects them into the
   prompt as context before forwarding to the LLM.
2. **Streams model output** back to the caller.

The system is stateless in process — all memory state lives in the configured memory store backend.

```
Client / REPL
     │  prompt
     ▼
┌──────────────────────────────────────┐
│           loci Contextualizer        │
│  1. query MemoryStore (semantic)     │
│  2. inject [MEMORY CONTEXT] block    │
│  3. forward enriched prompt ───────► │  Model Provider
│  ◄────────────────────────────────── │  (Ollama / any model provider)
│  4. stream response                  │
└──────────────────────────────────────┘
     │  response
     ▼
  Client / REPL
```

---

## Current State

> **Note:** loci is under active development. APIs, CLI commands, and trait signatures may change between versions without prior deprecation.

The following is fully implemented and working today.

### Workspace

| Crate                        | Path                                | Purpose                                                                              |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| `loci-core`                  | `crates/loci-core`                  | Traits, domain types, `Contextualizer`                                               |
| `loci-memory-store-qdrant`   | `crates/loci-memory-store-qdrant`   | Qdrant-backed `MemoryStore` with lifecycle-aware retrieval                           |
| `loci-model-provider-ollama` | `crates/loci-model-provider-ollama` | Ollama embedding + text generation model provider                                    |
| `loci-model-provider-openai` | `crates/loci-model-provider-openai` | OpenAI-compatible embedding + text generation model provider                         |
| `loci-config`                | `crates/loci-config`                | TOML config loading and secret resolution                                            |
| `loci-wire`                  | `crates/loci-wire`                  | Runtime wiring: builds concrete store and provider from `AppConfig`                  |
| `loci-server`                | `crates/loci-server`                | `loci-server` binary — OpenAI-compatible and Connect RPC HTTP server                 |
| `loci-cli`                   | `crates/loci-cli`                   | `loci` CLI binary — memory operations, LLM extraction, and contextualized generation |
| `loci-e2e-tests`             | `crates/loci-e2e-tests`             | End-to-end coverage for the CLI and server flows                                     |

### Core Abstractions (`loci-core`)

| Trait                         | Purpose                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `MemoryStore`                 | Add, get, query, promote, delete, and prune expired memory entries |
| `TextEmbedder`                | Embed text into a vector                                           |
| `EmbeddingModelProvider`      | Raw embedding model provider (HTTP, model name)                    |
| `TextGenerationModelProvider` | Raw text generation model provider                                 |

Key domain types: `MemoryEntry`, `MemoryInput`, `MemoryQuery`, `MemoryTrust`, `TrustEvidence`, `MemoryQueryMode`, `Score`, `Embedding`.

### Storage (`loci-memory-store-qdrant`)

`QdrantMemoryStore` uses [Qdrant](https://qdrant.tech/) for cosine-similarity vector search.

Features:

- Configurable deduplication (`similarity_threshold`) to reuse near-duplicates
- Two-variant trust model: `MemoryTrust::Extracted` (Bayesian confidence, subject to decay/discard/promotion) and `MemoryTrust::Fact` (confidence 1.0, no expiry)
- Per-kind TTL defaults and query-time expiry filtering
- Weighted retrieval ranking (`similarity * kind_weight`)
- Manual promotion path via `loci memory promote`
- Metadata filtering (AND semantics, exact match)
- Min score threshold and max result limits

### Memory Extraction (`loci-core`, `loci-cli`)

`MemoryExtractor` turns unstructured text into persisted memory entries using the configured text
model and classifier.

Features:

- `loci memory extract` accepts positional text, repeatable `--file` inputs, or stdin
- Optional sentence-aware chunking from `[memory.extraction.chunking]`
- Per-run overrides for `max_entries`, `min_confidence`, `guidelines`, and attached metadata
- Dual semantic search plus hit classification before insert/merge decisions
- Configurable merge strategy (`best_score` or LLM-based)
- Discard handling for low-confidence candidates and contradictions against existing `Fact` entries

### Model Providers (`loci-model-provider-ollama`, `loci-model-provider-openai`)

`OllamaModelProvider` implements both `EmbeddingModelProvider` and `TextGenerationModelProvider`
against a local [Ollama](https://ollama.com/) instance.

`OpenAIModelProvider` implements the same traits against any OpenAI-compatible HTTP API
(OpenAI, local proxies, etc.). Configure `endpoint` and optionally `api_key` in `config.toml`.

Default models in the generated config:

- Embedding: `qwen3-embedding:0.6b` (768 dimensions)
- Text generation: `qwen3.5:0.8b`

### Server (`loci-server`)

`loci-server` is a standalone HTTP server that exposes loci's memory and generation
capabilities over two APIs:

| Endpoint                           | Protocol          | Description                                                                   |
| ---------------------------------- | ----------------- | ----------------------------------------------------------------------------- |
| `GET  /v1/health`                  | HTTP              | Health check                                                                  |
| `POST /openai/v1/chat/completions` | OpenAI-compatible | Chat completions with automatic memory enrichment; supports streaming via SSE |
| Connect RPC endpoints              | Connect RPC       | Full memory CRUD (`memory.*`) and generate (`generate.*`) services            |

Any OpenAI-compatible client (e.g. Open WebUI, shell scripts using `curl`) can point its
base URL at `http://<host>:<port>/openai` and get transparent memory enrichment without
modification.

**Server flags:**

| Flag             | Env var            | Default                      | Description          |
| ---------------- | ------------------ | ---------------------------- | -------------------- |
| `--config`/`-c`  | `LOCI_CONFIG`      | `~/.config/loci/config.toml` | Path to config file  |
| `--host`         | `LOCI_SERVER_HOST` | `127.0.0.1`                  | Listen address       |
| `--port`         | `LOCI_SERVER_PORT` | `8080`                       | Listen port          |
| `--verbose`/`-v` |                    | off                          | Enable debug logging |

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (2024 edition)
- [Docker](https://www.docker.com/) (for Qdrant)
- [Ollama](https://ollama.com/) — install and run natively (see "Ollama (native)" below)

### Start infrastructure

```bash
docker compose up -d
```

This starts:

- **Qdrant** on `http://localhost:6333` (HTTP) / `http://localhost:6334` (gRPC)

Note: Ollama is not started by the Docker compose setup in this repository. Due to GPU usage constraints Ollama is meant be installed and run natively on your machine (see "Ollama (native)" below).

### Ollama (native)

Ollama must be installed and started natively (it is not started by `docker compose` in this repository due to GPU usage constraints). Install instructions and platform-specific details are available at https://ollama.com/. Once Ollama is running (the default HTTP API is `http://localhost:11434`), pull the required models:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:0.8b
```

### Configure

Generate a default config file:

```bash
cargo run --bin loci -- config init
# Written to: ~/.config/loci/config.toml
```

Edit the file to point to your preferred models. The generated file contains comments
explaining every option.

### Build

```bash
cargo build --release
# Binaries at: target/release/loci  (CLI)
#              target/release/loci-server
```

Or run directly:

```bash
cargo run --bin loci -- <subcommand>
cargo run --bin loci-server -- --host 0.0.0.0 --port 8080
```

---

## CLI Reference

### Global options

| Flag               | Env var       | Default                      | Description              |
| ------------------ | ------------- | ---------------------------- | ------------------------ |
| `--config` / `-c`  | `LOCI_CONFIG` | `~/.config/loci/config.toml` | Path to TOML config file |
| `--verbose` / `-v` |               | off                          | Enable debug logging     |

### `loci memory add`

Add a new memory entry.

```bash
loci memory add "The project uses Qdrant for vector storage"
loci memory add "Deployment target is Kubernetes" --meta env=production --meta team=platform
loci memory add "This is a curated fact" --kind fact --meta source=manual
```

| Argument / Flag                     | Description                                          |
| ----------------------------------- | ---------------------------------------------------- |
| `<content>`                         | Memory text (required positional argument)           |
| `--meta KEY=VALUE`                  | Metadata key-value pair (repeatable)                 |
| `--kind <fact \| extracted-memory>` | Optional kind override (default: `extracted-memory`) |

### `loci memory query`

Retrieve semantically similar memory entries.

```bash
loci memory query "vector database"
loci memory query "deployment" --max-results 3 --min-score 0.7 --filter env=production
loci memory query "platform"
```

| Argument / Flag      | Default      | Description                                 |
| -------------------- | ------------ | ------------------------------------------- |
| `<topic>`            | _(required)_ | Query topic                                 |
| `--max-results <n>`  | `10`         | Maximum number of results                   |
| `--min-score <f64>`  | `0.0`        | Minimum weighted score [0.0, 1.0]           |
| `--filter KEY=VALUE` | _(none)_     | Metadata filter (repeatable, AND semantics) |

### `loci memory get`

Fetch one memory entry by UUID.

```bash
loci memory get <uuid>
```

### `loci memory promote`

Promote a memory entry to `Fact` (confidence 1.0, no expiry).

```bash
loci memory promote <uuid>
```

| Argument / Flag | Description                |
| --------------- | -------------------------- |
| `<uuid>`        | Memory entry ID (required) |

### `loci memory delete`

Remove a memory by UUID.

```bash
loci memory delete <uuid>
```

### `loci memory prune-expired`

Remove **all** expired memory entries from the collection.

```bash
loci memory prune-expired
```

### `loci memory extract`

Extract discrete memory entries from a block of text using the configured LLM and persist them.

**Input sources (mutually exclusive):**

```bash
# Positional string
loci memory extract "The team uses Qdrant for vector storage and Ollama for embeddings."

# File(s) — use - for stdin, repeatable
loci memory extract -f notes.md
loci memory extract -f chapter1.md -f chapter2.md

# Stdin (auto-detected when no other input is given)
cat transcript.txt | loci memory extract
```

| Argument / Flag          | Default      | Description                                                          |
| ------------------------ | ------------ | -------------------------------------------------------------------- |
| `[TEXT]`                 | _(optional)_ | Text to extract from (positional). Mutually exclusive with `--file`. |
| `--file / -f <PATH>`     | _(none)_     | File to read input from. Use `-` for stdin. Repeatable.              |
| `--meta KEY=VALUE`       | _(none)_     | Metadata applied to every extracted entry (repeatable).              |
| `--max-entries <n>`      | _(none)_     | Hard cap on the number of entries extracted.                         |
| `--min-confidence <f64>` | _(none)_     | Discard extracted candidates below this LLM confidence score.        |
| `--guidelines <TEXT>`    | _(none)_     | Free-form instructions appended to the extraction prompt.            |

> **Note:** Chunking and thinking mode are configured in `config.toml` under `[memory.extraction]`,
> not as CLI flags.

**Output:**

```json
{
  "inserted": 2,
  "merged": 1,
  "discarded": 0
}
```

### `loci generate` (alias: `gen`)

Generate a response for a prompt, with optional memory retrieval and contextualization.

```bash
loci gen "What storage backend do we use?"
loci gen "Summarise our deployment setup" --max-memory-entries 8 --min-score 0.5
loci gen "Summarise production deployment" --filters env=production
loci gen "Answer in one paragraph" --system "Be concise." --system-mode replace
```

| Flag                       | Default      | Description                                         |
| -------------------------- | ------------ | --------------------------------------------------- |
| `<prompt>`                 | _(required)_ | Prompt text (positional)                            |
| `--system <TEXT>`          | _(none)_     | Override or extend the default system prompt        |
| `--system-mode <MODE>`     | `append`     | How `--system` interacts with the default prompt    |
| `--max-memory-entries <n>` | `5`          | Max memory entries to inject as context             |
| `--min-score <f64>`        | `0.5`        | Minimum weighted score for context memory entries   |
| `--memory-mode`            | 'auto'       | Memory query mode (`auto` and `off`)                |
| `--filters KEY=VALUE`      | _(none)_     | Metadata filter for retrieved memory (repeatable)   |
| `--debug-flags <FLAG>`     | _(none)_     | Extra debug output (repeatable; currently `memory`) |

### `loci config init`

Scaffold a default configuration file at the config path.

```bash
loci config init
loci --config /path/to/config.toml config init
```

### Key config sections

```toml
[resources.model_providers.ollama]
kind     = "ollama"
endpoint = "http://localhost:11434"

[resources.models.text.default]
provider = "ollama"
model    = "qwen3.5:0.8b"

[resources.models.embedding.default]
provider  = "ollama"
model     = "qwen3-embedding:0.6b"
dimension = 768

[resources.memory_stores.qdrant]
kind       = "qdrant"
url        = "http://localhost:6334"
collection = "memory_entries"

[generation.text]
model = "default"

[embedding]
model = "default"

[memory]
store = "qdrant"
# similarity_threshold = 0.95  # deduplicate by semantic similarity (0.0–1.0)

[memory.extraction]
model = "default"
# max_entries    = 20
# min_confidence = 0.7
# guidelines     = "Focus on technical facts only."
```

---

## Development

```bash
cargo check          # type-check workspace
cargo test           # run workspace tests
cargo clippy         # lint
cargo fmt --check    # formatting check

# Integration tests — requires Docker (Qdrant via testcontainers)
cargo test-it        # shorthand for 'cargo test --features integration,testing -- --test-threads=1'

# E2E tests — requires a running Ollama instance + Docker
cargo test-e2e       # shorthand for 'cargo test --features e2e,testing -- --test-threads=1'

# All tests (unit + integration + e2e)
cargo test-all       # shorthand for 'cargo test --features integration,e2e,testing -- --test-threads=1'
```

### E2E test prerequisites

The E2E tests require a running [Ollama](https://ollama.com) instance with models pulled:

```bash
ollama serve                         # start Ollama (if not already running)
ollama pull qwen3:0.6b               # text generation model
ollama pull qwen3-embedding:0.6b     # embedding model
```

Override models/URL via environment variables:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_TEXT_MODEL` (default: `qwen3:0.6b`)
- `OLLAMA_EMBEDDING_MODEL` (default: `qwen3-embedding:0.6b`)

---

## Roadmap

The items below are **planned** — they are not yet implemented.

### Scanner Integration

With extraction strategies in place, loci can begin ingesting knowledge from external
sources automatically. Scanners provide the raw content; extraction strategies decide
what is worth remembering.

A `Scanner` trait will define a pluggable interface for content sources. Planned built-in
scanners include:

- **Git repository scanner** — extract knowledge from commit history, diffs, READMEs,
  and code comments.
- **File system scanner** — watch directories for new or changed files and feed their
  content through extraction.

Scanners feed content into extraction strategies, which produce memory entries that flow
into the existing memory store with full lifecycle support (deduplication, Bayesian
confidence scoring, contradiction checks against `Fact` entries, and auto-discard).

### Session-Aware Memory Proxy

With rich memory populated by scanners and extraction, the next step is scoping retrieval
and injection per session.

A `SessionStore` trait (pluggable: in-process HashMap, SQLite, Redis, …) keyed by a
session ID. Each session carries configuration (filters, model preferences, context window
size) and a lightweight interaction history. The proxy itself remains stateless — no
per-session state is held in process memory beyond the current request.

### Enhanced REPL CLI

A chat-mode REPL for interactive sessions, building on session-aware memory:

- Multi-turn conversation with persistent session state
- Line editing, input history, and auto-complete (`rustyline` or similar)
- Formatted memory context display with relevance scores
- Session ID management directly from the REPL prompt

### Semantic Knowledge Graph

The long-term vision is to move beyond flat memory entries toward a semantic knowledge
graph that captures relationships between concepts. Building on the volume and variety
of data from scanners, extraction strategies, and multi-session interactions, the
knowledge graph will enable:

- Relationship-aware retrieval (not just similarity, but connectedness)
- Reasoning across related facts from different sources and sessions
- Richer context injection that preserves the structure of knowledge, not just
  individual statements

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy,
development workflow, and guidelines.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Copyright © 2026 Daniel Götten
