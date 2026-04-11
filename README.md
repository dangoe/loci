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

The following is fully implemented and working today.

### Workspace

| Crate                        | Path                                | Purpose                                                    |
| ---------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| `loci-core`                  | `crates/loci-core`                  | Traits, domain types, `Contextualizer`                     |
| `loci-memory-store-qdrant`   | `crates/loci-memory-store-qdrant`   | Qdrant-backed `MemoryStore` with lifecycle-aware retrieval |
| `loci-model-provider-ollama` | `crates/loci-model-provider-ollama` | Ollama embedding + text generation model provider          |
| `loci-config`                | `crates/loci-config`                | TOML config loading and secret resolution                  |
| `loci-cli`                   | `crates/loci-cli`                   | `loci` CLI binary for CRUD + prompt enhancement            |

### Core Abstractions (`loci-core`)

| Trait                         | Purpose                                                             |
| ----------------------------- | ------------------------------------------------------------------- |
| `MemoryStore`                 | Add, get, query, update, set tier, delete, prune expired memory entries |
| `TextEmbedder`                | Embed text into a vector                                            |
| `EmbeddingModelProvider`      | Raw embedding model provider (HTTP, model name)                     |
| `TextGenerationModelProvider` | Raw text generation model provider                                  |

Key domain types: `MemoryEntry`, `MemoryQueryResult`, `MemoryInput`, `MemoryQuery`, `MemoryTier`, `MemoryQueryMode`, `Score`, `Embedding`.

### Storage (`loci-memory-store-qdrant`)

`QdrantMemoryStore` uses [Qdrant](https://qdrant.tech/) for cosine-similarity vector search.

Features:

- Configurable deduplication (`similarity_threshold`) to reuse near-duplicates
- Tiered memory lifecycle (`Candidate`, `Stable`, `Core`; `Ephemeral` is request-scoped only)
- Per-tier TTL defaults and query-time expiry filtering
- Weighted retrieval ranking (`similarity * tier_weight`)
- Source-corroboration promotion (`Candidate -> Stable`) when the same fact is observed from a different `source` metadata value
- Manual curation path (`set_tier`) for promoting to `Core`
- Metadata filtering (AND semantics, exact match)
- Min score threshold and max result limits

### Model Providers (`loci-model-provider-ollama`)

`OllamaModelProvider` implements both `EmbeddingModelProvider` and `TextGenerationModelProvider`
against a local [Ollama](https://ollama.com/) instance.

Default models in the generated config:

- Embedding: `qwen3-embedding:0.6b` (768 dimensions)
- Text generation: `qwen3.5:0.8b`

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
# Binary at: target/release/loci
```

Or run directly:

```bash
cargo run --bin loci -- <subcommand>
```

---

## CLI Reference

### Global options

| Flag               | Env var       | Default                      | Description              |
| ------------------ | ------------- | ---------------------------- | ------------------------ |
| `--config` / `-c`  | `LOCI_CONFIG` | `~/.config/loci/config.toml` | Path to TOML config file |
| `--verbose` / `-v` |               | off                          | Enable debug logging     |

### `loci memory save`

Store a new memory.

```bash
loci memory save "The project uses Qdrant for vector storage"
loci memory save "Deployment target is Kubernetes" --meta env=production --meta team=platform
loci memory save "This is a curated fact" --tier core --meta source=manual
```

| Argument / Flag                        | Description                                |
| -------------------------------------- | ------------------------------------------ |
| `<content>`                            | Memory text (required positional argument) |
| `--meta KEY=VALUE`                     | Metadata key-value pair (repeatable)       |
| `--tier <candidate \| stable \| core>` | Optional persisted tier override           |

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

### `loci memory update`

Update an existing memory by UUID.

```bash
loci memory update <uuid> "Updated content" --meta key=value
loci memory update <uuid> --tier core
loci memory update <uuid> --meta source=manual
```

| Argument / Flag                        | Description                                       |
| -------------------------------------- | ------------------------------------------------- |
| `<uuid>`                               | Memory entry ID (required)                        |
| `[content]`                            | New content (optional positional argument)        |
| `--meta KEY=VALUE`                     | Replace metadata with provided pairs (repeatable) |
| `--tier <candidate \| stable \| core>` | Optional tier override                            |

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

### `loci generate` (alias: `gen`)

Generate a response for a prompt, with optional memory retrieval and contextualization.

```bash
loci gen "What storage backend do we use?"
loci gen "Summarise our deployment setup" --max-memory-entries 8 --min-score 0.5
```

| Flag                       | Default      | Description                                       |
| -------------------------- | ------------ | ------------------------------------------------- |
| `<prompt>`                 | _(required)_ | Prompt text (positional)                          |
| `--max-memory-entries <n>` | `5`          | Max memory entries to inject as context           |
| `--min-score <f64>`        | `0.5`        | Minimum weighted score for context memory entries |
| `--memory-mode`            | 'auto'       | Memory query mode (`auto` and `off`)              |
| `--debug-flags <FLAGS>`    | _(none)_     | Comma-separated debug flags (e.g. `memory`)       |

### `loci config init`

Scaffold a default configuration file at the config path.

```bash
loci config init
loci --config /path/to/config.toml config init
```

### Memory config keys

```toml
[memory]
store = "qdrant"
collection = "memory_entries"
# similarity_threshold = 0.95     # deduplicate by semantic similarity
# promotion_source_threshold = 2  # promote Candidate -> Stable when corroborated by a different source
```

---

## Development

```bash
cargo check          # type-check workspace
cargo test           # run all unit tests
cargo clippy         # lint
cargo fmt            # format

# Integration tests — requires Docker (Qdrant via testcontainers)
cargo test-it        # shorthand for 'cargo test --features integration -- --test-threads=1'

# E2E tests — requires a running Ollama instance + Docker
cargo test-e2e       # shorthand for 'cargo test --features e2e -- --test-threads=1'

# All tests (unit + integration + e2e)
cargo test-all       # shorthand for 'cargo test --features integration,e2e -- --test-threads=1'
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

The items below are **planned** — they are not yet implemented. They are listed in
dependency order: each step builds on the capabilities introduced by the previous ones.

### Phase 1: Memory Extraction Strategies

Currently, saving memories is a manual process. Before any automated ingestion can be
useful, loci needs the ability to distill raw content into meaningful memory entries.

A `MemoryExtractionStrategy` trait will process content asynchronously and produce
candidate memory entries. Two built-in strategies are planned:

- **`LlmSummarizationStrategy`** — sends content to the LLM with a system prompt that
  extracts factual statements as new memory entries.
- **`KeywordEntityStrategy`** — lightweight keyword and entity extraction that does not
  require an additional LLM call.

The trait is open for extension; custom strategies can be plugged in. Once extraction
strategies exist, they also enable automatic memory extraction from prompt/response pairs
during generation.

### Phase 2: Scanner Integration

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
into the existing memory store with full lifecycle support (deduplication, tiering,
source-corroboration promotion).

### Phase 3: Session-Aware Memory Proxy

With rich memory populated by scanners and extraction, the next step is scoping retrieval
and injection per session.

A `SessionStore` trait (pluggable: in-process HashMap, SQLite, Redis, …) keyed by a
session ID. Each session carries configuration (filters, model preferences, context window
size) and a lightweight interaction history. The proxy itself remains stateless — no
per-session state is held in process memory beyond the current request.

### Phase 4: Enhanced REPL CLI

A chat-mode REPL for interactive sessions, building on session-aware memory:

- Multi-turn conversation with persistent session state
- Line editing, input history, and auto-complete (`rustyline` or similar)
- Formatted memory context display with relevance scores
- Session ID management directly from the REPL prompt

### Phase 5: Protocol Layer

Expose loci as a network proxy that any client can target without modification. The
specific protocol is undecided; an **OpenAI-compatible API** is the leading candidate
for maximum client interoperability.

### Phase 6: Semantic Knowledge Graph

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
