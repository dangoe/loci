# Rust Development Extension: loci

Project-specific overrides and additions to `.github/skills/rust-development.md`.

## Build, Lint, and Test Commands

```bash
cargo check                                                # type-check workspace
cargo build                                                # workspace build
cargo test                                                 # run workspace tests
cargo test -p loci-core <name>                            # run one test by name
cargo test -p loci-memory-store-qdrant --features integration -- --test-threads=1
cargo clippy                                               # lint
cargo fmt --check                                          # formatting check
```

## Workspace Structure

| Crate | Path | Purpose |
|-------|------|---------|
| `loci-core` | `crates/loci-core` | Domain types and core traits (`MemoryStore`, `TextEmbedder`, model-provider traits) plus `Contextualizer` |
| `loci-memory-store-qdrant` | `crates/loci-memory-store-qdrant` | `QdrantMemoryStore` implementation with tiering and deduplication |
| `loci-model-provider-ollama` | `crates/loci-model-provider-ollama` | `OllamaModelProvider` for embeddings and text generation |
| `loci-config` | `crates/loci-config` | Config schema and loader (`env:` secret resolution) |
| `loci-cli` | `crates/loci-cli` | CLI entry point and command handling |

## Runtime Status

- Implemented at runtime: `qdrant` store + `ollama` provider.
- Config may parse `openai`, `anthropic`, and `markdown`, but runtime currently returns
  `UnsupportedKind` when those are selected.

## Core Domain Notes (`loci-core`)

- `MemoryStore` is text-centric: `save/get/query/update/set_tier/delete/clear`.
- `Memory` stores lifecycle fields (`tier`, `seen_count`, `first_seen`, `last_seen`, `expires_at`);
  embeddings are computed in store/provider layers, not stored on `Memory`.
- `Contextualizer` queries memories using `MemoryQueryMode::Use` and streams model output.

## Dependencies in Use

| Crate | Used in | Purpose |
|-------|---------|---------|
| `qdrant-client` | `loci-memory-store-qdrant` | Qdrant API client |
| `reqwest` | `loci-core`, `loci-model-provider-ollama` | HTTP model-provider communication |
| `clap` | `loci-cli` | CLI parsing |
| `chrono` | `loci-core`, `loci-memory-store-qdrant` | Timestamp and TTL handling |
| `serde` / `serde_json` / `toml` | config + providers | Config and payload serialization |
| `tokio` / `futures` / `async-stream` | core + cli + providers | Async runtime and streaming |
| `uuid` | core + store + cli | Memory IDs |
