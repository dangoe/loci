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

| Crate                        | Path                                      | Purpose                                                                                                   |
| ---------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `loci-core`                  | `crates/loci-core`                        | Domain types and core traits (`MemoryStore`, `TextEmbedder`, model-provider traits) plus `Contextualizer` |
| `loci-memory-store-qdrant`   | `crates/loci-memory-store-qdrant`         | `QdrantMemoryStore` implementation with Bayesian confidence scoring and deduplication                     |
| `loci-model-provider-ollama` | `crates/loci-model-provider-ollama`       | `OllamaModelProvider` for embeddings and text generation                                                  |
| `loci-model-provider-openai` | `crates/loci-model-provider-openai`       | `OpenAIModelProvider` for embeddings and text generation (OpenAI-compatible API)                          |
| `loci-config`                | `crates/loci-config`                      | Config schema and loader (`env:` secret resolution)                                                       |
| `loci-wire`                  | `crates/loci-wire`                        | Runtime wiring: builds concrete store and provider instances from `AppConfig`; exposes `AnyModelProvider` |
| `loci-server`                | `crates/loci-server`                      | `loci-server` binary; HTTP server with OpenAI-compatible API and Connect RPC endpoints                    |
| `loci-cli`                   | `crates/loci-cli`                         | CLI entry point and command handling                                                                      |
| `loci-e2e-tests`             | `crates/loci-e2e-tests`                   | End-to-end tests (requires Docker + Ollama)                                                               |

## Runtime Status

- Implemented at runtime: `qdrant` store + `ollama` and `openai`-compatible providers (via `loci-wire`).
- Config may parse `anthropic` and `markdown` store, but runtime currently returns
  `UnsupportedKind` when those are selected.

## Core Domain Notes (`loci-core`)

- `MemoryStore` is text-centric: `add_entry/get_entry/query/update_entry/set_entry_trust/delete_entry/prune_expired`.
- `MemoryEntry` stores lifecycle fields (`trust`, `seen_count`, `first_seen`, `last_seen`, `expires_at`);
  embeddings are computed in store/provider layers, not stored on `MemoryEntry`.
- `Contextualizer` queries memory entries using `MemoryQueryMode::Use` and streams model output.

## Memory Trust Model

`MemoryTrust` is a single enum that merges the old `MemoryKind` + `confidence` + Bayesian counters into one concept:

| Variant | Confidence | TTL | Retrieval weight | Notes |
| --- | --- | --- | --- | --- |
| `MemoryTrust::Extracted { confidence, evidence }` | Bayesian `(0.0, 1.0)` | 365 days | 0.8 | Default for LLM-extracted entries; subject to auto-discard and auto-promotion |
| `MemoryTrust::Fact` | 1.0 (fixed) | None (no expiry) | 1.0 | Curated or auto-promoted entries; never decayed or discarded |

Key rules:
- LLM confidence is clamped to the open interval `(0.0, 1.0)` using `clamp_confidence()`.
- Default confidence when the LLM omits it: `0.5`.
- `TrustEvidence` holds Bayesian counters (`alpha`, `beta`); score = α/(α+β).
- `TrustEvidence::from_confidence(c, seed_weight)` initialises α = c×W, β = (1−c)×W.
- Storage schema is unchanged: flat fields `kind`, `confidence`, `credibility_belief_alpha`, `credibility_belief_beta` (backward-compatible with existing Qdrant data).

## Memory Extraction Pipeline (`MemoryExtractionPipeline`)

Config fields (all in `PipelineConfig` / `[memory.extraction.pipeline]`):

| Field | Default | Purpose |
| --- | --- | --- |
| `bayesian_seed_weight` | 10.0 | W in the seed formula α=c×W, β=(1−c)×W |
| `max_counter_increment` | 5.0 | Cap on the increment applied per hit |
| `max_counter` | 100.0 | Absolute cap on α and β counters |
| `auto_discard_threshold` | 0.1 | score ≤ threshold → discard, no store write |
| `auto_promotion_threshold` | 0.9 | score ≥ threshold → store as `Fact` |
Pipeline result fields: `inserted`, `merged`, `promoted` (auto-promoted to `MemoryTrust::Fact`), `discarded` (with `DiscardReason`: `LowScore` or `ContradictsAFact`).

A candidate that contradicts an existing `MemoryTrust::Fact` entry is immediately discarded (`ContradictsAFact`) regardless of its score.

## Dependencies in Use

| Crate                                | Used in                                                          | Purpose                           |
| ------------------------------------ | ---------------------------------------------------------------- | --------------------------------- |
| `qdrant-client`                      | `loci-memory-store-qdrant`                                       | Qdrant API client                 |
| `reqwest`                            | `loci-core`, `loci-model-provider-ollama`, `loci-model-provider-openai` | HTTP model-provider communication |
| `clap`                               | `loci-cli`, `loci-server`                                        | CLI parsing                       |
| `chrono`                             | `loci-core`, `loci-memory-store-qdrant`                          | Timestamp and TTL handling        |
| `serde` / `serde_json` / `toml`      | config + providers                                               | Config and payload serialization  |
| `tokio` / `futures` / `async-stream` | core + cli + server + providers                                  | Async runtime and streaming       |
| `uuid`                               | core + store + cli                                               | Memory IDs                        |
| `axum`                               | `loci-server`                                                    | HTTP routing                      |
| `buffa` / `buffa-types`              | `loci-server`                                                    | Connect RPC support               |
| `tower-http`                         | `loci-server`                                                    | CORS and tracing middleware       |
