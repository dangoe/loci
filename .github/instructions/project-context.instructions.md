---
applyTo: "**"
---

# Project Context: loci

A Rust workspace for memory-augmented LLM prompts.

`loci` stores semantic memory entries, retrieves relevant entries from Qdrant, and enriches prompts via a
`Contextualizer` before forwarding to an LLM provider.

## Workspace

| Crate                         | Purpose                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| `loci-core`                   | Core traits and domain types (`MemoryStore`, `TextEmbedder`, `Contextualizer`)                        |
| `loci-memory-store-qdrant`    | Qdrant-backed memory store with deduplication, tiers, and metadata filtering                          |
| `loci-model-provider-ollama`  | Ollama model provider (embeddings + text generation, including streaming)                             |
| `loci-model-provider-openai`  | OpenAI-compatible model provider (embeddings + text generation, including streaming)                  |
| `loci-config`                 | TOML config schema, parsing, and `env:` secret resolution                                             |
| `loci-wire`                   | Runtime wiring: builds concrete store and provider instances from `AppConfig`                         |
| `loci-server`                 | `loci-server` binary; HTTP server exposing an OpenAI-compatible API and Connect RPC endpoints         |
| `loci-cli`                    | `loci` binary (`memory add/query/get/update/delete/prune-expired/extract`, `generate`, `config init`) |
| `loci-e2e-tests`              | End-to-end tests (requires Docker + Ollama)                                                           |

## Runtime Support

- Active memory backend: `qdrant`
- Active model providers: `ollama`, `openai`-compatible (via `loci-wire` / `loci-server`)
- `anthropic` and `markdown` store are parsed in config but currently return
  `UnsupportedKind` at runtime when selected.

## Quick Reference

```bash
cargo check                                                # type-check workspace
cargo test-it                                              # unit + integration tests (requires Docker)
cargo test-e2e                                             # e2e tests (requires Docker + Ollama)
cargo test-all                                             # all tests (unit + integration + e2e)
cargo clippy                                               # lint
cargo fmt --check                                          # formatting check
```

## Key References

- Rust development conventions: `.github/skills/rust-development.md`
- Testing conventions: `.github/skills/rust-testing.md`
- Project-specific development overrides: `.github/skills/rust-development.extension.md`
- Project-specific testing overrides: `.github/skills/rust-testing.extension.md`
