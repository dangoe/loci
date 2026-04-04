---
applyTo: "**"
---

# Project Context: loci

A Rust workspace for memory-augmented LLM prompts.

`loci` stores semantic memory entries, retrieves relevant entries from Qdrant, and enriches prompts via a
`Contextualizer` before forwarding to an LLM provider.

## Workspace

| Crate                        | Purpose                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------ |
| `loci-core`                  | Core traits and domain types (`MemoryStore`, `TextEmbedder`, `Contextualizer`) |
| `loci-memory-store-qdrant`   | Qdrant-backed memory store with deduplication, tiers, and metadata filtering   |
| `loci-model-provider-ollama` | Ollama model provider (embeddings + text generation, including streaming)      |
| `loci-config`                | TOML config schema, parsing, and `env:` secret resolution                      |
| `loci-cli`                   | `loci` binary (`memory`, `prompt`, `config init`)                              |

## Runtime Support

- Active memory backend: `qdrant`
- Active model provider: `ollama`
- `openai`, `anthropic`, and `markdown` store are parsed in config but currently return
  `UnsupportedKind` at runtime when selected.

## Quick Reference

```bash
cargo check                                                # type-check workspace
cargo test                                                 # run workspace tests
cargo test -p loci-core <name>                            # run one core test by name
cargo test -p loci-memory-store-qdrant --features integration -- --test-threads=1
cargo clippy                                               # lint
cargo fmt --check                                          # formatting check
```

## Key References

- Rust development conventions: `.github/skills/rust-development.md`
- Testing conventions: `.github/skills/rust-testing.md`
- Project-specific development overrides: `.github/skills/rust-development.extension.md`
- Project-specific testing overrides: `.github/skills/rust-testing.extension.md`
