---
applyTo: "**"
---

# Project Context: loci

A Rust library for semantic memory storage and retrieval. Text is embedded into vectors and stored
in a backend (currently Neo4j); retrieval uses cosine vector similarity.

## Workspace

| Crate | Purpose |
|-------|---------|
| `loci-core` | Traits, domain types, `MemoryService<S,E>` |
| `loci-neo4j` | Neo4j-backed `MemoryStore` implementation |

## Quick Reference

```bash
cargo check                          # type-check workspace
cargo test                           # run all tests
cargo test -p loci-core <name>  # run a single test
cargo clippy                         # lint
cargo fmt                            # format
```

## Key References

- Rust development conventions: `.github/skills/rust-development.md`
- Testing conventions: `.github/skills/rust-testing.md`
- Project-specific overrides: `.github/skills/rust-development.extension.md`
