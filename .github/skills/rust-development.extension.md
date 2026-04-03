# Rust Development Extension: loci

Project-specific overrides and additions to `.github/skills/rust-development.md`.

## Build & Test Commands

```bash
cargo build                            # debug build (all crates)
cargo build --release                  # release build
cargo check                            # fast type-check
cargo test                             # run all tests
cargo test -p loci-core           # run tests for a single crate
cargo test -p loci-core <name>    # run a single test by name (substring match)
cargo clippy                           # lint — must pass with zero warnings
cargo fmt                              # format code
cargo fmt --check                      # verify formatting (CI)
```

## Workspace Structure

| Crate | Path | Purpose |
|-------|------|---------|
| `loci-core` | `crates/loci-core` | `MemoryStore` trait, `EmbeddingPort` trait, domain types, `MemoryService<S,E>` |
| `loci-neo4j` | `crates/loci-neo4j` | `Neo4jMemoryStore` — `impl MemoryStore` backed by Neo4j vector index |

## Domain Types (loci-core)

| Type | Description |
|------|-------------|
| `Embedding` | Newtype over `Vec<f32>`; use `.values()` / `.dimension()` |
| `Memory` | `id`, `content`, `embedding`, `metadata`, `created_at` |
| `MemoryEntry` | Query result: `Memory` + `Score` |
| `MemoryQuery` | Pre-computed `Embedding` + `max_results` |
| `Score` | Validated `f64` in [0.0, 1.0]; constructed with `Score::new(v)?` |
| `MemoryService<S, E>` | Composes store + embedding; call `.memorize()`, `.retrieve()`, `.forget()` |

## Trait Design Rules

- `MemoryStore` and `EmbeddingPort` use native AFIT (`async fn` directly in trait).
- Both traits are `Send + Sync` supertrait-bounded.
- Production types are prefixed `Native`; test doubles are prefixed `Mock`.
- New backend crates follow the same pattern as `loci-neo4j`: one `Config` struct, one store struct, `impl MemoryStore for ...` in `store.rs`.

## Dependencies in Use

| Crate | Used in | Purpose |
|-------|---------|---------|
| `chrono` | both | `DateTime<Utc>` timestamps |
| `serde` | core | derive `Serialize`/`Deserialize` on domain types |
| `serde_json` | neo4j | metadata serialisation to/from JSON string node property |
| `uuid` | both | `Uuid::new_v4()` for memory IDs |
| `neo4rs` | neo4j | async Neo4j driver |
| `log` | neo4j | deduplication debug logging |
