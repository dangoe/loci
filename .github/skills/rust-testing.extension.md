# Rust Testing Extension: loci

Project-specific overrides and additions to `.github/skills/rust-testing.md`.

## Test Commands

```bash
cargo test                                   # run all tests across the workspace
cargo test -p loci-core                      # run tests for a single crate
cargo test -p loci-core <name>               # run tests matching substring
cargo test -p loci-config                    # config parser/loader tests
cargo test -p loci-server --features testing # server integration tests with mocks
cargo test -p loci-cli --features testing    # CLI integration tests with mocks
cargo test -p loci-memory-store-qdrant --features integration -- --test-threads=1
cargo test -- --nocapture                    # show println! output
cargo test -- --test-threads=1               # serialize tests when needed
```

## Test Dependencies in Use

| Crate | Used in | Purpose |
|-------|---------|---------|
| `pretty_assertions` | multiple crates | improved assertion diffs |
| `rstest` | `loci-core`, `loci-config` | parameterized tests |
| `tempfile` | `loci-config` | temp config files in unit tests |
| `tokio` | async tests | `#[tokio::test]` runtime |
| `testcontainers` | `loci-memory-store-qdrant`, `loci-e2e-tests` | real Qdrant container tests |

## Shared Test Infrastructure

Each crate that defines traits provides reusable mocks via a feature-gated `testing` module:

| Crate | testing.rs provides | Feature |
|---|---|---|
| **loci-core** | `MockStore`, `MockTextGenerationModelProvider`, `MockTextEmbedder`, `make_result()` | `testing` |
| **loci-memory-store-qdrant** | `start_qdrant_container()`, `start_store()`, `QDRANT_GRPC_PORT` | `testing` |
| **loci-model-provider-ollama** | `base_url()`, `text_model()`, `embedding_model()`, `ollama_provider()`, `ensure_ollama_available()` | `testing` |
| **loci-server** | `TestServer`, `minimal_app_config()`, `mock_config()` | `testing` |
| **loci-cli** | `TestCli`, `minimal_ollama_config()`, `mock_config()` | `testing` |

## Integration Tests

`loci-memory-store-qdrant` integration tests use
[testcontainers](https://crates.io/crates/testcontainers) to spin up a real Qdrant instance.
The whole test file is gated with `#![cfg(feature = "integration")]`:

```bash
cargo test -p loci-memory-store-qdrant --features integration -- --test-threads=1
cargo test -p loci-memory-store-qdrant --features integration test_saved_memory_is_returned_by_query -- --test-threads=1
```

Use `--test-threads=1` for integration tests to avoid resource contention from parallel containers.

`loci-server` integration tests use mock stores/providers and are gated with `#![cfg(feature = "testing")]`:

```bash
cargo test -p loci-server --features testing
```

## Cargo Aliases

The workspace defines cargo aliases in `.cargo/config.toml`:

```bash
cargo test-it                                    # cargo test --features integration,testing -- --test-threads=1
cargo test-e2e                                   # cargo test --features e2e,testing -- --test-threads=1
cargo test-all                                   # cargo test --features integration,e2e,testing -- --test-threads=1
```
