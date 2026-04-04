# Rust Testing Extension: loci

Project-specific overrides and additions to `.github/skills/rust-testing.md`.

## Test Commands

```bash
cargo test                                   # run all tests across the workspace
cargo test -p loci-core                      # run tests for a single crate
cargo test -p loci-core <name>               # run tests matching substring
cargo test -p loci-config                    # config parser/loader tests
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
| `testcontainers` | `loci-memory-store-qdrant` integration tests | real Qdrant container tests |

## Integration Tests

`loci-memory-store-qdrant` integration tests use
[testcontainers](https://crates.io/crates/testcontainers) to spin up a real Qdrant instance.
They are guarded with `#[cfg_attr(not(feature = "integration"), ignore)]`:

```bash
cargo test -p loci-memory-store-qdrant --features integration -- --test-threads=1
cargo test -p loci-memory-store-qdrant --features integration test_saved_memory_is_returned_by_query -- --test-threads=1
```

Use `--test-threads=1` for integration tests to avoid resource contention from parallel containers.
