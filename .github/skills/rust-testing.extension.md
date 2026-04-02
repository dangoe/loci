# Rust Testing Extension: ai-memory

Project-specific overrides and additions to `.github/skills/rust-testing.md`.

## Test Commands

```bash
cargo test                                   # run all tests across the workspace
cargo test -p ai-memory-core                 # run tests for a single crate
cargo test -p ai-memory-core <name>          # run tests matching substring
cargo test -- --nocapture                    # show println! output
cargo test -- --test-threads=1               # serialise tests (useful for I/O tests)
```

## Test Dependencies in Use

| Crate | Used in | Purpose |
|-------|---------|---------|
| _(none yet)_ | — | — |

## Integration Tests

`ai-memory-neo4j` integration tests use [testcontainers](https://crates.io/crates/testcontainers)
to spin up a real Neo4j 5 instance via Docker. They are marked `#[ignore]` and must be opted into
explicitly:

```bash
# Run all Neo4j integration tests (Docker must be running)
cargo test -p ai-memory-neo4j -- --ignored --test-threads=1

# Run a single integration test
cargo test -p ai-memory-neo4j <name> -- --ignored
```

`--test-threads=1` is recommended because each test starts its own container; running them
serially avoids exhausting Docker resources.
