# Rust Testing Extension: ai-memory

Project-specific overrides and additions to `.github/skills/rust-testing.md`.

## Test Commands

```bash
cargo test                      # run all tests
cargo test <name>               # run tests matching substring
cargo test -- --nocapture       # show println! output
cargo test -- --test-threads=1  # serialise tests (useful for filesystem tests)
```

## Test Dependencies in Use

> Update this section when dev-dependencies are added to `Cargo.toml`.

| Crate | Purpose |
|-------|---------|
| _(none yet)_ | — |
