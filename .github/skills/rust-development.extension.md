# Rust Development Extension: ai-memory

Project-specific overrides and additions to `.github/skills/rust-development.md`.

## Build & Test Commands

```bash
cargo build               # debug build
cargo build --release     # release build
cargo check               # fast type-check
cargo test                # run all tests
cargo test <name>         # run a single test by name (substring match)
cargo clippy              # lint — must pass with zero warnings
cargo fmt                 # format code
cargo fmt --check         # verify formatting (CI)
```

## Module Structure

> Update this section as the project grows.

| Module | Purpose |
|--------|---------|
| `main.rs` | CLI entry point |

## Dependencies in Use

> Update this section when dependencies are added to `Cargo.toml`.

| Crate | Purpose |
|-------|---------|
| _(none yet)_ | — |
