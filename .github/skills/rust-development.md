# Rust Development Skills

Check for a project-specific extension at `.github/skills/rust-development.extension.md` and apply it alongside this file.

## 1. Language & Idioms

- Target **Rust 2024 edition**; use all idioms it enables
- Prefer explicit domain types and small constructors over ad hoc maps/tuples
- Prefer `impl Trait` arguments or trait objects for dependency boundaries
- Use `From`/`Into` and `TryFrom`/`TryInto` where conversions are part of the API

## 2. Module Organisation

- `main.rs` is a thin entry point: CLI parsing, logger initialisation, and a single call to run the service
- Domain logic lives in `lib.rs` or dedicated modules; avoid putting logic in `main.rs`
- Visibility is as restrictive as possible: prefer `pub(crate)` or `pub(super)` over `pub`
- Put traits in the abstraction-owning module, implementations in backend crates/modules

## 3. Error Handling

- Use explicit domain errors (`Display` + `std::error::Error`) for public boundaries
- Add actionable context when mapping lower-level failures
- Propagate with `?`; never silently drop a `Result`
- Avoid `unwrap()`/`expect()` in normal runtime paths; if used for invariants, keep it narrowly scoped and obvious

## 4. Trait-Based Design

- Define traits for stable seams (store/provider/embedder boundaries), not for every helper
- Prefer injectable dependencies (trait object or generic) where testing/isolation needs it
- For public async traits, prefer RPITIT forms that preserve `Send` where needed:
  ```rust
  // trait
  fn save(&self, x: T) -> impl Future<Output = Result<U, E>> + Send + '_;

  // impl
  async fn save(&self, x: T) -> Result<U, E> { ... }
  ```

## 5. Dependencies

- Add a dependency only when the standard library cannot reasonably do the job
- Prefer focused crates over broad frameworks
- Gate optional platform integrations behind Cargo features
- Keep `[dev-dependencies]` separate; never put test-only crates in `[dependencies]`

## 6. Async & Concurrency

- Use async for network/store/model-provider boundaries
- Keep sync helpers sync unless they need async I/O
- Use `Arc`/`Mutex` intentionally and keep lock scopes tight

## 7. Config & CLI

- Parse CLI arguments with `clap` using the derive API (`#[derive(Parser)]`)
- Load configuration from TOML via `serde` with explicit defaults only where required
- Resolve `env:` secrets through config loading utilities where applicable

## 8. Logging

- Use the `log` façade (`log::info!`, `log::debug!`, `log::warn!`, `log::error!`)
- Initialise via the crate's logging setup (env_logger or optional journal logger feature)
- `info!` — important state transitions ("Service started", "Listening on …")
- `debug!` — routine operational detail ("Accepted connection", "Parsed command")
- `warn!` — recoverable issues ("Config file missing, using defaults")
- `error!` — failures requiring investigation ("Failed to bind socket")
- Do not log in tight loops without a guard; avoid flooding the journal

## 9. Naming Conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Functions & methods | `snake_case`, verb-first | `parse_command`, `setup_socket_directory` |
| Types & traits | `PascalCase` | `ParseError`, `UnixListenerFactory` |
| Enum variants | `PascalCase` | `Auto`, `High`, `GetPowerLevel` |
| Constants | `SCREAMING_SNAKE_CASE` | `DEFAULT_CARD`, `BUFFER_SIZE` |
| Modules | `snake_case` | `shell_integration`, `unix_socket` |

## 10. Comment Style

- Code is the primary documentation; choose names that make intent obvious
- Add `///` docs for public APIs where they improve discoverability/usage
- Use `//` inline comments sparingly and only to explain *why*, not *what*
