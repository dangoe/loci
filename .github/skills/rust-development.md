# Rust Development Skills

Check for a project-specific extension at `.github/skills/rust-development.extension.md` and apply it alongside this file.

## 1. Language & Idioms

- Target **Rust 2024 edition**; use all idioms it enables
- Use `#[derive(...)]` for `Debug`, `Clone`, `PartialEq`, `Serialize`, `Deserialize` wherever applicable
- Use `strum` derives (`EnumIter`, `Display`, `EnumString`) for enums that need iteration or string conversion
- Prefer `impl Trait` in function arguments; use generics when the concrete type must be named at the call site
- Use `From`/`Into` and `TryFrom`/`TryInto` to convert between domain types

## 2. Module Organisation

- `main.rs` is a thin entry point: CLI parsing, logger initialisation, and a single call to run the service
- Domain logic lives in `lib.rs` or dedicated modules; avoid putting logic in `main.rs`
- Submodules with complex test requirements split into `mod.rs`, `mocks.rs`, and `tests.rs`
- Visibility is as restrictive as possible: prefer `pub(crate)` or `pub(super)` over `pub`
- Traits are defined in the module that owns the abstraction, not the module that implements it

## 3. Error Handling

- Use `std::io::Error` / `io::Result<T>` for I/O operations
- Define custom error types with manual `std::error::Error` implementations when a module has a distinct error domain
- Chain errors with context: `.map_err(|e| io::Error::new(e.kind(), format!("context: {e}")))`
- Propagate with `?`; never silently drop a `Result`
- **No `unwrap()` or `expect()` in production code paths**; reserve them for tests and const initialisation

## 4. Trait-Based Design

- Define a trait for every external dependency or I/O operation (filesystem, shell, network)
- Provide a `Native` production implementation and a `Mock` test implementation for each trait
- Pass trait objects or generics to functions and structs — not concrete types — so tests can inject mocks
- Name traits after the capability they represent: `FileRead`, `FileWrite`, `ShellCommand`

## 5. Dependencies

- Add a dependency only when the standard library cannot reasonably do the job
- Prefer crates that expose derive macros to reduce boilerplate (`serde`, `clap`, `strum`)
- Gate platform-specific crates behind Cargo features; do not make them unconditional
- Keep `[dev-dependencies]` separate; never put test-only crates in `[dependencies]`
- Prefer well-maintained, minimal crates over feature-rich monoliths

## 6. Async & Concurrency

- **Synchronous by default.** Only reach for `tokio` or an async runtime when the domain genuinely requires concurrent I/O
- Use `std::sync` primitives (`Mutex`, `Arc`) for shared state in synchronous code
- Do not add `async`/`await` to functions that do not perform async I/O

## 7. Config & CLI

- Parse CLI arguments with `clap` using the derive API (`#[derive(Parser)]`)
- Provide both `short` and `long` forms for every flag; always include `help` and `long_help`
- Load configuration from a TOML file using `serde` with `#[serde(default)]` on every optional field
- On a missing or corrupt config file, log a warning and fall back to compiled-in defaults — do not exit

## 8. Logging

- Use the `log` façade (`log::info!`, `log::debug!`, `log::warn!`, `log::error!`)
- Initialise with `env_logger`; gate `systemd-journal-logger` behind a `journald` Cargo feature
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
| Mock types | `Mock` prefix | `MockFileRead`, `MockShellCommand` |
| Native (prod) types | `Native` prefix | `NativeFileRead` |

## 10. Comment Style

- Code is the primary documentation; choose names that make intent obvious
- Add `///` doc comments on all public items
- Use `//` inline comments sparingly and only to explain *why*, not *what*
