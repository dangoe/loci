# Rust Testing Skills

Check for a project-specific extension at `.github/skills/rust-testing.extension.md` and apply it alongside this file.

## 1. Test Location

- **Unit tests** — `#[cfg(test)] mod tests { ... }` at the bottom of the file under test
- **Complex modules** — split test helpers cleanly when inline tests become noisy
- **Integration tests** — `tests/` directory for tests that exercise the binary or multiple modules together

## 2. Test Frameworks

| Crate | Role |
|-------|------|
| `rstest` | Parameterised tests via `#[rstest]` and `#[values(...)]` |
| `pretty_assertions` | Drop-in replacement for `assert_eq!` with readable diffs |
| `tempfile` | Temporary files and directories for config/I/O tests |
| `tokio` | Async test runtime for async APIs |

## 3. Mock Strategy

- Implement the production trait for each mock type
- Mocks record calls and capture written data so tests can assert on them
- Gate mock types with `#[cfg(test)]` or place them in a `mocks.rs` file included only in test builds

```rust
#[cfg(test)]
pub(crate) struct MockFileWrite {
    pub written: RefCell<Vec<String>>,
}

#[cfg(test)]
impl FileWrite for MockFileWrite {
    fn write(&self, path: &str, content: &str) -> io::Result<()> {
        self.written.borrow_mut().push(content.to_string());
        Ok(())
    }
}
```

## 4. Parameterised Tests

Use `rstest` `#[values(...)]` to cover multiple inputs without duplicating test bodies:

```rust
#[rstest]
fn test_parse_level(#[values("low", "auto", "high")] input: &str) {
    assert!(parse_level(input).is_ok());
}
```

## 5. Naming

- Test functions: `snake_case`, descriptive of the scenario
- Prefer `test_<behavior>_when_<condition>` or `test_<what>_returns_<expected_outcome>`
- Test helper functions: no `test_` prefix; name them like production helpers

## 6. Coverage Expectations

Every module should cover:

- ✅ Happy path for each public function
- ✅ All error paths (missing file, corrupt input, permission denied, etc.)
- ✅ Boundary / edge values for parsed inputs
- ✅ Each enum variant for functions that branch on type
- ✅ Async success/error paths for async traits/providers
- ⚠️ Integration with external systems should be feature-gated and deterministic

## 7. Pre-Commit Verification

```bash
cargo test && cargo clippy && cargo fmt --check
```

All three must pass before committing.
