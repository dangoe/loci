# Contributing to loci

Contributions are welcome! Please open an issue to discuss significant changes before
submitting a pull request.

---

## Branching Strategy

loci follows a trunk-based branching model with four branch types. All branches are
short-lived except `main`.

### `main`

The single long-lived branch. `main` always reflects the latest released or
release-ready state. All work is integrated here via pull requests.

### `feat/<name>`

Feature branches for new functionality or enhancements.

- Branch from `main`
- Merge back into `main` via pull request
- Delete after merge
- Example: `feat/extraction-strategies`, `feat/git-scanner`

### `fix/<name>`

Bug fix branches for correcting defects.

- Branch from `main`
- Merge back into `main` via pull request
- Delete after merge
- Example: `fix/ttl-expiry-off-by-one`, `fix/qdrant-reconnect`

### `release/<version>`

Short-lived branches for release stabilization. Used when a release needs final
adjustments (version bumps, changelog updates, last-minute fixes) without blocking
ongoing feature work on `main`.

- Branch from `main`
- Only release-related commits (version bumps, changelog, critical fixes)
- Merge back into `main` via pull request, then tag the merge commit
- Delete after merge
- Example: `release/0.2.0`, `release/1.0.0`

---

## Commit Messages

Write concise commit messages that focus on the _why_ rather than the _what_. Use
imperative mood (e.g., "Add extraction trait" not "Added extraction trait").

---

## Development Workflow

```bash
# Fork, clone, create a branch
git checkout -b feat/my-feature

# Make changes, ensure quality
cargo check          # type-check
cargo test           # unit tests
cargo clippy         # lint
cargo fmt            # format

# All tests (requires Docker for Qdrant via testcontainers and Ollama)
cargo test-all

# Open a pull request against main
```

See the [README](README.md#development) for full test commands and prerequisites.

---

## License

By contributing, you agree that your contributions will be dual-licensed under
[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
