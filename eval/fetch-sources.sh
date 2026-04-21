#!/usr/bin/env bash
# fetch-sources.sh — Populate eval/sources/ with documents for extraction evaluation.
#
# This script must be run once before the first evaluation run. It creates all
# source documents in eval/sources/, which is git-ignored. Documents include
# Wikipedia articles (fetched over HTTP) and synthetic documents written inline.
#
# Usage:
#   ./eval/fetch-sources.sh
#
# Requires: curl, jq

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES_DIR="${SCRIPT_DIR}/sources"

# ── Dependency checks ────────────────────────────────────────────────────────

if ! command -v curl &>/dev/null; then
    echo "Error: 'curl' is required but not found in PATH." >&2
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: 'jq' is required but not found in PATH." >&2
    exit 1
fi

mkdir -p "${SOURCES_DIR}"

# ── Wikipedia articles ────────────────────────────────────────────────────────
# Fetched from the MediaWiki Action API (CC BY-SA 4.0).
# Content is not committed to the repository.

WIKI_UA="loci-eval/1.0 (https://github.com/dgoetten/loci; evaluation fixture fetcher)"

fetch_wikipedia() {
    local filename="$1"
    local title="$2"   # plain title, e.g. "Rust (programming language)"
    local out="${SOURCES_DIR}/${filename}.txt"
    echo "  Wikipedia: ${title} -> ${filename}.txt"
    curl -s -L -f -G \
        -H "User-Agent: ${WIKI_UA}" \
        --data-urlencode "titles=${title}" \
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json" \
    | jq -r '.query.pages | to_entries[0].value.extract' \
    > "${out}"
}

echo "Fetching Wikipedia articles ..."
fetch_wikipedia "wiki-biography-ada-lovelace"  "Ada Lovelace"
fetch_wikipedia "wiki-science-photosynthesis"  "Photosynthesis"
fetch_wikipedia "wiki-history-apollo-11"       "Apollo 11"
fetch_wikipedia "wiki-geography-amazon-river"  "Amazon River"
fetch_wikipedia "wiki-technology-rust-lang"    "Rust (programming language)"
echo ""

# ── Synthetic documents ───────────────────────────────────────────────────────

write_source() {
    local filename="$1"
    local content="$2"
    echo "  Writing: ${filename}.txt"
    printf '%s\n' "${content}" > "${SOURCES_DIR}/${filename}.txt"
}

echo "Writing synthetic documents ..."

# ── Technical: API documentation ─────────────────────────────────────────────

write_source "tech-api-docs" 'Nimbus API v2 Reference — Notifications Endpoint

Base URL: https://api.nimbus.example.com/v2

Authentication
All requests require a Bearer token in the Authorization header. Tokens expire after
24 hours and can be refreshed using the /auth/refresh endpoint. Rate limiting is
applied at 1,000 requests per minute per API key.

POST /notifications
Creates a new notification and queues it for delivery.

Request body (application/json):
  channel         string   required   One of: email, sms, push, webhook
  recipient       string   required   Email address, phone number, device token, or webhook URL
  subject         string   optional   Subject line for email notifications (max 200 chars)
  body            string   required   Notification body text (max 4,096 chars)
  priority        string   optional   low | normal | high. Defaults to normal.
  scheduled_at    string   optional   ISO 8601 timestamp. If omitted, delivers immediately.
  idempotency_key string   optional   Prevents duplicate delivery within a 24-hour window.

Response (201 Created):
  id          string   UUID of the created notification
  status      string   queued | scheduled
  created_at  string   ISO 8601 timestamp

Error responses:
  400 Bad Request   — Missing required fields or invalid channel
  401 Unauthorized  — Missing or expired token
  422 Unprocessable — Recipient format does not match channel type
  429 Too Many Requests — Rate limit exceeded; Retry-After header is set

GET /notifications/{id}
Returns the current status of a notification. Response fields include: id, channel,
recipient, status (queued | delivering | delivered | failed | cancelled), created_at,
delivered_at, failure_reason.

DELETE /notifications/{id}
Cancels a scheduled notification. Returns 409 Conflict if already delivered.

Webhooks
If channel is webhook, the system sends a POST to the recipient URL with a JSON
payload. The payload is signed via X-Nimbus-Signature (HMAC-SHA256). Endpoints must
respond with HTTP 200 within 5 seconds or the delivery is retried up to 3 times with
exponential backoff.'

# ── Technical: Architecture Decision Record ───────────────────────────────────

write_source "tech-adr" 'ADR-0012: Replace Redis-based Session Store with JWT Stateless Tokens

Status: Accepted
Date: 2025-11-04
Deciders: Felix Wagner (backend lead), Amira Hassan (security), Leon Park (infrastructure)

Context
Our current session management relies on Redis to store server-side session data.
Redis has introduced operational complexity: we run a Redis Sentinel cluster for high
availability, session data must be replicated across three availability zones, and
Redis accounts for roughly 18% of our monthly infrastructure cost.

The API serves exclusively first-party clients (our own web and mobile apps). We do
not expose third-party OAuth flows. All clients can handle token refresh.

Decision
We will replace the Redis session store with short-lived JWT access tokens (15-minute
expiry) and longer-lived opaque refresh tokens (30 days) stored in an HTTP-only
secure cookie. The refresh token is stored in PostgreSQL rather than Redis. Access
tokens are signed with RS256 using a 4096-bit key pair. Key rotation happens every
90 days via an automated job.

Consequences
- Eliminates the Redis Sentinel cluster and its operational overhead.
- Reduces infrastructure cost by an estimated 12-18%.
- Immediate token revocation is no longer possible for access tokens; revocation
  applies only at the next refresh cycle (max 15 minutes delay). Acceptable per the
  security team risk assessment for our threat model.
- All API handlers must validate the JWT signature and expiry on every request.
  Middleware handles this centrally.
- We maintain a short-lived revocation list in PostgreSQL (checked only during
  refresh) to handle logout and compromised token scenarios.

Alternatives Considered
- Keeping Redis but switching to Redis Cluster — rejected due to cost and complexity.
- Using Paseto tokens instead of JWT — deferred; ecosystem tooling less mature.'

# ── Technical: Changelog ──────────────────────────────────────────────────────

write_source "tech-changelog" 'Orion Data Pipeline — Changelog

v3.4.0 (2026-03-15)
Breaking changes:
- The --output-format flag now defaults to parquet instead of csv.
- Dropped support for Python 3.9. Minimum required version is now Python 3.11.

New features:
- Added native support for reading from Apache Iceberg tables via [source.iceberg].
- Incremental watermark checkpointing: pipelines can now resume from the last
  processed timestamp after a crash. Enable with checkpoint_interval = "5m".
- New CLI command: orion validate-schema validates source data against the declared
  schema before a pipeline run.

Bug fixes:
- Fixed a memory leak in the Parquet writer for pipelines running longer than 2 hours.
- Corrected a timezone handling bug where UTC-offset timestamps were converted to
  local time in certain locales.
- The S3 connector no longer retries on 404 errors.

v3.3.2 (2026-02-01)
- Security patch: Updated HTTP client to address CVE-2026-10441 (path traversal in
  multipart upload handling). All users on v3.3.x should upgrade immediately.

v3.3.0 (2026-01-08)
- Added Delta Lake as an output format. Requires delta-rs >= 0.18.
- New metric: orion_pipeline_lag_seconds available in the /metrics Prometheus endpoint.
- PostgreSQL source connection pool size now defaults to 10 (was 5).'

# ── Technical: Project README ─────────────────────────────────────────────────

write_source "tech-readme" 'ferroqueue — A persistent, zero-copy message queue for Rust

ferroqueue is an embedded message queue library for Rust applications. It stores
messages in a memory-mapped file, supports multiple concurrent readers, and guarantees
at-least-once delivery. Designed for durable, ordered message delivery without an
external broker.

Features
- Persistent storage via memory-mapped files (no external process required)
- Zero-copy reads: consumers read directly from mapped memory
- Multiple independent consumer groups, each tracking their own offset
- Configurable retention by message count or total size
- ACID-safe: uses fsync and atomic offset commits to survive crashes
- Supported platforms: Linux and macOS (Windows support planned)

Requirements
- Rust 1.75 or later (stable channel)
- Filesystem supporting mmap (ext4, APFS, tmpfs; network filesystems not supported)

Quick start

Add to Cargo.toml:
  ferroqueue = "0.5"

Create a queue and publish:
  let queue = Queue::open("/var/data/myqueue", QueueConfig::default())?;
  queue.publish(b"hello world")?;

Consume messages:
  let consumer = queue.consumer("my-group")?;
  while let Some(msg) = consumer.next()? {
      consumer.commit()?;
  }

Configuration (QueueConfig):
  segment_size    max size of a single segment file (default: 512 MB)
  max_segments    maximum number of retained segments (default: 8)
  sync_interval   how often to fsync (default: every 100 ms)
  compression     None | Lz4 | Zstd (default: None)

Running tests:
  cargo test                  # unit tests
  cargo test --features e2e   # end-to-end tests (creates temp files in /tmp)

License: MIT OR Apache-2.0 — MSRV: 1.75'

# ── Meetings: Daily standup ───────────────────────────────────────────────────

write_source "meeting-standup" 'Daily Standup Notes — Argus Team
Date: 2026-04-14 (Monday)
Attendees: Priya Sharma, Tom Belletti, Yuki Tanaka, Marcus Webb, Fatima Al-Rashid (PM)

Priya Sharma
Yesterday: Finished the integration tests for the invoice parsing service. All 47 pass.
Today: Starting the PDF renderer migration from pdfmake to puppeteer. Estimated 2 days.
Blockers: None.

Tom Belletti
Yesterday: Investigated memory spike in JIRA-2841. Root cause: the event listener in
the WebSocket handler is never cleaned up on disconnect.
Today: Shipping the fix for JIRA-2841 with a regression test.
Blockers: Needs code review from Priya before merging — she owns the WebSocket module.

Yuki Tanaka
Yesterday: On leave.
Today: Reviewing backlog PRs. Will prioritize database migration PR #447 as it blocks
the staging deployment.
Blockers: PR #447 needs sign-off from the data team. Fatima to follow up.

Marcus Webb
Yesterday: Pair-programmed with design on the new dashboard layout. Figma done.
Today: Beginning frontend implementation using React 19 and the existing component
library. Target: first screen wired up by Thursday.
Blockers: None.

Action items:
- Fatima: Ping the data team today for sign-off on PR #447.
- Tom: Have JIRA-2841 fix in review by EOD.
- Marcus: Demo new dashboard skeleton in Friday sprint review.

Next standup: Tuesday 2026-04-15, 09:30.'

# ── Meetings: Sprint planning ─────────────────────────────────────────────────

write_source "meeting-planning" 'Sprint 22 Planning — Helios Platform Team
Date: 2026-04-07
Facilitator: Sandra Krause (Engineering Manager)

Sprint goal: Ship the multi-tenant billing module to production by April 30.

Committed stories (34 story points total):

HEL-514 — Per-tenant usage metering API (8 pts, Diego)
Depends on the TimescaleDB schema migration from sprint 21 (complete).

HEL-521 — Stripe integration for invoice generation (8 pts, Noa + Chiara)
Noa handles the Stripe client; Chiara handles the webhook receiver. Must use
idempotency keys. Code-complete deadline: April 22 to allow QA time.

HEL-529 — Tenant admin UI: billing tab (5 pts, Kwame)
Requires HEL-514 to be merged first.

HEL-533 — End-to-end billing smoke tests in staging (5 pts, Diego + Noa)
Must pass before production deploy.

HEL-538 — Runbook and incident response guide for billing module (3 pts, Sandra)
Covers alerting thresholds, rollback procedure, and Stripe dashboard access.

HEL-541 — Update API rate limits for metering endpoint (2 pts, Diego)
Raise from 100 req/s to 5,000 req/s per tenant via API gateway config.

HEL-545 — Upgrade Go to 1.24 (3 pts, Chiara)
CI is pinned to Go 1.22. Low risk, pure toolchain change.

Decisions:
- Billing module launches in read-only mode for existing tenants; write access
  requires explicit opt-in by each tenant admin.
- Stripe webhook secret stored in Vault, not env vars. Diego to document Vault path.
- Billing tab hidden by server-side permission check (billing_enabled in tenant config),
  no feature flag needed.

Out of scope: payment failure recovery (sprint 23), multi-currency support (Q3).'

# ── Meetings: Chat transcript ─────────────────────────────────────────────────

write_source "conversation-chat" 'Slack — #backend-eng (2026-04-10)

Lena Hoffmann [10:03]
Heads-up: the deployment pipeline is broken. Build step 3 (Docker image push) is
failing with a 403 on ECR. The IAM role credentials for the CI runner expired.
Filed ticket: PLAT-2091.

Jonas Bauer [10:07]
Is this blocking the CVE hotfix that needs to go out today?

Lena Hoffmann [10:09]
Yes, blocks all deployments. Escalating to platform now. Hopefully ~2 hours.

Rania Khalil [10:12]
Can we push the image manually from a dev machine? The hotfix is 2 lines.

Jonas Bauer [10:15]
We should not bypass the pipeline — we would lose the vulnerability scan step.

Lena Hoffmann [10:41]
Platform fixed it: the IAM role policy had an expiry date set (should not have been),
now removed. Pipeline runner picks up new credentials within 10 minutes. Rebuilding.

Lena Hoffmann [11:02]
Pipeline is green. CVE hotfix building. ETA to production: ~20 minutes.

Lena Hoffmann [11:28]
CVE patch (commit a4f91c3) is live in production. All health checks passing.
Closing PLAT-2091.

Jonas Bauer [11:29]
Thanks. Can someone add a check to infra-as-code to prevent IAM role expiry dates?
This is the second time this has happened.

Rania Khalil [11:32]
On it — adding a Conftest policy rule. Opening a PR today. CI will fail if anyone
sets an expiry on an IAM role.'

# ── Personal: Preferences ─────────────────────────────────────────────────────

write_source "personal-preferences" 'My Setup and Preferences (last updated April 2026)

Editor & tooling
I use Neovim as my primary editor with the LazyVim configuration. My colorscheme is
Catppuccin Mocha. Terminal emulator: Alacritty. Multiplexer: tmux. Font: JetBrains
Mono Nerd Font at 14pt.

For Rust projects I rely on rust-analyzer, cargo-watch, and cargo-nextest instead of
the built-in test runner. I always enable clippy in CI with -D warnings.

Shell setup
Shell: zsh with Starship prompt. Tools: fzf for fuzzy history/file search, zoxide for
directory jumping, bat instead of cat. Key aliases: gs=git status, gl=git log
--oneline --graph, gd=git diff --staged.

Languages and stack
Preferred backend language: Rust. Scripting: Python 3.12+ or bash. I avoid JavaScript
on the backend; TypeScript on the frontend is acceptable but kept minimal.
Databases: PostgreSQL by default, SQLite for embedded use. I avoid ORMs and write SQL
directly with sqlx in Rust projects.

Workflow
I prefer small, focused PRs over large ones. I squash commits before merging. Code
review turnaround expectation: 1 business day. I write ADRs for significant
architectural decisions under docs/adr/.

I prefer GitHub Issues with a Kanban board over Jira. Standups should be async and
written, not a daily video call.

Hardware
Primary machine: ThinkPad X1 Carbon Gen 12 running Arch Linux. External display: 27"
4K monitor. Keyboard: ZSA Moonlander Mark II with Kailh Box White switches.'

# ── Personal: Goals ───────────────────────────────────────────────────────────

write_source "personal-goals" 'Goals and focus areas — Q2 2026

Professional goals

1. Ship loci v1.0 by end of June. Remaining: pipeline evaluation, documentation,
   publish to crates.io.

2. Benchmark vector databases (Qdrant vs Weaviate vs pgvector) by May 15 to confirm
   Qdrant is the right long-term choice.

3. Write two blog posts on Rust async patterns — one per month in April and May.

4. Complete the Rustlings exercises abandoned in February and get comfortable with
   lifetimes in complex scenarios (self-referential structs, GATs).

Health and personal

- Run at least 3 times per week. Current: 5 km. Target: 10 km by end of June.
- Read 2 technical books this quarter: "Designing Data-Intensive Applications"
  (halfway) and "The Art of PostgreSQL" (not started).
- No laptop after 21:30.

Habits to keep
- Weekly review every Sunday evening.
- Daily morning journaling (30-day streak as of April 1).
- No meetings before 10:00 — protected deep work block.

Not doing this quarter
- No new side projects. Only loci counts as active work.
- No rewrites of anything that currently works.'

# ── Personal: Project notes ───────────────────────────────────────────────────

write_source "personal-project-notes" 'Notes on the loci project — working notes

Architecture

The extraction pipeline is currently a synchronous call. This works for CLI use but
would be a bad fit for server-side extraction mid-conversation. Should eventually
be an async background task. Not for v1 — out of scope.

The Qdrant store is the only backend implemented. A SQLite backend for local/offline
use would be valuable. The trait is already defined so the implementation should be
straightforward. Add to backlog after v1.

Open questions
- Should memory entries have structured tags in addition to free-form metadata?
  A HashMap is flexible but hard to query. Structured tags would allow filtering like
  "all entries tagged #work". Leaning yes but not before v1.
- The decay system only runs when the user calls `loci memory decay` explicitly.
  A background daemon for server mode makes sense. For CLI, maybe a warning if decay
  has not run in N days.

Known rough edges
- Chunking: if a chunk spans a topic boundary, extracted entries may be context-free.
  Known limitation with no current fix.
- No deduplication across chunks within a single run. The same fact in two overlapping
  chunks is extracted twice. The pipeline similarity check handles it for persistent
  runs but not dry-run.
- E2E tests require Docker (testcontainers). CI skips them if Docker is unavailable.

Decisions made
- Config format: TOML, not YAML. More explicit, aligns with Rust ecosystem.
- CLI is the primary interface for v1. The HTTP proxy server is secondary and does
  not need to be fully documented for the initial release.
- No GUI, no TUI for v1. Pure CLI only.'

# ── Edge case: Very short ─────────────────────────────────────────────────────

write_source "edge-very-short" 'Maria prefers async communication over meetings and uses Obsidian for note-taking.
Her team uses Notion for project tracking and she personally dislikes it.'

# ── Edge case: Noisy ──────────────────────────────────────────────────────────

write_source "edge-noisy" 'Monday morning. Coffee. The usual. Traffic was fine today, which is a nice change.
Opened my laptop, 47 unread Slack messages — almost all of them people reacting to
GIFs in the #random channel. Spent the first 30 minutes clearing notifications.

Grabbed lunch at the Italian place around the corner. Penne arrabbiata, same as every
week. The weather was decent for a change — 17 degrees and sunny. On the way back I
noticed the bakery next to the office has closed. Third shop on that street this year.

Afternoon was mostly meetings. First one ran long. Second could have been an email.
By 16:00 I was struggling to focus. Put on brown noise, closed all tabs. Got about
45 minutes of actual work done.

Important: the team Kubernetes cluster is being migrated from AWS us-east-1 to
eu-west-1 on May 20, 2026. All services need region configuration updated before
that date. Migration approved by the infrastructure committee last Thursday, confirmed
by DevOps lead Samuel Chen.

Got home around 18:30. Made pasta. Watched something forgettable. Went to bed at 23:00.'

# ── Edge case: Sparse / procedural ───────────────────────────────────────────

write_source "edge-sparse" 'How to reset your password in the Acme Portal

Step 1: Go to the login page
Open your web browser and navigate to the Acme Portal login page.

Step 2: Click "Forgot password"
Below the password field, click "Forgot your password?" to open the reset page.

Step 3: Enter your registered email address
Type the email address associated with your account and click "Send reset link".

Step 4: Check your email
You will receive an email within a few minutes. Check your spam folder if needed.
The reset link is valid for 24 hours.

Step 5: Choose a new password
Your new password must be at least 12 characters, contain an uppercase letter, a
number, and a special character, and must not match any of your last 5 passwords.

Step 6: Log in with your new password
Return to the login page and sign in.

If you continue to have trouble, contact IT helpdesk at helpdesk@acme.example.com
or call extension 4400, Monday-Friday 08:00-17:00.'

# ── Edge case: Code-heavy ────────────────────────────────────────────────────

write_source "edge-code-heavy" '// config.rs — Application configuration loader
// NOTE: The config format switched from JSON to TOML in v0.4.0.
// All existing JSON configs must be migrated before upgrading.

use serde::Deserialize;
use std::path::PathBuf;

/// Top-level configuration. Loaded from ~/.config/vespera/config.toml by default.
/// The environment variable VESPERA_CONFIG overrides the default path.
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub cache: Option<CacheConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Bind address. Defaults to 127.0.0.1:8080.
    /// In production deployments use 0.0.0.0:443 behind a load balancer.
    pub bind: String,
    /// Max concurrent connections. Default: 1024.
    /// OS limit (ulimit -n) must be at least twice this value.
    pub max_connections: Option<usize>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL.
    /// The placeholder {{DB_PASSWORD}} is replaced from the DB_PASSWORD env var.
    pub url: String,
    /// Connection pool size. Default: 10. Typical range 20-50 for high traffic.
    /// Do not exceed max_connections on the PostgreSQL server (default 100).
    pub pool_size: Option<u32>,
    /// Maximum query execution time before cancellation. Default: 30s.
    pub statement_timeout_secs: Option<u64>,
}

// IMPORTANT: CacheConfig was added in v0.6.0 and is optional for backwards
// compatibility. Default TTL when cache is enabled: 5 minutes.
#[derive(Debug, Deserialize)]
pub struct CacheConfig {
    pub backend: CacheBackend,
    pub ttl_secs: Option<u64>,
    pub max_entries: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheBackend {
    Memory,
    Redis { url: String },
}'

# ── Edge case: Mixed languages ────────────────────────────────────────────────

write_source "edge-mixed-languages" 'Project Meridian — Status Update / Statusbericht (2026-04-10)

English summary:
Project Meridian is a data integration platform for the municipal government of
Frankfurt. Go-live date: September 1, 2026. The platform uses an event-driven
architecture with Apache Kafka. The team is distributed across Berlin and Frankfurt.

Deutsche Zusammenfassung:
Projekt Meridian ist eine Datenintegrationsplattform für die Stadtverwaltung Frankfurt.
Go-live-Datum: 1. September 2026. Die Plattform basiert auf einer ereignisgesteuerten
Architektur mit Apache Kafka. Das Team ist auf Berlin und Frankfurt verteilt.

Current status:
- Phase 1 (data ingestion): complete, running in staging.
- Phase 2 (transformation layer): 60% complete. Target completion: June 15.
- Phase 3 (reporting UI): not started. Budget approved at €120,000.

Key contacts:
- Project lead: Dr. Elena Becker (Berlin)
- Client contact: Herr Thomas Kunze (Stadtplanung Frankfurt)
- Technical lead: Arjun Mehta (Berlin)

Open risk:
Kafka cluster sizing depends on data volumes from the city legacy systems, not yet
fully measured. If volumes exceed estimates, infrastructure costs must be renegotiated.'

# ── Contradiction pairs ───────────────────────────────────────────────────────
# Each pair consists of a base document that establishes facts, followed by an
# update document that contradicts some of those facts. The eval pipeline
# processes them in alphabetical order (base before update) so that the update
# drives beta increments on the previously inserted entries.
#
# Probe query for the report (used by run-eval.sh):
#   contra-a: "Project Helios timeline and team lead"
#   contra-b: "primary database and caching layer"
#   contra-c: "engineering lead and team meeting schedule"

echo "Writing contradiction pairs ..."

write_source "contra-a-base" 'Project Helios — Status report (2026-02-10)

Project Helios is an internal developer platform being built by the infrastructure
team. The project lead is Marco Bianchi. The planned go-live date is May 15, 2026.
The initial deployment target is the company'"'"'s primary data centre in Frankfurt.
The project is funded for €340,000 through Q2 2026.

Current stack decisions:
- Container orchestration: Kubernetes 1.29 on bare metal
- Service mesh: Istio 1.21
- Observability: Grafana + Prometheus + Loki (the "LGTM stack")
- CI/CD: GitLab CI with ArgoCD for GitOps deployments

The team consists of 6 engineers and 1 part-time designer. Two contractors from
Infra GmbH are supporting the Kubernetes setup through March.

All production secrets are stored in HashiCorp Vault. The Vault cluster runs on
three dedicated nodes and is the single source of truth for credentials.'

write_source "contra-a-update" 'Project Helios — Status report (2026-04-05)

Project Helios has been restructured following the departure of Marco Bianchi, who
left the company on March 28. Sarah Chen has taken over as the new project lead.

The go-live date has been pushed back to November 3, 2026 due to scope changes and
the leadership transition. The deployment target has been expanded: Helios will now
launch in both Frankfurt and Amsterdam simultaneously to meet the new redundancy
requirements from the security audit.

The Istio service mesh has been replaced with Cilium after performance testing showed
unacceptable latency overhead under load. All service-to-service policies have been
migrated. The team size has grown to 9 engineers after the Infra GmbH contractors
were hired as full-time employees.

Budget has been revised upward to €510,000, extended through Q4 2026.'

write_source "contra-b-base" 'Atlas Backend — Architecture decisions (recorded 2025-09-12)

The Atlas backend service uses PostgreSQL 16 as its primary database. The database
runs on a managed RDS instance (db.r6g.2xlarge) in eu-central-1. All migrations are
managed with Flyway.

For caching, the team chose Redis 7 (managed via ElastiCache). Redis is used for
session storage, rate-limiting counters, and hot query results. The TTL for cached
query results is 5 minutes. The Redis cluster has 3 nodes for high availability.

The API is written in Go 1.22. The HTTP framework is chi. JSON serialisation uses
the standard library encoding/json.

Background jobs run on a dedicated worker fleet. The job queue is backed by
PostgreSQL using the pgqueue library. There is no separate message broker.

The service currently handles approximately 4,200 requests per second at peak. The
P99 latency is 38 ms measured at the load balancer.'

write_source "contra-b-update" 'Atlas Backend — Architecture update (recorded 2026-03-20)

The team has completed the migration from PostgreSQL to CockroachDB (v23.2) to
support active-active multi-region deployments. The migration was completed on
March 14 with zero downtime using a dual-write strategy. Flyway has been replaced
with golang-migrate, which has better support for CockroachDB dialect.

Redis has been removed entirely. Session storage now uses CockroachDB-backed tables
with a short TTL. Rate-limiting counters moved to an in-process token bucket
implementation (no external dependency). Hot query caching was eliminated after
profiling showed the cache hit rate was below 12% — insufficient to justify the
operational cost.

The service has been upgraded to Go 1.24. The JSON serialisation library was
switched from encoding/json to jsoniter for a measured 18% improvement in
serialisation throughput under load.'

write_source "contra-c-base" 'Orion team — Onboarding notes (2026-01-08)

Engineering lead: Alice Müller (alice@example.com). Alice is responsible for
architecture decisions, code review sign-off, and sprint planning facilitation.

Deployments are handled manually by Bob Tanaka. Bob coordinates with QA, runs the
deployment checklist, and monitors production for 30 minutes post-deploy. All
deployments require Bob'"'"'s explicit approval in the #deploy Slack channel.

The team meets every Monday at 10:00 CET for the weekly sync. The meeting is
mandatory for all full-time engineers and covers the week'"'"'s priorities, blockers,
and any architectural topics. Minutes are posted to Confluence within 24 hours.

Code is reviewed by at least two engineers before merge. Alice must approve all
changes to the core data pipeline. Security-sensitive changes additionally require
review by the security guild.'

write_source "contra-c-update" 'Orion team — Onboarding notes (2026-04-01)

Engineering lead: Carlos Rivera (carlos@example.com). Carlos took over from Alice
Müller, who moved to the platform team on March 1. Carlos is responsible for
architecture decisions, code review sign-off, and sprint planning.

Deployments are now fully automated via the CI/CD pipeline. The manual deployment
process run by Bob Tanaka was retired when the new GitOps workflow launched on
February 15. Bob has moved to the SRE team. All deployments trigger automatically
on merge to main, with a 10-minute canary phase and automatic rollback on error rate
spike.

The team weekly sync moved from Monday to Wednesday at 14:00 CET. The Monday slot
conflicted with company-wide all-hands. The meeting format is unchanged.

The two-reviewer rule remains, but Carlos'"'"' approval is now required for data
pipeline changes (previously Alice). The security guild review requirement is
unchanged.'

echo ""
echo "Done. $(find "${SOURCES_DIR}" -name '*.txt' | wc -l) source files written to ${SOURCES_DIR}"
