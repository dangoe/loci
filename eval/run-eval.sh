#!/usr/bin/env bash
# run-eval.sh — Evaluate loci memory extraction quality.
#
# Runs all source documents through the full extraction pipeline and records:
#   - Per-document: inserted / merged / promoted / discarded counts
#   - Final section: Bayesian store probes for the contradiction pairs, showing
#     alpha, beta, and derived Bayesian confidence for each matched entry.
#
# Prerequisites:
#   1. Run ./eval/fetch-sources.sh to populate eval/sources/
#   2. loci must be fully configured (loci.toml with a running store and models)
#   3. For Bayesian betas to appear, [memory.extraction.pipeline] must be
#      configured. Without it, loci uses the simple extractor which does not
#      update alpha/beta.
#   4. Start with an empty memory store for reproducible results. Entries from
#      previous runs will influence scoring of subsequent runs.
#
# Usage:
#   ./eval/run-eval.sh [extra loci flags...]
#
# Examples:
#   ./eval/run-eval.sh
#   ./eval/run-eval.sh --guidelines "Focus on technical facts only"
#   ./eval/run-eval.sh --max-entries 20
#
# Requires: cargo, jq
# Output:   eval/report-YYYYMMDD-HHMMSS.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCES_DIR="${SCRIPT_DIR}/sources"
REPORT_FILE="${SCRIPT_DIR}/report-$(date +%Y%m%d-%H%M%S).md"
BODY_FILE="$(mktemp -t loci-eval-body.XXXXXX.md)"
trap 'rm -f "${BODY_FILE}"' EXIT
LOCI_BIN="${WORKSPACE_DIR}/target/debug/loci"
EXTRA_ARGS=("$@")

# Outcome counters — fed to the banner at the top of the final report.
TOTAL=0
OK_COUNT=0
ERROR_COUNT=0
TOTAL_INSERTED=0
TOTAL_MERGED=0
TOTAL_PROMOTED=0
TOTAL_DISCARDED=0

# ── Dependency checks ─────────────────────────────────────────────────────────

if ! command -v jq &>/dev/null; then
    echo "Error: 'jq' is required but not found in PATH." >&2
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    echo "Error: 'cargo' is required but not found in PATH." >&2
    exit 1
fi

# ── Build the CLI binary ──────────────────────────────────────────────────────

echo "Building loci CLI ..."
cargo build -p loci-cli --manifest-path "${WORKSPACE_DIR}/Cargo.toml"
echo ""

if [[ ! -d "${SOURCES_DIR}" ]]; then
    echo "Error: sources directory not found. Run ./eval/fetch-sources.sh first." >&2
    exit 1
fi

mapfile -t FIXTURES < <(find "${SOURCES_DIR}" -maxdepth 1 -name '*.txt' | sort)

if [[ ${#FIXTURES[@]} -eq 0 ]]; then
    echo "Error: no .txt files in ${SOURCES_DIR}. Run ./eval/fetch-sources.sh first." >&2
    exit 1
fi

# ── Accumulators for summary table ────────────────────────────────────────────

declare -a SUMMARY_NAMES=()
declare -a SUMMARY_INSERTED=()
declare -a SUMMARY_MERGED=()
declare -a SUMMARY_PROMOTED=()
declare -a SUMMARY_DISCARDED=()

# ── Process each source document ─────────────────────────────────────────────
#
# Per-document output is accumulated in BODY_FILE; the final REPORT_FILE is
# assembled at the end so the banner can sit above the per-document details.

echo "Processing ${#FIXTURES[@]} documents ..."
echo ""

for FIXTURE in "${FIXTURES[@]}"; do
    DOCNAME="$(basename "${FIXTURE}")"
    TOTAL=$((TOTAL + 1))
    echo "  ${DOCNAME} ..."

    {
        echo "## ${DOCNAME}"
        echo ""
    } >>"${BODY_FILE}"

    # Source excerpt (first 500 chars)
    EXCERPT="$(head -c 500 "${FIXTURE}")"
    {
        echo "### Source excerpt"
        echo ""
        echo '```'
        echo "${EXCERPT}"
        if [[ ${#EXCERPT} -ge 500 ]]; then
            echo "[... truncated]"
        fi
        echo '```'
        echo ""
    } >>"${BODY_FILE}"

    # Run full pipeline extraction (no --dry-run)
    set +e
    RAW_OUTPUT="$("${LOCI_BIN}" memory extract -f "${FIXTURE}" "${EXTRA_ARGS[@]}" 2>&1)"
    EXIT_CODE=$?
    set -e

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        {
            echo "### Error"
            echo ""
            echo '```'
            echo "${RAW_OUTPUT}"
            echo '```'
            echo ""
            echo "---"
            echo ""
        } >>"${BODY_FILE}"
        SUMMARY_NAMES+=("${DOCNAME}")
        SUMMARY_INSERTED+=("ERR")
        SUMMARY_MERGED+=("-")
        SUMMARY_PROMOTED+=("-")
        SUMMARY_DISCARDED+=("-")
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi

    # Parse pipeline output: {"inserted": N, "merged": N, "promoted": N, "discarded": N}
    if echo "${RAW_OUTPUT}" | jq -e 'has("inserted")' &>/dev/null; then
        INS="$(echo "${RAW_OUTPUT}"   | jq '.inserted')"
        MRG="$(echo "${RAW_OUTPUT}"   | jq '.merged')"
        PRO="$(echo "${RAW_OUTPUT}"   | jq '.promoted')"
        DIS="$(echo "${RAW_OUTPUT}"   | jq '.discarded')"
        TOTAL_INSERTED=$((TOTAL_INSERTED + INS))
        TOTAL_MERGED=$((TOTAL_MERGED   + MRG))
        TOTAL_PROMOTED=$((TOTAL_PROMOTED + PRO))
        TOTAL_DISCARDED=$((TOTAL_DISCARDED + DIS))
        {
            echo "### Pipeline result"
            echo ""
            echo "Inserted: **${INS}** | Merged: **${MRG}** | Promoted: **${PRO}** | Discarded: **${DIS}**"
            echo ""
            echo "### Annotations"
            echo ""
            echo "<!-- Add your review notes here -->"
            echo ""
            echo "---"
            echo ""
        } >>"${BODY_FILE}"
        SUMMARY_NAMES+=("${DOCNAME}")
        SUMMARY_INSERTED+=("${INS}")
        SUMMARY_MERGED+=("${MRG}")
        SUMMARY_PROMOTED+=("${PRO}")
        SUMMARY_DISCARDED+=("${DIS}")

    # Fallback: simple extractor output {"added": [...], "failures": [...]}
    elif echo "${RAW_OUTPUT}" | jq -e 'has("added")' &>/dev/null; then
        ADDED="$(echo "${RAW_OUTPUT}" | jq '.added | length')"
        FAILS="$(echo "${RAW_OUTPUT}" | jq '.failures | length')"
        TOTAL_INSERTED=$((TOTAL_INSERTED + ADDED))
        {
            echo "### Simple extractor result"
            echo ""
            echo "> **Note:** pipeline not configured — alpha/beta will not be updated."
            echo ""
            echo "Added: **${ADDED}** | Failures: **${FAILS}**"
            echo ""
            echo "### Annotations"
            echo ""
            echo "<!-- Add your review notes here -->"
            echo ""
            echo "---"
            echo ""
        } >>"${BODY_FILE}"
        SUMMARY_NAMES+=("${DOCNAME}")
        SUMMARY_INSERTED+=("${ADDED}")
        SUMMARY_MERGED+=("-")
        SUMMARY_PROMOTED+=("-")
        SUMMARY_DISCARDED+=("-")

    else
        {
            echo "### Unexpected output"
            echo ""
            echo '```'
            echo "${RAW_OUTPUT}"
            echo '```'
            echo ""
            echo "---"
            echo ""
        } >>"${BODY_FILE}"
        SUMMARY_NAMES+=("${DOCNAME}")
        SUMMARY_INSERTED+=("?")
        SUMMARY_MERGED+=("-")
        SUMMARY_PROMOTED+=("-")
        SUMMARY_DISCARDED+=("-")
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi

    OK_COUNT=$((OK_COUNT + 1))
done

# ── Bayesian probe queries ────────────────────────────────────────────────────
#
# Query the store for topics that span the contradiction pairs.  Each probe
# shows the matched entries with their alpha/beta counters and derived Bayesian
# confidence so you can verify that contradictions drove beta updates.
#
# Expected pattern after both halves of a pair are processed:
#   - The original entry has beta > 0 (contradiction bumped it)
#   - The contradicting entry has beta = 0 or a lower score

PROBE_TOPICS=(
    "Project Helios timeline and team lead"
    "primary database and caching layer"
    "engineering lead and team meeting schedule"
)

{
    echo "## Bayesian store probes"
    echo ""
    echo "Entries matched for the three contradiction-pair topics."
    echo "A non-zero **beta** indicates the pipeline detected a contradiction against"
    echo "an existing entry and downgraded its Bayesian confidence."
    echo ""
} >>"${BODY_FILE}"

for PROBE in "${PROBE_TOPICS[@]}"; do
    {
        echo "### Query: \"${PROBE}\""
        echo ""
    } >>"${BODY_FILE}"

    set +e
    PROBE_OUT="$("${LOCI_BIN}" memory query "${PROBE}" --max-results 10 2>&1)"
    PROBE_EXIT=$?
    set -e

    if [[ ${PROBE_EXIT} -ne 0 ]]; then
        {
            echo '```'
            echo "${PROBE_OUT}"
            echo '```'
            echo ""
        } >>"${BODY_FILE}"
        continue
    fi

    # Check for empty result
    if ! echo "${PROBE_OUT}" | jq -e 'type == "array"' &>/dev/null || \
       [[ "$(echo "${PROBE_OUT}" | jq 'length')" -eq 0 ]]; then
        {
            echo "_No entries matched._"
            echo ""
        } >>"${BODY_FILE}"
        continue
    fi

    {
        echo "| Content | confidence | alpha | beta | bayes_conf | kind | retrieval_score |"
        echo "|---------|------------|-------|------|------------|------|-----------------|"
    } >>"${BODY_FILE}"

    while IFS= read -r ROW; do
        CONTENT="$(    echo "${ROW}" | jq -r '.content')"
        CONF="$(        echo "${ROW}" | jq -r '(.confidence // "–")')"
        ALPHA="$(       echo "${ROW}" | jq -r '(.review.alpha // "–")')"
        BETA="$(        echo "${ROW}" | jq -r '(.review.beta  // "–")')"
        BAYES="$(       echo "${ROW}" | jq -r '
            if (.review.alpha != null) and (.review.beta != null)
                and ((.review.alpha + .review.beta) > 0)
            then ((.review.alpha / (.review.alpha + .review.beta)) * 100 | round / 100)
            else "–"
            end')"
        KIND="$(        echo "${ROW}" | jq -r '.kind')"
        SCORE="$(       echo "${ROW}" | jq -r '(.score // "–")')"
        echo "| ${CONTENT} | ${CONF} | ${ALPHA} | ${BETA} | ${BAYES} | ${KIND} | ${SCORE} |" >>"${BODY_FILE}"
    done < <(echo "${PROBE_OUT}" | jq -c '.[]')

    echo "" >>"${BODY_FILE}"
done

# ── Assemble final report ─────────────────────────────────────────────────────
#
# Layout: header → status banner → per-document body → summary table.
# The banner sits at the top so regressions are visible without scrolling.

FAILED_TOTAL=${ERROR_COUNT}
if [[ ${FAILED_TOTAL} -eq 0 ]]; then
    STATUS_LINE="**Status:** PASS — all ${TOTAL} documents processed without error."
else
    STATUS_LINE="**Status:** FAIL — ${FAILED_TOTAL}/${TOTAL} documents errored."
fi

{
    echo "# loci Memory Extraction Evaluation Report"
    echo ""
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "**Command:** \`${LOCI_BIN} memory extract${EXTRA_ARGS:+ ${EXTRA_ARGS[*]}}\`"
    echo ""
    echo "**Sources:** ${SOURCES_DIR}"
    echo ""
    echo "${STATUS_LINE}"
    echo ""
    echo "**Pipeline totals:** inserted ${TOTAL_INSERTED} | merged ${TOTAL_MERGED} | promoted ${TOTAL_PROMOTED} | discarded ${TOTAL_DISCARDED}"
    echo ""
    echo "---"
    echo ""
} >"${REPORT_FILE}"

cat "${BODY_FILE}" >>"${REPORT_FILE}"

{
    echo "## Summary"
    echo ""
    echo "| Document | Inserted | Merged | Promoted | Discarded |"
    echo "|----------|----------|--------|----------|-----------|"
} >>"${REPORT_FILE}"

for i in "${!SUMMARY_NAMES[@]}"; do
    echo "| ${SUMMARY_NAMES[$i]} | ${SUMMARY_INSERTED[$i]} | ${SUMMARY_MERGED[$i]} | ${SUMMARY_PROMOTED[$i]} | ${SUMMARY_DISCARDED[$i]} |" >>"${REPORT_FILE}"
done

echo "" >>"${REPORT_FILE}"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "Report written to: ${REPORT_FILE}"

if [[ ${FAILED_TOTAL} -ne 0 ]]; then
    exit 1
fi
