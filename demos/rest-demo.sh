#!/usr/bin/env bash
# MemMe REST demo — the preferred local trial path.
#
# What it proves, in order:
#   1. Write memories for one owner and two pets (local ONNX embeddings, no API key).
#   2. Recall them in the right scope: Momo's promise is invisible to Luna.
#   3. Stop the server completely, restart it on the same SQLite file, and
#      recall the same results again — memory survives restarts.
#
# Requirements: macOS or Linux (x64 / arm64) and a Rust toolchain.
# First run downloads the compact bge-small-zh-v1.5 ONNX model (~100 MB);
# later runs start instantly and work fully offline.
#
# Reset the demo data:  rm -rf ~/.cache/memme-demo   (or $MEMME_DEMO_DIR)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${MEMME_DEMO_DIR:-$HOME/.cache/memme-demo}"
PORT="${MEMME_DEMO_PORT:-18070}"
API_KEY="demo-key"
BASE="http://127.0.0.1:${PORT}"

say()    { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
note()   { printf '   \033[2m%s\033[0m\n' "$*"; }
passed() { printf '   \033[1;32mPASS\033[0m %s\n' "$*"; }
failed() { printf '   \033[1;31mFAIL\033[0m %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

FAILURES=0
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Preflight ──────────────────────────────────────────────────────────────
say "Preflight"
case "$(uname -s):$(uname -m)" in
  Darwin:x86_64|Darwin:arm64|Darwin:aarch64|Linux:x86_64|Linux:aarch64|Linux:arm64) ;;
  *) echo "Unsupported host: $(uname -s) $(uname -m). This demo needs macOS or Linux on x64/arm64." >&2; exit 1 ;;
esac
command -v cargo >/dev/null 2>&1 || { echo "cargo not found. Install Rust from https://rustup.rs first." >&2; exit 1; }

export MEMME_VEXDB_LITE_EXTENSION
MEMME_VEXDB_LITE_EXTENSION="$(cd "${ROOT}" && bash scripts/download-vexdb-lite-extension.sh)"
note "VexDB-Lite extension: ${MEMME_VEXDB_LITE_EXTENSION}"

export ORT_DYLIB_PATH
ORT_DYLIB_PATH="$(cd "${ROOT}" && bash scripts/download-onnx-runtime.sh)"
note "ONNX Runtime: ${ORT_DYLIB_PATH}"

if [[ ! -f "${ROOT}/target/release/memme-server" ]]; then
  say "Building memme-server (first build takes a few minutes)"
fi
(cd "${ROOT}" && cargo build --release -p memme-server)

mkdir -p "${DEMO_DIR}"
DB="${DEMO_DIR}/demo-memory.db"

# ── Helpers ────────────────────────────────────────────────────────────────
start_server() {
  MEMME_API_KEY="${API_KEY}" "${ROOT}/target/release/memme-server" \
    --host 127.0.0.1 --port "${PORT}" --db-path "${DB}" >"${DEMO_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    if curl -fsS "${BASE}/health" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "Server did not become healthy; log tail:" >&2
  tail -20 "${DEMO_DIR}/server.log" >&2 || true
  exit 1
}

stop_server() {
  say "Stopping the server (full process exit)"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

post_events() { # session agent message
  local session="$1" agent="$2" content="$3"
  local agent_json=""
  if [[ -n "${agent}" ]]; then agent_json="\"agent_id\":\"${agent}\","; fi
  local body
  body="$(curl -sS -w '\n%{http_code}' "${BASE}/v1/events" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"${session}\",\"user_id\":\"owner-001\",${agent_json}\"app_id\":\"demo\",\"messages\":[{\"event_id\":\"${session}-e1\",\"role\":\"user\",\"content\":${content}}]}")"
  local code
  code="$(tail -n1 <<<"${body}")"
  if [[ "${code}" != "200" ]]; then
    echo "POST /v1/events (${session}) returned HTTP ${code}:" >&2
    head -n -1 <<<"${body}" >&2
    exit 1
  fi
}

recall() { # agent query -> prints JSON
  curl -fsS "${BASE}/v1/recall" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"$1\",\"user_id\":\"owner-001\",\"agent_id\":\"$2\",\"limit\":5}"
}

# ── 1. Write ───────────────────────────────────────────────────────────────
say "Starting server on a fresh database"
[[ -f "${DB}" ]] && note "reusing existing ${DB} (delete ${DEMO_DIR} to reset)"
start_server
note "first recall below may download the local ONNX model (~100 MB, once)"

post_events "s-momo-1" "momo" '"Momo和主人有一个约定：每天晚上八点一起看一集纪录片。"'
post_events "s-luna-1" "luna" '"Luna最喜欢的玩具是红色的毛线球。"'
post_events "s-owner-1" ""     '"主人对花生严重过敏。"'
passed "wrote: Momo's promise, Luna's toy, owner's allergy"

# ── 2. Recall in scope ─────────────────────────────────────────────────────
say "Recall as Momo"
MOMO_JSON="$(recall "晚上八点要做什么？" "momo")"
echo "${MOMO_JSON}" | head -c 600; echo; echo

say "Recall as Luna — Momo's promise must NOT appear"
LUNA_JSON="$(recall "晚上八点要做什么？" "luna")"
echo "${LUNA_JSON}" | head -c 600; echo; echo

say "Recall owner-global fact as Momo (owner memory is shared with pets)"
ALLERGY_JSON="$(recall "准备零食要注意什么？" "momo")"
echo "${ALLERGY_JSON}" | head -c 600; echo; echo

if echo "${MOMO_JSON}"   | grep -q "纪录片"; then passed "Momo sees his promise"; else failed "Momo's promise not recalled"; fi
if echo "${LUNA_JSON}"   | grep -q "纪录片"; then failed "Momo's promise leaked to Luna"; else passed "no leak to Luna"; fi
if echo "${LUNA_JSON}"   | grep -q "毛线球"; then passed "Luna sees her toy"; else failed "Luna's toy not recalled"; fi
if echo "${ALLERGY_JSON}"| grep -q "花生"; then passed "owner-global allergy visible to Momo"; else failed "owner-global allergy not recalled"; fi

# ── 3. Restart and recall again ────────────────────────────────────────────
stop_server
say "Restarting the server on the same SQLite file"
start_server

MOMO_JSON_2="$(recall "晚上八点要做什么？" "momo")"
LUNA_JSON_2="$(recall "晚上八点要做什么？" "luna")"
echo "${MOMO_JSON_2}" | head -c 600; echo; echo

if echo "${MOMO_JSON_2}" | grep -q "纪录片"; then passed "after restart, Momo still remembers"; else failed "memory lost after restart"; fi
if echo "${LUNA_JSON_2}" | grep -q "纪录片"; then failed "isolation lost after restart"; else passed "isolation survives restarts"; fi

# ── Summary ────────────────────────────────────────────────────────────────
stop_server
say "Summary"
if [[ "${FAILURES}" -eq 0 ]]; then
  echo "   All checks passed. Data file: ${DB}"
  echo "   Reset the demo with: rm -rf ${DEMO_DIR}"
else
  echo "   ${FAILURES} check(s) failed. See ${DEMO_DIR}/server.log"
  echo "   Common causes: missing VexDB-Lite extension (set MEMME_VEXDB_LITE_EXTENSION),"
  echo "   port ${PORT} already in use (set MEMME_DEMO_PORT), or an interrupted model download (rerun)."
  exit 1
fi
