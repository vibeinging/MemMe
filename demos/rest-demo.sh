#!/usr/bin/env bash
# MemMe REST demo — the preferred local trial path.
#
# What it proves, in order:
#   1. Write memories for one owner and two pets (local ONNX embeddings, no API key).
#   2. Recall them in the right scope: Momo's promise is invisible to Luna.
#   3. Stop the server completely, restart it on the same SQLite file, and
#      recall the same results again — memory survives restarts.
#
# Requirements: macOS or Linux (x64 / arm64), Rust, Python 3 and curl.
# First run downloads the local bge-small-zh-v1.5 ONNX model before serving.
# Later runs reuse the model cache under MEMME_DEMO_DIR.
#
# Reset the demo data:  rm -rf ~/.cache/memme-demo   (or $MEMME_DEMO_DIR)

set -euo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="${MEMME_DEMO_DIR:-$HOME/.cache/memme-demo}"
PORT="${MEMME_DEMO_PORT:-18070}"
START_TIMEOUT="${MEMME_DEMO_START_TIMEOUT:-600}"
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
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Preflight ──────────────────────────────────────────────────────────────
say "Preflight"
case "$(uname -s):$(uname -m)" in
  Darwin:x86_64|Darwin:arm64|Darwin:aarch64|Linux:x86_64|Linux:aarch64|Linux:arm64) ;;
  *) echo "Unsupported host: $(uname -s) $(uname -m). This demo needs macOS or Linux on x64/arm64." >&2; exit 1 ;;
esac
command -v cargo >/dev/null 2>&1 || { echo "cargo not found. Install Rust from https://rustup.rs first." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required for the demo checks." >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
[[ "${START_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "MEMME_DEMO_START_TIMEOUT must be a positive number of seconds." >&2; exit 1; }

# Fail before downloading, building, or writing anything to an existing service.
python3 - "${PORT}" <<'PY'
import socket, sys
try:
    port = int(sys.argv[1])
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
except (OSError, ValueError) as error:
    sys.exit(f"Cannot use demo port: {error}. Set MEMME_DEMO_PORT to a free port.")
PY
API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
DEMO_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${DEMO_DIR}")"

export MEMME_VEXDB_LITE_EXTENSION
MEMME_VEXDB_LITE_EXTENSION="$(cd "${PROJECT_ROOT}" && env -u GITHUB_ENV bash scripts/download-vexdb-lite-extension.sh)"
note "VexDB-Lite extension: ${MEMME_VEXDB_LITE_EXTENSION}"

export ORT_DYLIB_PATH
ORT_DYLIB_PATH="$(cd "${PROJECT_ROOT}" && env -u GITHUB_ENV bash scripts/download-onnx-runtime.sh)"
note "ONNX Runtime: ${ORT_DYLIB_PATH}"

say "Building memme-server (first build takes a few minutes)"
(cd "${PROJECT_ROOT}" && cargo build --locked --release -p memme-server)
TARGET_DIR="$(cd "${PROJECT_ROOT}" && cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
SERVER_BIN="${TARGET_DIR}/${CARGO_BUILD_TARGET:+${CARGO_BUILD_TARGET}/}release/memme-server"

mkdir -p "${DEMO_DIR}"
DB="${DEMO_DIR}/demo-memory.db"

# ── Helpers ────────────────────────────────────────────────────────────────
start_server() {
  (
    cd "${DEMO_DIR}"
    unset LLM_API_KEY LLM_URL OPENAI_API_KEY EMBEDDING_API_KEY EMBEDDING_URL
    export MEMME_API_KEY="${API_KEY}"
    exec "${SERVER_BIN}" --host 127.0.0.1 --port "${PORT}" --db-path "${DB}" \
      --embedding-provider onnx --onnx-embedding-model bge-small-zh-v15
  ) >"${DEMO_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Demo server exited during startup:" >&2
      tail -20 "${DEMO_DIR}/server.log" >&2 || true
      exit 1
    fi
    # /health is public: authenticate using this run's unique key before writing.
    if curl -fsS --connect-timeout 1 --max-time 5 "${BASE}/v1/recall" \
      -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
      -d '{"query":"ready","user_id":"demo-readiness","limit":1}' >/dev/null 2>&1; then
      kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Demo server exited after readiness check." >&2; exit 1; }
      return 0
    fi
    sleep 0.5
  done
  echo "Server did not become healthy; log tail:" >&2
  tail -20 "${DEMO_DIR}/server.log" >&2 || true
  exit 1
}

stop_server() {
  say "Stopping the server (full process exit)"
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "Demo server exited unexpectedly." >&2; exit 1; }
  kill "${SERVER_PID}"
  local code=0
  wait "${SERVER_PID}" || code=$?
  SERVER_PID=""
  [[ "${code}" -eq 0 || "${code}" -eq 143 ]] || { echo "Unexpected server exit: ${code}" >&2; exit 1; }
}

post_events() { # session agent message
  local session="$1" agent="$2" content="$3"
  local agent_json=""
  if [[ -n "${agent}" ]]; then agent_json="\"agent_id\":\"${agent}\","; fi
  local body
  body="$(curl -sS --max-time 30 -w '\n%{http_code}' "${BASE}/v1/events" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"${session}\",\"user_id\":\"owner-001\",${agent_json}\"app_id\":\"demo\",\"messages\":[{\"event_id\":\"${session}-e1\",\"role\":\"user\",\"content\":${content}}]}")"
  local code
  code="$(tail -n1 <<<"${body}")"
  if [[ "${code}" != "200" ]]; then
    echo "POST /v1/events (${session}) returned HTTP ${code}:" >&2
    sed '$d' <<<"${body}" >&2
    exit 1
  fi
}

recall() { # query agent -> prints JSON
  curl -fsS --max-time 30 "${BASE}/v1/recall" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"$1\",\"user_id\":\"owner-001\",\"agent_id\":\"$2\",\"limit\":5}"
}

contains_memory() {
  python3 -c 'import json,sys; data=json.load(sys.stdin); sys.exit(0 if data.get("success") is True and any(sys.argv[1] in m["content"] for m in data["data"]["memories"]) else 1)' "$1"
}

# ── 1. Write ───────────────────────────────────────────────────────────────
say "Starting server (first model download can take several minutes)"
[[ -f "${DB}" ]] && note "reusing existing ${DB} (delete ${DEMO_DIR} to reset)"
note "startup log: ${DEMO_DIR}/server.log; timeout: ${START_TIMEOUT}s"
start_server

post_events "s-momo-1" "momo" '"Momo和主人有一个约定：每天晚上八点一起看一集纪录片。"'
post_events "s-luna-1" "luna" '"Luna最喜欢的玩具是红色的毛线球。"'
post_events "s-owner-1" ""     '"主人对花生严重过敏。"'
[[ -f "${DB}" ]] || { echo "The requested demo database was not created." >&2; exit 1; }
passed "wrote: Momo's promise, Luna's toy, owner's allergy"

# ── 2. Recall in scope ─────────────────────────────────────────────────────
say "Recall as Momo"
MOMO_JSON="$(recall "晚上八点要做什么？" "momo")"
printf '%.600s\n\n' "${MOMO_JSON}"

say "Recall as Luna — Momo's promise must NOT appear"
LUNA_JSON="$(recall "晚上八点要做什么？" "luna")"
printf '%.600s\n\n' "${LUNA_JSON}"

say "Recall owner-global fact as Momo (owner memory is shared with pets)"
ALLERGY_JSON="$(recall "准备零食要注意什么？" "momo")"
printf '%.600s\n\n' "${ALLERGY_JSON}"

if contains_memory "纪录片" <<<"${MOMO_JSON}"; then passed "Momo sees his promise"; else failed "Momo's promise not recalled"; fi
if contains_memory "纪录片" <<<"${LUNA_JSON}"; then failed "Momo's promise leaked to Luna"; else passed "no leak to Luna"; fi
if contains_memory "毛线球" <<<"${LUNA_JSON}"; then passed "Luna sees her toy"; else failed "Luna's toy not recalled"; fi
if contains_memory "花生" <<<"${ALLERGY_JSON}"; then passed "owner-global allergy visible to Momo"; else failed "owner-global allergy not recalled"; fi

# ── 3. Restart and recall again ────────────────────────────────────────────
stop_server
say "Restarting the server on the same SQLite file"
start_server

MOMO_JSON_2="$(recall "晚上八点要做什么？" "momo")"
LUNA_JSON_2="$(recall "晚上八点要做什么？" "luna")"
printf '%.600s\n\n' "${MOMO_JSON_2}"

if contains_memory "纪录片" <<<"${MOMO_JSON_2}"; then passed "after restart, Momo still remembers"; else failed "memory lost after restart"; fi
if contains_memory "纪录片" <<<"${LUNA_JSON_2}"; then failed "isolation lost after restart"; else passed "isolation survives restarts"; fi

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
