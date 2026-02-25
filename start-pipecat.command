#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Pipecat POC (Phase 1)..."

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

if [ -f "pipecat_poc/config.env" ]; then
  set -a
  source "pipecat_poc/config.env"
  set +a
fi

_parse_host_port_from_url() {
  local url="$1"
  local clean="${url#http://}"
  clean="${clean#https://}"
  local hostport="${clean%%/*}"
  local host="${hostport%%:*}"
  local port="${hostport##*:}"
  if [ -z "$host" ]; then
    host="127.0.0.1"
  fi
  if [ "$host" = "$port" ] || [ -z "$port" ]; then
    port="8787"
  fi
  echo "$host" "$port"
}

_is_listening() {
  local host="$1"
  local port="$2"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | grep -q "$host\\|\\*:$port"
}

if command -v uv >/dev/null 2>&1; then
  echo "Using uv project environment (pipecat_poc)"
  echo "Tip: run 'uv sync --project pipecat_poc' once if Pipecat is not installed yet."
  echo "Default mode (from env): ${PIPECAT_POC_MODE:-doctor}"
  echo ""

  # CLI mode remains available when args are passed.
  if [ "$#" -gt 0 ]; then
    exec uv run --project pipecat_poc python -m pipecat_poc.main "$@"
  fi

  # Web UI mode (double-click default)
  POCKET_URL="${POCKET_TTS_URL:-${LOCAL_TTS_URL:-http://127.0.0.1:8787}}"
  read -r POCKET_HOST POCKET_PORT <<< "$(_parse_host_port_from_url "$POCKET_URL")"
  POC_WEB_HOST="${PIPECAT_POC_WEB_HOST:-127.0.0.1}"
  POC_WEB_PORT="${PIPECAT_POC_WEB_PORT:-8002}"
  POC_WEB_URL="http://${POC_WEB_HOST}:${POC_WEB_PORT}"
  POCKET_LOG="${SCRIPT_DIR}/.pocket-tts.log"
  POC_WEB_LOG="${SCRIPT_DIR}/.pipecat-poc-web.log"
  POCKET_PID=""
  WEB_PID=""

  cleanup() {
    if [ -n "$WEB_PID" ] && kill -0 "$WEB_PID" 2>/dev/null; then
      kill "$WEB_PID" 2>/dev/null || true
    fi
    if [ -n "$POCKET_PID" ] && kill -0 "$POCKET_PID" 2>/dev/null; then
      kill "$POCKET_PID" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  if _is_listening "$POCKET_HOST" "$POCKET_PORT"; then
    echo "pocket-tts already running on ${POCKET_HOST}:${POCKET_PORT}"
  else
    POCKET_TTS_BIN=""
    if command -v pocket-tts >/dev/null 2>&1; then
      POCKET_TTS_BIN="$(command -v pocket-tts)"
    elif [ -x "$HOME/.local/bin/pocket-tts" ]; then
      POCKET_TTS_BIN="$HOME/.local/bin/pocket-tts"
    fi

    if [ -z "$POCKET_TTS_BIN" ]; then
      echo "Warning: pocket-tts not found. Web UI will start, but voice-loop TTS may fail."
    else
      echo "Starting pocket-tts on ${POCKET_HOST}:${POCKET_PORT}..."
      "$POCKET_TTS_BIN" serve --host "$POCKET_HOST" --port "$POCKET_PORT" >"$POCKET_LOG" 2>&1 &
      POCKET_PID=$!
      for _ in {1..120}; do
        if _is_listening "$POCKET_HOST" "$POCKET_PORT"; then
          echo "pocket-tts ready on ${POCKET_HOST}:${POCKET_PORT}"
          break
        fi
        sleep 0.25
      done
      if ! _is_listening "$POCKET_HOST" "$POCKET_PORT"; then
        echo "Warning: pocket-tts did not start in time (see $POCKET_LOG)"
      fi
    fi
  fi

  if _is_listening "$POC_WEB_HOST" "$POC_WEB_PORT"; then
    echo "Pipecat web UI already running on ${POC_WEB_URL}"
  else
    echo "Starting Pipecat web UI on ${POC_WEB_URL}..."
    uv run --project pipecat_poc python -m uvicorn pipecat_poc.web:app --host "$POC_WEB_HOST" --port "$POC_WEB_PORT" >"$POC_WEB_LOG" 2>&1 &
    WEB_PID=$!
    for _ in {1..80}; do
      if _is_listening "$POC_WEB_HOST" "$POC_WEB_PORT"; then
        echo "Pipecat web UI ready on ${POC_WEB_URL}"
        break
      fi
      sleep 0.25
    done
    if ! _is_listening "$POC_WEB_HOST" "$POC_WEB_PORT"; then
      echo "Error: Pipecat web UI failed to start (see $POC_WEB_LOG)"
      read -r "REPLY?Press Enter to close..."
      exit 1
    fi
  fi

  sleep 0.5
  open "$POC_WEB_URL" >/dev/null 2>&1 || true

  echo ""
  echo "Pipecat POC Web UI is running."
  echo "  URL: $POC_WEB_URL"
  echo "  pocket-tts: $POCKET_URL"
  echo "  Logs: $POC_WEB_LOG / $POCKET_LOG"
  echo ""
  echo "Press Ctrl+C to stop services launched by this script."
  wait
fi

if [ -x ".venv/bin/python" ]; then
  PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  echo "Error: no Python interpreter found."
  read -r "REPLY?Press Enter to close..."
  exit 1
fi

echo "uv not found, falling back to plain Python (Pipecat deps may be missing)."
echo "Using Python: $PY_BIN"
echo "Stable app remains available via ./start.command"
echo ""

exec "$PY_BIN" -m pipecat_poc.main "$@"
