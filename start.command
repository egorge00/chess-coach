#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

POCKET_TTS_PID=""

if [ ! -d ".venv" ]; then
  echo "Erreur: .venv introuvable dans $SCRIPT_DIR"
  echo "Lance d'abord l'installation initiale."
  read -r "REPLY?Appuie sur Entree pour fermer..."
  exit 1
fi

source .venv/bin/activate

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

cleanup() {
  if [ -n "$POCKET_TTS_PID" ]; then
    kill "$POCKET_TTS_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# Start local pocket-tts if available and not already running.
TTS_URL="${LOCAL_TTS_URL:-http://127.0.0.1:8787}"
TTS_HOSTPORT="${TTS_URL#http://}"
TTS_HOSTPORT="${TTS_HOSTPORT#https://}"
TTS_HOSTPORT="${TTS_HOSTPORT%%/*}"
TTS_HOST="${TTS_HOSTPORT%%:*}"
TTS_PORT="${TTS_HOSTPORT##*:}"
[ -n "$TTS_HOST" ] || TTS_HOST="127.0.0.1"
[ -n "$TTS_PORT" ] || TTS_PORT="8787"

POCKET_TTS_BIN="$(command -v pocket-tts 2>/dev/null || true)"
if [ -z "$POCKET_TTS_BIN" ] && [ -x "$HOME/.local/bin/pocket-tts" ]; then
  POCKET_TTS_BIN="$HOME/.local/bin/pocket-tts"
fi

if [ -n "$POCKET_TTS_BIN" ]; then
  if lsof -nP -iTCP:$TTS_PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "pocket-tts deja en cours sur $TTS_HOST:$TTS_PORT"
  else
    echo "Demarrage pocket-tts sur $TTS_HOST:$TTS_PORT..."
    "$POCKET_TTS_BIN" serve --host "$TTS_HOST" --port "$TTS_PORT" > "$SCRIPT_DIR/.pocket-tts.log" 2>&1 &
    POCKET_TTS_PID=$!
    i=0
    while ! lsof -nP -iTCP:$TTS_PORT -sTCP:LISTEN >/dev/null 2>&1; do
      i=$((i + 1))
      if [ $i -ge 40 ]; then
        echo "Attention: pocket-tts ne semble pas avoir demarre. Voir .pocket-tts.log"
        break
      fi
      sleep 0.5
    done
    if lsof -nP -iTCP:$TTS_PORT -sTCP:LISTEN >/dev/null 2>&1; then
      echo "pocket-tts pret sur $TTS_HOST:$TTS_PORT"
    fi
  fi
else
  echo "pocket-tts introuvable (optionnel)."
fi

echo "Demarrage du serveur..."
echo "Ouvre ensuite: http://127.0.0.1:8001"
echo "Arret: Ctrl+C"

# Ouvre la page dans le navigateur apres un court delai, puis laisse uvicorn au premier plan.
( sleep 1; open "http://127.0.0.1:8001" >/dev/null 2>&1 || true ) &
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001
