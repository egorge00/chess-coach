#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

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

echo "Demarrage du serveur..."
echo "Ouvre ensuite: http://127.0.0.1:8001"
echo "Arret: Ctrl+C"

# Ouvre la page dans le navigateur apres un court delai, puis laisse uvicorn au premier plan.
( sleep 1; open "http://127.0.0.1:8001" >/dev/null 2>&1 || true ) &
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001
