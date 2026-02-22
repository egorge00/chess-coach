# chess-coach-poc

POC local: une app FastAPI qui sert une page unique HTML/JS pour jouer les blancs, recevoir un commentaire coach, des suggestions, et un coup des noirs choisi par LLM.

## Prerequis

- Python 3.10+
- Cle API Mistral
- Cle API Gradium

## Variables d'environnement

```bash
export MISTRAL_API_KEY="<ta_cle_mistral>"
export GRADIUM_API_KEY="<ta_cle_gradium>"
# Optionnel: eu (defaut) ou us
export GRADIUM_REGION="eu"
```

## Installation

```bash
cd "/Users/ErwanGorge/Documents/New project/chess-coach-poc"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer en local

```bash
uvicorn main:app --reload --port 8001
```

Puis ouvrir: [http://127.0.0.1:8001](http://127.0.0.1:8001)

## Endpoints

- `GET /` page web unique
- `GET /health` -> `{ "ok": true }`
- `POST /coach_move`
- `POST /ask`
- `POST /transcribe`
- `POST /tts`

## Notes

- Les cles API restent uniquement serveur (`os.getenv`) et ne sont jamais exposees au navigateur.
- Timeout Mistral configure a 20 secondes.
- Si le coup noir propose est illegal, le serveur fait un fallback sur un coup legal aleatoire.

## Lancement en 1 clic (macOS)

1. Duplique `.env.example` en `.env` puis colle tes cles API.
2. Double-clique `start.command`.
3. Le serveur se lance automatiquement sur `http://127.0.0.1:8001`.

Pour arreter le serveur: `Ctrl + C` dans la fenetre Terminal ouverte.
