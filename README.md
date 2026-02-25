# Chess Coach

A voice-powered chess coach: you play White, the coach comments on your move, plays Black's response, then gives you a simple plan for the next move.

## Why this project

I wanted a fast prototype for chess coaching with:
- a clean, enjoyable UI
- short pedagogical feedback
- voice interaction (mic + transcription + TTS)
- a minimal Python backend

## Demo Highlights

- Interactive chessboard (White side)
- Coach feedback on your last move
- Black reply selected by an LLM
- Clear next-move advice
- Free-form follow-up questions to the coach (microphone)
- Audio playback of coach responses (Pocket TTS by Kyutai)

## Screenshots

### Main UI
The main coaching screen: interactive board, move-by-move feedback, voice controls, and question panel.

![Chess Coach main UI](docs/screenshots/hero-ui.png)

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Single-page HTML / CSS / vanilla JS
- **Chess logic**: `chess.js`
- **Realtime voice orchestration**: Pipecat
- **LLM**: Mistral (`mistral-small-latest`)
- **STT (streaming)**: Voxtral (`voxtral-mini-transcribe-realtime-2602`)
- **TTS**: Pocket TTS (Kyutai, local)

## Architecture

- `main.py`
  - API endpoints (`/coach_move`, `/ask`, `/transcribe`, `/tts`)
  - Embedded web page (HTML/CSS/JS)
- `start.command`
  - One-click macOS launcher (opens the browser automatically)

## Local Setup

### Prerequisites

- Python 3.10+ (3.9 may also work depending on your environment)
- A Mistral API key
- Pocket TTS (Kyutai) running locally for voice output

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
MISTRAL_API_KEY="<your_mistral_key>"
LOCAL_TTS_URL="http://127.0.0.1:8787"
# Optional local Pocket TTS voice preset
# LOCAL_TTS_VOICE="alba"
```

### Installation

```bash
cd /path/to/chess-coach-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run locally

Terminal option:

```bash
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Then open [http://127.0.0.1:8001](http://127.0.0.1:8001)

macOS one-click option:

1. Double-click `start.command`
2. The server starts
3. Your browser opens automatically at `http://127.0.0.1:8001`

## API Endpoints

- `GET /`
- `GET /health`
- `POST /coach_move`
- `POST /ask`
- `POST /transcribe`
- `POST /tts`

## Security / Notes

- API keys stay server-side (`os.getenv`)
- `.env` is ignored by Git (`.gitignore`)
- Fallback to a random legal move if an LLM Black move is invalid
- The microphone stream is released after recording stops
