# Pipecat POC (Phase 1)

This folder contains a **separate Pipecat proof of concept** so the current Chess Coach app can remain stable.

## Goal (Phase 1)

Validate a **voice-first Pipecat pipeline** without touching the chess UI:
- audio input
- LLM response
- audio output

This phase is intentionally isolated from `/main.py`.

## Why this is separate

- The current app is working and should remain the fallback (`start.command`)
- Pipecat introduces more complexity (streaming, async pipelines, transport setup)
- This POC lets us experiment safely and compare UX/latency before integrating

## Progressive roadmap

1. **Phase 1 (this folder)**: minimal voice-only Pipecat POC
2. **Phase 2**: plug local `pocket-tts` into the Pipecat voice pipeline
3. **Phase 3**: add Voxtral streaming STT
4. **Phase 4**: connect the chess coaching logic
5. **Phase 1.6**: realtime voice loop (STT -> LLM -> TTS)

## Current files

- `main.py`: Phase 1 doctor (checks Python / Pipecat install / Pocket TTS reachability)
- `config.example.env`: POC-specific environment variables
- `pyproject.toml`: isolated `uv` project for Pipecat dependencies

## First-time setup

Copy the POC config file (optional but recommended):

```bash
cp pipecat_poc/config.example.env pipecat_poc/config.env
```

Install the POC dependencies in an isolated environment:

```bash
uv sync --project pipecat_poc
```

This now also installs the Mistral Python SDK with realtime support for the
`mic-realtime` mode (Voxtral streaming STT).

### Local audio support (Phase 1.3)

For microphone tests with Pipecat local audio prerequisites:

```bash
brew install portaudio
uv add --project pipecat_poc 'pipecat-ai[local]'
```

## Run (Phase 1)

Use the launcher at the repo root:

```bash
./start-pipecat.command
```

At this stage, the launcher runs a **doctor script** that validates:
- Python/runtime info
- Pocket TTS local server reachability
- whether `pipecat-ai` is installed in the dedicated POC env

### Demo mode (minimal Pipecat pipeline)

This is the first runnable Pipecat pipeline in the POC (text-in -> processor -> text-out).

One-off:

```bash
./start-pipecat.command --mode demo --demo-text "Hello from Pipecat"
```

Or set it in `pipecat_poc/config.env`:

```bash
PIPECAT_POC_MODE=demo
```

### Demo + Pocket TTS (Phase 1.2)

This mode keeps the same minimal Pipecat pipeline, but also calls your local
`pocket-tts` server and saves a WAV file generated from the pipeline output.

Make sure `pocket-tts` is running first (for example on `127.0.0.1:8787`).

```bash
./start-pipecat.command --mode demo-tts --demo-text "Hello from Pipecat"
```

Auto-play the generated WAV on macOS:

```bash
./start-pipecat.command --mode demo-tts --demo-text "Hello from Pipecat" --autoplay
```

Output files are saved to:

```text
pipecat_poc/output/demo-tts-YYYYMMDD-HHMMSS.wav
```

### Microphone doctor (Phase 1.3 prep)

Checks whether local audio prerequisites are installed and lists audio devices.

```bash
./start-pipecat.command --mode mic-doctor
```

### Microphone record test (Phase 1.3)

Records a short WAV from your default microphone (no STT yet).

```bash
./start-pipecat.command --mode mic-record --duration 3
```

Output file:

```text
pipecat_poc/output/mic-sample-YYYYMMDD-HHMMSS.wav
```

### Microphone -> Mistral transcription (Phase 1.4, batch STT)

Records a short WAV from your default microphone, then sends it to Mistral
audio transcription (Voxtral batch, not streaming yet).

Requires:
- `MISTRAL_API_KEY` in `.env` (already used by the main app)
- local audio support installed (`pipecat-ai[local]`)

```bash
./start-pipecat.command --mode mic-transcribe --duration 4
```

Outputs:
- WAV file in `pipecat_poc/output/`
- transcript `.txt` file in `pipecat_poc/output/`

Optional model override:

```bash
PIPECAT_POC_TRANSCRIBE_MODEL=voxtral-mini-latest
```

### Microphone -> Mistral realtime transcription (Phase 1.5, Voxtral streaming STT)

Streams microphone audio to Mistral realtime transcription (Voxtral) and prints
transcript deltas as they arrive.

Requires:
- `MISTRAL_API_KEY` in `.env`
- local audio support installed (`pipecat-ai[local]`)
- POC dependencies synced (`uv sync --project pipecat_poc`)

```bash
./start-pipecat.command --mode mic-realtime --duration 6
```

Tuning options:

```bash
./start-pipecat.command --mode mic-realtime --duration 6 --chunk-ms 200 --target-delay-ms 800
```

Output:
- transcript `.txt` file in `pipecat_poc/output/` (when text is received)

Default realtime model:

```bash
PIPECAT_POC_REALTIME_MODEL=voxtral-mini-transcribe-realtime-2602
```

Optional model override:

```bash
PIPECAT_POC_REALTIME_MODEL=<another-realtime-model>
```

### Realtime voice loop (Phase 1.6)

End-to-end local voice loop:
- microphone -> Voxtral realtime STT
- transcript -> Mistral chat reply
- reply -> local Pocket TTS WAV

Requires:
- `pocket-tts` running locally
- `MISTRAL_API_KEY` in `.env`

```bash
./start-pipecat.command --mode voice-loop --duration 6 --autoplay
```

Chess coach mode (recommended): provide a FEN so the reply uses a chess-coach
prompt similar to the main app `/ask` endpoint.

```bash
./start-pipecat.command --mode voice-loop --duration 6 --autoplay --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

You can also set a default position:

```bash
PIPECAT_POC_FEN="startpos-fen-here"
```

Outputs:
- `pipecat_poc/output/voice-loop-*.transcript.txt`
- `pipecat_poc/output/voice-loop-*.reply.txt`
- `pipecat_poc/output/voice-loop-*.wav`

Optional chat model override:

```bash
PIPECAT_POC_CHAT_MODEL=mistral-small-latest
```

## Notes for future integration

- Keep the current app as the stable version:
  - `./start.command`
- Keep the Pipecat POC separate until:
  - startup is reliable
  - TTS/STT are validated
  - latency is better than the current flow
