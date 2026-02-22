"""
Environment variables required:
- MISTRAL_API_KEY: API key for Mistral chat + transcription
- GRADIUM_API_KEY: API key for Gradium TTS

Optional for explicit Gradium voices:
- GRADIUM_VOICE_ID_WHITE
- GRADIUM_VOICE_ID_BLACK

Run locally:
uvicorn main:app --reload
"""

import base64
import io
import json
import os
import random
import re
import unicodedata
from typing import Literal, Optional

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

app = FastAPI(title="Chess Coach POC")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY", "").strip()
GRADIUM_REGION = os.getenv("GRADIUM_REGION", "eu").strip().lower()
MISTRAL_CHAT_MODEL = "mistral-small-latest"
MISTRAL_TRANSCRIBE_MODEL = "voxtral-mini-latest"
MISTRAL_TIMEOUT_SECONDS = 20

# Keep these constant per requirement.
# Public sample voice shown in Gradium docs.
DEFAULT_GRADIUM_VOICE = "YTpq7expH9539ERJ"
VOICE_WHITE = os.getenv("GRADIUM_VOICE_ID_WHITE", DEFAULT_GRADIUM_VOICE).strip()
VOICE_BLACK = os.getenv("GRADIUM_VOICE_ID_BLACK", DEFAULT_GRADIUM_VOICE).strip()


class CoachMoveRequest(BaseModel):
    fen: str
    last_move_uci: str
    legal_moves_uci: list[str] = Field(min_length=1)
    elo: int


class CoachMoveResponse(BaseModel):
    black_move_uci: str
    white_feedback: str
    black_advice: str


class AskRequest(BaseModel):
    fen: str
    question: str


class AskResponse(BaseModel):
    answer: str


class TTSRequest(BaseModel):
    text: str
    color: Literal["white", "black"]
    voice_id: Optional[str] = None
    speed: Optional[float] = None


def ascii_only(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    encoded = normalized.encode("ascii", "ignore").decode("ascii")
    encoded = re.sub(r"\s+", " ", encoded).strip()
    return encoded


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def fallback_coach_response(
    legal_moves_uci: list[str], white_feedback: Optional[str] = None
) -> CoachMoveResponse:
    return CoachMoveResponse(
        black_move_uci=random.choice(legal_moves_uci),
        white_feedback=white_feedback
        or "Your move stands, but destiny still demands more precision.",
        black_advice="Breathe, develop a piece, and keep your king safe.",
    )


def call_mistral_chat(messages: list[dict], temperature: float = 0.7) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY missing")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": MISTRAL_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    url = "https://api.mistral.ai/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=MISTRAL_TIMEOUT_SECONDS)
    if resp.status_code >= 400:
        # Compatibility fallback for accounts/endpoints that reject response_format.
        payload_no_format = dict(payload)
        payload_no_format.pop("response_format", None)
        resp2 = requests.post(url, headers=headers, json=payload_no_format, timeout=MISTRAL_TIMEOUT_SECONDS)
        if resp2.status_code >= 400:
            detail = ascii_only(resp2.text)[:220] or ascii_only(resp.text)[:220] or "Mistral call error"
            raise HTTPException(
                status_code=502,
                detail=f"Mistral call error ({resp2.status_code}): {detail}",
            )
        resp = resp2

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise HTTPException(status_code=502, detail="Empty Mistral response")
    return content


def call_mistral_transcribe(filename: str, content_type: str, file_bytes: bytes) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY missing")

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    files = {
        "file": (filename or "audio.webm", io.BytesIO(file_bytes), content_type or "audio/webm"),
    }
    data = {"model": MISTRAL_TRANSCRIBE_MODEL}

    resp = requests.post(
        "https://api.mistral.ai/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data,
        timeout=MISTRAL_TIMEOUT_SECONDS,
    )
    if resp.status_code >= 400:
        detail = ascii_only(resp.text)[:220] or "Transcription error"
        raise HTTPException(status_code=502, detail=f"Transcription error ({resp.status_code}): {detail}")

    payload = resp.json()
    text = payload.get("text") or payload.get("transcript") or ""
    return ascii_only(text)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.post("/coach_move", response_model=CoachMoveResponse)
def coach_move(req: CoachMoveRequest) -> CoachMoveResponse:
    if not req.legal_moves_uci:
        raise HTTPException(status_code=400, detail="legal_moves_uci is empty")

    system_prompt = (
        "You are a chess coach ON WHITE'S SIDE (the user). "
        "You MUST answer in strict JSON, with no surrounding text."
    )

    white_feedback = ""
    try:
        # Step 1: evaluate white move only, without anticipating black's reply.
        feedback_prompt = (
            "Context:\n"
            f"fen={req.fen}\n"
            f"last_move_uci={req.last_move_uci}\n\n"
            "Required rules:\n"
            "- Evaluate ONLY the White move that was just played.\n"
            "- Do NOT anticipate Black's upcoming move and do not mention any future Black move.\n"
            "- white_feedback: max 1 sentence, conversational English, humorous + dramatic, ASCII only, no emoji.\n"
            "- Use simple criteria: king safety, center, development, tactical threats, material.\n"
            "- Avoid fluff: be concrete, specific, and instructive.\n\n"
            "Expected strict JSON format:\n"
            '{"white_feedback":"..."}'
        )
        raw_feedback = call_mistral_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": feedback_prompt},
            ],
            temperature=0.25,
        )
        parsed_feedback = extract_json_object(raw_feedback)
        white_feedback = ascii_only(str(parsed_feedback.get("white_feedback", "")))
        white_feedback = re.split(r"(?<=[.!?])\s+", white_feedback)[0].strip()[:180]
        if not white_feedback:
            white_feedback = "Your move is playable, but tragedy lurks behind every neglected square."

        # Step 2: choose black move and provide next-move advice for white.
        black_prompt = (
            "Context:\n"
            f"fen={req.fen}\n"
            f"last_move_uci={req.last_move_uci}\n"
            f"elo={req.elo}\n"
            f"legal_moves_uci={json.dumps(req.legal_moves_uci, ensure_ascii=True)}\n\n"
            "Required rules:\n"
            "- Choose a legal move for BLACK.\n"
            "- black_move_uci must be EXACTLY one item from legal_moves_uci.\n"
            "- Elo 800: sometimes imprecise. Elo 1200: reasonable. Elo 1600: more solid and active.\n"
            "- black_advice: max 1 sentence, conversational English, humorous + dramatic, ASCII only, no emoji.\n"
            "- black_advice must help WHITE on their NEXT move after YOUR Black move.\n"
            "- Give ONE clear action (develop, defend, castle, contest the center, etc.).\n\n"
            "Expected strict JSON format:\n"
            '{"black_move_uci":"...","black_advice":"..."}'
        )
        raw_black = call_mistral_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": black_prompt},
            ],
            temperature=0.35,
        )
        parsed_black = extract_json_object(raw_black)
        black_move = str(parsed_black.get("black_move_uci", "")).strip()
        black_advice = ascii_only(str(parsed_black.get("black_advice", "")))
        black_advice = re.split(r"(?<=[.!?])\s+", black_advice)[0].strip()[:180]
        if black_move not in req.legal_moves_uci:
            return fallback_coach_response(req.legal_moves_uci, white_feedback=white_feedback)
        if not black_advice:
            black_advice = "Develop a piece calmly and turn this chaos into a plan."

        return CoachMoveResponse(
            black_move_uci=black_move,
            white_feedback=white_feedback,
            black_advice=black_advice,
        )
    except HTTPException:
        raise
    except Exception:
        return fallback_coach_response(
            req.legal_moves_uci,
            white_feedback=white_feedback or None,
        )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = ascii_only(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    system_prompt = (
        "You are a pedagogical chess coach. Answer in English, short, clear, ASCII only, no emoji."
    )
    user_prompt = (
        f"fen={req.fen}\n"
        f"question={question}\n"
        "Constraints: max 4 sentences, no long variations."
    )

    try:
        raw = call_mistral_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
        )
        parsed = extract_json_object(raw)
        answer = ""
        for key in ("answer", "reponse", "response", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                answer = ascii_only(value)
                break
        if not answer:
            # If Mistral returned another shape, prefer a flat string value before
            # falling back to the raw payload (which can cause JSON to be spoken by TTS).
            for value in parsed.values():
                if isinstance(value, str) and value.strip():
                    answer = ascii_only(value)
                    break
        if not answer:
            # Last resort if the model did not return usable JSON fields.
            answer = ascii_only(raw)
        return AskResponse(answer=answer[:700])
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Coach response error")


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing file")
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    text = call_mistral_transcribe(file.filename, file.content_type or "audio/webm", file_bytes)
    return {"text": text}


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    if not GRADIUM_API_KEY:
        raise HTTPException(status_code=500, detail="GRADIUM_API_KEY missing")

    safe_text = ascii_only(req.text)
    if not safe_text:
        raise HTTPException(status_code=400, detail="empty text")

    voice_id = (req.voice_id or "").strip()
    if not voice_id:
        voice_id = VOICE_WHITE if req.color == "white" else VOICE_BLACK
    speed = req.speed
    if speed is not None:
        speed = max(0.5, min(2.0, float(speed)))

    try:
        headers = {
            "Authorization": f"Bearer {GRADIUM_API_KEY}",
            "X-API-Key": GRADIUM_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }

        # Gradium API docs use regional hosts.
        region_hosts = ["us", "eu"] if GRADIUM_REGION == "us" else ["eu", "us"]
        candidate_urls: list[str] = []
        for region in region_hosts:
            candidate_urls.extend(
                [
                    f"https://{region}.api.gradium.ai/api/post/speech/tts",
                ]
            )
        payload_variants: list[dict] = [
            {"text": safe_text, "output_format": "wav", "only_audio": True},
            {"input": safe_text, "format": "wav"},
            {"text": safe_text},
        ]
        if voice_id:
            payload_variants = [
                {**p, "voice_id": voice_id} for p in payload_variants
            ] + [{**p, "voice": voice_id} for p in payload_variants]
        if speed is not None:
            payload_variants = (
                [{**p, "speed": speed} for p in payload_variants]
                + [{**p, "speaking_rate": speed} for p in payload_variants]
            )

        last_error = "TTS error"
        for url in candidate_urls:
            for payload in payload_variants:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=20,
                )
                if resp.status_code >= 400:
                    body = ascii_only(resp.text)[:220]
                    last_error = f"TTS {resp.status_code}: {body or 'empty response'}"
                    continue

                ctype = (resp.headers.get("content-type") or "").lower()
                if "audio/wav" in ctype or "audio/" in ctype:
                    return Response(content=resp.content, media_type="audio/wav")

                # Some APIs may return JSON with base64 audio or an audio URL.
                if "application/json" in ctype:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_error = "TTS: invalid JSON"
                        continue

                    b64 = data.get("audio_base64") or data.get("audio")
                    if isinstance(b64, str) and b64.strip():
                        try:
                            audio_bytes = base64.b64decode(b64)
                        except Exception:
                            last_error = "TTS: invalid base64 audio"
                            continue
                        return Response(content=audio_bytes, media_type="audio/wav")

                    audio_url = data.get("audio_url")
                    if isinstance(audio_url, str) and audio_url.startswith("http"):
                        get_resp = requests.get(audio_url, timeout=20)
                        if get_resp.status_code < 400:
                            return Response(content=get_resp.content, media_type="audio/wav")
                        last_error = f"TTS audio_url {get_resp.status_code}"
                        continue

                    msg = data.get("detail") or data.get("message") or data.get("error")
                    last_error = f"TTS JSON without audio: {ascii_only(str(msg or 'unknown'))[:200]}"
                    continue

                last_error = f"TTS: unsupported format ({ctype or 'unknown'})"

        raise HTTPException(status_code=502, detail=last_error)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS error: {ascii_only(str(e))[:220]}")


HTML_PAGE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chess Coach POC</title>
  <style>
    :root {
      --bg-0: #efe6d8;
      --bg-1: #d8c4ab;
      --bg-2: #f6f1e7;
      --card: rgba(255, 252, 245, 0.88);
      --card-strong: #fffdf8;
      --text: #241d14;
      --muted: #6c5c49;
      --line: rgba(91, 68, 38, 0.16);
      --line-strong: rgba(91, 68, 38, 0.28);
      --light: #dcc7a1;
      --dark: #b78b63;
      --accent: #0f766e;
      --accent-2: #134e4a;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --danger: #b91c1c;
      --danger-soft: rgba(185, 28, 28, 0.12);
      --ok: #15803d;
      --ok-soft: rgba(21, 128, 61, 0.12);
      --target: rgba(15, 118, 110, 0.3);
      --target-ring: rgba(15, 118, 110, 0.85);
      --shadow: 0 16px 48px rgba(34, 23, 12, 0.12);
    }
    body[data-theme="dark"] {
      --bg-0: #0e1520;
      --bg-1: #182331;
      --bg-2: #0f1724;
      --card: rgba(27, 34, 47, 0.9);
      --card-strong: #1f2838;
      --text: #e6eaf2;
      --muted: #aeb7c8;
      --line: rgba(173, 191, 216, 0.12);
      --line-strong: rgba(173, 191, 216, 0.22);
      --light: #cbb59d;
      --dark: #7f6046;
      --accent: #22c55e;
      --accent-2: #15803d;
      --accent-soft: rgba(34, 197, 94, 0.14);
      --danger: #f87171;
      --danger-soft: rgba(248, 113, 113, 0.14);
      --ok: #4ade80;
      --ok-soft: rgba(74, 222, 128, 0.14);
      --target: rgba(34, 197, 94, 0.35);
      --target-ring: rgba(74, 222, 128, 0.85);
      --shadow: 0 18px 52px rgba(0, 0, 0, 0.34);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 18px;
      color: var(--text);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(800px 400px at 8% -5%, rgba(255, 255, 255, 0.6), transparent 60%),
        radial-gradient(700px 420px at 100% 10%, rgba(97, 52, 22, 0.08), transparent 70%),
        linear-gradient(180deg, var(--bg-0), var(--bg-1) 46%, var(--bg-2));
      min-height: 100vh;
    }
    .app-shell {
      max-width: 1240px;
      margin: 0 auto;
    }
    .app-header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: end;
      margin-bottom: 16px;
    }
    .title-block h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.05;
      letter-spacing: 0.2px;
      font-family: "Georgia", "Iowan Old Style", serif;
    }
    .title-block p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .status-strip {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 32px;
      border-radius: 999px;
      padding: 6px 11px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.42);
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
      backdrop-filter: blur(5px);
    }
    body[data-theme="dark"] .chip { background: rgba(255, 255, 255, 0.03); }
    .chip .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--muted);
      box-shadow: 0 0 0 0 rgba(0,0,0,0);
    }
    .chip.is-live .dot {
      background: var(--danger);
      animation: pulseDot 1.2s infinite ease-out;
    }
    .chip.is-busy .dot { background: #d97706; }
    .chip.is-ok .dot { background: var(--ok); }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 560px) minmax(320px, 1fr);
      gap: 16px;
      align-items: start;
    }
    .stack { display: grid; gap: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .card-title {
      margin: 0;
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .card-body { padding: 14px; }
    .board-card .card-body { padding: 12px; }
    .board-shell {
      background: linear-gradient(180deg, rgba(80, 52, 28, 0.12), rgba(80, 52, 28, 0.04));
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 10px;
    }
    table { border-collapse: collapse; }
    #board {
      width: 100%;
      table-layout: fixed;
      border-radius: 6px;
      overflow: hidden;
      border: 4px solid #6f5138;
      box-shadow: inset 0 0 0 1px rgba(25, 18, 12, 0.3);
    }
    #board td {
      width: 56px;
      height: 56px;
      text-align: center;
      vertical-align: middle;
      font-size: 42px;
      line-height: 1;
      cursor: pointer;
      user-select: none;
      position: relative;
      font-family: "Times New Roman", "Segoe UI Symbol", "Apple Symbols", serif;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.18), 0 1px 2px rgba(0, 0, 0, 0.18);
      transition: filter 140ms ease, transform 90ms ease, box-shadow 120ms ease;
    }
    #board td:hover { filter: brightness(1.04); }
    #board td:active { transform: scale(0.97); }
    #board .light { background: #dcc7a1; }
    #board .dark { background: #b78b63; }
    #board .selected {
      outline: 3px solid var(--accent);
      outline-offset: -3px;
      box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.55);
    }
    #board .legal-target {
      box-shadow: inset 0 0 0 3px var(--target-ring);
      position: relative;
    }
    #board .legal-target::after {
      content: "";
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: var(--target);
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
    #board .last-from { box-shadow: inset 0 0 0 3px rgba(255, 255, 255, 0.25); }
    #board .last-to {
      box-shadow: inset 0 0 0 3px rgba(255, 255, 255, 0.25);
      animation: pulseLastMove 520ms ease-out;
    }
    #board .rank-mark,
    #board .file-mark {
      position: absolute;
      font-size: 11px;
      font-weight: 700;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      text-shadow: none;
      opacity: 0.8;
      pointer-events: none;
    }
    #board .rank-mark { left: 4px; top: 3px; }
    #board .file-mark { right: 4px; bottom: 3px; }
    #board .light .rank-mark, #board .light .file-mark { color: #8f6847; }
    #board .dark .rank-mark, #board .dark .file-mark { color: #f2dfbf; }
    .board-meta {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin-top: 10px;
      align-items: center;
    }
    .board-state {
      display: grid;
      gap: 8px;
    }
    .status-line {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: var(--card-strong);
      min-height: 44px;
      font-weight: 600;
    }
    .status-line small {
      display: block;
      color: var(--muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.35px;
      margin-bottom: 2px;
    }
    .board-controls {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .control-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .control-group {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--card-strong);
      color: var(--muted);
      font-weight: 700;
      font-size: 13px;
    }
    .control-group label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: inherit;
      cursor: pointer;
    }
    #elo {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card-strong);
      padding: 6px 8px;
      min-width: 84px;
      color: var(--text);
    }
    #log {
      min-height: 120px;
      max-height: 240px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      background: var(--card-strong);
      color: var(--text);
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px;
    }
    #coach-commentary {
      display: grid;
      gap: 10px;
    }
    .coach-tile {
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--card-strong);
      padding: 12px;
      min-height: 76px;
      transition: transform 160ms ease, box-shadow 180ms ease, border-color 160ms ease;
    }
    .coach-tile.flash {
      transform: translateY(-1px);
      border-color: var(--line-strong);
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
    }
    .coach-tile .label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.35px;
      margin-bottom: 5px;
    }
    .coach-tile .value {
      line-height: 1.35;
      font-size: 15px;
    }
    .coach-tile.empty .value { color: var(--muted); }
    .coach-history {
      margin-top: 10px;
      border-top: 1px dashed var(--line);
      padding-top: 10px;
      display: grid;
      gap: 6px;
    }
    .coach-history .history-item {
      font-size: 12px;
      line-height: 1.3;
      color: var(--muted);
      padding: 7px 9px;
      border-radius: 10px;
      background: rgba(255,255,255,0.25);
      border: 1px solid var(--line);
    }
    #ask-answer {
      min-height: 66px;
      line-height: 1.35;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: var(--card-strong);
      padding: 10px 12px;
    }
    .ask-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
      align-items: center;
    }
    .mic-status-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
      align-items: center;
    }
    .mic-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      background: var(--card-strong);
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
    }
    .mic-pill .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--muted);
    }
    .mic-pill.recording { border-color: rgba(185,28,28,0.25); background: var(--danger-soft); }
    .mic-pill.recording .dot { background: var(--danger); animation: pulseDot 1.2s infinite ease-out; }
    .mic-pill.busy { background: rgba(217, 119, 6, 0.1); }
    .mic-pill.busy .dot { background: #d97706; }
    .mic-pill.ready { background: var(--ok-soft); }
    .mic-pill.ready .dot { background: var(--ok); }
    .mic-timer {
      min-width: 56px;
      text-align: center;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 10px 13px;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(180deg, var(--accent), var(--accent-2));
      box-shadow: 0 8px 18px rgba(17, 94, 89, 0.24);
      transition: transform 120ms ease, filter 120ms ease, box-shadow 160ms ease;
    }
    button:hover { filter: brightness(1.05); }
    button:active { transform: translateY(1px); }
    button.secondary {
      background: var(--card-strong);
      color: var(--text);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    button.danger-soft {
      background: linear-gradient(180deg, #ef4444, #b91c1c);
      box-shadow: 0 8px 18px rgba(185, 28, 28, 0.24);
    }
    input[type=text] {
      width: 100%;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card-strong);
      color: var(--text);
    }
    .voice-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .voice-grid select {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card-strong);
      color: var(--text);
      padding: 8px 10px;
      width: 100%;
    }
    .voice-actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    details.audio-details {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.18);
    }
    details.audio-details[open] { background: rgba(255, 255, 255, 0.26); }
    details.audio-details summary {
      list-style: none;
      cursor: pointer;
      padding: 12px 14px;
      font-weight: 800;
      font-size: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    details.audio-details summary::-webkit-details-marker { display: none; }
    .summary-right {
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }
    .audio-details .details-body {
      padding: 0 14px 14px;
      border-top: 1px solid var(--line);
    }
    .section-note {
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 10px;
      line-height: 1.35;
    }
    @keyframes pulseDot {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }
      75% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    @keyframes pulseLastMove {
      0% { transform: scale(0.94); filter: brightness(1.12); }
      100% { transform: scale(1); filter: brightness(1); }
    }
    @media (max-width: 920px) {
      .layout { grid-template-columns: 1fr; }
      .app-header {
        grid-template-columns: 1fr;
        align-items: start;
      }
      .status-strip { justify-content: flex-start; }
      #board td {
        width: clamp(42px, 10vw, 56px);
        height: clamp(42px, 10vw, 56px);
        font-size: clamp(30px, 7.4vw, 42px);
      }
      .board-meta { grid-template-columns: 1fr; }
      .control-row { align-items: stretch; }
      .control-group { width: 100%; justify-content: space-between; }
      .ask-actions button { flex: 1 1 140px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div class="title-block">
        <h1>Chess Coach</h1>
        <p>You play White. The coach comments on your move, plays Black, then gives you a simple plan.</p>
      </div>
      <div class="status-strip">
        <div id="turn-chip" class="chip"><span class="dot"></span><span>Turn: White</span></div>
        <div id="pending-chip" class="chip"><span class="dot"></span><span>Coach ready</span></div>
      </div>
    </header>

    <div class="layout">
      <section class="stack">
        <div class="card board-card">
          <div class="card-body">
            <div class="board-shell">
              <table id="board"></table>
            </div>
            <div class="board-meta">
              <div class="board-state">
                <div id="app-status" class="status-line">
                  <small>Session status</small>
                  <span>Loading...</span>
                </div>
              </div>
              <button id="reset-btn" class="secondary">New game</button>
            </div>
            <div class="board-controls">
              <div class="control-row">
                <div class="control-group">
                  <span>Black strength</span>
                  <select id="elo">
                    <option value="800">800</option>
                    <option value="1200" selected>1200</option>
                    <option value="1600">1600</option>
                  </select>
                </div>
                <div class="control-group">
                  <label><input type="checkbox" id="audio-toggle" checked> Coach voice</label>
                </div>
                <div class="control-group">
                  <label><input type="checkbox" id="theme-toggle"> Dark theme</label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="stack">
        <div class="card">
          <div class="card-body">
            <div class="card-header">
              <h2 class="card-title">Move Coach</h2>
            </div>
            <div id="coach-commentary"></div>
            <div id="coach-history" class="coach-history"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="card-header">
              <h2 class="card-title">Ask The Coach</h2>
            </div>
            <p class="section-note">Type your question or use the microphone. Transcription starts automatically after you stop recording.</p>
            <input id="ask-input" type="text" placeholder="Example: What is my plan here?" />
            <div class="ask-actions">
              <button id="ask-btn">Send question</button>
              <button id="mic-btn" title="Click to start/stop">Mic: start</button>
            </div>
            <div class="mic-status-row">
              <div id="mic-state" class="mic-pill ready"><span class="dot"></span><span>Mic ready</span></div>
              <div id="mic-timer" class="mic-pill mic-timer"><span>00:00</span></div>
            </div>
            <div style="height:10px"></div>
            <div id="ask-answer"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <details class="audio-details">
              <summary>
                <span>Audio settings (Gradium)</span>
                <span class="summary-right">Voice, speed, test</span>
              </summary>
              <div class="details-body">
                <p class="section-note">These settings only affect the coach voice (TTS). API keys stay server-side.</p>
                <div class="voice-grid">
                  <select id="voice-black-select"></select>
                  <select id="voice-speed-select"></select>
                  <input id="voice-black-custom" type="text" placeholder="custom coach voice_id" style="display:none" />
                </div>
                <div class="voice-actions">
                  <button id="voice-save-btn" class="secondary">Save voice</button>
                  <button id="voice-test-btn">Test coach voice</button>
                </div>
              </div>
            </details>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="card-header">
              <h2 class="card-title">Session Log</h2>
            </div>
            <div id="log"></div>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script type="module">
    import { Chess } from 'https://cdn.jsdelivr.net/npm/chess.js@1.4.0/+esm';
    const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const PIECES = {
      p: '♟', r: '♜', n: '♞', b: '♝', q: '♛', k: '♚',
      P: '♙', R: '♖', N: '♘', B: '♗', Q: '♕', K: '♔'
    };

    const boardEl = document.getElementById('board');
    const logEl = document.getElementById('log');
    const commentaryEl = document.getElementById('coach-commentary');
    const coachHistoryEl = document.getElementById('coach-history');
    const askAnswerEl = document.getElementById('ask-answer');
    const appStatusEl = document.getElementById('app-status');
    const turnChipEl = document.getElementById('turn-chip');
    const pendingChipEl = document.getElementById('pending-chip');
    const micStateEl = document.getElementById('mic-state');
    const micTimerEl = document.getElementById('mic-timer');

    const audioToggle = document.getElementById('audio-toggle');
    const themeToggle = document.getElementById('theme-toggle');
    const eloSelect = document.getElementById('elo');
    const resetBtn = document.getElementById('reset-btn');
    const micBtn = document.getElementById('mic-btn');
    const askBtn = document.getElementById('ask-btn');
    const askInput = document.getElementById('ask-input');
    const voiceBlackSelect = document.getElementById('voice-black-select');
    const voiceSpeedSelect = document.getElementById('voice-speed-select');
    const voiceBlackCustomInput = document.getElementById('voice-black-custom');
    const voiceSaveBtn = document.getElementById('voice-save-btn');
    const voiceTestBtn = document.getElementById('voice-test-btn');
    const VOICE_OPTIONS = [
      { value: '', label: 'Server default' },
      { value: 'YTpq7expH9539ERJ', label: 'Emma (en-US, feminine)' },
      { value: 'LFZvm12tW_z0xfGo', label: 'Kent (en-US, masculine)' },
      { value: 'jtEKaLYNn6iif5PR', label: 'Sydney (en-US, feminine)' },
      { value: 'KWJiFWu2O9nMPYcR', label: 'John (en-US, masculine)' },
      { value: 'ubuXFxVQwVYnZQhy', label: 'Eva (en-GB, feminine)' },
      { value: 'm86j6D7UZpGzHsNu', label: 'Jack (en-GB, masculine)' },
      { value: 'b35yykvVppLXyw_l', label: 'Elise (fr-FR, feminine)' },
      { value: 'axlOaUiFyOZhy4nv', label: 'Leo (fr-FR, masculine)' },
      { value: '-uP9MuGtBqAvEyxI', label: 'Mia (de-DE, feminine)' },
      { value: '0y1VZjPabOBU3rWy', label: 'Maximilian (de-DE, masculine)' },
      { value: 'B36pbz5_UoWn4BDl', label: 'Valentina (es-MX, feminine)' },
      { value: 'xu7iJ_fn2ElcWp2s', label: 'Sergio (es-ES, masculine)' },
      { value: 'pYcGZz9VOo4n2ynh', label: 'Alice (pt-BR, feminine)' },
      { value: 'M-FvVo9c-jGR4PgP', label: 'Davi (pt-BR, masculine)' },
      { value: '__custom__', label: 'Custom...' }
    ];
    const SPEED_OPTIONS = [
      { value: '1.0', label: 'Speed: normal' },
      { value: '1.15', label: 'Speed: fast' },
      { value: '1.3', label: 'Speed: very fast' },
    ];

    const chess = new Chess();
    let selectedSquare = null;
    let pending = false;
    let legalTargets = new Set();
    let lastMoveSquares = { from: null, to: null };
    let mediaRecorder = null;
    let chunks = [];
    let voiceSettings = { voice_id: '', speed: 1.0 };
    let coachState = { white_feedback: '', black_advice: '' };
    let coachHistory = [];
    let micTimerInterval = null;
    let micStartedAt = 0;

    function log(msg) {
      const now = new Date().toLocaleTimeString();
      logEl.textContent = `[${now}] ${msg}\n` + logEl.textContent;
    }

    function setAppStatus(text, tone = '') {
      if (!appStatusEl) return;
      const span = appStatusEl.querySelector('span') || appStatusEl;
      span.textContent = text;
      appStatusEl.style.borderColor = tone === 'busy'
        ? 'rgba(217, 119, 6, 0.25)'
        : tone === 'error'
          ? 'rgba(185, 28, 28, 0.25)'
          : 'var(--line)';
      appStatusEl.style.background = tone === 'busy'
        ? 'rgba(217, 119, 6, 0.07)'
        : tone === 'error'
          ? 'var(--danger-soft)'
          : 'var(--card-strong)';
    }

    function setChip(chipEl, text, tone) {
      if (!chipEl) return;
      const label = chipEl.querySelector('span:last-child');
      if (label) label.textContent = text;
      chipEl.classList.remove('is-live', 'is-busy', 'is-ok');
      if (tone === 'live') chipEl.classList.add('is-live');
      if (tone === 'busy') chipEl.classList.add('is-busy');
      if (tone === 'ok') chipEl.classList.add('is-ok');
    }

    function updateTurnChip() {
      const turn = chess.turn() === 'w' ? 'White' : 'Black';
      setChip(turnChipEl, `Turn: ${turn}`, chess.turn() === 'w' ? 'ok' : 'busy');
    }

    function formatMicTimer(ms) {
      const totalSec = Math.max(0, Math.floor(ms / 1000));
      const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
      const ss = String(totalSec % 60).padStart(2, '0');
      return `${mm}:${ss}`;
    }

    function updateMicTimer() {
      const label = micTimerEl && micTimerEl.querySelector('span');
      if (!label) return;
      if (!micStartedAt) {
        label.textContent = '00:00';
        return;
      }
      label.textContent = formatMicTimer(Date.now() - micStartedAt);
    }

    function startMicTimer() {
      micStartedAt = Date.now();
      updateMicTimer();
      if (micTimerInterval) clearInterval(micTimerInterval);
      micTimerInterval = setInterval(updateMicTimer, 250);
    }

    function stopMicTimer() {
      micStartedAt = 0;
      if (micTimerInterval) {
        clearInterval(micTimerInterval);
        micTimerInterval = null;
      }
      updateMicTimer();
    }

    function setMicState(kind, text) {
      if (micStateEl) {
        micStateEl.classList.remove('recording', 'busy', 'ready');
        if (kind === 'recording') micStateEl.classList.add('recording');
        else if (kind === 'busy') micStateEl.classList.add('busy');
        else micStateEl.classList.add('ready');
        const label = micStateEl.querySelector('span:last-child');
        if (label) label.textContent = text;
      }
      if (micBtn) {
        micBtn.classList.remove('danger-soft', 'secondary');
        if (kind === 'recording') {
          micBtn.classList.add('danger-soft');
          micBtn.textContent = 'Mic: stop';
        } else {
          micBtn.textContent = 'Mic: start';
        }
      }
    }

    function flash(el) {
      if (!el) return;
      el.classList.remove('flash');
      void el.offsetWidth;
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 220);
    }

    function legalMovesToUci() {
      return chess.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion ? m.promotion : ''));
    }

    function squareColor(fileIdx, rank) {
      return ((fileIdx + rank) % 2 === 0) ? 'light' : 'dark';
    }

    function refreshLegalTargets() {
      legalTargets = new Set();
      if (!selectedSquare) return;
      const moves = chess.moves({ square: selectedSquare, verbose: true });
      for (const m of moves) legalTargets.add(m.to);
    }

    function setTheme(theme) {
      const safeTheme = theme === 'dark' ? 'dark' : 'light';
      document.body.dataset.theme = safeTheme;
      themeToggle.checked = safeTheme === 'dark';
      localStorage.setItem('chess_theme', safeTheme);
    }

    function renderBoard() {
      const board = chess.board();
      boardEl.innerHTML = '';

      for (let rankIndex = 0; rankIndex < 8; rankIndex++) {
        const tr = document.createElement('tr');

        for (let fileIdx = 0; fileIdx < 8; fileIdx++) {
          const td = document.createElement('td');
          const rank = 8 - rankIndex;
          const square = FILES[fileIdx] + String(rank);

          td.dataset.square = square;
          td.className = squareColor(fileIdx, rank);
          if (selectedSquare === square) td.classList.add('selected');
          if (legalTargets.has(square)) td.classList.add('legal-target');
          if (lastMoveSquares.from === square) td.classList.add('last-from');
          if (lastMoveSquares.to === square) td.classList.add('last-to');

          const piece = board[rankIndex][fileIdx];
          td.textContent = piece ? PIECES[piece.color === 'w' ? piece.type.toUpperCase() : piece.type] : '';

          if (fileIdx === 0) {
            const rankMark = document.createElement('span');
            rankMark.className = 'rank-mark';
            rankMark.textContent = String(rank);
            td.appendChild(rankMark);
          }
          if (rank === 1) {
            const fileMark = document.createElement('span');
            fileMark.className = 'file-mark';
            fileMark.textContent = FILES[fileIdx];
            td.appendChild(fileMark);
          }

          td.addEventListener('click', () => onSquareClick(square));
          tr.appendChild(td);
        }

        boardEl.appendChild(tr);
      }
      updateTurnChip();
    }

    function renderCoach() {
      commentaryEl.innerHTML = '';
      const cards = [
        {
          key: 'white_feedback',
          label: 'Your move (analysis)',
          empty: 'Play a move as White to get your first coach feedback.',
          value: coachState.white_feedback,
          className: 'coach-white',
        },
        {
          key: 'black_advice',
          label: 'Suggested plan (after Black reply)',
          empty: 'The coach will suggest one simple action for your next move here.',
          value: coachState.black_advice,
          className: 'coach-plan',
        },
      ];

      for (const item of cards) {
        const tile = document.createElement('div');
        tile.className = `coach-tile ${item.className}`;
        if (!item.value) tile.classList.add('empty');
        tile.dataset.role = item.key;

        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = item.label;
        const value = document.createElement('div');
        value.className = 'value';
        value.textContent = item.value || item.empty;

        tile.appendChild(label);
        tile.appendChild(value);
        commentaryEl.appendChild(tile);
      }

      renderCoachHistory();
    }

    function renderCoachHistory() {
      if (!coachHistoryEl) return;
      coachHistoryEl.innerHTML = '';
      if (!coachHistory.length) {
        const empty = document.createElement('div');
        empty.className = 'history-item';
        empty.textContent = 'History: your last 3 coach tips will appear here.';
        coachHistoryEl.appendChild(empty);
        return;
      }
      for (const item of coachHistory.slice(0, 3)) {
        const d = document.createElement('div');
        d.className = 'history-item';
        d.textContent = item;
        coachHistoryEl.appendChild(d);
      }
    }

    function setCoachWhiteFeedback(text) {
      coachState.white_feedback = (text || '').trim();
      coachState.black_advice = '';
      renderCoach();
      flash(commentaryEl.querySelector('[data-role="white_feedback"]'));
    }

    function setCoachBlackAdvice(text) {
      coachState.black_advice = (text || '').trim();
      if (coachState.black_advice) {
        coachHistory.unshift(coachState.black_advice);
        coachHistory = coachHistory.slice(0, 3);
      }
      renderCoach();
      flash(commentaryEl.querySelector('[data-role="black_advice"]'));
    }

    function resetCoach() {
      coachState = { white_feedback: '', black_advice: '' };
      coachHistory = [];
      renderCoach();
    }

    function loadVoiceSettings() {
      try {
        const raw = localStorage.getItem('gradium_voice_settings');
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed.voice_id === 'string') {
          voiceSettings = {
            voice_id: parsed.voice_id,
            speed: Number(parsed.speed || 1.0) || 1.0,
          };
        } else if (parsed && parsed.black && typeof parsed.black.voice_id === 'string') {
          // Backward compatibility with previous saved structure.
          voiceSettings = {
            voice_id: parsed.black.voice_id,
            speed: Number(parsed.black.speed || 1.0) || 1.0,
          };
        }
      } catch (_) {
        // Keep defaults.
      }
    }

    function renderVoiceSettings() {
      renderVoiceSelect(voiceBlackSelect, voiceBlackCustomInput, voiceSettings.voice_id || '');
      renderSpeedSelect();
    }

    function renderSpeedSelect() {
      voiceSpeedSelect.innerHTML = '';
      for (const opt of SPEED_OPTIONS) {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        voiceSpeedSelect.appendChild(o);
      }
      const current = String(voiceSettings.speed || 1.0);
      if ([...voiceSpeedSelect.options].some(o => o.value === current)) {
        voiceSpeedSelect.value = current;
      } else {
        voiceSpeedSelect.value = '1.0';
      }
    }

    function renderVoiceSelect(selectEl, customInputEl, currentVoiceId) {
      selectEl.innerHTML = '';
      const known = new Set(VOICE_OPTIONS.map(o => o.value));
      for (const opt of VOICE_OPTIONS) {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        selectEl.appendChild(o);
      }
      if (currentVoiceId && !known.has(currentVoiceId)) {
        const customOpt = document.createElement('option');
        customOpt.value = currentVoiceId;
        customOpt.textContent = `Custom (${currentVoiceId.slice(0, 8)}...)`;
        selectEl.insertBefore(customOpt, selectEl.lastElementChild);
        selectEl.value = currentVoiceId;
        customInputEl.style.display = 'none';
      } else if (currentVoiceId && known.has(currentVoiceId)) {
        selectEl.value = currentVoiceId;
        customInputEl.style.display = 'none';
      } else {
        selectEl.value = '';
        customInputEl.style.display = 'none';
      }
      customInputEl.value = currentVoiceId || '';
    }

    function readVoiceIdFromControls(selectEl, customInputEl) {
      if (selectEl.value === '__custom__') return (customInputEl.value || '').trim();
      return (selectEl.value || '').trim();
    }

    function onVoiceSelectChange(selectEl, customInputEl) {
      if (selectEl.value === '__custom__') {
        customInputEl.style.display = 'block';
        customInputEl.focus();
      } else {
        customInputEl.style.display = 'none';
      }
    }

    function saveVoiceSettings() {
      voiceSettings = {
        voice_id: readVoiceIdFromControls(voiceBlackSelect, voiceBlackCustomInput),
        speed: Number(voiceSpeedSelect.value || '1.0') || 1.0,
      };
      localStorage.setItem('gradium_voice_settings', JSON.stringify(voiceSettings));
      log('Voice settings saved');
    }

    async function playTTS(text, color) {
      if (!audioToggle.checked || !text) return;
      try {
        const resp = await fetch('/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            color,
            voice_id: voiceSettings.voice_id || undefined,
            speed: voiceSettings.speed,
          })
        });
        if (!resp.ok) {
          let detail = 'TTS error';
          try {
            const data = await resp.json();
            if (data && data.detail) detail = String(data.detail);
          } catch (_) {
            // Keep default detail.
          }
          throw new Error(detail);
        }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        await new Promise((resolve, reject) => {
          let settled = false;
          const done = () => {
            if (settled) return;
            settled = true;
            URL.revokeObjectURL(url);
            resolve();
          };
          audio.onended = done;
          audio.onerror = () => {
            if (settled) return;
            settled = true;
            URL.revokeObjectURL(url);
            reject(new Error('audio playback failed'));
          };
          audio.play().catch((err) => {
            if (settled) return;
            settled = true;
            URL.revokeObjectURL(url);
            reject(err);
          });
        });
      } catch (e) {
        log(`Gradium TTS unavailable: ${e && e.message ? e.message : 'unknown error'}`);
      }
    }

    function isWhitePiece(square) {
      const p = chess.get(square);
      return p && p.color === 'w';
    }

    function toUci(moveObj) {
      return moveObj.from + moveObj.to + (moveObj.promotion ? moveObj.promotion : '');
    }

    async function onWhiteMoveApplied(lastMoveUci) {
      pending = true;
      setAppStatus('Coach is analyzing your move and preparing Black reply...', 'busy');
      setChip(pendingChipEl, 'Coach thinking', 'busy');
      try {
        const payload = {
          fen: chess.fen(),
          last_move_uci: lastMoveUci,
          legal_moves_uci: legalMovesToUci(),
          elo: Number(eloSelect.value)
        };

        const resp = await fetch('/coach_move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!resp.ok) {
          let detail = 'Coach error';
          try {
            const data = await resp.json();
            if (data && data.detail) detail = String(data.detail);
          } catch (_) {
            // Keep generic detail.
          }
          throw new Error(detail);
        }
        const data = await resp.json();

        setCoachWhiteFeedback(data.white_feedback);
        await playTTS(data.white_feedback || '', 'black');

        const black = chess.move({
          from: data.black_move_uci.slice(0, 2),
          to: data.black_move_uci.slice(2, 4),
          promotion: data.black_move_uci.length > 4 ? data.black_move_uci.slice(4, 5) : undefined
        });

        if (!black) {
          log('Received invalid Black move.');
        } else {
          lastMoveSquares = { from: black.from, to: black.to };
          renderBoard();
          log(`Black: ${toUci(black)}`);
          setCoachBlackAdvice(data.black_advice || '');
          await playTTS(data.black_advice || '', 'black');
        }
      } catch (e) {
        log(`Coach unavailable: ${e && e.message ? e.message : 'unknown error'}`);
        setCoachWhiteFeedback('Error, please try again.');
        setAppStatus('Coach error. Try another move or restart the server.', 'error');
      } finally {
        selectedSquare = null;
        refreshLegalTargets();
        pending = false;
        renderBoard();
        if (!coachState.white_feedback.startsWith('Error')) {
          setAppStatus('Your turn: apply the tip or ask the coach a question.');
        }
        setChip(pendingChipEl, 'Coach ready', 'ok');
      }
    }

    async function onSquareClick(square) {
      if (pending) return;
      if (chess.turn() !== 'w') return;

      if (!selectedSquare) {
        if (isWhitePiece(square)) {
          selectedSquare = square;
          refreshLegalTargets();
          renderBoard();
        }
        return;
      }

      if (selectedSquare === square) {
        selectedSquare = null;
        refreshLegalTargets();
        renderBoard();
        return;
      }

      const move = chess.move({ from: selectedSquare, to: square, promotion: 'q' });
      if (!move) {
        if (isWhitePiece(square)) {
          selectedSquare = square;
        } else {
          selectedSquare = null;
        }
        refreshLegalTargets();
        renderBoard();
        return;
      }

      const whiteUci = toUci(move);
      log(`White: ${whiteUci}`);
      lastMoveSquares = { from: whiteUci.slice(0, 2), to: whiteUci.slice(2, 4) };
      selectedSquare = null;
      refreshLegalTargets();
      renderBoard();
      await onWhiteMoveApplied(whiteUci);
    }

    async function askCoach(question) {
      const clean = (question || '').trim();
      if (!clean) return;

      askAnswerEl.textContent = '...';
       setChip(pendingChipEl, 'Answering question', 'busy');
       setAppStatus('Coach is preparing a short answer to your question...', 'busy');
      try {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fen: chess.fen(), question: clean })
        });
        if (!resp.ok) throw new Error('ask error');

        const data = await resp.json();
        askAnswerEl.textContent = data.answer || '';
        await playTTS(data.answer || '', 'black');
        setAppStatus('Coach answer ready. You can play or ask another question.');
      } catch (_) {
        askAnswerEl.textContent = 'Coach question error';
        setAppStatus('Coach question error. Try again in a few seconds.', 'error');
      } finally {
        setChip(pendingChipEl, 'Coach ready', 'ok');
      }
    }

    askBtn.addEventListener('click', () => askCoach(askInput.value));
    askInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') askCoach(askInput.value);
    });

    resetBtn.addEventListener('click', () => {
      chess.reset();
      selectedSquare = null;
      lastMoveSquares = { from: null, to: null };
      refreshLegalTargets();
      resetCoach();
      askAnswerEl.textContent = '';
      setAppStatus('New game. You play White.');
      setChip(pendingChipEl, 'Coach ready', 'ok');
      log('New game');
      renderBoard();
    });

    themeToggle.addEventListener('change', () => {
      setTheme(themeToggle.checked ? 'dark' : 'light');
    });
    voiceBlackSelect.addEventListener('change', () => {
      onVoiceSelectChange(voiceBlackSelect, voiceBlackCustomInput);
    });
    voiceSaveBtn.addEventListener('click', saveVoiceSettings);
    voiceTestBtn.addEventListener('click', async () => {
      saveVoiceSettings();
      await playTTS('Coach voice test.', 'black');
    });

    async function ensureRecorder() {
      if (mediaRecorder) return true;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        log('Microphone not supported');
        setMicState('ready', 'Microphone not supported');
        return false;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorder = recorder;
        recorder.ondataavailable = e => {
          if (e.data && e.data.size > 0) chunks.push(e.data);
        };
        recorder.onstop = async () => {
          // Release microphone immediately after recording stops so macOS/browser
          // no longer show the mic as active during transcription/TTS.
          stream.getTracks().forEach(track => track.stop());
          if (mediaRecorder === recorder) mediaRecorder = null;

          const blob = new Blob(chunks, { type: 'audio/webm' });
          chunks = [];
          if (!blob.size) {
            stopMicTimer();
            setMicState('ready', 'Mic ready');
            return;
          }

          const form = new FormData();
          form.append('file', blob, 'question.webm');

          try {
            stopMicTimer();
            setMicState('busy', 'Transcribing...');
            setAppStatus('Transcribing your question...', 'busy');
            const tr = await fetch('/transcribe', { method: 'POST', body: form });
            if (!tr.ok) throw new Error('transcribe error');
            const data = await tr.json();
            const text = (data.text || '').trim();
            if (!text) {
              log('Empty transcription');
              setMicState('ready', 'Mic ready');
              setAppStatus('Empty transcription. Try again with a shorter question.');
              return;
            }
            askInput.value = text;
            setMicState('busy', 'Sending to coach...');
            await askCoach(text);
            setMicState('ready', 'Mic ready');
          } catch (_) {
            log('Microphone/transcription error');
            setMicState('ready', 'Mic ready');
            setAppStatus('Microphone/transcription error. Try again.', 'error');
          }
        };
        return true;
      } catch (_) {
        log('Microphone access denied');
        setMicState('ready', 'Microphone access denied');
        return false;
      }
    }

    async function startRec() {
      const ok = await ensureRecorder();
      if (!ok) return;
      if (mediaRecorder.state === 'inactive') {
        chunks = [];
        mediaRecorder.start();
        log('Recording...');
        startMicTimer();
        setMicState('recording', 'Recording...');
        setAppStatus('Microphone recording in progress. Click to stop.', 'busy');
      }
    }

    function stopRec() {
      if (!mediaRecorder) return;
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        log('Recording stopped');
        setMicState('busy', 'Stopping...');
      } else if (mediaRecorder.state === 'inactive') {
        // Cleanup stale recorder instance if it exists.
        try {
          mediaRecorder.stream.getTracks().forEach(track => track.stop());
        } catch (_) {
          // Ignore cleanup errors.
        }
        mediaRecorder = null;
        stopMicTimer();
        setMicState('ready', 'Mic ready');
      }
    }

    micBtn.addEventListener('click', async () => {
      if (pending) return;
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRec();
      } else {
        await startRec();
      }
    });

    setTheme(localStorage.getItem('chess_theme') || 'light');
    loadVoiceSettings();
    renderVoiceSettings();
    renderCoach();
    renderBoard();
    setMicState('ready', 'Mic ready');
    setChip(pendingChipEl, 'Coach ready', 'ok');
    setAppStatus('Ready. You play White.');
    log('Ready. You play White.');
  </script>
</body>
</html>
"""
