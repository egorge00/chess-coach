from __future__ import annotations

import asyncio
import html
import os
import requests
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

APP = FastAPI(title="Pipecat POC UI")
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

DEFAULT_REALTIME_MODEL = "voxtral-mini-transcribe-realtime-2602"
DEFAULT_CHAT_MODEL = "mistral-small-latest"
DEFAULT_POCKET_URL = "http://127.0.0.1:8787"
DEFAULT_CHUNK_MS = 200
DEFAULT_TARGET_DELAY_MS = 800
DEFAULT_DURATION = 6.0
SERVER_BOOT_ID = str(int(time.time()))


class RunRequest(BaseModel):
    mode: str = "voice-loop"
    duration: float = DEFAULT_DURATION
    demo_text: str = "Hello from Pipecat"
    autoplay: bool = True
    chunk_ms: int = DEFAULT_CHUNK_MS
    target_delay_ms: int = DEFAULT_TARGET_DELAY_MS
    fen: str = ""


class CoachCommentaryRequest(BaseModel):
    phase: str  # white_move | black_reply
    fen: str
    move_text: str = ""


class SpeakRequest(BaseModel):
    text: str


ALLOWED_MODES = {"voice-loop", "voice-turn", "mic-realtime", "doctor"}


def _start_board_table_html() -> str:
    rows = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]
    pieces = {
        "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
        "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
    }
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    parts: list[str] = []
    for r in range(8):
        parts.append("<tr>")
        for c in range(8):
            cls = "light" if (r + c) % 2 == 0 else "dark"
            piece = rows[r][c]
            coord = ""
            if r == 7 or c == 0:
                coord_text = (str(8 - r) if c == 0 else "") + ((" " if c == 0 and r == 7 else "") + files[c] if r == 7 else "")
                coord = f'<span class="coord">{html.escape(coord_text)}</span>'
            piece_text = pieces.get(piece, "") if piece else ""
            sq = f"{files[c]}{8-r}"
            parts.append(f'<td class="{cls}" data-square="{sq}">{piece_text}{coord}</td>')
        parts.append("</tr>")
    return "".join(parts)


def _env_config() -> dict[str, Any]:
    return {
        "pocket_tts_url": os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", DEFAULT_POCKET_URL)),
        "pocket_tts_voice": os.getenv("POCKET_TTS_VOICE", os.getenv("LOCAL_TTS_VOICE", "")).strip() or "server default",
        "realtime_model": os.getenv("PIPECAT_POC_REALTIME_MODEL", DEFAULT_REALTIME_MODEL).strip() or DEFAULT_REALTIME_MODEL,
        "chat_model": os.getenv("PIPECAT_POC_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL,
        "chunk_ms": int(os.getenv("PIPECAT_POC_UI_CHUNK_MS", str(DEFAULT_CHUNK_MS)) or DEFAULT_CHUNK_MS),
        "target_delay_ms": int(os.getenv("PIPECAT_POC_UI_TARGET_DELAY_MS", str(DEFAULT_TARGET_DELAY_MS)) or DEFAULT_TARGET_DELAY_MS),
        "duration": float(os.getenv("PIPECAT_POC_UI_DURATION", str(DEFAULT_DURATION)) or DEFAULT_DURATION),
    }


def _build_cmd(req: RunRequest) -> list[str]:
    if req.mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="unsupported mode")

    cmd = [sys.executable, "-m", "pipecat_poc.main", "--mode", req.mode]
    if req.mode in {"voice-loop", "voice-turn", "mic-realtime"}:
        cmd += ["--duration", str(max(1.0, min(30.0, float(req.duration))))]
        cmd += ["--chunk-ms", str(max(50, min(1000, int(req.chunk_ms))))]
        cmd += ["--target-delay-ms", str(max(100, min(5000, int(req.target_delay_ms))))]
    if req.mode in {"voice-loop", "voice-turn"}:
        cmd += ["--autoplay"]
        if (req.fen or "").strip():
            cmd += ["--fen", req.fen.strip()]
    return cmd


def _run_subprocess(req: RunRequest) -> dict[str, Any]:
    cmd = _build_cmd(req)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-70000:],
        "stderr": proc.stderr[-30000:],
        "cmd": cmd,
    }


def _safe_read_text(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_text(errors="replace")[:limit]
    except Exception:
        return ""


def _list_output_files() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for path in sorted(OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        if not path.is_file():
            continue
        st = path.stat()
        items.append({"name": path.name, "size": st.st_size, "mtime": int(st.st_mtime)})
    return items


def _latest_group(prefix: str) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(OUTPUT_DIR.glob(f"{prefix}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {"prefix": prefix, "found": False}

    latest = candidates[0]
    stem = latest.name
    for suffix in (".transcript.txt", ".reply.txt", ".txt", ".wav"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    group = [p for p in candidates if p.name.startswith(stem)]
    out: dict[str, Any] = {
        "prefix": prefix,
        "found": True,
        "stem": stem,
        "mtime": int(max((p.stat().st_mtime for p in group), default=latest.stat().st_mtime)),
    }

    transcript_file = next((p for p in group if p.name.endswith(".transcript.txt")), None)
    reply_file = next((p for p in group if p.name.endswith(".reply.txt")), None)
    wav_file = next((p for p in group if p.suffix.lower() == ".wav"), None)
    generic_txt = next((p for p in group if p.name.endswith(".txt") and p not in {transcript_file, reply_file}), None)

    if transcript_file:
        out["transcript"] = _safe_read_text(transcript_file)
        out["transcript_file"] = transcript_file.name
    elif generic_txt:
        out["transcript"] = _safe_read_text(generic_txt)
        out["transcript_file"] = generic_txt.name
    if reply_file:
        out["reply"] = _safe_read_text(reply_file)
        out["reply_file"] = reply_file.name
    if wav_file:
        out["audio_file"] = wav_file.name
    return out


def _call_mistral_chat(messages: list[dict[str, str]], *, model: str, temperature: float = 0.4) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY missing")
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": messages,
        },
        timeout=45,
    )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mistral chat HTTP {resp.status_code}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Mistral returned no choices")
    msg = choices[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Mistral returned empty content")
    return text[:400]


def _pocket_tts_speak_bytes(text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    pocket_url = os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", DEFAULT_POCKET_URL)).rstrip("/")
    if not pocket_url.endswith("/tts"):
        pocket_url = pocket_url + "/tts"
    form = {"text": text}
    voice = os.getenv("POCKET_TTS_VOICE", os.getenv("LOCAL_TTS_VOICE", "")).strip()
    if voice:
        form["voice_url"] = voice
    try:
        resp = requests.post(pocket_url, data=form, timeout=45)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pocket TTS connection error: {e}") from e
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Pocket TTS HTTP {resp.status_code}")
    return resp.content


@APP.get("/", response_class=HTMLResponse)
def index() -> str:
    cfg = _env_config()
    start_board_html = _start_board_table_html()
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Chess Coach (Streaming Pipecat)</title>
  <style>
    :root {{
      --swatch-1:#c89678; --swatch-2:#d2b9a5; --swatch-3:#ead8c7;
      --swatch-4:#b3b29f; --swatch-5:#a5a58c; --swatch-6:#727963;
      --bg-0:#ead8c7; --bg-2:#f3e8dc; --card:rgba(245,240,231,.9); --card-strong:#fbf8f2;
      --text:#2e2a24; --muted:#6d6a5f; --line:rgba(114,121,99,.18); --accent:#7d866c; --accent-2:#6d755d;
      --ok:#6f8b5c; --warn:#b88b63; --danger:#b85d5d; --shadow:0 14px 40px rgba(55,56,47,.08);
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--text); background: radial-gradient(1200px 420px at 15% -10%, rgba(255,255,255,.7), transparent 65%), linear-gradient(180deg,var(--bg-2),var(--bg-0) 38%,#d9d2bf); font-family: ui-rounded, system-ui, -apple-system, sans-serif; }}
    .app-shell {{ max-width:1400px; margin:0 auto; padding:16px; }}
    .app-header {{ display:grid; grid-template-columns:1fr auto; gap:14px; align-items:start; margin-bottom:14px; }}
    .title-block h1 {{ margin:0; font-family: Georgia, 'Times New Roman', serif; font-size: clamp(1.9rem,4vw,3rem); line-height:.95; }}
    .title-block p {{ margin:8px 0 0; color:var(--muted); font-size:12px; font-style:italic; }}
    .status-strip {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
    .chip {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:10px 14px; border:1px solid var(--line); background:rgba(255,255,255,.58); font-weight:700; }}
    .chip .dot {{ width:9px; height:9px; border-radius:999px; background:#8d7b67; }}
    .chip.ok .dot {{ background:var(--ok); }} .chip.busy .dot {{ background:var(--warn); }} .chip.error .dot {{ background:var(--danger); }}
    .layout {{ display:grid; grid-template-columns:minmax(360px,.95fr) minmax(420px,1.1fr); gap:16px; }}
    .stack {{ display:grid; gap:16px; align-content:start; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:22px; box-shadow:var(--shadow); overflow:hidden; }}
    .card.subtle {{ background:rgba(255,252,245,.62); box-shadow:0 8px 22px rgba(44,28,11,.05); }}
    .card.subtle .card-title {{ color:#74624d; font-size:13px; }}
    .card-body {{ padding:18px; }}
    .card-header {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }}
    .card-title {{ margin:0; text-transform:uppercase; letter-spacing:.02em; font-weight:900; color:#615747; font-size:15px; }}
    .section-note {{ color:var(--muted); font-size:12px; line-height:1.4; margin:0 0 10px; }}
    .board-shell {{ border-radius:16px; border:1px solid var(--line); background:rgba(255,255,255,.45); padding:10px; }}
    .board-frame {{ border-radius:14px; border:1px solid var(--line); background:rgba(255,255,255,.35); padding:8px; overflow:auto; }}
    #board {{ border-collapse:collapse; margin:0 auto; }}
    #board td {{ width: clamp(42px, 8vw, 56px); height: clamp(42px, 8vw, 56px); text-align:center; vertical-align:middle; font-size: clamp(29px,6.5vw,40px); line-height:1; position:relative; user-select:none; cursor:pointer; }}
    #board td.light {{ background:#d8c29d; }} #board td.dark {{ background:#ba8f63; }}
    #board td .coord {{ position:absolute; left:4px; bottom:2px; font-size:10px; color:rgba(58,39,20,.55); font-weight:800; }}
    #board td.selected {{ outline:3px solid rgba(15,118,110,.65); outline-offset:-3px; }}
    .board-meta {{ display:grid; grid-template-columns:1fr auto; gap:12px; margin-top:12px; align-items:end; }}
    .status-line small {{ display:block; color:var(--muted); font-style:italic; font-weight:400; letter-spacing:0; font-size:10px; }}
    .status-line span {{ display:block; margin-top:4px; color:var(--muted); font-style:italic; font-weight:400; font-size:12px; }}
    button {{ border:0; border-radius:12px; padding:10px 13px; font-weight:700; cursor:pointer; color:#fff; background:linear-gradient(180deg,var(--accent),var(--accent-2)); box-shadow:0 8px 18px rgba(109,117,93,.24); }}
    button.secondary {{ background:var(--card-strong); color:var(--text); border:1px solid var(--line); box-shadow:none; }}
    button:disabled {{ opacity:.55; cursor:not-allowed; box-shadow:none; }}
    .board-controls {{ margin-top:12px; }}
    .control-row {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .control-group {{ border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.48); padding:10px; display:grid; gap:6px; }}
    .control-group span {{ color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }}
    .coach-card {{ border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.46); padding:12px; display:grid; gap:6px; margin-top:10px; }}
    .coach-card:first-child {{ margin-top:0; }}
    .coach-card h3 {{ margin:0; font-size:12px; font-weight:900; color:#716654; text-transform:uppercase; letter-spacing:.03em; }}
    .coach-card p {{ margin:0; font-size:15px; line-height:1.4; }}
    .coach-card.empty p {{ color:var(--muted); }}
    .history-list {{ display:grid; gap:8px; margin-top:10px; }}
    .history-item {{ border:1px dashed var(--line); border-radius:12px; padding:10px; background:rgba(255,255,255,.34); }}
    .history-item small {{ color:var(--muted); display:block; margin-bottom:4px; }} .history-item p {{ margin:0; font-size:13px; line-height:1.35; }}
    .ask-actions {{ display:flex; gap:8px; flex-wrap:wrap; }} .ask-actions button {{ flex:1 1 140px; }}
    .voice-orb-wrap {{ margin-top:12px; border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.4); padding:14px; display:grid; gap:8px; justify-items:center; }}
    .voice-orb {{ width:72px; height:72px; border-radius:999px; position:relative; background:radial-gradient(circle at 35% 30%, #cfd2be, #8f9479 62%, #727963); box-shadow:0 10px 24px rgba(114,121,99,.22); }}
    .voice-orb .voice-piece {{ position:absolute; inset:0; display:grid; place-items:center; font-size:34px; color:rgba(255,255,255,.96); text-shadow:0 2px 8px rgba(0,0,0,.22); }}
    .voice-orb::before {{ content:''; position:absolute; inset:-8px; border-radius:999px; border:2px solid rgba(126,134,108,.18); }}
    .voice-orb::after {{ content:''; position:absolute; inset:-14px; border-radius:999px; border:2px solid rgba(126,134,108,.1); }}
    .voice-orb.idle {{ opacity:.85; filter:saturate(.8); }}
    .voice-orb.listening {{ animation: pulse-listen 1.1s ease-out infinite; }}
    .voice-orb.thinking {{ animation: pulse-think 1.5s linear infinite; }}
    .voice-orb.speaking {{ animation: pulse-speak .45s ease-in-out infinite alternate; }}
    .voice-orb.error {{ background:radial-gradient(circle at 35% 30%, #fca5a5, #b91c1c 70%, #7f1d1d); }}
    .voice-orb.listening .voice-piece {{ animation: piece-listen .9s ease-in-out infinite alternate; }}
    .voice-orb.thinking .voice-piece {{ animation: piece-think 1.2s ease-in-out infinite; }}
    .voice-orb.speaking .voice-piece {{ animation: piece-speak .3s ease-in-out infinite alternate; }}
    .voice-orb-label {{ font-size:13px; font-weight:900; color:#5d4a35; }}
    .voice-orb-sub {{ font-size:12px; color:var(--muted); text-align:center; }}
    @keyframes pulse-listen {{
      0% {{ transform:scale(1); box-shadow:0 10px 24px rgba(114,121,99,.18); }}
      70% {{ transform:scale(1.04); box-shadow:0 0 0 14px rgba(126,134,108,.1), 0 10px 24px rgba(114,121,99,.22); }}
      100% {{ transform:scale(1); box-shadow:0 0 0 0 rgba(126,134,108,0), 0 10px 24px rgba(114,121,99,.18); }}
    }}
    @keyframes pulse-think {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
    @keyframes pulse-speak {{
      from {{ transform:scale(1.0); box-shadow:0 10px 24px rgba(114,121,99,.2); }}
      to {{ transform:scale(1.08); box-shadow:0 0 0 10px rgba(126,134,108,.12), 0 12px 28px rgba(114,121,99,.25); }}
    }}
    @keyframes piece-listen {{ from {{ transform:translateY(0px); }} to {{ transform:translateY(-2px); }} }}
    @keyframes piece-think {{ 0%,100% {{ transform:rotate(-6deg) scale(.98); }} 50% {{ transform:rotate(6deg) scale(1.02); }} }}
    @keyframes piece-speak {{ from {{ transform:scale(1); }} to {{ transform:scale(1.12); }} }}
    .mic-status-row {{ display:flex; gap:8px; align-items:center; margin-top:10px; flex-wrap:wrap; }}
    .mic-pill {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:7px 10px; border:1px solid var(--line); background:var(--card-strong); font-size:12px; font-weight:700; }}
    .mic-pill .dot {{ width:8px; height:8px; border-radius:999px; background:#8d7b67; }}
    .mic-pill.ready .dot {{ background:var(--ok); }} .mic-pill.busy .dot {{ background:var(--warn); }} .mic-pill.error .dot {{ background:var(--danger); }}
    details.slim {{ border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.18); margin-top:10px; }}
    details.slim summary {{ list-style:none; cursor:pointer; padding:12px 14px; font-weight:800; font-size:14px; display:flex; justify-content:space-between; }}
    details.slim summary::-webkit-details-marker {{ display:none; }}
    .details-body {{ padding:0 14px 14px; border-top:1px solid var(--line); }}
    .param-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }}
    .param-card {{ border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,.4); padding:10px; }}
    .param-card b {{ display:block; font-size:12px; color:#6b5946; text-transform:uppercase; }} .param-card span {{ display:block; margin-top:4px; font-size:14px; }} .param-card small {{ display:block; color:var(--muted); font-size:11px; margin-top:4px; }}
    .file-list {{ border:1px solid var(--line); background:rgba(255,255,255,.4); border-radius:12px; padding:10px; max-height:180px; overflow:auto; font-size:13px; margin-top:8px; }}
    .file-list ul {{ margin:0; padding-left:18px; }} .file-list li {{ margin:4px 0; }}
    .log-shell {{ background:#17120d; color:#efe7db; border-radius:16px; border:1px solid rgba(255,255,255,.08); padding:14px; min-height:220px; max-height:420px; overflow:auto; }}
    pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; line-height:1.35; }}
    @media (max-width:980px) {{ .layout{{grid-template-columns:1fr;}} .app-header{{grid-template-columns:1fr;}} .status-strip{{justify-content:flex-start;}} .board-meta{{grid-template-columns:1fr;}} .control-row{{grid-template-columns:1fr;}} .param-grid{{grid-template-columns:1fr;}} }}
  </style>
</head>
<body>
  <div class=\"app-shell\">
    <header class=\"app-header\">
      <div class=\"title-block\">
        <h1>Chess Coach</h1>
        <p>Voxtral + Mistral LLM + Pipecat + Pocket TTS</p>
      </div>
      <div class=\"status-strip\">
        <div id=\"turn-chip\" class=\"chip\"><span class=\"dot\"></span><span>Turn: White</span></div>
        <div id=\"pending-chip\" class=\"chip ok\"><span class=\"dot\"></span><span>Coach ready</span></div>
      </div>
    </header>

    <div class=\"layout\">
      <section class=\"stack\">
        <div class=\"card\"><div class=\"card-body\">
          <div class=\"board-shell\"><div class=\"board-frame\"><table id=\"board\">{start_board_html}</table></div></div>
          <div class=\"board-meta\">
            <div id=\"app-status\" class=\"status-line\"><small>Session status</small><span>Play White on the board, then ask the coach in voice loop.</span></div>
            <button id=\"new-game-btn\" class=\"secondary\" type=\"button\">New game</button>
          </div>
        </div></div>
      </section>

      <section class=\"stack\">
        <div class=\"card\"><div class=\"card-body\">
          <div class=\"card-header\"><h2 class=\"card-title\">Voice Status</h2></div>
          <div class=\"voice-orb-wrap\" style=\"margin-top:0;\">
            <div id=\"voice-orb\" class=\"voice-orb idle\" aria-hidden=\"true\"><span class=\"voice-piece\">♕</span></div>
            <div id=\"voice-orb-label\" class=\"voice-orb-label\">Coach idle</div>
            <div id=\"voice-orb-sub\" class=\"voice-orb-sub\">The coach animation reacts for both move commentary and voice questions.</div>
          </div>
          <div class=\"mic-status-row\">
            <div id=\"mic-state\" class=\"mic-pill ready\"><span class=\"dot\"></span><span>Mic ready</span></div>
            <div id=\"pipe-state\" class=\"mic-pill\"><span class=\"dot\"></span><span>Pipeline idle</span></div>
            <div id=\"audio-state\" class=\"mic-pill\"><span class=\"dot\"></span><span>No audio yet</span></div>
          </div>
          <audio id=\"audio-player\" preload=\"none\" style=\"display:none\"></audio>
        </div></div>

        <div class=\"card\"><div class=\"card-body\">
          <div class=\"card-header\"><h2 class=\"card-title\">Move Coach</h2></div>
          <div id=\"transcript-card\" class=\"coach-card empty\"><h3>Your move (analysis)</h3><p>No move commentary yet.</p></div>
          <div id=\"reply-card\" class=\"coach-card empty\"><h3>Black reply + next tip</h3><p>No Black reply commentary yet.</p></div>
        </div></div>

        <div class=\"card\"><div class=\"card-body\">
          <div class=\"card-header\"><h2 class=\"card-title\">Ask The Coach</h2></div>
          <div class=\"ask-actions\">
            <button id=\"voice-loop-btn\" type=\"button\">Ask coach (voice loop)</button>
          </div>
          <div class=\"coach-card\" style=\"margin-top:10px;\">
            <h3>Conversation</h3>
            <div id=\"history-list\" class=\"history-list\"></div>
          </div>
        </div></div>

        <div class=\"card subtle\"><div class=\"card-body\">
          <div class=\"card-header\"><h2 class=\"card-title\">Settings and Logs</h2></div>
          <details class=\"slim\">
            <summary><span>Pipecat Settings</span><span style=\"color:#6c5c49;font-size:12px;\">Latency, chunk, models</span></summary>
            <div class=\"details-body\">
              <div class=\"param-grid\">
                <div class=\"param-card\"><b>Realtime STT model</b><span>{html.escape(cfg['realtime_model'])}</span><small>Voxtral realtime websocket transcription</small></div>
                <div class=\"param-card\"><b>Coach chat model</b><span>{html.escape(cfg['chat_model'])}</span><small>Generates coach answer from transcript + board position</small></div>
                <div class=\"param-card\"><b>Pocket TTS URL</b><span>{html.escape(cfg['pocket_tts_url'])}</span><small>Local TTS backend (no credits)</small></div>
                <div class=\"param-card\"><b>Pocket TTS voice</b><span>{html.escape(cfg['pocket_tts_voice'])}</span><small>Current local voice preset</small></div>
                <div class=\"param-card\"><b>Chunk size</b><span>{cfg['chunk_ms']} ms</span><small>Mic audio chunk size sent to realtime STT</small></div>
                <div class=\"param-card\"><b>Target streaming delay</b><span>{cfg['target_delay_ms']} ms</span><small>Latency/quality tradeoff (lower=faster, higher=smoother)</small></div>
              </div>
              <p class=\"section-note\" style=\"margin-top:10px;\">Chosen defaults: chunk <strong>{cfg['chunk_ms']}ms</strong>, target delay <strong>{cfg['target_delay_ms']}ms</strong>, voice loop duration <strong>{cfg['duration']}s</strong> per question.</p>
            </div>
          </details>
          <details class=\"slim\">
            <summary><span>Session Log</span><span id=\"run-status-inline\" style=\"color:#6c5c49;font-size:12px;\">Idle</span></summary>
            <div class=\"details-body\">
              <div class=\"log-shell\"><pre id=\"log\">Ready.</pre></div>
              <div id=\"recent-files\" class=\"file-list\">No files yet.</div>
            </div>
          </details>
        </div></div>
      </section>
    </div>
  </div>

  <script>
    const chessLoadError = '';

    const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const PIECES = {{ p:'♟', r:'♜', n:'♞', b:'♝', q:'♛', k:'♚', P:'♙', R:'♖', N:'♘', B:'♗', Q:'♕', K:'♔' }};
    const FILES = ['a','b','c','d','e','f','g','h'];
    const HISTORY_KEY = 'pipecat_stream_history_v1';
    const LAST_STEM_KEY = 'pipecat_stream_last_stem_v1';
    const SERVER_BOOT_ID = '{SERVER_BOOT_ID}';
    const SERVER_BOOT_TS = Number(SERVER_BOOT_ID || '0');
    const SERVER_BOOT_KEY = 'pipecat_stream_server_boot_v1';
    const FIXED_RUN = {{ mode:'voice-turn', duration:{cfg['duration']}, autoplay:true, chunk_ms:{cfg['chunk_ms']}, target_delay_ms:{cfg['target_delay_ms']} }};

    const boardEl = document.getElementById('board');
    const appStatusEl = document.getElementById('app-status');
    const turnChipEl = document.getElementById('turn-chip');
    const pendingChipEl = document.getElementById('pending-chip');
    const micStateEl = document.getElementById('mic-state');
    const pipeStateEl = document.getElementById('pipe-state');
    const audioStateEl = document.getElementById('audio-state');
    const transcriptCardEl = document.getElementById('transcript-card');
    const replyCardEl = document.getElementById('reply-card');
    const historyListEl = document.getElementById('history-list');
    const logEl = document.getElementById('log');
    const recentFilesEl = document.getElementById('recent-files');
    const audioPlayerEl = document.getElementById('audio-player');
    const runStatusInlineEl = document.getElementById('run-status-inline');
    const voiceLoopBtnEl = document.getElementById('voice-loop-btn');
    const voiceOrbEl = document.getElementById('voice-orb');
    const voiceOrbLabelEl = document.getElementById('voice-orb-label');
    const voiceOrbSubEl = document.getElementById('voice-orb-sub');

    const game = null;
    const fallbackState = parseFenRows(START_FEN);
    let selectedSquare = null;
    let blackAutoPending = false;
    let turnCoachingBusy = false;
    let voiceLoopTimers = [];
    let liveAskActive = false;
    let liveAskStopRequested = false;

    function setStatus(text) {{ const el = appStatusEl.querySelector('span'); if (el) el.textContent = text; }}
    function setChip(el, text, tone='') {{ el.className = 'chip ' + tone; el.querySelector('span:last-child').textContent = text; }}
    function setPill(el, text, tone='') {{ el.className = 'mic-pill ' + tone; el.querySelector('span:last-child').textContent = text; }}
    function setVoiceOrb(mode, label, sub='') {{
      if (voiceOrbEl) voiceOrbEl.className = 'voice-orb ' + (mode || 'idle');
      if (voiceOrbLabelEl) voiceOrbLabelEl.textContent = label || 'Coach idle';
      if (voiceOrbSubEl) voiceOrbSubEl.textContent = sub || '';
    }}
    function setAskButtonUi() {{
      if (!voiceLoopBtnEl) return;
      if (turnCoachingBusy && !liveAskActive) {{
        voiceLoopBtnEl.textContent = 'Ask coach (wait for tip)';
      }} else {{
        voiceLoopBtnEl.textContent = liveAskActive ? 'Stop live ask' : 'Ask coach (live)';
      }}
      voiceLoopBtnEl.disabled = !!turnCoachingBusy && !liveAskActive;
      voiceLoopBtnEl.classList.toggle('secondary', liveAskActive);
    }}
    function setCoachCard(el, title, text, emptyText) {{ el.className = (text && String(text).trim()) ? 'coach-card' : 'coach-card empty'; el.querySelector('h3').textContent = title; el.querySelector('p').textContent = (text && String(text).trim()) ? String(text).trim() : emptyText; }}
    function escapeHtml(s) {{ return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
    function currentFen() {{ return game ? game.fen() : fallbackFen(); }}
    function summarizeRunFailure(output) {{
      const lines = String(output || '').split(/\\r?\\n/).map((x) => x.trim()).filter(Boolean);
      const match = lines.find((l) =>
        l.startsWith('Error:') ||
        l.startsWith('Realtime transcription failed:') ||
        l.startsWith('Coach reply failed:') ||
        l.startsWith('Pocket TTS failed:') ||
        l.includes('No transcript captured')
      );
      if (match) return match;

      const noisePatterns = [
        /pipecat:<module>/i,
        /^Pipecat POC /,
        /^This POC is isolated/,
        /^Stable app launcher:/,
        /^Environment$/,
        /^Python:/,
        /^Platform:/,
        /^Pocket TTS /,
        /^MISTRAL_API_KEY /,
        /^Microphone transcription test /,
        /^Voice loop test /,
        /^Flow: /,
        /^Duration: /,
        /^Model: /,
        /^Chunk size: /,
        /^Target streaming delay: /,
        /^Recording \\+ streaming\\.{3}/,
        /^Realtime summary$/,
        /^Events seen:/,
        /^Transcript deltas received:/,
        /^Transcript$/,
        /^Saved /,
      ];
      const useful = lines.filter((l) => !noisePatterns.some((p) => p.test(l)));
      return useful[useful.length - 1] || (lines.length ? lines[lines.length - 1] : 'Unknown error');
    }}
    function clearVoiceLoopPhaseUi() {{
      for (const t of voiceLoopTimers) clearTimeout(t);
      voiceLoopTimers = [];
    }}
    function startVoiceLoopPhaseUi(durationSec) {{
      clearVoiceLoopPhaseUi();
      if (!durationSec || durationSec <= 0) {{
        setPill(micStateEl, 'Mic open', 'busy');
        setStatus('Listening... speak naturally. The coach will respond when you pause.');
        setVoiceOrb('listening', 'Listening', 'Speak naturally. The turn ends automatically after a short silence.');
        return;
      }}
      const durationMs = Math.max(1000, Math.round((durationSec || 6) * 1000));
      const startedAt = Date.now();
      let listeningDone = false;
      const tick = () => {{
        const elapsed = Date.now() - startedAt;
        const leftMs = Math.max(0, durationMs - elapsed);
        const left = (leftMs / 1000).toFixed(1);
        if (!listeningDone) {{
          setPill(micStateEl, 'Mic open (listening ' + left + 's)', 'busy');
          setStatus('Listening... speak now. The mic closes automatically after capture.');
          setVoiceOrb('listening', 'Listening', 'Mic open. Speak your follow-up question.');
        }}
        if (leftMs > 0) {{
          voiceLoopTimers.push(setTimeout(tick, 150));
        }} else if (!listeningDone) {{
          listeningDone = true;
          setPill(micStateEl, 'Mic closed', 'ready');
          setPill(pipeStateEl, 'Transcribing (Voxtral realtime)', 'busy');
          setStatus('Mic closed. Transcribing and preparing the coach reply...');
          setVoiceOrb('thinking', 'Transcribing', 'Mic closed. Converting your speech to text.');
          voiceLoopTimers.push(setTimeout(() => {{
            setPill(pipeStateEl, 'Coach thinking', 'busy');
            setStatus('Coach is generating the answer...');
            setVoiceOrb('thinking', 'Coach thinking', 'Generating a short coaching answer.');
          }}, 900));
          voiceLoopTimers.push(setTimeout(() => {{
            setPill(audioStateEl, 'Generating voice (Pocket TTS)', 'busy');
            setStatus('Generating the spoken response...');
            setVoiceOrb('thinking', 'Preparing voice', 'Pocket TTS is generating the spoken reply.');
          }}, 1700));
        }}
      }};
      tick();
    }}

    function loadHistory() {{ try {{ return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }} catch {{ return []; }} }}
    function saveHistory(items) {{ localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 12))); }}
    function pushHistory(item) {{
      const h = loadHistory();
      h.unshift({{ ...item, ts: Number(item?.ts || Date.now()) }});
      saveHistory(h);
      renderHistory();
    }}
    function clearUiCacheForNewServerBoot() {{
      localStorage.removeItem(HISTORY_KEY);
      localStorage.removeItem(LAST_STEM_KEY);
      localStorage.setItem(SERVER_BOOT_KEY, SERVER_BOOT_ID);
    }}
    function renderHistory() {{
      const h = loadHistory().sort((a, b) => Number(b?.ts || 0) - Number(a?.ts || 0));
      historyListEl.innerHTML = h.slice(0,5).map((x)=>'<div class="history-item"><small>'+escapeHtml(x.when||'')+'</small><p><strong>You:</strong> '+escapeHtml(x.transcript||'')+'</p><p><strong>Coach:</strong> '+escapeHtml(x.reply||'')+'</p></div>').join('');
    }}

    function parseFenRows(fen) {{
      const parts = (fen || START_FEN).trim().split(/\\s+/);
      const placement = parts[0] || START_FEN.split(' ')[0];
      const side = ((parts[1] || 'w').toLowerCase() === 'b') ? 'Black' : 'White';
      const rows = placement.split('/').map((row) => {{
        const cells = [];
        for (const ch of row) {{
          if (/^[1-8]$/.test(ch)) {{
            for (let i = 0; i < Number(ch); i++) cells.push(null);
          }} else {{
            cells.push(ch);
          }}
        }}
        return cells;
      }});
      return {{ rows, side }};
    }}

    function fallbackFen() {{
      const rows = [];
      for (let r = 0; r < 8; r++) {{
        let out = '';
        let empty = 0;
        for (let c = 0; c < 8; c++) {{
          const p = fallbackState.rows[r][c];
          if (!p) {{ empty += 1; continue; }}
          if (empty) {{ out += String(empty); empty = 0; }}
          out += p;
        }}
        if (empty) out += String(empty);
        rows.push(out);
      }}
      const side = fallbackState.side === 'Black' ? 'b' : 'w';
      return rows.join('/') + ' ' + side + ' - - 0 1';
    }}

    function parseSquare(square) {{
      if (!/^[a-h][1-8]$/.test(square || '')) return null;
      return {{ c: FILES.indexOf(square[0]), r: 8 - Number(square[1]) }};
    }}

    async function fetchCoachCommentary(phase, moveText) {{
      const res = await fetch('/coach_commentary', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ phase, fen: currentFen(), move_text: moveText || '' }}),
      }});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'coach commentary failed');
      return String(data.text || '').trim();
    }}

    async function speakCommentary(text) {{
      const clean = String(text || '').trim();
      if (!clean) return;
      setPill(audioStateEl, 'Speaking', 'busy');
      setVoiceOrb('speaking', 'Coach speaking', 'Playing the spoken coach response.');
      const res = await fetch('/speak', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ text: clean }}),
      }});
      if (!res.ok) {{
        let msg = 'TTS failed';
        try {{ const e = await res.json(); msg = e.detail || msg; }} catch {{}}
        throw new Error(msg);
      }}
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      audioPlayerEl.src = url;
      await new Promise((resolve) => {{
        const done = () => {{ audioPlayerEl.removeEventListener('ended', done); resolve(); }};
        audioPlayerEl.addEventListener('ended', done, {{ once: true }});
        audioPlayerEl.play().catch(() => resolve());
      }});
      setPill(audioStateEl, 'Audio ready', 'ready');
      setVoiceOrb('idle', 'Coach ready', 'You can play your move or ask a follow-up.');
    }}

    function fallbackBlackCandidates() {{
      const moves = [];
      const knightOffsets = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
      for (let r = 0; r < 8; r++) {{
        for (let c = 0; c < 8; c++) {{
          const p = fallbackState.rows[r][c];
          if (!p || p !== p.toLowerCase()) continue;
          // Very simple pseudo-legal fallback moves (good enough for UI flow).
          if (p === 'p') {{
            const nr = r + 1;
            if (nr < 8 && !fallbackState.rows[nr][c]) moves.push({{ from:[r,c], to:[nr,c] }});
            for (const dc of [-1,1]) {{
              const nc = c + dc;
              if (nr < 8 && nc >= 0 && nc < 8) {{
                const t = fallbackState.rows[nr][nc];
                if (t && t === t.toUpperCase()) moves.push({{ from:[r,c], to:[nr,nc] }});
              }}
            }}
          }} else if (p === 'n') {{
            for (const [dr, dc] of knightOffsets) {{
              const nr = r + dr, nc = c + dc;
              if (nr < 0 || nr > 7 || nc < 0 || nc > 7) continue;
              const t = fallbackState.rows[nr][nc];
              if (!t || t === t.toUpperCase()) moves.push({{ from:[r,c], to:[nr,nc] }});
            }}
          }} else {{
            for (const [dr, dc] of [[1,0],[0,1],[0,-1],[1,1],[1,-1]]) {{
              const nr = r + dr, nc = c + dc;
              if (nr < 0 || nr > 7 || nc < 0 || nc > 7) continue;
              const t = fallbackState.rows[nr][nc];
              if (!t || t === t.toUpperCase()) moves.push({{ from:[r,c], to:[nr,nc] }});
            }}
          }}
        }}
      }}
      return moves;
    }}

    function applyFallbackMove(move) {{
      const [fr, fc] = move.from;
      const [tr, tc] = move.to;
      const piece = fallbackState.rows[fr][fc];
      fallbackState.rows[tr][tc] = piece;
      fallbackState.rows[fr][fc] = null;
    }}

    function renderBoard() {{
      if (!boardEl) return;
      boardEl.innerHTML = '';
      let rows = null;
      let side = 'White';
      if (game) {{
        rows = game.board().map((r) => r.map((p) => p ? (p.color === 'w' ? p.type.toUpperCase() : p.type) : null));
        side = (game.turn()==='w' ? 'White' : 'Black');
      }} else {{
        rows = fallbackState.rows;
        side = fallbackState.side;
      }}
      for (let r = 0; r < 8; r++) {{
        const tr = document.createElement('tr');
        for (let c = 0; c < 8; c++) {{
          const td = document.createElement('td');
          const sq = FILES[c] + String(8-r);
          td.dataset.square = sq;
          td.className = ((r+c)%2===0) ? 'light' : 'dark';
          if (selectedSquare === sq) td.classList.add('selected');
          const p = rows[r][c];
          if (p) td.textContent = PIECES[p] || '';
          if (r === 7 || c === 0) {{
            const coord = document.createElement('span'); coord.className = 'coord';
            coord.textContent = (c===0 ? String(8-r) : '') + (r===7 ? ((c===0?' ':'') + FILES[c]) : '');
            td.appendChild(coord);
          }}
          td.addEventListener('click', () => onSquareClick(sq));
          tr.appendChild(td);
        }}
        boardEl.appendChild(tr);
      }}
      setChip(turnChipEl, 'Turn: ' + side, 'ok');
      if (!game && chessLoadError) {{
        setStatus('Board loaded in fallback mode (chess.js unavailable). ' + chessLoadError);
      }}
      if (game) {{
        setStatus('Board ready. Play White, then ask the coach.');
      }}
    }}

    function onSquareClick(square) {{
      if (!game) {{
        if (fallbackState.side !== 'White') {{ setStatus('It is Black to move. Wait for the automatic Black reply.'); return; }}
        const pos = parseSquare(square);
        if (!pos) return;
        const piece = fallbackState.rows[pos.r][pos.c];
        if (!selectedSquare) {{
          if (!piece || piece !== piece.toUpperCase()) {{ setStatus('Select a White piece first.'); return; }}
          selectedSquare = square;
          renderBoard();
          setStatus('Selected ' + square + '. Click a destination square.');
          return;
        }}
        if (selectedSquare === square) {{ selectedSquare = null; renderBoard(); setStatus('Selection cleared.'); return; }}
        const from = parseSquare(selectedSquare);
        const moving = fallbackState.rows[from.r][from.c];
        if (!moving) {{ selectedSquare = null; renderBoard(); setStatus('Selection lost. Try again.'); return; }}
        const whiteMoveText = selectedSquare + square;
        fallbackState.rows[pos.r][pos.c] = moving;
        fallbackState.rows[from.r][from.c] = null;
        fallbackState.side = 'Black';
        selectedSquare = null;
        renderBoard();
        setStatus('White move applied locally. Coach will comment, then Black will reply.');
        runTurnCoaching(whiteMoveText);
        return;
      }}
      if (blackAutoPending) {{ setStatus('Black is replying... wait a moment.'); return; }}
      if (game.turn() !== 'w') {{ setStatus('It is Black to move. Wait for the automatic Black reply.'); return; }}
      const piece = game.get(square);
      if (!selectedSquare) {{
        if (!piece || piece.color !== 'w') {{ setStatus('Select a White piece first.'); return; }}
        selectedSquare = square;
        renderBoard();
        setStatus('Selected ' + square + '. Click a destination square.');
        return;
      }}
      if (selectedSquare === square) {{ selectedSquare = null; renderBoard(); setStatus('Selection cleared.'); return; }}
      const move = game.move({{ from: selectedSquare, to: square, promotion: 'q' }});
      if (!move) {{
        if (piece && piece.color === 'w') {{ selectedSquare = square; renderBoard(); setStatus('Selected ' + square + '. Click a destination square.'); }}
        else {{ setStatus('Illegal move. Try another square.'); }}
        return;
      }}
      selectedSquare = null;
      renderBoard();
      setStatus('White played ' + move.san + '. Coach will comment, then Black will reply.');
      runTurnCoaching(move.san);
    }}

    async function triggerBlackAutoMove() {{
      if (!game) {{
        if (fallbackState.side !== 'Black') return '';
        blackAutoPending = true;
        setChip(pendingChipEl, 'Black is thinking', 'busy');
        setPill(pipeStateEl, 'Black auto move', 'busy');
        return await new Promise((resolve) => {{
          setTimeout(() => {{
            const candidates = fallbackBlackCandidates();
            if (!candidates.length) {{
              blackAutoPending = false;
              setChip(pendingChipEl, 'Game over', 'error');
              setPill(pipeStateEl, 'Pipeline idle', '');
              setStatus('No fallback Black move available.');
              resolve('');
              return;
            }}
            const move = candidates[Math.floor(Math.random() * candidates.length)];
            const fromSq = FILES[move.from[1]] + String(8 - move.from[0]);
            const toSq = FILES[move.to[1]] + String(8 - move.to[0]);
            applyFallbackMove(move);
            fallbackState.side = 'White';
            blackAutoPending = false;
            renderBoard();
            setPill(pipeStateEl, 'Pipeline idle', '');
            resolve(fromSq + toSq);
          }}, 500);
        }});
      }}
      if (game.turn() !== 'b') return '';
      blackAutoPending = true;
      setChip(pendingChipEl, 'Black is thinking', 'busy');
      setPill(pipeStateEl, 'Black auto move', 'busy');
      return await new Promise((resolve) => {{
        setTimeout(() => {{
          const moves = game.moves();
          if (!moves.length) {{
            blackAutoPending = false;
            setChip(pendingChipEl, 'Game over', 'error');
            setPill(pipeStateEl, 'Pipeline idle', '');
            setStatus('No legal Black move available.');
            resolve('');
            return;
          }}
          const chosen = moves[Math.floor(Math.random() * moves.length)];
          const blackMove = game.move(chosen);
          renderBoard();
          blackAutoPending = false;
          setPill(pipeStateEl, 'Pipeline idle', '');
          resolve(blackMove ? blackMove.san : chosen);
        }}, 500);
      }});
    }}

    async function runVoiceLoopOnce() {{
      const body = {{ ...FIXED_RUN, fen: game ? game.fen() : fallbackFen() }};
      setChip(pendingChipEl, 'Coach running', 'busy');
      setPill(micStateEl, 'Mic open', 'busy');
      setPill(pipeStateEl, 'Voice loop running', 'busy');
      setPill(audioStateEl, 'Waiting for reply audio', '');
      setVoiceOrb('listening', 'Listening', 'Speak now. The mic will close automatically.');
      runStatusInlineEl.textContent = 'Running voice-loop...';
      setStatus('Starting voice-loop on current board position...');
      logEl.textContent = 'Running voice-loop...';
      startVoiceLoopPhaseUi(body.mode === 'voice-turn' ? 0 : body.duration);
      try {{
        const res = await fetch('/run', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body) }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        clearVoiceLoopPhaseUi();
        logEl.textContent = (data.stdout || '') + ((data.stderr || '') ? '\\n\\n[stderr]\\n' + data.stderr : '');
        if (data.ok) {{
          await refreshLatest();
          setChip(pendingChipEl, 'Coach ready', 'ok');
          setPill(micStateEl, 'Mic closed', 'ready');
          setPill(pipeStateEl, 'Voice loop done', 'ready');
          setStatus('Voice-loop completed. Mic is closed. You can keep talking or play the next move.');
          setVoiceOrb('idle', 'Coach ready', 'Voice question completed.');
          runStatusInlineEl.textContent = 'Completed';
          return {{ ok: true, silent: false, reason: '' }};
        }} else {{
          const reason = summarizeRunFailure((data.stdout || '') + '\\n' + (data.stderr || ''));
          await refreshLatest();
          setChip(pendingChipEl, 'Run failed', 'error');
          setPill(micStateEl, 'Mic closed', 'ready');
          setPill(pipeStateEl, 'Pipeline error', 'error');
          setStatus('Voice-loop failed: ' + reason);
          setVoiceOrb('error', 'Voice loop failed', reason);
          runStatusInlineEl.textContent = 'Failed';
          return {{ ok: false, silent: /No transcript captured|No transcript text received/i.test(reason), reason }};
        }}
      }} catch (e) {{
        clearVoiceLoopPhaseUi();
        setChip(pendingChipEl, 'Run error', 'error');
        setPill(micStateEl, 'Mic closed', 'ready');
        setPill(pipeStateEl, 'Pipeline error', 'error');
        setStatus('Error: ' + e.message);
        setVoiceOrb('error', 'Voice loop error', String(e.message || e));
        logEl.textContent = String(e);
        runStatusInlineEl.textContent = 'Error';
        return {{ ok: false, silent: false, reason: String(e && e.message ? e.message : e) }};
      }} finally {{
        setPill(micStateEl, 'Mic ready', 'ready');
      }}
    }}

    async function toggleAskCoachLive() {{
      if (liveAskActive) {{
        liveAskStopRequested = true;
        setStatus('Stopping live ask after the current coach reply...');
        setChip(pendingChipEl, 'Stopping live ask', 'busy');
        return;
      }}

      liveAskActive = true;
      liveAskStopRequested = false;
      setAskButtonUi();
      setStatus('Live ask started. Speak naturally. It will stop automatically after silence.');
      setVoiceOrb('listening', 'Listening', 'Live ask is on. Speak naturally.');
      setChip(pendingChipEl, 'Live ask active', 'busy');

      try {{
        while (!liveAskStopRequested) {{
          const result = await runVoiceLoopOnce();
          await refreshLatest();
          if (liveAskStopRequested) break;
          if (!result || result.silent) {{
            setStatus('Live ask stopped (silence detected). You can continue playing.');
            setChip(pendingChipEl, 'Coach ready', 'ok');
            setVoiceOrb('idle', 'Coach ready', 'Live ask stopped after silence.');
            break;
          }}
          if (!result.ok) {{
            setStatus('Live ask stopped: ' + (result.reason || 'Unknown error'));
            break;
          }}
          setStatus('Live ask is still on. Ask another question or stay silent to stop.');
          setChip(pendingChipEl, 'Live ask active', 'busy');
          setVoiceOrb('listening', 'Listening', 'Live ask continues. Speak again or stay silent to stop.');
        }}
      }} finally {{
        liveAskActive = false;
        liveAskStopRequested = false;
        setAskButtonUi();
        if ((pendingChipEl.querySelector('span:last-child')?.textContent || '').includes('Stopping live ask')) {{
          setChip(pendingChipEl, 'Coach ready', 'ok');
          setVoiceOrb('idle', 'Coach ready', 'Live ask stopped.');
        }}
      }}
    }}

    async function runTurnCoaching(whiteMoveText) {{
      if (turnCoachingBusy) return;
      turnCoachingBusy = true;
      setAskButtonUi();
      try {{
        setChip(pendingChipEl, 'Coach commenting your move', 'busy');
        setPill(pipeStateEl, 'Coach commentary', 'busy');
        setStatus('Coach is commenting your move...');

        const whiteComment = await fetchCoachCommentary('white_move', whiteMoveText);
        setCoachCard(transcriptCardEl, 'Your move (analysis)', whiteComment, 'No move commentary yet.');
        await speakCommentary(whiteComment);

        setStatus('Black is playing...');
        const blackMoveText = await triggerBlackAutoMove();
        if (!blackMoveText) {{
          setChip(pendingChipEl, 'Coach ready', 'ok');
          setPill(pipeStateEl, 'Pipeline idle', '');
          return;
        }}

        setChip(pendingChipEl, 'Coach explaining Black move', 'busy');
        setPill(pipeStateEl, 'Coach commentary', 'busy');
        setStatus('Coach is explaining Black\\'s move and your next step...');
        const blackComment = await fetchCoachCommentary('black_reply', blackMoveText);
        setCoachCard(replyCardEl, 'Black reply + next tip', blackComment, 'No Black reply commentary yet.');
        await speakCommentary(blackComment);

        setChip(pendingChipEl, 'Coach ready', 'ok');
        setPill(pipeStateEl, 'Pipeline idle', '');
        setStatus('Black replied and coach explained the plan. You can play White or ask a follow-up.');
      }} catch (e) {{
        setChip(pendingChipEl, 'Coach error', 'error');
        setPill(pipeStateEl, 'Pipeline error', 'error');
        setStatus('Turn coaching failed: ' + (e && e.message ? e.message : e));
      }} finally {{
        turnCoachingBusy = false;
        setAskButtonUi();
      }}
    }}

    async function refreshOutputs() {{
      try {{
        const res = await fetch('/outputs'); const data = await res.json();
        const files = data.files || [];
        recentFilesEl.innerHTML = files.length ? ('<ul>' + files.slice(0,12).map(f => '<li><code>'+escapeHtml(f.name)+'</code> <span style="color:#6c5c49;">('+f.size+' bytes)</span></li>').join('') + '</ul>') : 'No files yet.';
      }} catch {{ recentFilesEl.textContent = 'Failed to load outputs.'; }}
    }}

    async function refreshLatest() {{
      await refreshOutputs();
      try {{
        const res = await fetch('/latest'); const data = await res.json();
        const latest = (data.voice_loop && data.voice_loop.found) ? data.voice_loop : null;
        if (!latest) {{
          audioPlayerEl.removeAttribute('src'); audioPlayerEl.load();
          setPill(audioStateEl, 'No audio yet', '');
          return;
        }}
        if ((Number(latest.mtime || 0) + 1) < SERVER_BOOT_TS) {{
          audioPlayerEl.removeAttribute('src'); audioPlayerEl.load();
          setPill(audioStateEl, 'No audio yet', '');
          return;
        }}
        const transcript = latest.transcript || '';
        const reply = latest.reply || '';
        if ((transcript || reply) && latest.stem && localStorage.getItem(LAST_STEM_KEY) !== latest.stem) {{
          pushHistory({{ when: new Date().toLocaleTimeString(), transcript, reply, ts: Date.now() }});
          localStorage.setItem(LAST_STEM_KEY, latest.stem);
        }}
        if (latest.audio_file) {{
          audioPlayerEl.src = '/artifact/' + encodeURIComponent(latest.audio_file);
          setPill(audioStateEl, 'Audio ready', 'ready');
        }} else {{
          audioPlayerEl.removeAttribute('src'); audioPlayerEl.load();
          setPill(audioStateEl, 'No audio output', '');
        }}
      }} catch {{ setStatus('Failed to load latest outputs.'); }}
    }}

    function resetGame() {{
      if (game) game.reset();
      fallbackState.rows = parseFenRows(START_FEN).rows;
      fallbackState.side = 'White';
      selectedSquare = null;
      blackAutoPending = false;
      localStorage.removeItem(HISTORY_KEY);
      localStorage.removeItem(LAST_STEM_KEY);
      renderHistory();
      renderBoard();
      setCoachCard(transcriptCardEl, 'Your move (analysis)', '', 'No move commentary yet.');
      setCoachCard(replyCardEl, 'Black reply + next tip', '', 'No Black reply commentary yet.');
      setStatus('New game. Play White on the board, then ask the coach.');
      setChip(pendingChipEl, 'Coach ready', 'ok');
      setPill(pipeStateEl, 'Pipeline idle', '');
      logEl.textContent = 'Conversation reset (UI history only).';
    }}

    document.getElementById('voice-loop-btn').addEventListener('click', toggleAskCoachLive);
    document.getElementById('new-game-btn').addEventListener('click', resetGame);

    clearUiCacheForNewServerBoot();
    setPill(micStateEl, 'Mic ready', 'ready');
    setPill(pipeStateEl, 'Pipeline idle', '');
    setPill(audioStateEl, 'No audio yet', '');
    setVoiceOrb('idle', 'Coach idle', 'Press “Ask coach” to start recording.');
    setAskButtonUi();
    renderBoard();
    renderHistory();
    refreshLatest();
  </script>
</body>
</html>"""


@APP.post('/run')
async def run_mode(req: RunRequest) -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(_run_subprocess, req))


@APP.post('/coach_commentary')
def coach_commentary(req: CoachCommentaryRequest) -> dict[str, str]:
    phase = (req.phase or "").strip()
    fen = (req.fen or "").strip()
    move_text = (req.move_text or "").strip()
    if phase not in {"white_move", "black_reply"}:
        raise HTTPException(status_code=400, detail="invalid phase")
    if not fen:
        raise HTTPException(status_code=400, detail="missing fen")

    model = os.getenv("PIPECAT_POC_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    if phase == "white_move":
        system = (
            "You are a chess coach for a beginner/intermediate player (around 1000-1200 Elo). "
            "Comment only the player's last White move in simple English. "
            "Be practical, short, and spoken-friendly. "
            "Give at most 2 short sentences. No long variations. No markdown."
        )
        user = (
            f"fen={fen}\n"
            f"white_move={move_text}\n"
            "Task: briefly comment the White move and mention one simple idea."
        )
    else:
        system = (
            "You are a chess coach for a beginner/intermediate player (around 1000-1200 Elo). "
            "After Black's move, explain Black's idea and give ONE simple next move idea for White. "
            "Keep it short, practical, and spoken-friendly. "
            "Give at most 2 short sentences. No long variations. No markdown."
        )
        user = (
            f"fen={fen}\n"
            f"black_move={move_text}\n"
            "Task: explain Black's move simply, then give one clear next-step tip for White."
        )

    text = _call_mistral_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.4,
    )
    return {"text": text}


@APP.post('/speak')
def speak(req: SpeakRequest) -> Response:
    audio = _pocket_tts_speak_bytes(req.text)
    return Response(content=audio, media_type="audio/wav")


@APP.get('/outputs')
def outputs() -> dict[str, Any]:
    return {'files': _list_output_files()}


@APP.get('/latest')
def latest() -> dict[str, Any]:
    return {
        'voice_loop': _latest_group('voice-loop'),
        'mic_realtime': _latest_group('mic-realtime'),
    }


@APP.get('/artifact/{name}')
def artifact(name: str):
    safe = Path(name).name
    path = OUTPUT_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail='file not found')
    if path.suffix.lower() not in {'.wav', '.txt'} and not path.name.endswith('.reply.txt') and not path.name.endswith('.transcript.txt'):
        raise HTTPException(status_code=400, detail='unsupported file type')
    return FileResponse(path)


app = APP
