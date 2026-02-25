"""
Pipecat POC (Phase 1) - setup doctor / bootstrap runner.

This script is intentionally lightweight but "real":
- it validates the local environment for a Pipecat experiment
- it confirms whether `pipecat-ai` is installed in the POC env
- it checks Pocket TTS local server reachability

The actual voice pipeline implementation will be added in Phase 1.1/1.2
without touching the stable Chess Coach app.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import io
import os
import platform
import subprocess
import socket
import sys
import time
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

import requests
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _probe_tcp(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_host_port(url: str) -> tuple[str, int]:
    url = (url or "").strip()
    if not url:
        return ("127.0.0.1", 8787)
    clean = url.replace("http://", "").replace("https://", "")
    hostport = clean.split("/", 1)[0]
    if ":" in hostport:
        host, port_s = hostport.rsplit(":", 1)
        try:
            return host or "127.0.0.1", int(port_s)
        except ValueError:
            return (host or "127.0.0.1", 8787)
    return (hostport or "127.0.0.1", 8787)


def _import_pipecat_version() -> tuple[bool, str]:
    try:
        import importlib.metadata as md

        version = md.version("pipecat-ai")
        return True, version
    except Exception as e:
        return False, str(e)


def _print_header() -> None:
    print("Pipecat POC (Phase 1) - Doctor")
    print("")
    print("This POC is isolated from the stable Chess Coach app.")
    print("Stable app launcher: ./start.command")
    print("")


def _print_env_summary() -> None:
    pocket_tts_url = os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8787"))
    mistral_key = os.getenv("MISTRAL_API_KEY", "")
    host, port = _parse_host_port(pocket_tts_url)

    print("Environment")
    print(f"  Python: {platform.python_version()} ({sys.executable})")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Pocket TTS URL: {pocket_tts_url}")
    print(f"  Pocket TTS listening: {'yes' if _probe_tcp(host, port) else 'no'} ({host}:{port})")
    print(f"  MISTRAL_API_KEY present: {'yes' if bool(mistral_key.strip()) else 'no'}")
    print("")


def _print_pipecat_status() -> bool:
    ok, info = _import_pipecat_version()
    print("Pipecat")
    if ok:
        print(f"  pipecat-ai installed: yes (v{info})")
        print("  Phase 1 next step: replace this doctor script with a minimal voice pipeline.")
        print("")
        return True

    print("  pipecat-ai installed: no")
    print(f"  reason: {info}")
    print("")
    print("Install inside the dedicated POC environment:")
    print("  uv sync --project pipecat_poc")
    print("")
    return False


def _print_phase1_next_steps(pipecat_installed: bool) -> None:
    print("Planned Phase 1 progression")
    print("  1. Pipecat import + env validation (this doctor)")
    print("  2. Minimal voice-only pipeline (no chess UI integration)")
    print("  3. Pocket TTS local output in Pipecat")
    print("  4. Voxtral streaming STT")
    print("")
    if not pipecat_installed:
        print("Action now")
        print("  Run: uv sync --project pipecat_poc")
        print("  Then: ./start-pipecat.command")
        print("")


def _normalize_pocket_tts_endpoint(url: str) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8787/tts"
    if base.endswith("/tts"):
        return base
    return f"{base}/tts"


def _pocket_tts_generate_wav_bytes(text: str, voice: str | None = None) -> bytes:
    url = _normalize_pocket_tts_endpoint(
        os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8787"))
    )
    form = {"text": text}
    if voice and voice.strip():
        form["voice_url"] = voice.strip()

    resp = requests.post(url, data=form, timeout=45)
    if resp.status_code >= 400:
        raise RuntimeError(f"Pocket TTS HTTP {resp.status_code}: {resp.text[:300]}")
    ctype = (resp.headers.get("content-type") or "").lower()
    if "audio/" not in ctype and "wav" not in ctype:
        raise RuntimeError(f"Pocket TTS returned unexpected content-type: {ctype or 'unknown'}")
    return resp.content


def _mistral_transcribe_wav(wav_path: Path, model: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")

    headers = {"Authorization": f"Bearer {api_key}"}
    with wav_path.open("rb") as f:
        file_bytes = f.read()

    files = {
        "file": (wav_path.name, io.BytesIO(file_bytes), "audio/wav"),
    }
    data = {"model": model}
    resp = requests.post(
        "https://api.mistral.ai/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Mistral transcription HTTP {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    text = (payload.get("text") or payload.get("transcript") or "").strip()
    if not text:
        raise RuntimeError("Empty transcription response")
    return text


def _mistral_chat_reply(user_text: str, model: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a chess coach for a beginner/intermediate player (~1000-1200 Elo). "
                    "Reply in English in 1-2 short spoken sentences focused on the very next move or next simple plan. "
                    "Keep it practical, simple, and encouraging. No JSON. No markdown."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Mistral chat HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Mistral chat returned no choices")
    msg = choices[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise RuntimeError("Mistral chat returned empty content")
    return text


def _mistral_chess_coach_reply(question: str, fen: str, model: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")
    question = (question or "").strip()
    fen = (fen or "").strip()
    if not question:
        raise RuntimeError("Empty question")
    if not fen:
        raise RuntimeError("Missing FEN for chess coach mode")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.5,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a pedagogical chess coach for a beginner/intermediate player (~1000-1200 Elo). "
                    "Answer in English, short, clear, and practical. Focus on the next move (or one simple next-step plan), "
                    "not long variations. ASCII only, no emoji."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"fen={fen}\n"
                    f"question={question}\n"
                    "Constraints: max 2 short sentences, explain only the next move/idea, no long variations."
                ),
            },
        ],
    }
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Mistral chat HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Mistral chat returned no choices")
    msg = choices[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise RuntimeError("Mistral chat returned empty content")
    return text


def _import_mistral_realtime():
    try:
        from mistralai import Mistral
        try:
            # Preferred for realtime helper types in newer SDKs.
            from mistralai.extra.realtime import AudioFormat
        except Exception:
            from mistralai.models import AudioFormat

        return Mistral, AudioFormat
    except Exception as e:
        raise RuntimeError(
            "Mistral realtime client is not installed in the Pipecat POC environment. "
            "Run `uv sync --project pipecat_poc` after updating dependencies."
        ) from e


def _import_pyaudio():
    try:
        import pyaudio  # type: ignore

        return pyaudio
    except Exception as e:
        raise RuntimeError(
            "PyAudio is not installed in the Pipecat POC environment. "
            "Install local audio support first."
        ) from e


def _list_audio_devices() -> list[dict]:
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    devices: list[dict] = []
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            devices.append(
                {
                    "index": idx,
                    "name": info.get("name", ""),
                    "max_input_channels": int(info.get("maxInputChannels", 0) or 0),
                    "max_output_channels": int(info.get("maxOutputChannels", 0) or 0),
                    "default_sample_rate": int(info.get("defaultSampleRate", 0) or 0),
                }
            )
    finally:
        pa.terminate()
    return devices


def _record_mic_wav(duration_secs: float, output_path: Path, sample_rate: int = 16000) -> tuple[int, int]:
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    channels = 1
    frames_per_buffer = max(320, int(sample_rate * 0.02))  # ~20ms
    chunks: list[bytes] = []

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    try:
        total_chunks = max(1, int(duration_secs * sample_rate / frames_per_buffer))
        for _ in range(total_chunks):
            chunks.append(stream.read(frames_per_buffer, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(chunks))

    return sample_rate, len(chunks)


async def _iter_mic_pcm_chunks(
    *,
    duration_secs: float,
    sample_rate: int = 16000,
    chunk_ms: int = 200,
) -> AsyncIterator[bytes]:
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    channels = 1
    frames_per_buffer = max(320, int(sample_rate * (chunk_ms / 1000.0)))

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    try:
        total_chunks = max(1, int(duration_secs * 1000 / chunk_ms))
        for _ in range(total_chunks):
            chunk = await asyncio.to_thread(
                stream.read, frames_per_buffer, False  # exception_on_overflow
            )
            yield chunk
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


async def _iter_mic_pcm_chunks_until_turn_end(
    *,
    sample_rate: int = 16000,
    chunk_ms: int = 200,
    turn_end_silence_ms: int = 1000,
    idle_stop_ms: int = 2500,
    max_turn_secs: float = 20.0,
    energy_threshold: int = 220,
) -> AsyncIterator[bytes]:
    pyaudio = _import_pyaudio()
    pa = pyaudio.PyAudio()
    channels = 1
    frames_per_buffer = max(320, int(sample_rate * (chunk_ms / 1000.0)))

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    started_at = time.monotonic()
    last_voice_at: float | None = None
    speech_started = False
    try:
        while True:
            chunk = await asyncio.to_thread(stream.read, frames_per_buffer, False)
            now = time.monotonic()
            yield chunk

            try:
                rms = int(audioop.rms(chunk, 2))
            except Exception:
                rms = 0
            if rms >= energy_threshold:
                speech_started = True
                last_voice_at = now

            if not speech_started and (now - started_at) * 1000 >= idle_stop_ms:
                break
            if speech_started and last_voice_at is not None and (now - last_voice_at) * 1000 >= turn_end_silence_ms:
                break
            if (now - started_at) >= max_turn_secs:
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


class DemoTraceProcessor(FrameProcessor):
    """Logs frame flow for the POC demo."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            print(f"[trace:{direction.name.lower()}] {frame.__class__.__name__}: {frame.text}")
        elif isinstance(frame, EndFrame):
            print(f"[trace:{direction.name.lower()}] EndFrame(reason={frame.reason})")
        await self.push_frame(frame, direction)


class DemoCoachProcessor(FrameProcessor):
    """Very small Pipecat processor that turns user text into coach text."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only react to downstream plain TextFrame inputs in this demo.
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            user_text = (frame.text or "").strip()
            if not user_text:
                await self.push_frame(frame, direction)
                return

            coach_text = (
                "Pipecat POC coach: nice. "
                f"You said '{user_text}'. "
                "Phase 1.1 is working: frames are flowing through the pipeline."
            )
            await self.push_frame(TextFrame(coach_text), direction)
            return

        await self.push_frame(frame, direction)


class DemoSinkProcessor(FrameProcessor):
    """Terminal processor that prints output text."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            print("")
            print("Pipeline output")
            print(f"  {frame.text}")
            print("")
        elif direction == FrameDirection.DOWNSTREAM and isinstance(frame, EndFrame):
            print("Pipeline finished.")
        await self.push_frame(frame, direction)


class DemoPocketTTSProcessor(FrameProcessor):
    """Generate a local WAV file using pocket-tts from downstream text frames."""

    def __init__(
        self,
        *,
        output_dir: Path,
        voice: str | None = None,
        autoplay: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_dir = output_dir
        self._voice = (voice or "").strip()
        self._autoplay = autoplay

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if text:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = self._output_dir / f"demo-tts-{ts}.wav"
                wav_bytes = await asyncio.to_thread(_pocket_tts_generate_wav_bytes, text, self._voice or None)
                out_path.write_bytes(wav_bytes)
                print(f"Pocket TTS output saved: {out_path}")
                print(f"  bytes: {len(wav_bytes)}")
                if self._autoplay:
                    await asyncio.to_thread(_try_autoplay_audio, out_path)
        await self.push_frame(frame, direction)


def _try_autoplay_audio(path: Path) -> None:
    system = platform.system().lower()
    try:
        if system == "darwin":
            subprocess.run(["afplay", str(path)], check=False)
            return
        if system == "linux":
            subprocess.run(["xdg-open", str(path)], check=False)
            return
        print("Autoplay not supported on this platform.")
    except Exception as e:
        print(f"Autoplay failed: {e}")


async def _run_demo_pipeline(user_text: str) -> None:
    pipeline = Pipeline(
        [
            DemoTraceProcessor(name="DemoTraceIn"),
            DemoCoachProcessor(name="DemoCoach"),
            DemoTraceProcessor(name="DemoTraceOut"),
            DemoSinkProcessor(name="DemoSink"),
        ]
    )

    task = PipelineTask(
        pipeline,
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
        params=PipelineParams(
            enable_heartbeats=False,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    async def _push_frames():
        await asyncio.sleep(0.05)
        await task.queue_frame(TextFrame(user_text))
        await task.queue_frame(EndFrame(reason="demo complete"))

    runner = PipelineRunner(handle_sigint=True)
    await asyncio.gather(runner.run(task), _push_frames())


async def _run_demo_tts_pipeline(user_text: str, *, autoplay: bool = False) -> None:
    output_dir = Path(__file__).parent / "output"
    pocket_voice = os.getenv("POCKET_TTS_VOICE", os.getenv("LOCAL_TTS_VOICE", "")).strip()

    pipeline = Pipeline(
        [
            DemoTraceProcessor(name="DemoTraceIn"),
            DemoCoachProcessor(name="DemoCoach"),
            DemoPocketTTSProcessor(
                name="DemoPocketTTS",
                output_dir=output_dir,
                voice=pocket_voice or None,
                autoplay=autoplay,
            ),
            DemoSinkProcessor(name="DemoSink"),
        ]
    )

    task = PipelineTask(
        pipeline,
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
        params=PipelineParams(
            enable_heartbeats=False,
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    async def _push_frames():
        await asyncio.sleep(0.05)
        await task.queue_frame(TextFrame(user_text))
        await task.queue_frame(EndFrame(reason="demo-tts complete"))

    runner = PipelineRunner(handle_sigint=True)
    await asyncio.gather(runner.run(task), _push_frames())


def _run_mic_doctor() -> int:
    print("Microphone doctor (Phase 1.3 prep)")
    print("")
    try:
        devices = _list_audio_devices()
    except RuntimeError as e:
        print(f"PyAudio status: missing ({e})")
        print("")
        print("Install local audio support (macOS):")
        print("  brew install portaudio")
        print("  uv add --project pipecat_poc 'pipecat-ai[local]'")
        print("")
        return 1

    print("PyAudio status: ok")
    print("")
    print("Audio devices:")
    for d in devices:
        io_flags = []
        if d["max_input_channels"] > 0:
            io_flags.append(f"in:{d['max_input_channels']}")
        if d["max_output_channels"] > 0:
            io_flags.append(f"out:{d['max_output_channels']}")
        if not io_flags:
            io_flags.append("no-io")
        print(f"  [{d['index']}] {d['name']} ({' '.join(io_flags)}) sr={d['default_sample_rate']}")
    print("")
    print("Next: run `--mode mic-record --duration 3` to capture a short mic sample.")
    return 0


def _run_mic_record(duration_secs: float) -> int:
    print("Microphone record test (Phase 1.3)")
    print(f"Duration: {duration_secs:.1f}s")
    out_dir = Path(__file__).parent / "output"
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"mic-sample-{ts}.wav"
    try:
        print("Recording... speak now")
        sample_rate, num_chunks = _record_mic_wav(duration_secs, out_path)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Run `--mode mic-doctor` for install guidance.")
        return 1
    except Exception as e:
        print(f"Recording failed: {e}")
        return 1

    print(f"Saved: {out_path}")
    print(f"Sample rate: {sample_rate}")
    print(f"Chunks: {num_chunks}")
    print("Tip: on macOS, play it with `afplay <file.wav>`")
    return 0


def _run_mic_transcribe(duration_secs: float) -> int:
    print("Microphone transcription test (Phase 1.4, batch STT)")
    print(f"Duration: {duration_secs:.1f}s")
    model = os.getenv("PIPECAT_POC_TRANSCRIBE_MODEL", "voxtral-mini-latest").strip() or "voxtral-mini-latest"
    out_dir = Path(__file__).parent / "output"
    ts = time.strftime("%Y%m%d-%H%M%S")
    wav_path = out_dir / f"mic-transcribe-{ts}.wav"
    txt_path = out_dir / f"mic-transcribe-{ts}.txt"

    try:
        print("Recording... speak now")
        sample_rate, num_chunks = _record_mic_wav(duration_secs, wav_path)
        print(f"Saved audio: {wav_path}")
        print(f"Sample rate: {sample_rate}")
        print(f"Chunks: {num_chunks}")
        print(f"Transcribing with Mistral model: {model}")
        transcript = _mistral_transcribe_wav(wav_path, model=model)
    except RuntimeError as e:
        print(f"Error: {e}")
        if "PyAudio" in str(e):
            print("Run `--mode mic-doctor` for local audio setup guidance.")
        return 1
    except Exception as e:
        print(f"Transcription failed: {e}")
        return 1

    txt_path.write_text(transcript + "\n")
    print("")
    print("Transcript")
    print(f"  {transcript}")
    print("")
    print(f"Saved transcript: {txt_path}")
    return 0


async def _capture_mic_realtime_transcript(
    duration_secs: float,
    *,
    chunk_ms: int,
    target_delay_ms: int,
    print_events: bool = True,
) -> tuple[int, str, int, bool]:
    default_realtime_model = "voxtral-mini-transcribe-realtime-2602"
    model = os.getenv("PIPECAT_POC_REALTIME_MODEL", default_realtime_model).strip() or default_realtime_model
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        print("Error: MISTRAL_API_KEY is missing")
        return 1, "", 0, False

    try:
        Mistral, AudioFormat = _import_mistral_realtime()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1, "", 0, False

    sample_rate = 16000
    if print_events:
        print(f"Model: {model}")
        print(f"Chunk size: {chunk_ms} ms")
        print(f"Target streaming delay: {target_delay_ms} ms")
        print("Recording + streaming... speak now")
        print("")

    client = Mistral(api_key=api_key)
    transcript_parts: list[str] = []
    events_seen = 0
    had_delta = False
    final_text = ""

    try:
        audio_stream = _iter_mic_pcm_chunks(
            duration_secs=duration_secs,
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
        )
        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=sample_rate)
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=audio_stream,
            model=model,
            audio_format=audio_format,
            target_streaming_delay_ms=target_delay_ms,
        ):
            events_seen += 1
            etype = getattr(event, "type", "") or ""
            direct_text = str(getattr(event, "text", "") or getattr(event, "transcript", "") or "")
            data = getattr(event, "data", None)
            nested_text = ""
            if isinstance(data, dict):
                nested_text = str(data.get("delta") or data.get("text") or data.get("transcript") or "")
            elif data is not None:
                nested_text = str(
                    getattr(data, "delta", "")
                    or getattr(data, "text", "")
                    or getattr(data, "transcript", "")
                    or ""
                )
            text_value = (direct_text or nested_text or "").strip()

            if etype == "transcription.text.delta" and text_value:
                had_delta = True
                transcript_parts.append(text_value)
                if print_events:
                    print(text_value, end="", flush=True)
            elif etype == "transcription.done":
                if text_value:
                    final_text = text_value
                if print_events:
                    print("\n[realtime-event] transcription.done")
            elif etype == "transcription.language":
                lang = getattr(event, "audio_language", "") or text_value
                if print_events and lang:
                    print(f"[realtime-event] transcription.language: {lang}")
            elif etype and print_events:
                if "error" in etype.lower():
                    print(f"\n[realtime-event] {etype}: {data}")
                elif "end" in etype.lower() or "complete" in etype.lower():
                    print(f"\n[realtime-event] {etype}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130, "", events_seen, had_delta
    except RuntimeError as e:
        print(f"Error: {e}")
        if "PyAudio" in str(e):
            print("Run `--mode mic-doctor` for local audio setup guidance.")
        return 1, "", events_seen, had_delta
    except Exception as e:
        msg = str(e)
        print(f"\nRealtime transcription failed: {msg}")
        if "1008" in msg or "policy violation" in msg.lower():
            print(
                "Hint: this usually means the selected model is not enabled for realtime transcription. "
                "Try PIPECAT_POC_REALTIME_MODEL=voxtral-mini-transcribe-realtime-2602."
            )
        return 1, "", events_seen, had_delta

    transcript = (final_text or "".join(part for part in transcript_parts if part)).strip()
    return 0, transcript, events_seen, had_delta


async def _capture_mic_realtime_transcript_until_turn_end(
    *,
    chunk_ms: int,
    target_delay_ms: int,
    print_events: bool = True,
    turn_end_silence_ms: int = 1000,
    idle_stop_ms: int = 2500,
    max_turn_secs: float = 20.0,
    energy_threshold: int = 220,
) -> tuple[int, str, int, bool]:
    default_realtime_model = "voxtral-mini-transcribe-realtime-2602"
    model = os.getenv("PIPECAT_POC_REALTIME_MODEL", default_realtime_model).strip() or default_realtime_model
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        print("Error: MISTRAL_API_KEY is missing")
        return 1, "", 0, False

    try:
        Mistral, AudioFormat = _import_mistral_realtime()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1, "", 0, False

    sample_rate = 16000
    if print_events:
        print(f"Model: {model}")
        print(f"Chunk size: {chunk_ms} ms")
        print(f"Target streaming delay: {target_delay_ms} ms")
        print(f"Turn-end silence: {turn_end_silence_ms} ms")
        print(f"Idle stop (no speech): {idle_stop_ms} ms")
        print("Recording + streaming... speak now")
        print("")

    client = Mistral(api_key=api_key)
    transcript_parts: list[str] = []
    events_seen = 0
    had_delta = False
    final_text = ""

    try:
        audio_stream = _iter_mic_pcm_chunks_until_turn_end(
            sample_rate=sample_rate,
            chunk_ms=chunk_ms,
            turn_end_silence_ms=turn_end_silence_ms,
            idle_stop_ms=idle_stop_ms,
            max_turn_secs=max_turn_secs,
            energy_threshold=energy_threshold,
        )
        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=sample_rate)
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=audio_stream,
            model=model,
            audio_format=audio_format,
            target_streaming_delay_ms=target_delay_ms,
        ):
            events_seen += 1
            etype = getattr(event, "type", "") or ""
            direct_text = str(getattr(event, "text", "") or getattr(event, "transcript", "") or "")
            data = getattr(event, "data", None)
            nested_text = ""
            if isinstance(data, dict):
                nested_text = str(data.get("delta") or data.get("text") or data.get("transcript") or "")
            elif data is not None:
                nested_text = str(
                    getattr(data, "delta", "")
                    or getattr(data, "text", "")
                    or getattr(data, "transcript", "")
                    or ""
                )
            text_value = (direct_text or nested_text or "").strip()

            if etype == "transcription.text.delta" and text_value:
                had_delta = True
                transcript_parts.append(text_value)
                if print_events:
                    print(text_value, end="", flush=True)
            elif etype == "transcription.done":
                if text_value:
                    final_text = text_value
                if print_events:
                    print("\n[realtime-event] transcription.done")
            elif etype == "transcription.language":
                lang = getattr(event, "audio_language", "") or text_value
                if print_events and lang:
                    print(f"[realtime-event] transcription.language: {lang}")
            elif etype and print_events:
                if "error" in etype.lower():
                    print(f"\n[realtime-event] {etype}: {data}")
                elif "end" in etype.lower() or "complete" in etype.lower():
                    print(f"\n[realtime-event] {etype}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130, "", events_seen, had_delta
    except RuntimeError as e:
        print(f"Error: {e}")
        if "PyAudio" in str(e):
            print("Run `--mode mic-doctor` for local audio setup guidance.")
        return 1, "", events_seen, had_delta
    except Exception as e:
        msg = str(e)
        print(f"\nRealtime transcription failed: {msg}")
        if "1008" in msg or "policy violation" in msg.lower():
            print(
                "Hint: this usually means the selected model is not enabled for realtime transcription. "
                "Try PIPECAT_POC_REALTIME_MODEL=voxtral-mini-transcribe-realtime-2602."
            )
        return 1, "", events_seen, had_delta

    transcript = (final_text or "".join(part for part in transcript_parts if part)).strip()
    return 0, transcript, events_seen, had_delta


async def _run_mic_realtime(duration_secs: float, *, chunk_ms: int, target_delay_ms: int) -> int:
    print("Microphone transcription test (Phase 1.5, Voxtral realtime STT)")
    print(f"Duration: {duration_secs:.1f}s")
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    txt_path = out_dir / f"mic-realtime-{ts}.txt"
    status, transcript, events_seen, had_delta = await _capture_mic_realtime_transcript(
        duration_secs,
        chunk_ms=chunk_ms,
        target_delay_ms=target_delay_ms,
        print_events=True,
    )
    if status != 0:
        return status
    print("")
    print("")
    print("Realtime summary")
    print(f"  Events seen: {events_seen}")
    print(f"  Transcript deltas received: {'yes' if had_delta else 'no'}")
    if transcript:
        txt_path.write_text(transcript + "\n")
        print(f"  Saved transcript: {txt_path}")
        print("")
        print("Transcript")
        print(f"  {transcript}")
    else:
        print("  No transcript text received.")
    return 0


def _run_voice_loop(
    duration_secs: float,
    *,
    chunk_ms: int,
    target_delay_ms: int,
    autoplay: bool,
    fen: str | None = None,
) -> int:
    print("Voice loop test (Phase 1.6)")
    print("Flow: mic realtime STT -> Mistral coach reply -> Pocket TTS")
    print(f"Duration: {duration_secs:.1f}s")
    fen = (fen or os.getenv("PIPECAT_POC_FEN", "")).strip()
    if fen:
        print("Coach mode: chess (FEN provided)")
    else:
        print("Coach mode: generic (no FEN provided)")

    pocket_tts_url = os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8787"))
    host, port = _parse_host_port(pocket_tts_url)
    if not _probe_tcp(host, port):
        print(f"Pocket TTS is not reachable at {host}:{port}")
        print("Start pocket-tts first (or use your main app launcher).")
        return 1

    status, transcript, events_seen, had_delta = asyncio.run(
        _capture_mic_realtime_transcript(
            duration_secs,
            chunk_ms=chunk_ms,
            target_delay_ms=target_delay_ms,
            print_events=True,
        )
    )
    if status != 0:
        return status

    print("")
    print("")
    print("Realtime summary")
    print(f"  Events seen: {events_seen}")
    print(f"  Transcript deltas received: {'yes' if had_delta else 'no'}")
    transcript = transcript.strip()
    if not transcript:
        print("  No transcript text received.")
        return 1

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    transcript_path = out_dir / f"voice-loop-{ts}.transcript.txt"
    reply_path = out_dir / f"voice-loop-{ts}.reply.txt"
    wav_path = out_dir / f"voice-loop-{ts}.wav"
    transcript_path.write_text(transcript + "\n")

    print("")
    print("Transcript")
    print(f"  {transcript}")

    chat_model = os.getenv("PIPECAT_POC_CHAT_MODEL", "mistral-small-latest").strip() or "mistral-small-latest"
    print("")
    print(f"Generating coach reply with Mistral model: {chat_model}")
    try:
        if fen:
            reply = _mistral_chess_coach_reply(transcript, fen, chat_model)
        else:
            reply = _mistral_chat_reply(transcript, chat_model)
    except Exception as e:
        print(f"Coach reply failed: {e}")
        return 1
    reply_path.write_text(reply + "\n")
    print("Coach reply")
    print(f"  {reply}")

    pocket_voice = os.getenv("POCKET_TTS_VOICE", os.getenv("LOCAL_TTS_VOICE", "")).strip()
    print("")
    print("Generating local TTS with Pocket TTS...")
    try:
        wav_bytes = _pocket_tts_generate_wav_bytes(reply, pocket_voice or None)
    except Exception as e:
        print(f"Pocket TTS failed: {e}")
        return 1
    wav_path.write_bytes(wav_bytes)
    print(f"Saved transcript: {transcript_path}")
    print(f"Saved reply: {reply_path}")
    print(f"Saved audio: {wav_path}")
    if autoplay:
        _try_autoplay_audio(wav_path)
    return 0


def _run_voice_turn(
    *,
    chunk_ms: int,
    target_delay_ms: int,
    autoplay: bool,
    fen: str | None = None,
) -> int:
    print("Voice turn test (live ask turn-taking)")
    print("Flow: open mic -> end on silence -> Mistral coach reply -> Pocket TTS")
    fen = (fen or os.getenv("PIPECAT_POC_FEN", "")).strip()
    if fen:
        print("Coach mode: chess (FEN provided)")
    else:
        print("Coach mode: generic (no FEN provided)")

    pocket_tts_url = os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8787"))
    host, port = _parse_host_port(pocket_tts_url)
    if not _probe_tcp(host, port):
        print(f"Pocket TTS is not reachable at {host}:{port}")
        print("Start pocket-tts first (or use your main app launcher).")
        return 1

    status, transcript, events_seen, had_delta = asyncio.run(
        _capture_mic_realtime_transcript_until_turn_end(
            chunk_ms=chunk_ms,
            target_delay_ms=target_delay_ms,
            print_events=True,
            turn_end_silence_ms=int(os.getenv("PIPECAT_POC_TURN_END_MS", "1000") or "1000"),
            idle_stop_ms=int(os.getenv("PIPECAT_POC_IDLE_STOP_MS", "2500") or "2500"),
            max_turn_secs=float(os.getenv("PIPECAT_POC_MAX_TURN_SECS", "20") or "20"),
            energy_threshold=int(os.getenv("PIPECAT_POC_ENERGY_THRESHOLD", "220") or "220"),
        )
    )
    if status != 0:
        return status

    print("")
    print("")
    print("Realtime summary")
    print(f"  Events seen: {events_seen}")
    print(f"  Transcript deltas received: {'yes' if had_delta else 'no'}")
    transcript = transcript.strip()
    if not transcript:
        print("  No transcript text received.")
        return 1

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    transcript_path = out_dir / f"voice-loop-{ts}.transcript.txt"
    reply_path = out_dir / f"voice-loop-{ts}.reply.txt"
    wav_path = out_dir / f"voice-loop-{ts}.wav"
    transcript_path.write_text(transcript + "\n")

    print("")
    print("Transcript")
    print(f"  {transcript}")

    chat_model = os.getenv("PIPECAT_POC_CHAT_MODEL", "mistral-small-latest").strip() or "mistral-small-latest"
    print("")
    print(f"Generating coach reply with Mistral model: {chat_model}")
    try:
        if fen:
            reply = _mistral_chess_coach_reply(transcript, fen, chat_model)
        else:
            reply = _mistral_chat_reply(transcript, chat_model)
    except Exception as e:
        print(f"Coach reply failed: {e}")
        return 1
    reply_path.write_text(reply + "\n")
    print("Coach reply")
    print(f"  {reply}")

    pocket_voice = os.getenv("POCKET_TTS_VOICE", os.getenv("LOCAL_TTS_VOICE", "")).strip()
    print("")
    print("Generating local TTS with Pocket TTS...")
    try:
        wav_bytes = _pocket_tts_generate_wav_bytes(reply, pocket_voice or None)
    except Exception as e:
        print(f"Pocket TTS failed: {e}")
        return 1
    wav_path.write_bytes(wav_bytes)
    print(f"Saved transcript: {transcript_path}")
    print(f"Saved reply: {reply_path}")
    print(f"Saved audio: {wav_path}")
    if autoplay:
        _try_autoplay_audio(wav_path)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pipecat POC doctor")
    parser.add_argument("--quiet", action="store_true", help="Only return exit code checks")
    parser.add_argument(
        "--mode",
        choices=(
            "doctor",
            "demo",
            "demo-tts",
            "mic-doctor",
            "mic-record",
            "mic-transcribe",
            "mic-realtime",
            "voice-loop",
            "voice-turn",
        ),
        default=os.getenv("PIPECAT_POC_MODE", "doctor"),
        help="Run mode (default: doctor)",
    )
    parser.add_argument(
        "--demo-text",
        default="Hello coach, this is a Pipecat test.",
        help="Input text used in demo mode",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Auto-play generated WAV in demo-tts mode (uses afplay on macOS)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Recording duration in seconds for mic modes",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="Realtime mic chunk size in ms (mic-realtime mode)",
    )
    parser.add_argument(
        "--target-delay-ms",
        type=int,
        default=800,
        help="Target streaming delay in ms for Mistral realtime STT (mic-realtime mode)",
    )
    parser.add_argument(
        "--fen",
        default=os.getenv("PIPECAT_POC_FEN", ""),
        help="Optional chess FEN for voice-loop chess coach mode",
    )
    args = parser.parse_args(argv)

    if args.mode == "doctor":
        if not args.quiet:
            _print_header()
            _print_env_summary()
        pipecat_installed = _print_pipecat_status() if not args.quiet else _import_pipecat_version()[0]
        if not args.quiet:
            _print_phase1_next_steps(pipecat_installed)
        return 0

    if args.mode == "mic-doctor":
        _print_header()
        _print_env_summary()
        return _run_mic_doctor()

    if args.mode == "mic-record":
        _print_header()
        _print_env_summary()
        return _run_mic_record(max(0.5, min(15.0, args.duration)))

    if args.mode == "mic-transcribe":
        _print_header()
        _print_env_summary()
        return _run_mic_transcribe(max(0.5, min(20.0, args.duration)))

    if args.mode == "mic-realtime":
        _print_header()
        _print_env_summary()
        return asyncio.run(
            _run_mic_realtime(
                max(1.0, min(30.0, args.duration)),
                chunk_ms=max(50, min(1000, args.chunk_ms)),
                target_delay_ms=max(100, min(5000, args.target_delay_ms)),
            )
        )

    if args.mode == "voice-loop":
        _print_header()
        _print_env_summary()
        return _run_voice_loop(
            max(1.0, min(30.0, args.duration)),
            chunk_ms=max(50, min(1000, args.chunk_ms)),
            target_delay_ms=max(100, min(5000, args.target_delay_ms)),
            autoplay=args.autoplay,
            fen=args.fen,
        )

    if args.mode == "voice-turn":
        _print_header()
        _print_env_summary()
        return _run_voice_turn(
            chunk_ms=max(50, min(1000, args.chunk_ms)),
            target_delay_ms=max(100, min(5000, args.target_delay_ms)),
            autoplay=args.autoplay,
            fen=args.fen,
        )

    # Demo modes
    _print_header()
    _print_env_summary()
    pipecat_installed, info = _import_pipecat_version()
    if not pipecat_installed:
        print("Cannot run demo: pipecat-ai is not installed in pipecat_poc/.venv")
        print(f"Reason: {info}")
        print("Run: uv sync --project pipecat_poc")
        return 1

    print(f"Pipecat {args.mode} mode running with pipecat-ai v{info}")
    print(f"Demo input: {args.demo_text}")
    print("")
    if args.mode == "demo":
        asyncio.run(_run_demo_pipeline(args.demo_text))
        return 0

    # demo-tts
    pocket_tts_url = os.getenv("POCKET_TTS_URL", os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:8787"))
    host, port = _parse_host_port(pocket_tts_url)
    if not _probe_tcp(host, port):
        print(f"Pocket TTS is not reachable at {host}:{port}")
        print("Start it first (or use your main app launcher which starts it automatically).")
        return 1
    asyncio.run(_run_demo_tts_pipeline(args.demo_text, autoplay=args.autoplay))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
