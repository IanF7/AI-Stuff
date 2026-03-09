import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd


PIPER_PATH = os.getenv("PIPER_PATH", "").strip()
PIPER_MODEL = os.getenv("PIPER_MODEL", "").strip()

_speech_queue = queue.Queue()
_worker_thread = None
_worker_started = False
_stop_event = threading.Event()

_sample_rate = None
_sample_rate_lock = threading.Lock()


def _require_paths() -> None:
    if not PIPER_PATH:
        raise RuntimeError("PIPER_PATH is not set.")
    if not PIPER_MODEL:
        raise RuntimeError("PIPER_MODEL is not set.")

    if not Path(PIPER_PATH).exists():
        raise RuntimeError(f"Piper executable not found: {PIPER_PATH}")
    if not Path(PIPER_MODEL).exists():
        raise RuntimeError(f"Piper model not found: {PIPER_MODEL}")

    model_json = Path(PIPER_MODEL + ".json")
    if not model_json.exists():
        raise RuntimeError(f"Piper model config not found: {model_json}")


def _get_sample_rate() -> int:
    global _sample_rate

    with _sample_rate_lock:
        if _sample_rate is not None:
            return _sample_rate

        _require_paths()
        model_json = Path(PIPER_MODEL + ".json")
        data = json.loads(model_json.read_text(encoding="utf-8"))

        # Piper voice configs include audio.sample_rate
        sr = data.get("audio", {}).get("sample_rate")
        if not isinstance(sr, int):
            raise RuntimeError("Could not read audio.sample_rate from Piper model json.")

        _sample_rate = sr
        return _sample_rate


def _prepare_text_for_speech(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", " code omitted. ", text, flags=re.DOTALL)

    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # URLs
    text = re.sub(r"https?://\S+", " link ", text)

    # Simple markdown bullets
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_speakable_chunks(buffer: str):
    """
    Returns (chunks, remainder).

    Low-latency strategy:
    - Prefer complete sentences for natural delivery.
    - Also allow newline / semicolon / colon to flush.
    - If the assistant keeps talking without a hard stop,
      flush a long clause early to reduce latency.
    """
    chunks = []
    start = 0
    split_chars = {".", "!", "?", "\n", ";", ":"}

    for i, ch in enumerate(buffer):
        if ch in split_chars:
            chunk = buffer[start:i + 1].strip()
            if chunk:
                chunks.append(chunk)
            start = i + 1

    remainder = buffer[start:].strip()

    # Force an early split if a clause gets too long.
    # This keeps speech starting quickly without going token-by-token.
    if len(remainder) > 120:
        split_idx = max(
            remainder.rfind(", "),
            remainder.rfind(" "),
        )
        if split_idx > 40:
            early_chunk = remainder[:split_idx].strip()
            remainder = remainder[split_idx:].strip()
            if early_chunk:
                chunks.append(early_chunk)

    return chunks, remainder


def _synthesize_raw_pcm(text: str) -> bytes:
    """
    Runs Piper on one text chunk and returns raw 16-bit mono PCM bytes.
    """
    text = _prepare_text_for_speech(text)
    if not text:
        return b""

    _require_paths()

    cmd = [
        PIPER_PATH,
        "--model", PIPER_MODEL,
        "--output_raw",
    ]

    result = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Piper failed: {stderr}")

    return result.stdout


def _play_pcm_blocking(raw_pcm: bytes) -> None:
    """
    Plays 16-bit mono PCM returned by Piper.
    """
    if not raw_pcm:
        return

    if _stop_event.is_set():
        return

    sample_rate = _get_sample_rate()
    audio = np.frombuffer(raw_pcm, dtype=np.int16)

    # sounddevice expects shape (frames, channels) for explicit mono channel playback
    audio = audio.reshape(-1, 1)

    sd.play(audio, samplerate=sample_rate, blocking=True)
    sd.stop()


def _speech_worker():
    while True:
        item = _speech_queue.get()

        if item is None:
            _speech_queue.task_done()
            break

        try:
            if not _stop_event.is_set():
                raw_pcm = _synthesize_raw_pcm(item)
                _play_pcm_blocking(raw_pcm)
        except Exception as e:
            print(f"\n[TTS Error] {e}")
        finally:
            _speech_queue.task_done()


def start_tts_worker() -> None:
    global _worker_thread, _worker_started

    if _worker_started:
        return

    _stop_event.clear()
    _worker_thread = threading.Thread(target=_speech_worker, daemon=True)
    _worker_thread.start()
    _worker_started = True


def stop_tts_worker() -> None:
    global _worker_thread, _worker_started

    if not _worker_started:
        return

    _stop_event.set()
    clear_tts_queue()
    _speech_queue.put(None)
    _speech_queue.join()

    if _worker_thread is not None:
        _worker_thread.join(timeout=2.0)

    _worker_thread = None
    _worker_started = False


def queue_speak(text: str) -> None:
    text = _prepare_text_for_speech(text)
    if not text:
        return

    start_tts_worker()

    # Optional backlog control:
    # if speech falls too far behind, drop stale queued chunks
    if _speech_queue.qsize() > 3:
        clear_tts_queue()

    _speech_queue.put(text)


def clear_tts_queue() -> None:
    sd.stop()

    while True:
        try:
            item = _speech_queue.get_nowait()
            _speech_queue.task_done()
            if item is None:
                break
        except queue.Empty:
            break