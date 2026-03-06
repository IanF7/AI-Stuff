import os
from pathlib import Path
import pyttsx3

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        rate = int(os.getenv("TTS_RATE", "200"))
        volume = float(os.getenv("TTS_VOLUME", "1.0"))
        _engine.setProperty("rate", rate)
        _engine.setProperty("volume", volume)
    return _engine

def speak_text(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    engine = get_engine()
    engine.say(text)
    engine.runAndWait()

def save_text_to_file(text: str, filename: str = "reply.wav") -> Path | None:
    text = (text or "").strip()
    if not text:
        return
    
    engine = get_engine()
    out_path = AUDIO_DIR / filename
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return out_path
