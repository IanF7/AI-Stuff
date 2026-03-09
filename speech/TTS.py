import os
import re
import queue
import tempfile
import threading
import subprocess
import winsound
from pathlib import Path

def pop_tts_chunk(buffer: str):
    if not buffer.strip():
        return None, buffer
    
    sentence_match = re.search(r"(.+?[.!?\n])(\s|$)", buffer, re.DOTALL)
    if sentence_match:
        end = sentence_match.end(1)
        chunk = buffer[:end].strip()
        remaining = buffer[end:].lstrip()
        return chunk, remaining
    
    if len(buffer) > 140:
        split_points = [buffer.rfind(x) for x in [", ", "; ", ": "]]
        split_at = max(split_points)
        if split_at >= 60:
            chunk = buffer[:split_at].strip()
            remaining = buffer[split_at + 1:].lstrip()
            return chunk, remaining
        
    return None, buffer

class PiperTTS:
    def __init__(self, piper_exe: str, model_path: str, enabled: bool = True):
        self.piper_exe = str(piper_exe)
        self.model_path = str(model_path)
        self.enabled = enabled
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.text_thread = threading.Thread(target = self.text_worker, daemon = True)
        self.audio_thread = threading.Thread(target = self.audio_worker, daemon = True)
        self.text_thread.start()
        self.audio_thread.start()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def speak(self, text: str):
        if not self.enabled:
            return
        
        cleaned_text = " ".join(text.strip().split())
        if cleaned_text:
            self.text_queue.put(cleaned_text)

    def pause(self):
        self.text_queue.join()
        self.audio_queue.join()

    def close(self):
        self.pause()
        self.text_queue.put(None)
        self.text_thread.join()
        self.audio_queue.put(None)
        self.audio_thread.join()

    def text_worker(self):
        while True:
            text = self.text_queue.get()

            if text is None:
                self.text_queue.task_done()
                break

            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as tmp:
                    wav_path = tmp.name
                subprocess.run([
                    self.piper_exe, "-m", self.model_path, "-f", wav_path],
                    input = text.encode("utf-8"),
                    check = True,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL
                )
                self.audio_queue.put(wav_path)
                wav_path = None
            except Exception as e:
                print(f"\n[TTS ERROR] {e}\n")
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception as e:
                        pass
            finally:
                self.text_queue.task_done()

    def audio_worker(self):
        while True:
            wav_path = self.audio_queue.get()

            if wav_path is None:
                self.audio_queue.task_done()
                break

            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            except Exception as e:
                print(f"\n[TTS PLAYBACK ERROR] {e}\n")
            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception as e:
                        pass
                self.audio_queue.task_done()
    

