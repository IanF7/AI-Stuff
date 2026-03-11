import os
import re
import queue
import tempfile
import threading
import subprocess
import winsound
from pathlib import Path

def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("```", " ")
    text = text.replace("`", "")
    text = re.sub(r"\.{3,}", "…", text)

    # CHANGED:
    # Normalize curly quotes/apostrophes from LLM output into plain ASCII quotes.
    # This makes the regex rules below work consistently on text like:
    # “Alright… I’ll keep it simple.”
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

    def expand_acronym(match):
        word = match.group(0)
        return ".".join(list(word)) + "."

    text = re.sub(r"\b[A-Z]{2,}\b", expand_acronym, text)
    text = re.sub(r'([.!?])(["\'])', r'\2\1', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pop_tts_chunk(buffer: str, first_chunk: bool = False):
    if not buffer.strip():
        return None, buffer

    working = buffer.lstrip()
    clause_threshold = 90 if first_chunk else 130
    forced_threshold = 180 if first_chunk else 220
    lead_ins = (
        "however,",
        "for example,",
        "for instance,",
        "in fact,",
        "well,",
        "so,",
        "also,",
        "then,",
        "still,",
        "actually,",
        "basically,",
        "anyway,",
    )

    def is_chunk_short(text: str) -> bool:
        t = text.strip().lower()
        if len(t) < 35:
            return True

        if t.startswith(lead_ins) and len(t) < 70:
            return True

        if t.endswith(",") and len(t) < 70:
            return True

        return False

    sentence_match = re.search(
        r'.+?(?:\.{3,}|…|[.!?])(?=(?:["\'])?(?:\s|$))',
        working,
        re.DOTALL
    )
    if sentence_match:
        first_sentence = working[:sentence_match.end()].strip()
        rest = working[sentence_match.end():].lstrip()
        if len(first_sentence) < 55 and rest:
            second_match = re.search(
                r'.+?[.!?](?=(?:["\'])?(?:\s|$))',
                rest,
                re.DOTALL
            )
            if second_match:
                end = sentence_match.end() + second_match.end()
                chunk = working[:end].strip()
                remaining = working[end:].lstrip()
                return chunk, remaining

        return first_sentence, rest

    if len(working) >= clause_threshold:
        split_positions = [working.rfind(mark) for mark in ["; ", ": ", ", "]]
        split_at = max(split_positions)
        if split_at >= 55:
            chunk = working[:split_at + 1].strip()
            remaining = working[split_at + 1:].lstrip()
            if not is_chunk_short(chunk):
                return chunk, remaining

    if len(working) >= forced_threshold:
        split_at = working.rfind(" ", 0, 170 if first_chunk else 200)
        if split_at >= 80:
            chunk = working[:split_at].strip()
            remaining = working[split_at:].lstrip()
            return chunk, remaining

    return None, buffer

class PiperTTS:
    def __init__(self, piper_exe: str, model_path: str, enabled: bool = True):
        self.piper_exe = str(piper_exe)
        self.model_path = str(model_path)
        self.enabled = enabled
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.output_dir_obj = tempfile.TemporaryDirectory()
        self.output_dir = self.output_dir_obj.name
        self.piper = self.start_piper()
        self.text_thread = threading.Thread(target = self.text_worker, daemon = True)
        self.audio_thread = threading.Thread(target = self.audio_worker, daemon = True)
        self.text_thread.start()
        self.audio_thread.start()

    def start_piper(self):
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW

        return subprocess.Popen(
            [self.piper_exe, "-m", self.model_path, "--output_dir", self.output_dir,],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            bufsize = 1,
            creationflags = creationflags,
        )

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def speak(self, text: str):
        if not self.enabled:
            return
        
        cleaned_text = normalize_text(text)
        if cleaned_text:
            self.text_queue.put(cleaned_text)

    def flush(self):
        self.text_queue.join()
        self.audio_queue.join()

    def close(self):
        self.flush()
        self.text_queue.put(None)
        self.text_thread.join()
        self.audio_queue.put(None)
        self.audio_thread.join()

        try:
            if self.piper.stdin:
                self.piper.stdin.close()
        except Exception:
            pass

        self.output_dir_obj.cleanup()

    def text_worker(self):
        while True:
            text = self.text_queue.get()

            if text is None:
                self.text_queue.task_done()
                break

            try:
                if not self.piper or self.piper.poll() is not None:
                    self.piper = self.start_piper()
                
                self.piper.stdin.write(text + "\n")
                self.piper.stdin.flush()
                wav_path = self.piper.stdout.readline().strip()
                if wav_path:
                    if not os.path.isabs(wav_path):
                        wav_path = os.path.join(self.output_dir, wav_path)

                    self.audio_queue.put(wav_path)
            except Exception as e:
                print(f"\n[TTS ERROR] {e}\n")
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
    

