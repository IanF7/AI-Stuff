import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

sample_rate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frames_per_block = int(sample_rate * block_duration)
frames_per_chunk = int(sample_rate * chunk_duration)

audio_queue = queue.Queue()
text_queue = queue.Queue()
audio_buffer = []

model = WhisperModel("small.en", device = "cpu", compute_type = "float32")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate = sample_rate, channels = channels, dtype = "float32",callback = audio_callback, blocksize = frames_per_block):
        print("Listening...")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []
            audio_data = audio_data.flatten().astype(np.float32)
            segments, _ = model.transcribe(audio_data, language = "en", beam_size = 1, vad_filter= True, no_speech_threshold = 0.6, condition_on_previous_text = False)
            text = " ".join(segment.text for segment in segments)
            if text:
                text_queue.put(text)

def start_stt():
    threading.Thread(target = recorder, daemon = True).start()
    threading.Thread(target = transcriber, daemon = True).start()

def get_text():
    return text_queue.get()
