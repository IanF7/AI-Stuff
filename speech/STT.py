import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

class SpeechToText:
    def __init__(self, model_size = "base", sample_rate = 16000, silence_duration = 0.4, device = "cpu"):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.model = WhisperModel(model_size, device = device, compute_type = "int8")
        self.vad_model = load_silero_vad()
        self.running = True
        self.worker = threading.Thread(target = self.process_audio, daemon = True)
        self.worker.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            return
        self.audio_queue.put(indata.copy())

    def listen(self):
        self.stream = sd.InputStream(samplerate = self.sample_rate, channels = 1, callback = self.audio_callback, blocksize = 1600)
        self.stream.start()
        print("Listening for speech...")

    def process_audio(self):
        buffer = np.zeros(0, dtype = np.float32)
        while self.running:
            audio_chunk = self.audio_queue.get()
            audio_chunk = audio_chunk.flatten()
            buffer = np.concatenate((buffer, audio_chunk))
            if len(buffer) < self.sample_rate:
                continue

            timestamps = get_speech_timestamps(buffer, self.vad_model, sampling_rate = self.sample_rate)
            if not timestamps:
                buffer = np.zeros(0, dtype = np.float32)
                continue

            end = timestamps[-1]['end']
            if len(buffer) - end > int(self.silence_duration * self.sample_rate):
                speech_audio = buffer[:end]
                buffer = buffer[end:]
                text = self.transcribe(speech_audio)
                if text:
                    self.text_queue.put(text)

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, beam_size = 1, language = "en", condition_on_previous_text = False)
        text = "".join([segment.text for segment in segments]).strip()
        return text
    
    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()