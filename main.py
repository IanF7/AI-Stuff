import os
from openai import OpenAI
from pathlib import Path

from memory.short_term_memory import load_stm, save_stm, stm_build_instructions, stm_updater_model
from memory.long_term_memory import ltm_updater_model, save_ltm
from speech.TTS import PiperTTS, pop_tts_chunk

BASE_DIR = Path(__file__).resolve().parent
CORE_PERSONALITY = (BASE_DIR / "personality/core_personality.txt").read_text(encoding="utf-8").strip()
SECONDARY_PERSONALITY = (BASE_DIR / "personality/secondary_personality.txt").read_text(encoding="utf-8").strip()

PIPER_EXE = BASE_DIR / "piper" / "piper.exe"
PIPER_VOICE = BASE_DIR / "speech" / "voices" / "en_US-amy-medium.onnx"

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
previous_response_id = None
tts = PiperTTS(piper_exe = str(PIPER_EXE), model_path = str(PIPER_VOICE), enabled = True)

while True:
    user_text = input("User> ").strip()
    if user_text in {"/quit"}:
        stm = load_stm()
        ltm_result = ltm_updater_model(stm, SECONDARY_PERSONALITY, client)
        save_ltm(ltm_result)
        tts.close()
        break
    if user_text == "/tts on":
        os.environ["tts_enabled"] = "true"
        tts.set_enabled(True)
        print("TTS enabled.\n")
        continue
    if user_text == "/tts off":
        os.environ["tts_enabled"] = "false"
        tts.set_enabled(False)
        print("TTS disabled.\n")
        continue

    stm = load_stm()
    instructions = stm_build_instructions(stm)
    speech_buffer = ""
    first_chunk = True

    with client.responses.stream(
        model = "gpt-5.2",
        instructions = instructions,
        input = user_text,
        previous_response_id = previous_response_id,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                print(delta, end="", flush=True)
                speech_buffer += delta
                while True:
                    chunk, speech_buffer = pop_tts_chunk(speech_buffer, first_chunk = first_chunk)
                    if not chunk:
                        break
                    tts.speak(chunk)
                    first_chunk = False

        final = stream.get_final_response()

    print("\n")
    previous_response_id = final.id

    assistant_text = final.output_text.strip()

    if speech_buffer.strip():
        tts.speak(speech_buffer)

    stm = stm_updater_model(stm, user_text, assistant_text, client)
    save_stm(stm)

