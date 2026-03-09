import os
from openai import OpenAI
from pathlib import Path

from memory.short_term_memory import load_stm, save_stm, stm_build_instructions, stm_updater_model
from memory.long_term_memory import ltm_updater_model, save_ltm, load_ltm
from speech.TTS import queue_speak, extract_speakable_chunks

BASE_DIR = Path(__file__).resolve().parent
CORE_PERSONALITY = (BASE_DIR / "personality/core_personality.txt").read_text(encoding="utf-8").strip()
SECONDARY_PERSONALITY = (BASE_DIR / "personality/secondary_personality.txt").read_text(encoding="utf-8").strip()

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
previous_response_id = None

tts_enabled = os.getenv("tts_enabled", "true").lower() in {"1", "true", "yes", "on"}

while True:
    user_text = input("User> ").strip()
    if user_text in {"/quit"}:
        stm = load_stm()
        ltm_result = ltm_updater_model(stm, SECONDARY_PERSONALITY, client)
        save_ltm(ltm_result)
        break
    if user_text == "/tts on":
        os.environ["tts_enabled"] = "true"
        print("TTS enabled.\n")
        continue
    if user_text == "/tts off":
        os.environ["tts_enabled"] = "false"
        print("TTS disabled.\n")
        continue

    stm = load_stm()
    instructions = stm_build_instructions(stm)
    tts_enabled = os.getenv("tts_enabled", "true").lower() in {"1", "true", "yes", "on"}
    text_parts = []
    speech_buffer = ""

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
                text_parts.append(delta)
                if tts_enabled:
                    speech_buffer += delta
                    chunks, speech_buffer = extract_speakable_chunks(speech_buffer)

                    for chunk in chunks:
                        queue_speak(chunk)

        final = stream.get_final_response()

    print("\n")
    previous_response_id = final.id

    assistant_text = final.output_text.strip()

    if tts_enabled and speech_buffer.strip():
        queue_speak(speech_buffer.strip())

    stm = stm_updater_model(stm, user_text, assistant_text, client)
    save_stm(stm)

