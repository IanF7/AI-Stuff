import os
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from pathlib import Path

from STM import load_stm, save_stm, stm_build_instructions, stm_updater_model
from LTM import ltm_updater_model, save_ltm, load_ltm
from speech.TTS import speak_text

BASE_DIR = Path(__file__).resolve().parent
CORE_PERSONALITY = (BASE_DIR / "personality/core_personality.txt").read_text(encoding="utf-8").strip()
SECONDARY_PERSONALITY = (BASE_DIR / "personality/secondary_personality.txt").read_text(encoding="utf-8").strip()

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
previous_response_id = None

TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

while True:
    user_text = input("User> ").strip()
    if user_text in {"/quit"}:
        stm = load_stm()
        ltm_result = ltm_updater_model(stm, SECONDARY_PERSONALITY, client)
        save_ltm(ltm_result)
        break
    if user_text == "/tts on":
        os.environ["TTS_ENABLED"] = "true"
        print("TTS enabled.\n")
        continue
    if user_text == "/tts off":
        os.environ["TTS_ENABLED"] = "false"
        print("TTS disabled.\n")
        continue

    stm = load_stm()
    instructions = stm_build_instructions(stm)

    with client.responses.stream(
        model = "gpt-5.2",
        instructions = instructions,
        input = user_text,
        previous_response_id = previous_response_id,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        final = stream.get_final_response()

    print("\n")
    previous_response_id = final.id

    assistant_text = final.output_text.strip()

    if TTS_ENABLED and assistant_text:
        try:
            speak_text(assistant_text)
        except Exception as e:
            print(f"[TTS Error] {e}")

    stm = stm_updater_model(stm, user_text, assistant_text, client)
    save_stm(stm)

