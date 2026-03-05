import json
import os
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent
CORE_PERSONALITY = (BASE_DIR / "personality/core_personality.txt").read_text(encoding="utf-8").strip()
SECONDARY_PERSONALITY = (BASE_DIR / "personality/secondary_personality.txt").read_text(encoding="utf-8").strip()

STM_PATH = BASE_DIR / "memory/short_term_memory.json"
STM_STORAGE_WINDOW = timedelta(hours = 24)
DEFAULT_STM = {
    "updated_at": None,
    "session_summary": "",
    "open_loops": [],
    "user_facts": {},
    "last_turn": {"user": "", "assistant": ""}
  }

STM_UPDATER_INSTRUCTIONS = """
    You are a memory updater for an AI assistant. Your job is to update the assistant's short-term 
    working memory (STM). Return ONLY valid JSON. No markdown. No commentary.

    STM must follow this schema:
    {
        "updated_at": "<ISO-8601 UTC timestamp>",
        "session_summary": "<1-3 sentences, max 600 chars>",
        "open_loops": ["<max 5 items>"],
        "user_facts": { "<key>": "<value>", "...": "..." },  // max ~10 keys, short strings
        "last_turn": { "user": "<max 300 chars>", "assistant": "<max 300 chars>" }
    }

    Rules:
    - Keep it compact. Do not store full transcripts.
    - Only keep facts from the last 24 hours; if something is older or irrelevant, drop it.
    - Prefer stable, useful facts: name, preferences, current goals, constraints.
    - If there is nothing to add, keep fields as-is but update updated_at and last_turn.
"""

def get_time() -> datetime:
    return datetime.now(timezone.utc)

def load_stm() -> list[dict]:
    if not STM_PATH.exists():
        return DEFAULT_STM.copy()
    text = STM_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return DEFAULT_STM.copy()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return DEFAULT_STM.copy()
        return data
    except json.JSONDecodeError:
        return DEFAULT_STM.copy()
    
def save_stm(stm: list[dict]) -> None:
    STM_PATH.write_text(json.dumps(stm, ensure_ascii=False, indent=2), encoding="utf-8")

def stm_build_instructions(stm: list[dict]) -> str:
    stm_block = json.dumps(stm, ensure_ascii=False, indent=2)
    parts = [
        "CORE PERSONALITY:\n", CORE_PERSONALITY,
        "SECONDARY PERSONALITY:\n", SECONDARY_PERSONALITY,
        "SHORT-TERM MEMORY:\n", stm_block,
        "Rules for memory usage:\n "
        "- Use short-term memory when relevant to recall events from the past 24 hours.\n "
        "- If memory conflicts with user, ask for clarification."
    ]
    return "\n\n".join(parts).strip()

def stm_updater_model(stm: dict, user_input: str, assistant_response: str) -> dict:
    updater_input = {
        "stm_before": stm,
        "last_turn": {"user": user_input, "assistant": assistant_response},
        "now_utc": get_time().isoformat(),
    }

    updater_output = client.responses.create(
        model="gpt-5.2",
        instructions=STM_UPDATER_INSTRUCTIONS,
        input = json.dumps(updater_input, ensure_ascii=False)
    )

    text = updater_output.output_text.strip()
    try:
        new_stm = json.loads(text)
        if isinstance(new_stm, dict):
            return new_stm
    except json.JSONDecodeError:
        pass

    stm["updated_at"] = get_time().isoformat()
    stm["last_turn"] = {"user":user_text[:300], "assistant": assistant_response[:300]}
    return stm

previous_response_id = None

while True:
    user_text = input("User> ").strip()
    if user_text in {"/quit"}:
        break

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
    stm = stm_updater_model(stm, user_text, assistant_text)
    save_stm(stm)

