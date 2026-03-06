import json
import os
from datetime import datetime, timezone, timedelta
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CORE_PERSONALITY = (BASE_DIR / "personality/core_personality.txt").read_text(encoding="utf-8").strip()
SECONDARY_PERSONALITY = (BASE_DIR / "personality/secondary_personality.txt").read_text(encoding="utf-8").strip()

LTM_PATH = BASE_DIR / "memory/long_term_memory.json"
LTM_PATH.parent.mkdir(parents = True, exist_ok = True)

LTM_UPDATER_INSTRUCTIONS = """
    You are a memory consolidation system for an AI assistant.
    Your job is to review the assistant's short-term memory snapshot and decide:
        1. Which items should be saved into long-term memory.
        2. Whether the assistant's secondary personality should be updated.
    Return ONLY valid JSON. No markdown. No commentary.
    Output schema:
    {
    "ltm_entries": [
    {
        "timestamp": "<ISO-8601 UTC timestamp>",
        "type": "<preference|identity|goal|constraint|relationship|project|other>",
        "summary": "<concise long-term memory summary, max 200 chars>",
        "importance": "<low|medium|high>"
    }
    ],
    "update_sp": <true|false>,
    "new_sp": "<very short additive patch, empty if no update>"
    }

    Rules:
    - Save only durable, useful information to long-term memory.
    - Do not save trivial turn-by-turn chatter.
    - Never save secrets, API keys, passwords, tokens, or highly sensitive information.
    - Update secondary personality only in rare cases where a repeated or important experience should change long-term conversational tendencies.
    - Any secondary personality update must be additive, compact, and must not conflict with the core personality.
    - Most of the time, update_sp should be false.
"""

def get_time() -> datetime:
    return datetime.now(timezone.utc)

def load_ltm() -> list[dict]:
    if not LTM_PATH.exists():
        return []
    text = LTM_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return data
    except json.JSONDecodeError:
        return []

def ltm_updater_model(stm: dict, current_sp: str, client) -> dict:
    updater_input = {
        "stm": stm,
        "current_sp": current_sp,
        "now_utc": get_time().isoformat(),
    }

    response = client.responses.create(
        model="gpt-5.2",
        instructions=LTM_UPDATER_INSTRUCTIONS,
        input = json.dumps(updater_input, ensure_ascii=False)
    )

    text = response.output_text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return{
        "ltm_entries": [],
        "update_sp": False,
        "new_sp": ""
    }

def save_ltm(result: dict) -> None:
    ltm = load_ltm()

    new_entries = result.get("ltm_entries", [])
    if isinstance(new_entries, list):
        ltm.extend(new_entries)
        LTM_PATH.write_text(json.dumps(ltm, ensure_ascii=False, indent=2), encoding="utf-8")
    
    update_sp = result.get("update_sp", False)
    update = result.get("new_sp", "").strip()
    if update_sp and update:
        current_sp = SECONDARY_PERSONALITY
        updated_sp = current_sp.rstrip() + "\n" + update + "\n"
        (BASE_DIR / "personality/secondary_personality.txt").write_text(updated_sp, encoding="utf-8")
