"""Tool definitions for AI agents.

Tools are functions that agents can call to interact with external systems.
Each tool is annotated with type hints that AutoGen uses to generate
function schemas for the LLM.
"""

import json
from typing import Annotated

import requests

from .config import config


def fetch_siyuan_notes(
    block_id: Annotated[str, "The unique 22-character Siyuan block ID to fetch."],
) -> str:
    """Fetch markdown content from a Siyuan Notes document or block.

    This tool retrieves knowledge from the local Siyuan Notes instance,
    keeping all data local (no cloud APIs, privacy-preserving).
    """
    headers = (
        {"Authorization": f"Token {config.SIYUAN_API_TOKEN}"}
        if config.SIYUAN_API_TOKEN
        else {}
    )
    payload = {"id": block_id}

    try:
        response = requests.post(
            config.SIYUAN_API_URL, headers=headers, json=payload, timeout=10
        )
        response_data = response.json()

        if response_data.get("code") == 0:
            data = response_data.get("data", {})
            return json.dumps(data, indent=2, ensure_ascii=False)
        return f"Siyuan Error: {response_data.get('msg')}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Siyuan. Is it running?"
    except Exception as e:
        return f"Error fetching notes: {str(e)}"


def push_to_anki(
    front_text: Annotated[str, "The text for the front of the flashcard."],
    back_text: Annotated[str, "The text for the back of the flashcard."],
) -> str:
    """Push a flashcard to Anki via the AnkiConnect API.

    Cards are added to the configured deck with duplicate prevention enabled.
    """
    payload = {
        "action": "addNote",
        "version": 6,
        "params": {
            "note": {
                "deckName": config.ANKI_DECK_NAME,
                "modelName": "Basic",
                "fields": {"Front": front_text, "Back": back_text},
                "options": {"allowDuplicate": False},
            }
        },
    }

    try:
        response = requests.post(config.ANKI_CONNECT_URL, json=payload, timeout=10)
        response_data = response.json()

        if response.status_code == 200 and response_data.get("error") is None:
            return f"Card added successfully with ID: {response_data['result']}"
        return f"Failed to add card: {response_data.get('error', 'Unknown error')}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Anki. Is AnkiConnect running?"
    except Exception as e:
        return f"Error pushing to Anki: {str(e)}"
