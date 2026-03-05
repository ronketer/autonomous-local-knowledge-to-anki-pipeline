"""Agent definitions and team orchestration.

This module implements a multi-agent system using AutoGen's SelectorGroupChat
pattern with a custom routing function for deterministic agent handoffs.

Architecture:
    User -> Knowledge_Manager -> Card_Writer -> Card_Reviewer -> Admin -> [loop or save]

Agentic Design Patterns Used:
    1. Tool Use: Agents call external APIs (Siyuan, Anki)
    2. Reflection: Card_Reviewer critiques the Card_Writer's output
    3. Multi-Agent Collaboration: Specialized agents with distinct roles
    4. Human-in-the-Loop: Admin provides final approval before saving
"""

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .config import config
from .tools import fetch_siyuan_notes, push_to_anki

# System prompts following best practices for instruction clarity
KNOWLEDGE_MANAGER_PROMPT = """You are the orchestrator of a flashcard creation pipeline.

Your responsibilities:
1. When the user requests notes, call fetch_siyuan_notes with the provided block ID.
2. Pass the fetched content to the Card_Writer for processing.
3. ONLY call push_to_anki after the Admin explicitly says 'APPROVE'.
4. After successfully pushing all cards, output TERMINATE to end the session.

Important: Never push cards without human approval."""

CARD_WRITER_PROMPT = """You are an expert at creating effective Anki flashcards.

Follow the Minimum Information Principle (SuperMemo's 20 Rules):
1. ATOMIC: Each card's back must contain exactly ONE fact.
2. CONCISE: Use minimal words while preserving meaning.
3. NO SETS: Never ask for lists of items - split into separate cards.
4. CLOZE DELETIONS: Use [...] for fill-in-the-blank when appropriate.

BAD EXAMPLE:
  Front: What do we know about dog domestication?
  Back: Dogs were domesticated 15,000-30,000 years ago, making them the first domesticated species.

GOOD EXAMPLES:
  Front: Dogs were domesticated between [...] and 30,000 years ago.
  Back: 15,000

  Front: What was the first species domesticated by humans?
  Back: Dogs

Output your flashcards as JSON in this exact format:
{"cards": [{"front": "question", "back": "answer"}, ...]}

Respond with ONLY the JSON, no other text."""

CARD_REVIEWER_PROMPT = """You are a quality reviewer for Anki flashcards.

A GOOD card follows the Minimum Information Principle:
- The BACK contains exactly ONE atomic fact (a single word, number, or short phrase)
- Example GOOD back: "Dogs" or "15,000 years ago" or "mitochondria"
- Example BAD back: "Dogs were domesticated 15,000 years ago and were the first species"

Check each card:
1. Is the back a SINGLE atomic answer? If yes = PASS
2. Does the front ask for a list? If yes = FAIL (split into separate cards)
3. Are there formatting artifacts like {#id} or markdown? If yes = FAIL

If ALL cards PASS, respond with exactly: APPROVED
If ANY card FAILS, explain which card failed and why, then say: REJECTED"""


def create_model_client() -> OpenAIChatCompletionClient:
    """Create the LLM client configured for the local inference server."""
    return OpenAIChatCompletionClient(
        model=config.LLM_MODEL_ID,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model_info={
            "json_output": False,  # Small models may struggle with strict JSON
            "vision": False,
            "function_calling": True,
            "structured_output": False,  # Ollama doesn't support native structured output
            "family": "unknown",
        },
        # Disable qwen3.5 "thinking mode" for faster responses
        extra_create_args={"options": {"num_ctx": 4096}},
    )


def create_agents(model_client: OpenAIChatCompletionClient) -> dict[str, ChatAgent]:
    """Create all agents for the pipeline."""
    return {
        "knowledge_manager": AssistantAgent(
            name="Knowledge_Manager",
            model_client=model_client,
            tools=[fetch_siyuan_notes, push_to_anki],
            system_message=KNOWLEDGE_MANAGER_PROMPT,
        ),
        "card_writer": AssistantAgent(
            name="Card_Writer",
            model_client=model_client,
            description="Drafts Anki flashcards from raw notes.",
            system_message=CARD_WRITER_PROMPT,
        ),
        "card_reviewer": AssistantAgent(
            name="Card_Reviewer",
            model_client=model_client,
            description="Critiques flashcards for quality.",
            system_message=CARD_REVIEWER_PROMPT,
        ),
        "admin": UserProxyAgent(
            name="Admin",
            description="Human reviewer who approves flashcards before saving.",
            input_func=lambda prompt: input(
                "\n" + "="*50 + "\n"
                "🎴 HUMAN REVIEW REQUIRED\n"
                "="*50 + "\n"
                "Options:\n"
                "  • APPROVE - Save cards to Anki\n"
                "  • REJECT  - Send back for revision\n"
                "  • (or type feedback for the Card_Writer)\n"
                "="*50 + "\n"
                "Your decision: "
            ),
        ),
    }


def selector_func(messages: list) -> str | None:
    """Custom routing function for deterministic agent handoffs.

    This implements a state machine for the conversation flow:
    User -> Knowledge_Manager -> Card_Writer -> Card_Reviewer -> Admin -> [loop]

    Returns:
        The name of the next agent to speak, or None for default selection.
    """
    if not messages or messages[-1].source == "user":
        return "Knowledge_Manager"

    last = messages[-1]

    # Manager fetches notes -> Writer processes them
    if last.source == "Knowledge_Manager":
        return "Card_Writer"

    # Writer creates cards -> Reviewer checks quality
    if last.source == "Card_Writer":
        return "Card_Reviewer"

    # Reviewer approves -> Human review; Reviewer rejects -> Writer revises
    if last.source == "Card_Reviewer":
        if "APPROVED" in last.content:
            return "Admin"
        return "Card_Writer"

    # Human approves -> Manager saves; Human rejects -> Writer revises
    if last.source == "Admin":
        if "APPROVE" in last.content.upper():
            return "Knowledge_Manager"
        return "Card_Writer"

    return None
