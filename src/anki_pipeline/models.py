"""Pydantic models for structured output from AI agents.

Using structured output ensures consistent, validated data flow
between agents in the multi-agent system.
"""

from pydantic import BaseModel, Field


class Flashcard(BaseModel):
    """A single Anki flashcard following the Minimum Information Principle."""

    front: str = Field(description="The question or prompt side of the card")
    back: str = Field(description="The answer - should contain ONE atomic fact only")


class FlashcardList(BaseModel):
    """Collection of flashcards generated from source material."""

    cards: list[Flashcard] = Field(description="List of atomic flashcards")
