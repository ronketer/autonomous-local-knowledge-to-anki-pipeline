#!/usr/bin/env python3
'''Autonomous Local Knowledge to Anki Pipeline.

A multi-agent AI system that extracts knowledge from Siyuan Notes
and creates optimized Anki flashcards using AutoGen.

This project demonstrates:
- Multi-agent orchestration with specialized roles
- Tool use for external API integration
- Reflection pattern for quality assurance
- Human-in-the-loop approval workflow
- Local-first, privacy-preserving AI (no cloud LLM dependencies)

Usage:
    1. Ensure Ollama is running (ollama serve) with a model pulled
    2. Ensure Siyuan Notes is running with API enabled
    3. Ensure Anki is running with AnkiConnect plugin
    4. Set TARGET_BLOCK_ID in your .env file
    5. Run: python main.py
'''

import asyncio
import sys

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from src.anki_pipeline.agents import create_agents, create_model_client, selector_func
from src.anki_pipeline.config import config


async def main() -> int:
    '''Run the flashcard generation pipeline.'''
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            print(f'Configuration Error: {error}')
        print('\nPlease check your .env file. See .env.example for reference.')
        return 1

    print(f'Starting pipeline for Siyuan block: {config.TARGET_BLOCK_ID}\n')

    # Initialize components
    model_client = create_model_client()
    agents = create_agents(model_client)

    # Assemble the multi-agent team with Reflection pattern
    # selector_func enables: Writer ↔ Reviewer revision loop until APPROVED
    team = SelectorGroupChat(
        participants=[
            agents['knowledge_manager'],
            agents['card_writer'],
            agents['card_reviewer'],
            agents['admin'],
        ],
        model_client=model_client,
        selector_func=selector_func,  # Custom routing for reflection loop
        termination_condition=TextMentionTermination('TERMINATE'),
    )

    # Run the pipeline
    task = (
        f"Fetch the notes for Siyuan block ID '{config.TARGET_BLOCK_ID}'. "
        'Draft the Anki cards, and once approved, save them.'
    )

    await Console(team.run_stream(task=task))
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
