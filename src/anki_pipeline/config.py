"""Configuration management for the Anki Pipeline.

All sensitive data is loaded from environment variables to ensure
no credentials are exposed in the codebase (security-first approach).
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # LLM Server Configuration (Ollama with Intel GPU via Vulkan)
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "qwen3.5:4b")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "placeholder")

    # Siyuan Notes Configuration
    SIYUAN_API_URL: str = os.getenv(
        "SIYUAN_API_URL", "http://127.0.0.1:6806/api/block/getBlockKramdown"
    )
    SIYUAN_API_TOKEN: str = os.getenv("SIYUAN_API_TOKEN", "")
    TARGET_BLOCK_ID: str = os.getenv("TARGET_BLOCK_ID", "")

    # Anki Configuration
    ANKI_CONNECT_URL: str = os.getenv("ANKI_CONNECT_URL", "http://localhost:8765")
    ANKI_DECK_NAME: str = os.getenv("ANKI_DECK_NAME", "Default")

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration. Returns list of missing variables."""
        errors = []
        if not cls.TARGET_BLOCK_ID:
            errors.append("TARGET_BLOCK_ID is required")
        return errors


config = Config()
