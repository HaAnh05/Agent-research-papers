from __future__ import annotations

import os
from typing import Any, Optional, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from config import config


def extract_text_from_response(content: Any) -> str:
    """Extract plain text from LLM response content (handles str or list of content parts)."""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif hasattr(item, "text"):
                text_parts.append(getattr(item, "text"))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()
    return str(content).strip()


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = config.TEMPERATURE,
) -> BaseChatModel:
    """Instantiate and return a ChatModel instance according to provider and model settings."""
    chosen_provider = (provider or config.DEFAULT_PROVIDER).lower()
    chosen_model = model or config.DEFAULT_MODEL

    if chosen_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = config.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env")
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            google_api_key=api_key,
            temperature=temperature,
        )

    elif chosen_provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env")
        return ChatOpenAI(
            model=chosen_model,
            api_key=api_key,
            temperature=temperature,
        )

    elif chosen_provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = config.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or .env")
        return ChatOpenAI(
            model=chosen_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )

    elif chosen_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment or .env")
        return ChatAnthropic(
            model=chosen_model,
            api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {chosen_provider}")
