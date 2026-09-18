from __future__ import annotations

import os
import json
import re
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

from config import config, resolved_provider


ModelT = TypeVar("ModelT", bound=BaseModel)


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
    *,
    json_mode: bool = False,
) -> BaseChatModel:
    """Instantiate and return a ChatModel instance according to provider and model settings."""
    chosen_provider = resolved_provider(provider)
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
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "api_key": api_key,
            "temperature": temperature,
        }
        if config.OPENAI_API_BASE:
            kwargs["base_url"] = config.OPENAI_API_BASE
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif chosen_provider == "zai":
        from langchain_openai import ChatOpenAI

        # During the migration from the old OpenAI-compatible alias, accept
        # OPENAI_API_KEY only when OPENAI_API_BASE explicitly points to Z.AI.
        legacy_zai_key = (
            config.OPENAI_API_KEY
            if "api.z.ai" in (config.OPENAI_API_BASE or "").lower()
            else ""
        )
        api_key = config.ZAI_API_KEY or legacy_zai_key
        if not api_key:
            raise ValueError("ZAI_API_KEY is not set in environment or .env")

        kwargs = {
            "model": chosen_model,
            "api_key": api_key,
            "base_url": config.ZAI_API_BASE or "https://api.z.ai/api/paas/v4/",
            "temperature": temperature,
        }
        # Z.AI's OpenAI-compatible API supports JSON mode.  This is opt-in so
        # free-form report/refinement calls remain ordinary text completions.
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif chosen_provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = config.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or .env")
        kwargs = {
            "model": chosen_model,
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif chosen_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment or .env")
        return ChatAnthropic(
            model=chosen_model,
            temperature=temperature,
            api_key=api_key,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {chosen_provider}")


def _schema_json(schema: Type[ModelT]) -> str:
    """Return a JSON schema string for both Pydantic v1 and v2 models."""

    if hasattr(schema, "model_json_schema"):
        payload = schema.model_json_schema()
    else:  # pragma: no cover - retained for older supported environments
        payload = schema.schema()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_prompt(prompt: str, schema: Type[ModelT]) -> str:
    """Tell JSON-mode providers exactly which object shape to return."""

    return (
        f"{prompt}\n\n"
        "Return only one valid JSON object. Do not wrap it in Markdown fences "
        "and do not add commentary. The object must satisfy this JSON Schema:\n"
        f"{_schema_json(schema)}"
    )


def _decode_json_object(value: Any) -> Any:
    """Decode a provider response without exposing its raw contents in errors."""

    if isinstance(value, dict):
        return value
    # LangChain returns AIMessage for ChatOpenAI.  Prefer its content payload
    # over ``str(message)``, which includes metadata and is not JSON.
    if hasattr(value, "content"):
        value = getattr(value, "content")
    text = extract_text_from_response(value)
    # Be tolerant of models that still return a fenced JSON block despite the
    # explicit instruction above.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some compatible gateways prepend a short sentence.  Decode the first
        # complete object while still refusing arbitrary Python expressions.
        start = text.find("{")
        if start >= 0:
            try:
                decoded, _ = json.JSONDecoder().raw_decode(text[start:])
                return decoded
            except json.JSONDecodeError:
                pass
        raise ValueError("LLM returned invalid JSON for the requested schema") from None


def _validate_model(value: Any, schema: Type[ModelT]) -> ModelT:
    """Validate structured output locally, including Z.AI JSON responses."""

    if isinstance(value, schema):
        return value
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    return schema.parse_obj(value)  # pragma: no cover - Pydantic v1 fallback


def invoke_structured_output(
    prompt: str,
    schema: Type[ModelT],
    *,
    llm: BaseChatModel | None = None,
    provider: Optional[str] = None,
) -> ModelT:
    """Invoke a provider-aware structured call and validate it locally.

    Native ``with_structured_output`` remains the default for existing
    providers.  Z.AI currently documents OpenAI-compatible JSON mode rather
    than the tool/schema strategy used by some LangChain integrations, so its
    request is sent with ``response_format={"type": "json_object"}`` and the
    returned object is validated with the supplied Pydantic model here.
    """

    chosen_provider = resolved_provider(provider)
    model = llm or get_llm(provider=chosen_provider, json_mode=chosen_provider == "zai")

    if chosen_provider == "zai":
        json_model: Any = model
        # When a caller injects a model, bind JSON mode at invocation time.  A
        # model made by get_llm(json_mode=True) already carries the same
        # request option; binding is harmless for LangChain and guarded for
        # lightweight fakes used by tests.
        if hasattr(model, "bind"):
            try:
                json_model = model.bind(response_format={"type": "json_object"})
            except Exception:
                json_model = model
        try:
            response = json_model.invoke(_json_prompt(prompt, schema))
        except TypeError:
            # Small test doubles and a few older wrappers do not accept bound
            # kwargs; they can still return JSON for local validation.
            response = model.invoke(_json_prompt(prompt, schema))
        return _validate_model(_decode_json_object(response), schema)

    structured_model = model.with_structured_output(schema)
    response = structured_model.invoke(prompt)
    if isinstance(response, schema):
        return response
    # Native integrations may return a dict, an AIMessage, or a JSON string.
    decoded = _decode_json_object(response) if not isinstance(response, dict) else response
    return _validate_model(decoded, schema)


# Short alias for callers that prefer the verb used in LangChain examples.
invoke_structured = invoke_structured_output
