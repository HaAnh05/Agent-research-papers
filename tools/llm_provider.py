from __future__ import annotations

import os
import json
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional, Type, TypeVar
from pydantic import BaseModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tracers.context import register_configure_hook

from config import config, resolved_provider


ModelT = TypeVar("ModelT", bound=BaseModel)


# --- Per-call metrics (duration and token usage) ------------------------------

def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _message_usage(message: Any) -> dict[str, int]:
    """Token counts the provider reported for one AI message; absent values are omitted.

    ``usage_metadata`` is the LangChain standard (ChatOpenAI fills it from
    ``prompt_tokens`` / ``completion_tokens`` and
    ``completion_tokens_details.reasoning_tokens``; Gemini and Anthropic fill it
    too).  ``response_metadata["token_usage"]`` is the raw OpenAI-compatible
    payload and serves as a fallback.
    """

    usage: dict[str, int] = {}
    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict):
        for key, field in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
            number = _count(meta.get(key))
            if number is not None:
                usage[field] = number
        details = meta.get("output_token_details")
        reasoning = _count(details.get("reasoning")) if isinstance(details, dict) else None
        if reasoning is not None:
            usage["reasoningTokens"] = reasoning
    raw = (getattr(message, "response_metadata", None) or {}).get("token_usage")
    if isinstance(raw, dict):
        for key, field in (("prompt_tokens", "inputTokens"), ("completion_tokens", "outputTokens")):
            number = _count(raw.get(key))
            if number is not None:
                usage.setdefault(field, number)
        details = raw.get("completion_tokens_details")
        reasoning = _count(details.get("reasoning_tokens")) if isinstance(details, dict) else None
        if reasoning is not None:
            usage.setdefault("reasoningTokens", reasoning)
    return usage


class _UsageCollector(BaseCallbackHandler):
    """Sums the token usage of every chat-model call made while it is active."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.totals: dict[str, int] = {}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
            message = generation.message
        except (AttributeError, IndexError):
            return
        if not isinstance(message, AIMessage):
            return
        usage = _message_usage(message)
        with self._lock:
            for key, number in usage.items():
                self.totals[key] = self.totals.get(key, 0) + number


_USAGE_COLLECTOR: ContextVar[_UsageCollector | None] = ContextVar("llm_usage_collector", default=None)
# Registered once: LangChain then attaches the active collector to every model call in this context.
register_configure_hook(_USAGE_COLLECTOR, inheritable=True)


@contextmanager
def track_llm_call(kind: str, sink: list[dict[str, Any]] | None = None, **extra: Any) -> Iterator[dict[str, Any]]:
    """Measure one LLM call: ``{kind, ms, inputTokens?, outputTokens?, reasoningTokens?}``.

    Wrap exactly one call (``invoke_structured_output``, ``invoke_text`` or a
    ``stream`` loop).  ``ms`` is wall time and always set; the token fields appear
    only when the provider returned them (a fake model or a stream without usage
    data gives ``ms`` alone).  The record is appended to ``sink`` when the block
    ends, also when the call raised.  ``extra`` adds identifying fields (``paperId``).
    """

    record: dict[str, Any] = {"kind": kind, **{key: value for key, value in extra.items() if value}}
    collector = _UsageCollector()
    token = _USAGE_COLLECTOR.set(collector)
    started = time.perf_counter()
    try:
        yield record
    finally:
        _USAGE_COLLECTOR.reset(token)
        record["ms"] = max(0, int((time.perf_counter() - started) * 1000))
        record.update(collector.totals)
        if sink is not None:
            sink.append(record)


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


def _json_suffix(schema: Type[ModelT]) -> str:
    """Static JSON-mode instruction + schema text for one schema."""

    return (
        "Return only one valid JSON object. Do not wrap it in Markdown fences "
        "and do not add commentary. The object must satisfy this JSON Schema:\n"
        f"{_schema_json(schema)}"
    )


def _json_prompt(prompt: str, schema: Type[ModelT]) -> str:
    """Tell JSON-mode providers exactly which object shape to return."""

    return f"{prompt}\n\n{_json_suffix(schema)}"


def _role_messages(system: str, user: str) -> list[Any]:
    """Build the ``[system, user]`` message pair.

    Verified against the installed integrations: ``langchain_google_genai``
    maps a leading ``SystemMessage`` to ``system_instruction``,
    ``langchain_anthropic`` maps it to the ``system`` parameter, and
    ``langchain_openai`` (Z.AI, OpenRouter, OpenAI) sends role ``system``.
    """

    return [SystemMessage(content=system), HumanMessage(content=user)]


def invoke_text(llm: Any, system: str, user: str) -> Any:
    """Free-form text call with a system and a user message; returns the response."""

    return llm.invoke(_role_messages(system, user))


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
    system: str | None = None,
) -> ModelT:
    """Invoke a provider-aware structured call and validate it locally.

    With ``system=None`` the request is exactly the historical single user
    string.  With ``system`` the request is ``[system, user]`` where ``prompt``
    is the user message; for Z.AI the JSON Schema text is appended to the
    system message (it is static per call type).

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
        payload: Any = (
            _json_prompt(prompt, schema)
            if system is None
            else _role_messages(f"{system}\n\n{_json_suffix(schema)}", prompt)
        )
        try:
            response = json_model.invoke(payload)
        except TypeError:
            # Small test doubles and a few older wrappers do not accept bound
            # kwargs; they can still return JSON for local validation.
            response = model.invoke(payload)
        return _validate_model(_decode_json_object(response), schema)

    structured_model = model.with_structured_output(schema)
    response = structured_model.invoke(prompt if system is None else _role_messages(system, prompt))
    if isinstance(response, schema):
        return response
    # Native integrations may return a dict, an AIMessage, or a JSON string.
    decoded = _decode_json_object(response) if not isinstance(response, dict) else response
    return _validate_model(decoded, schema)


# Short alias for callers that prefer the verb used in LangChain examples.
invoke_structured = invoke_structured_output
