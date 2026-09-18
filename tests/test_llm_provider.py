from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import tools.llm_provider as provider
from config import config


class Output(BaseModel):
    answer: int


class FakeChat:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.bound = None
        self.response = SimpleNamespace(content='{"answer": 7}')
        self.__class__.instances.append(self)

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, prompt):
        return self.response


def test_zai_factory_uses_explicit_base_key_and_json_mode(monkeypatch):
    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChat
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(config, "ZAI_API_KEY", "zai-test-secret")
    monkeypatch.setattr(config, "ZAI_API_BASE", "https://api.z.ai/api/paas/v4/")
    monkeypatch.setattr(config, "DEFAULT_MODEL", "glm-5.3-flash")

    model = provider.get_llm("zai", json_mode=True)
    assert model.kwargs["api_key"] == "zai-test-secret"
    assert model.kwargs["base_url"] == "https://api.z.ai/api/paas/v4/"
    assert model.kwargs["model"] == "glm-5.3-flash"
    assert model.kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}


def test_zai_structured_response_is_locally_validated(monkeypatch):
    model = FakeChat()
    result = provider.invoke_structured_output("return an answer", Output, llm=model, provider="zai")
    assert result.answer == 7
    assert model.bound == {"response_format": {"type": "json_object"}}

    model.response = SimpleNamespace(content='{"answer": "not-an-int"}')
    with pytest.raises(ValidationError):
        provider.invoke_structured_output("return an answer", Output, llm=model, provider="zai")
