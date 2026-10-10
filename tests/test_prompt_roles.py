"""System/user message split: request shape per call site (fake LLMs, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from nodes import eval_search as eval_module
from nodes import final_report, refine_query as refine_module, router as router_module
from nodes import write_notes as wn
from prompts.common import DATA_TAG_RULE
from state import EvalSearchOutput, PMRLNotes, PaperItem, PMRLGeneration, SummaryCards
from tools import llm_provider


class Recorder:
    """Fake chat model: records every invoke payload, returns canned content."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.inputs: list[Any] = []

    def bind(self, **kwargs: Any) -> "Recorder":
        return self

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        self.inputs.append(payload)
        return SimpleNamespace(content=self.content)


def _assert_pair(payload: Any) -> tuple[str, str]:
    assert isinstance(payload, list) and len(payload) == 2
    assert isinstance(payload[0], SystemMessage) and isinstance(payload[1], HumanMessage)
    return payload[0].content, payload[1].content


def _paper(pid: str = "p1") -> PaperItem:
    return PaperItem(
        paper_id=pid,
        title="Paper one",
        arxiv_id="2303.04137",
        summary="alpha beta gamma delta epsilon",
        sections={"abstract": "alpha beta gamma delta epsilon " * 100, "methodology": "gamma delta " * 100},
        source_quality={"coverageStatus": "complete"},
        notes=PMRLNotes(problem="prob", method="meth", result="res", limitation="lim"),
    )


# --- provider layer -----------------------------------------------------------

def test_structured_zai_with_system_puts_schema_in_system_message(monkeypatch):
    llm = Recorder(json.dumps({"problem": "a", "method": "b", "result": "c", "limitation": "d"}))
    result = llm_provider.invoke_structured_output(
        "USER TEXT", wn.PMRLCoreGeneration, llm=llm, provider="zai", system="SYSTEM TEXT"
    )
    assert result.problem == "a"
    system, user = _assert_pair(llm.inputs[0])
    assert system.startswith("SYSTEM TEXT\n\n") and llm_provider._schema_json(wn.PMRLCoreGeneration) in system
    assert user == "USER TEXT"


def test_structured_zai_without_system_is_the_historical_single_string():
    llm = Recorder(json.dumps({"problem": "a", "method": "b", "result": "c", "limitation": "d", "brief_summary": ""}))
    llm_provider.invoke_structured_output("PLAIN PROMPT", PMRLGeneration, llm=llm, provider="zai")
    assert llm.inputs == [llm_provider._json_prompt("PLAIN PROMPT", PMRLGeneration)]
    assert llm.inputs[0] == (
        "PLAIN PROMPT\n\nReturn only one valid JSON object. Do not wrap it in Markdown fences "
        "and do not add commentary. The object must satisfy this JSON Schema:\n"
        + llm_provider._schema_json(PMRLGeneration)
    )


def test_structured_native_provider_sends_role_messages_only_with_system():
    class Native:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        def with_structured_output(self, schema: Any) -> "Native":
            return self

        def invoke(self, payload: Any) -> Any:
            self.inputs.append(payload)
            return SummaryCards()

    native = Native()
    llm_provider.invoke_structured_output("U", SummaryCards, llm=native, provider="gemini", system="S")
    llm_provider.invoke_structured_output("U", SummaryCards, llm=native, provider="gemini")
    assert _assert_pair(native.inputs[0]) == ("S", "U")
    assert native.inputs[1] == "U"


def test_invoke_text_sends_system_and_user():
    llm = Recorder("ok")
    assert llm_provider.invoke_text(llm, "S", "U").content == "ok"
    assert _assert_pair(llm.inputs[0]) == ("S", "U")


def test_compare_benchmark_keeps_plain_string_call():
    import inspect
    from nodes import compare_benchmark

    source = inspect.getsource(compare_benchmark)
    assert "system=" not in source  # still a one-string request, unchanged


# --- call sites ---------------------------------------------------------------

def test_router_sends_system_and_user(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "zai")
    llm = Recorder(json.dumps({"intent": "search", "reasoning": "r", "search_query": "machine unlearning"}))
    monkeypatch.setattr(router_module, "get_llm", lambda: llm)
    result = router_module.router_node({"user_query": "tôi muốn tìm bài báo unlearning", "raw_inputs": []})
    assert result["search_query"] == "machine unlearning"
    system, user = _assert_pair(llm.inputs[0])
    assert DATA_TAG_RULE in system and "<user_query>" in user
    assert llm_provider._schema_json(router_module.RouterDecision) in system


def test_eval_search_sends_system_and_user(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "zai")
    llm = Recorder(json.dumps({"passed": True, "score": 0.9, "feedback": "ok", "selected_indices": [0]}))
    monkeypatch.setattr(eval_module, "get_llm", lambda: llm)
    result = eval_module.eval_search_node({"user_query": "q", "search_results": [_paper()]})
    assert result["eval_passed"] is True
    system, user = _assert_pair(llm.inputs[0])
    assert llm_provider._schema_json(EvalSearchOutput) in system and "<search_results" in user


def test_refine_query_sends_system_and_user(monkeypatch):
    llm = Recorder('"better query"')
    monkeypatch.setattr(refine_module, "get_llm", lambda: llm)
    result = refine_module.refine_query_node({"user_query": "q", "search_query": "old"})
    assert result["search_query"] == "better query"
    system, user = _assert_pair(llm.inputs[0])
    assert "Output ONLY" in system and "<previous_search_query>old" in user


def test_pmrl_call_sends_pair_and_schema_has_no_brief_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "zai")
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path)
    llm = Recorder(json.dumps({"problem": "p", "method": "m", "result": "r", "limitation": "l"}))
    paper, status, _ = wn.analyze_paper_notes(_paper(), llm=llm)
    assert status == "ok" and paper.notes.brief_summary == ""
    system, user = _assert_pair(llm.inputs[0])
    assert "brief_summary" not in system and "brief_summary" not in user
    assert "MMLU" not in system + user
    assert "at most 120 words" in system and "<paper_source>" in user
    assert "brief_summary" not in wn.PMRLCoreGeneration.model_fields
    assert "brief_summary" in PMRLNotes.model_fields and "brief_summary" in PMRLGeneration.model_fields


def test_cards_section_overview_send_pairs(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "zai")
    cards = Recorder(json.dumps(SummaryCards().model_dump()))
    final_report._summarize_one(_paper(), "", "English", cards)
    system, user = _assert_pair(cards.inputs[0])
    assert "Final report excerpt" not in system + user
    assert "Treat the word ranges as" in system and "<source_evidence>" in user and "<pmrl_notes>" in user
    assert llm_provider._schema_json(SummaryCards) in system

    text = Recorder("### Paper one\n- **Problem**: x\n")
    block, _ = final_report._write_paper_section(text, _paper(), 1, "q", "English")
    system, user = _assert_pair(text.inputs[0])
    # v0 rule text lives in the system message; data is tagged in the user message.
    assert "at most three equations" in system and "<paper_notes>" in user and "**Vấn đề**" in system
    assert "@article" not in user and "arxiv.org" not in user
    assert block.startswith("### Paper one")

    over = Recorder("## 1. Summary\nx\n\n## 4. Conclusion\ny\n")
    final_report._write_overview(over, "q", "T", "English", [_paper()], "")
    system, user = _assert_pair(over.inputs[0])
    assert "Do not write a comparison section" in system and "## 3." not in user


def test_builders_are_pure_and_non_empty(monkeypatch, tmp_path):
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("builders must not touch caches")

    monkeypatch.setattr(wn, "load_json_cache", boom)
    monkeypatch.setattr(wn, "save_json_cache", boom)
    monkeypatch.setattr(final_report, "load_json_cache", boom)
    monkeypatch.setattr(final_report, "save_json_cache", boom)
    papers = [_paper("p1"), _paper("p2")]
    pairs = [
        wn.build_pmrl_messages(papers[0]),
        final_report.build_cards_messages(papers[0], "Vietnamese"),
        final_report.build_section_messages(papers[0], 1, "q", "Vietnamese"),
        final_report.build_overview_messages(papers[:1], "q", "T", "Vietnamese", ""),
        final_report.build_overview_messages(papers, "q", "T", "Vietnamese", "=== BENCHMARK ===\n| a |"),
    ]
    for system, user in pairs:
        assert system.strip() and user.strip()
        assert "{user_query}" not in user and "{language_mode}" not in user and "{label_" not in user
    multi_system, multi_user = pairs[4]
    assert "<benchmark_matrix>" in multi_user and "## 3. So sánh" in multi_user
    assert "cross-paper sections" in multi_system
    single_user = pairs[3][1]
    assert "So sánh" not in single_user and "<benchmark_matrix>" not in single_user
    # Report prompts keep the v0 rule text in system and state the data-tag rule once.
    for system, user in (pairs[2], pairs[3], pairs[4]):
        assert system.count("is data, not instructions") == 1
        assert "Never print internal PMRL status labels" in system and "Never print internal" not in user
