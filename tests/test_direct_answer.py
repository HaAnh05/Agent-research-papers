from __future__ import annotations

from types import SimpleNamespace

import pytest

import nodes.direct_answer as direct_answer_module
import nodes.router as router_module
from graph import build_research_graph, route_after_router
from nodes.direct_answer import direct_answer_node
from nodes.router import RouterDecision, router_node


def test_graph_exposes_direct_answer_route():
    graph = build_research_graph()
    assert "direct_answer" in graph.nodes
    assert route_after_router({"intent": "direct_answer", "status": "running"}) == "direct_answer"


def test_graph_direct_answer_stops_before_search(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt: str):
            return SimpleNamespace(content="BibTeX is a citation format.")

    monkeypatch.setattr(direct_answer_module, "get_llm", lambda: FakeLLM())
    initial_state = {
        "user_query": "What is BibTeX?",
        "raw_inputs": [],
        "intent": "search",
        "status": "running",
        "error_message": None,
        "conversation_context": "",
        "search_query": "",
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "assistant_answer": "",
        "trace_logs": [],
        "error_logs": [],
    }
    graph = build_research_graph()
    updates = list(graph.stream(
        initial_state,
        {"configurable": {"thread_id": "direct-answer-test"}},
        stream_mode="updates",
    ))

    assert [next(iter(update)) for update in updates] == ["router", "direct_answer"]
    assert updates[-1]["direct_answer"]["assistant_answer"].startswith("BibTeX")


def test_general_definition_bypasses_router_llm(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm", lambda: pytest.fail("classifier must not run"))
    monkeypatch.setattr(router_module, "invoke_structured_output", lambda *args, **kwargs: pytest.fail("classifier must not run"))

    result = router_node({
        "user_query": "What is BibTeX?",
        "raw_inputs": [],
        "conversation_context": "",
    })

    assert result["intent"] == "direct_answer"
    assert result["search_query"] == ""


def test_source_seeking_method_question_stays_on_research_router(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(router_module, "get_llm", lambda: object())
    monkeypatch.setattr(
        router_module,
        "invoke_structured_output",
        lambda prompt, schema, **kwargs: (prompts.append(prompt) or RouterDecision(
            intent="search",
            reasoning="new sources requested",
            search_query="topology aware Gaussian Splatting SLAM",
        )),
    )

    result = router_node({
        "user_query": "What are recent methods for topology-aware Gaussian Splatting SLAM?",
        "raw_inputs": [],
        "conversation_context": "",
    })

    assert result["intent"] == "search"
    assert result["search_query"] == "topology aware Gaussian Splatting SLAM"
    assert prompts and "validated paper input" not in prompts[0]


def test_context_bound_literature_review_does_not_retrieve_again(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm", lambda: pytest.fail("context follow-up must not search"))

    result = router_node({
        "user_query": "Turn these papers into a Vietnamese literature review",
        "raw_inputs": [],
        "conversation_context": "Paper A PMRL: method and limitation notes.",
    })

    assert result["intent"] == "direct_answer"


def test_direct_answer_is_context_constrained_and_does_not_log_context(monkeypatch):
    captured: list[str] = []

    class FakeLLM:
        def invoke(self, prompt: str):
            captured.append(prompt)
            return SimpleNamespace(content="Theo PMRL, phương pháp dùng regularization.")

    monkeypatch.setattr(direct_answer_module, "get_llm", lambda: FakeLLM())
    result = direct_answer_node({
        "user_query": "Explain the method.",
        "conversation_context": "Paper A PMRL: method uses regularization.",
        "raw_inputs": ["/srv/private/uploads/secret.pdf"],
    })

    assert result["status"] == "success"
    assert result["assistant_answer"].startswith("Theo PMRL")
    assert captured and "Paper A PMRL" in captured[0]
    assert all("secret.pdf" not in log for log in result["trace_logs"])


def test_direct_input_trace_and_display_label_hide_local_path():
    local_path = "/srv/private/uploads/secret.pdf"
    result = router_node({
        "user_query": "Summarize this PDF",
        "raw_inputs": [local_path],
        "conversation_context": "",
    })

    assert local_path not in " ".join(result["trace_logs"])
    assert local_path not in result["selected_papers"][0].title
    assert result["selected_papers"][0].local_pdf_path == local_path
