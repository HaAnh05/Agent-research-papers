"""Public performance measurements remain numeric and content-free."""

from __future__ import annotations

import json
import time

import nodes.eval_search as eval_module
import nodes.search as search_module
from state import EvalSearchOutput, PaperItem


def test_arxiv_search_emits_numeric_timing_without_query(monkeypatch):
    def fake_search(**_kwargs):
        time.sleep(0.005)
        return []

    monkeypatch.setattr(search_module, "arxiv_search", fake_search)
    result = search_module.search_papers_node({"search_query": "private research query", "retry_count": 0})

    timing = result["node_timings"]["search_papers"]
    assert isinstance(timing["durationMs"], int)
    assert isinstance(timing["arxivMs"], int)
    assert timing["durationMs"] >= timing["arxivMs"] >= 0
    assert "private research query" not in json.dumps(timing)


def test_evaluation_emits_numeric_llm_timing_without_paper_text(monkeypatch):
    monkeypatch.setattr(eval_module, "get_llm", lambda: object())

    def fake_invoke(*_args, **_kwargs):
        time.sleep(0.005)
        return EvalSearchOutput(passed=True, score=0.9, feedback="relevant", selected_indices=[0])

    monkeypatch.setattr(eval_module, "invoke_structured_output", fake_invoke)
    paper = PaperItem(paper_id="arxiv_1706.03762", title="Fixture", summary="private PDF content")
    result = eval_module.eval_search_node({"user_query": "query", "search_results": [paper], "retry_count": 0})

    timing = result["node_timings"]["eval_search"]
    assert isinstance(timing["durationMs"], int)
    assert isinstance(timing["llmMs"], int)
    assert timing["durationMs"] >= timing["llmMs"] >= 0
    assert "private PDF content" not in json.dumps(timing)
