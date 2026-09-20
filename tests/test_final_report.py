from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config import config
from nodes import final_report
from state import PaperItem, PMRLNotes, SummaryCards


class FakeLLM:
    def __init__(self, calls: list | None = None):
        self.calls = calls if calls is not None else []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(content="# Collision-safe report\n")


def _cached_paper() -> PaperItem:
    words = "alpha beta gamma delta epsilon"
    return PaperItem(
        paper_id="p1",
        title=f"Cache test paper {words}",
        summary=words,
        sections={"abstract": (words + " ") * 200, "methodology": (words + " ") * 200},
        source_quality={"coverageStatus": "complete"},
        notes=PMRLNotes(problem="p", method="m", result="r", limitation="l"),
    )


def _card(word: str, count: int) -> str:
    return ((word + " ") * count).strip() + "."


def test_final_report_uses_unique_readable_filename_and_trace(monkeypatch, tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(final_report, "get_llm", lambda: FakeLLM())

    state = {
        "user_query": "collision test",
        "selected_papers": [],
        "benchmark_matrix": None,
    }

    first = final_report.final_report_node(state)
    second = final_report.final_report_node(state)

    first_name = first["trace_logs"][-1].rsplit(": ", 1)[-1]
    second_name = second["trace_logs"][-1].rsplit(": ", 1)[-1]
    assert first_name != second_name
    assert first_name.startswith("report_collision_test_")
    assert second_name.startswith("report_collision_test_")
    assert first_name.endswith(".md") and second_name.endswith(".md")
    assert (reports_dir / first_name).is_file()
    assert (reports_dir / second_name).is_file()


def test_final_report_runs_summary_in_parallel_without_report_text(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path / "summaries")
    (tmp_path / "summaries").mkdir()
    calls: list = []
    monkeypatch.setattr(final_report, "get_llm", lambda: FakeLLM(calls))

    seen_prompts: list = []

    def fake_structured(prompt: str, schema, **kwargs):
        seen_prompts.append(prompt)
        return SummaryCards(
            tldr=_card("alpha", 25),
            problem=_card("beta", 40),
            method=_card("gamma", 40),
            key_results=_card("delta", 40),
            why_it_matters=_card("epsilon", 40),
        )

    monkeypatch.setattr(final_report, "invoke_structured_output", fake_structured)

    state = {"user_query": "parallel test", "selected_papers": [_cached_paper()], "benchmark_matrix": None}
    result = final_report.final_report_node(state)

    assert result["status"] == "success"
    # Synthesis ran exactly once; the summary prompt never saw the report text,
    # proving the two LLM calls were decoupled and could overlap.
    assert len(calls) == 1
    assert len(seen_prompts) == 1
    assert "# Collision-safe report" not in seen_prompts[0]
    paper = result["selected_papers"][0]
    assert paper.notes.summary_quality == "complete"
    assert paper.notes.summary_cards.tldr.startswith("alpha")


def test_final_report_reuses_cache_on_identical_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path / "summaries")
    (tmp_path / "summaries").mkdir()
    calls: list = []
    monkeypatch.setattr(final_report, "get_llm", lambda: FakeLLM(calls))

    def working_structured(prompt: str, schema, **kwargs):
        return SummaryCards(
            tldr=_card("alpha", 25),
            problem=_card("beta", 40),
            method=_card("gamma", 40),
            key_results=_card("delta", 40),
            why_it_matters=_card("epsilon", 40),
        )

    monkeypatch.setattr(final_report, "invoke_structured_output", working_structured)

    def fake_structured(prompt: str, schema, **kwargs):
        raise AssertionError("summary LLM must not run on a full cache hit")

    state = {"user_query": "cache test", "selected_papers": [_cached_paper()], "benchmark_matrix": None}
    first = final_report.final_report_node(state)
    assert first["node_timings"]["final_report"].get("cache") is not True
    assert first["status"] == "success"

    monkeypatch.setattr(final_report, "invoke_structured_output", fake_structured)

    monkeypatch.setattr(final_report, "invoke_structured_output", fake_structured)
    calls.clear()
    # Identical synthesis input (same notes content) must hit the cache even
    # though the summary LLM is now rigged to explode.
    second_state = {"user_query": "cache test", "selected_papers": [_cached_paper()], "benchmark_matrix": None}
    second = final_report.final_report_node(second_state)

    assert calls == []
    assert second["final_report"] == first["final_report"]
    assert second["node_timings"]["final_report"].get("cache") is True
    assert second["node_timings"]["final_report"]["llmMs"] == 0
    assert second["status"] == "success"
