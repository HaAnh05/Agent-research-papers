from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config import config
from nodes import final_report
from state import PaperItem, PMRLNotes


_OVERVIEW_OK = (
    "## 1. Tóm tắt\n- Tổng quan.\n\n## 3. Kết luận\n- Kết luận.\n"
)


class FakeLLM:
    def __init__(self, calls: list | None = None):
        self.calls = calls if calls is not None else []

    def invoke(self, prompt):
        # Calls send [SystemMessage, HumanMessage]; flatten for matching.
        prompt = prompt if isinstance(prompt, str) else "\n".join(str(m.content) for m in prompt)
        self.calls.append(prompt)
        # The report sends a paper-block prompt and an overview prompt.
        if "ONE paper block" in prompt:
            return SimpleNamespace(content="### Fake paper\n- **Vấn đề**: p\n")
        if "summary and conclusion of a research report" in prompt:
            return SimpleNamespace(content=_OVERVIEW_OK)
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


def test_final_report_uses_unique_readable_filename_and_trace(monkeypatch, tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(final_report, "get_llm", lambda: FakeLLM())

    state = {
        "user_query": "collision test",
        "selected_papers": [],
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


def test_final_report_reuses_cache_on_identical_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path / "summaries")
    (tmp_path / "summaries").mkdir()
    calls: list = []
    monkeypatch.setattr(final_report, "get_llm", lambda: FakeLLM(calls))

    state = {"user_query": "cache test", "selected_papers": [_cached_paper()]}
    first = final_report.final_report_node(state)
    assert first["node_timings"]["final_report"].get("cache") is not True
    assert first["status"] == "success"

    calls.clear()
    # Identical synthesis input (same notes content) must hit the cache.
    second_state = {"user_query": "cache test", "selected_papers": [_cached_paper()]}
    second = final_report.final_report_node(second_state)

    assert calls == []
    assert second["final_report"] == first["final_report"]
    assert second["node_timings"]["final_report"].get("cache") is True
    assert second["node_timings"]["final_report"]["llmMs"] == 0
    assert second["status"] == "success"


def test_report_cache_key_versions_the_report(monkeypatch):
    paper = _cached_paper()
    first_key, _ = final_report._report_cache_key(
        "cache version", "Report", "English", "breakdown", [paper]
    )
    monkeypatch.setattr(final_report, "REPORT_VERSION", "next-version")
    second_key, _ = final_report._report_cache_key(
        "cache version", "Report", "English", "breakdown", [paper]
    )
    assert first_key != second_key


