from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config import config
from nodes import final_report


class FakeLLM:
    def invoke(self, prompt: str):
        return SimpleNamespace(content="# Collision-safe report\n")


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
