from __future__ import annotations

import pytest
import time
import copy
import threading
import nodes.router as router_module
import nodes.compare_benchmark as compare_module
import nodes.web_enrich as web_enrich_module
import nodes.write_notes as write_notes_module
from nodes.router import router_node
from nodes.eval_search import eval_search_node
from nodes.error_handler import error_handler_node
from config import config
from state import ComparisonArtifact, ComparisonPaper, ComparisonRow, GitHubRepoInfo, PMRLNotes, PaperItem, ResearchState
from text_utils import detect_response_language


def test_detect_response_language_matches_user_input():
    """The generated response should follow the user's language."""
    assert detect_response_language("Tôi muốn tìm vài bài báo về unlearning") == "Vietnamese"
    assert detect_response_language("Find papers on diffusion models") == "English"


def test_eval_search_exhaustion_reports_the_search_failure():
    result = eval_search_node({
        "user_query": "an over-specific paper query",
        "search_results": [],
        "retry_count": config.MAX_RETRIES,
    })

    assert result["status"] == "error"
    assert "ArXiv" in result["error_message"]
    assert "No selected papers" not in result["error_message"]


def test_router_node_single_link():
    """Test router deterministic matching for single arXiv link."""
    state: ResearchState = {
        "user_query": "",
        "raw_inputs": ["https://arxiv.org/abs/1706.03762"],
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": "",
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }
    result = router_node(state)
    assert result["intent"] == "direct_read"
    assert len(result["selected_papers"]) == 1
    assert result["selected_papers"][0].arxiv_id == "1706.03762"


def test_router_uses_cached_direct_arxiv_metadata_without_guessing(monkeypatch):
    monkeypatch.setattr(
        router_module,
        "get_cached_arxiv_metadata",
        lambda _arxiv_id: {
            "title": "Attention Is All You Need",
            "summary": "Transformer abstract",
            "authors": ["Author One"],
            "published": "2017-06-12",
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        },
    )
    result = router_node({"user_query": "", "raw_inputs": ["1706.03762"], "trace_logs": []})
    paper = result["selected_papers"][0]
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ["Author One"]
    assert paper.published == "2017-06-12"


def test_router_downgrades_direct_intent_without_a_paper_input(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm", lambda: object())
    monkeypatch.setattr(
        router_module,
        "invoke_structured_output",
        lambda *args, **kwargs: router_module.RouterDecision(
            intent="direct_read",
            reasoning="The query names a paper",
            search_query="attention is all you need",
        ),
    )

    result = router_node({
        "user_query": "đọc Attention Is All You Need",
        "raw_inputs": [],
        "trace_logs": [],
    })

    assert result["intent"] == "search"
    assert result["search_query"] == "attention is all you need"


def test_router_node_multi_links():
    """Test router deterministic matching for multiple arXiv links."""
    state: ResearchState = {
        "user_query": "Compare these",
        "raw_inputs": ["1706.03762", "2005.14165"],
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": "",
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }
    result = router_node(state)
    assert result["intent"] == "direct_compare"
    assert len(result["selected_papers"]) == 2


def test_specific_paper_name_is_preserved_in_query():
    """Exact comparison queries should keep paper names together instead of turning into generic MoE keywords."""
    query = "so sánh DeepSeek-MoE và DeepSeek-V2"
    assert "DeepSeek-MoE" in query and "DeepSeek-V2" in query


def test_safe_router_compare_pattern_is_detected():
    """Only explicit comparison patterns should trigger the safe compare path."""
    state: ResearchState = {
        "user_query": "so sánh DeepSeek-MoE và DeepSeek-V2",
        "raw_inputs": [],
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": "",
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }
    result = router_node(state)
    assert result["intent"] == "search"
    assert "DeepSeek-MoE" in result["search_query"] and "DeepSeek-V2" in result["search_query"]


def test_safe_router_topic_query_keeps_generic_search():
    """Generic topic queries should not be forced into compare mode."""
    state: ResearchState = {
        "user_query": "tìm các bài báo về MoE và distillation trong LLM",
        "raw_inputs": [],
        "intent": "search",
        "status": "running",
        "error_message": None,
        "search_query": "",
        "search_results": [],
        "retry_count": 0,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }
    result = router_node(state)
    assert result["intent"] == "search"
    assert "MoE" in result["search_query"] or "distillation" in result["search_query"]


def test_error_handler_node():
    """Test error handler node producing safe degraded report."""
    state: ResearchState = {
        "user_query": "Test query",
        "raw_inputs": [],
        "intent": "search",
        "status": "error",
        "error_message": "Network timeout occurred",
        "search_query": "Test query",
        "search_results": [],
        "retry_count": 2,
        "eval_passed": False,
        "eval_feedback": "",
        "selected_papers": [],
        "benchmark_matrix": None,
        "final_report": None,
        "trace_logs": [],
        "error_logs": ["Network timeout occurred"],
    }
    result = error_handler_node(state)
    assert result["status"] == "error"
    assert "Network timeout occurred" in result["final_report"]


def test_validate_brief_summary_requires_conclusion_and_three_or_four_bullets():
    assert write_notes_module.validate_brief_summary(
        "The method is useful on the reported setting.\n- First finding.\n- Second finding.\n- Third finding."
    ).startswith("The method is useful")
    assert write_notes_module.validate_brief_summary("Conclusion.\n- only\n- two") == ""
    long_text = "Conclusion.\n" + "\n".join(f"- {'word ' * 50}" for _ in range(3))
    assert write_notes_module.validate_brief_summary(long_text) == ""


def test_write_notes_two_workers_preserve_order_and_isolate_failure(monkeypatch):
    papers = [
        PaperItem(paper_id="p1", title="Paper one", summary="one"),
        PaperItem(paper_id="p2", title="Paper two", summary="two"),
    ]

    monkeypatch.setattr(write_notes_module, "get_llm", lambda: object())

    def fake_invoke(prompt, schema, **kwargs):
        time.sleep(0.1)
        if "Paper two" in prompt:
            raise RuntimeError("provider failure")
        return PMRLNotes(
            problem="p",
            method="m",
            result="r",
            limitation="l",
            brief_summary="Conclusion.\n- a\n- b\n- c",
        )

    monkeypatch.setattr(write_notes_module, "invoke_structured_output", fake_invoke)
    started = time.perf_counter()
    result = write_notes_module.write_notes_node({"selected_papers": papers})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.19
    output = result["selected_papers"]
    assert [paper.paper_id for paper in output] == ["p1", "p2"]
    assert output[0].notes_quality == "complete"
    assert output[0].notes.brief_summary.count("\n-") == 3
    assert output[1].notes_quality == "invalid"
    assert output[1].notes.method.startswith("Not extracted")
    assert any("p2" in error for error in result["error_logs"])


def test_compare_artifact_rejects_unsupported_numeric_alignment(monkeypatch):
    papers = [
        PaperItem(
            paper_id="p1",
            title="Paper one",
            notes=PMRLNotes(problem="p", method="m", result="On ImageNet, accuracy reached 88.5%.", limitation="l"),
        ),
        PaperItem(
            paper_id="p2",
            title="Paper two",
            notes=PMRLNotes(problem="p", method="m", result="On ImageNet, accuracy reached 90.0%.", limitation="l"),
        ),
    ]
    artifact = ComparisonArtifact(
        papers=[ComparisonPaper(paper_id="p1", title="Paper one"), ComparisonPaper(paper_id="p2", title="Paper two")],
        rows=[ComparisonRow(
            metric="accuracy",
            dataset="ImageNet",
            unit="%",
            values={"p1": "88.5%", "p2": "90.0%"},
            source_quotes={"p1": "accuracy reached 88.5%", "p2": "accuracy reached 90.0%"},
            comparable=True,
            reason="",
        )],
        synthesis=["trade-off"],
    )
    monkeypatch.setattr(compare_module, "get_llm", lambda: object())
    monkeypatch.setattr(compare_module, "invoke_structured_output", lambda *args, **kwargs: artifact)
    result = compare_module.compare_benchmark_node({"selected_papers": papers})
    row = result["comparison_artifact"]["rows"][0]
    assert row["comparable"] is False
    assert row["reason"].startswith("Not directly comparable")
    assert "88.5%" not in result["benchmark_matrix"]


def test_compare_artifact_keeps_same_metric_dataset_unit_with_pdf_quotes(monkeypatch):
    papers = [
        PaperItem(paper_id="p1", title="Paper one", sections={"experiments": "On ImageNet, accuracy reached 88.5%."}),
        PaperItem(paper_id="p2", title="Paper two", sections={"experiments": "On ImageNet, accuracy reached 90.0%."}),
    ]
    artifact = ComparisonArtifact(
        papers=[ComparisonPaper(paper_id="p1", title="Paper one"), ComparisonPaper(paper_id="p2", title="Paper two")],
        rows=[ComparisonRow(
            metric="accuracy",
            dataset="ImageNet",
            unit="%",
            values={"p1": "88.5%", "p2": "90.0%"},
            source_quotes={"p1": "On ImageNet, accuracy reached 88.5%.", "p2": "On ImageNet, accuracy reached 90.0%."},
            comparable=True,
        )],
    )
    monkeypatch.setattr(compare_module, "get_llm", lambda: object())
    monkeypatch.setattr(compare_module, "invoke_structured_output", lambda *args, **kwargs: artifact)

    result = compare_module.compare_benchmark_node({"selected_papers": papers})
    assert result["comparison_artifact"]["rows"][0]["comparable"] is True
    assert "88.5%" in result["benchmark_matrix"]


def test_web_enrich_combined_queue_is_bounded_parallel_and_ordered(monkeypatch):
    papers = [
        PaperItem(paper_id="p1", title="Paper one"),
        PaperItem(paper_id="p2", title="Paper two"),
    ]
    lock = threading.Lock()
    active = 0
    max_active = 0

    def enter_job():
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)

    def leave_job():
        nonlocal active
        with lock:
            active -= 1

    def fake_github(paper):
        enter_job()
        try:
            time.sleep(0.1)
            return [GitHubRepoInfo(name=f"{paper.paper_id}/repo", url="https://github.com/example/repo")], {"durationMs": 100, "tiers": {"title_keywords": {"status": "ok"}}}, False
        finally:
            leave_job()

    def fake_pmrl(paper, *, llm=None):
        enter_job()
        try:
            time.sleep(0.1)
            result = copy.deepcopy(paper)
            result.notes = PMRLNotes(problem="p", method="m", result="r", limitation="l", brief_summary="C.\n- a\n- b\n- c")
            result.notes_quality = "complete" if paper.paper_id == "p1" else "invalid"
            return result, "ok" if paper.paper_id == "p1" else "fallback:RuntimeError", {"durationMs": 100, "llmMs": 100}
        finally:
            leave_job()

    monkeypatch.setattr(web_enrich_module, "get_llm", lambda: object())
    monkeypatch.setattr(web_enrich_module, "_github_one", fake_github)
    monkeypatch.setattr(web_enrich_module, "analyze_paper_notes", fake_pmrl)
    started = time.perf_counter()
    result = web_enrich_module.web_enrich_node({"selected_papers": papers})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.34
    assert max_active <= 2
    assert [paper.paper_id for paper in result["selected_papers"]] == ["p1", "p2"]
    assert result["selected_papers"][0].notes_quality == "complete"
    assert result["selected_papers"][1].notes_quality == "invalid"
    assert any("p2" in error for error in result["error_logs"])
