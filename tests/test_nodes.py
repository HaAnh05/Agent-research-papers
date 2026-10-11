from __future__ import annotations

import pytest
import time
import copy
import threading
import nodes.router as router_module
import nodes.web_enrich as web_enrich_module
import nodes.write_notes as write_notes_module
from nodes.router import router_node
from nodes.eval_search import eval_search_node
from nodes.error_handler import error_handler_node
from config import config
from state import GitHubRepoInfo, PMRLNotes, PaperItem, ResearchState
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


def test_text_only_question_always_routes_to_paper_search(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(router_module, "get_llm", lambda: object())
    monkeypatch.setattr(
        router_module,
        "invoke_structured_output",
        lambda prompt, schema, **kwargs: (prompts.append(prompt) or router_module.RouterDecision(
            intent="search",
            search_query="BibTeX scholarly publishing",
        )),
    )

    result = router_node({
        "user_query": "What is BibTeX?",
        "raw_inputs": [],
        "conversation_context": "Previous paper notes are available.",
    })

    assert result["intent"] == "search"
    assert result["search_query"] == "BibTeX scholarly publishing"
    assert prompts


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
        "final_report": None,
        "trace_logs": [],
        "error_logs": [],
    }
    result = router_node(state)
    assert result["intent"] == "direct_read"
    assert len(result["selected_papers"]) == 2


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
        "final_report": None,
        "trace_logs": [],
        "error_logs": ["Network timeout occurred"],
    }
    result = error_handler_node(state)
    assert result["status"] == "error"
    assert "Network timeout occurred" in result["final_report"]


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
        )

    monkeypatch.setattr(write_notes_module, "invoke_structured_output", fake_invoke)
    started = time.perf_counter()
    result = write_notes_module.write_notes_node({"selected_papers": papers})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.19
    output = result["paper_notes"]
    assert list(output) == ["p1", "p2"]
    assert output["p1"]["notes_quality"] == "complete"
    assert output["p2"]["notes_quality"] == "invalid"
    assert output["p2"]["notes"].method.startswith("Not extracted")
    assert any("p2" in error for error in result["error_logs"])


def test_pmrl_cache_hit_skips_llm_call(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path)
    paper = PaperItem(paper_id="p1", title="Cached paper", summary="cached content words here")

    calls: list = []

    def fake_invoke(prompt, schema, **kwargs):
        calls.append(prompt)
        return PMRLNotes(
            problem="cached problem",
            method="cached method",
            result="cached result",
            limitation="cached limitation",
        )

    monkeypatch.setattr(write_notes_module, "invoke_structured_output", fake_invoke)

    first_paper, first_status, first_timing = write_notes_module.analyze_paper_notes(paper)
    assert first_status == "ok"
    assert first_timing.get("llmMs", 0) > 0 or "cache" not in first_timing
    assert first_paper.notes is not None and first_paper.notes.problem == "cached problem"

    # Second identical call must reuse the cache without touching the provider.
    monkeypatch.setattr(write_notes_module, "get_llm", lambda: (_ for _ in ()).throw(AssertionError("LLM must not run on cache hit")))
    second_paper, second_status, second_timing = write_notes_module.analyze_paper_notes(paper)
    assert second_status == "ok:cache"
    assert second_timing.get("llmMs") == 0
    assert len(calls) == 1
    assert second_paper.notes is not None and second_paper.notes.problem == "cached problem"
    assert second_paper.notes_quality == "complete"


def test_web_enrich_runs_github_in_parallel_skips_html_repos_and_keeps_order(monkeypatch):
    html_paper = PaperItem(
        paper_id="p0", title="Paper zero", source_quality={"codeUrls": ["https://github.com/example/own"]},
    )
    papers = [html_paper, PaperItem(paper_id="p1", title="Paper one"), PaperItem(paper_id="p2", title="Paper two")]
    searched: list[str] = []

    def fake_github(paper):
        searched.append(paper.paper_id)
        time.sleep(0.1)
        return [GitHubRepoInfo(name=f"{paper.paper_id}/repo", url="https://github.com/example/repo")], {"durationMs": 100, "tiers": {"title_keywords": {"status": "ok"}}}, False

    monkeypatch.setattr(web_enrich_module, "_github_one", fake_github)
    started = time.perf_counter()
    result = web_enrich_module.web_enrich_node({"selected_papers": papers})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.19  # the two searches overlap
    assert sorted(searched) == ["p1", "p2"]  # p0 already has a repo link in its HTML
    enrichment = result["paper_enrichment"]
    assert list(enrichment) == ["p0", "p1", "p2"]
    assert "selected_papers" not in result
    assert enrichment["p0"]["repo_source"] == "html" and enrichment["p0"]["github_repos"] == []
    assert enrichment["p1"]["repo_source"] == "github_search" and len(enrichment["p1"]["github_repos"]) == 1
    assert all(item["bibtex"] for item in enrichment.values())
