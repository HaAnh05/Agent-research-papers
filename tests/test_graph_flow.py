from __future__ import annotations

import pytest
from graph import build_research_graph, route_after_router, route_after_eval_search, route_after_read_paper
from state import ResearchState, PaperItem
from config import config
from text_utils import compact_message


def test_compact_message_removes_emoji_and_collapses_whitespace():
    """Message output should stay short and clean without emoji/stickers."""
    assert compact_message("✅ [router] Đã tìm thấy 2 bài báo  ") == "[router] Đã tìm thấy 2 bài báo"


def test_graph_compilation():
    """Verify that the StateGraph compiles cleanly without syntax or schema errors."""
    graph = build_research_graph()
    assert graph is not None
    assert "router" in graph.nodes
    assert set(graph.nodes) == {
        "__start__",
        "router",
        "search_papers",
        "eval_search",
        "refine_query",
        "read_paper",
        "web_enrich",
        "write_notes",
        "citation_graph",
        "final_report",
        "error_handler",
    }
    assert "search_papers" in graph.nodes
    assert "eval_search" in graph.nodes
    assert "read_paper" in graph.nodes
    assert "web_enrich" in graph.nodes
    assert "write_notes" in graph.nodes
    assert "compare_benchmark" not in graph.nodes
    assert "final_report" in graph.nodes


def test_routing_logic():
    """Test conditional edge route functions."""
    # Test route_after_router
    direct_paper = PaperItem(paper_id="direct", title="Direct paper")
    assert route_after_router({"intent": "direct_read", "selected_papers": [direct_paper], "status": "running"}) == "read_paper"
    assert route_after_router({"intent": "direct_read", "selected_papers": [direct_paper], "status": "running"}) == "read_paper"
    assert route_after_router({"intent": "direct_read", "selected_papers": [], "status": "running"}) == "search_papers"
    assert route_after_router({"intent": "search", "status": "running"}) == "search_papers"
    assert route_after_router({"status": "error"}) == "error_handler"

    # Test route_after_eval_search
    assert route_after_eval_search({"eval_passed": True, "status": "running"}) == "read_paper"
    assert route_after_eval_search({"eval_passed": False, "retry_count": 0, "status": "running"}) == "refine_query"
    # Hard-stop at max retries without sending an empty selection to PDF parsing.
    assert route_after_eval_search({
        "eval_passed": False,
        "retry_count": config.MAX_RETRIES,
        "selected_papers": [],
        "status": "running",
    }) == "error_handler"

    # Test route_after_read_paper: fan out to the three parallel branches
    dummy_paper = PaperItem(paper_id="p1", title="Test Paper 1")
    assert set(route_after_read_paper({"selected_papers": [dummy_paper], "status": "running"})) == {
        "write_notes", "citation_graph", "web_enrich",
    }
    assert route_after_read_paper({"selected_papers": [dummy_paper], "status": "error"}) == "error_handler"
