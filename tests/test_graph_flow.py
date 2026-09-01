from __future__ import annotations

import pytest
from graph import build_research_graph, route_after_router, route_after_eval_search, route_after_write_notes
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
    assert "search_papers" in graph.nodes
    assert "eval_search" in graph.nodes
    assert "read_paper" in graph.nodes
    assert "web_enrich" in graph.nodes
    assert "write_notes" in graph.nodes
    assert "compare_benchmark" in graph.nodes
    assert "final_report" in graph.nodes


def test_routing_logic():
    """Test conditional edge route functions."""
    # Test route_after_router
    assert route_after_router({"intent": "direct_read", "status": "running"}) == "read_paper"
    assert route_after_router({"intent": "direct_compare", "status": "running"}) == "read_paper"
    assert route_after_router({"intent": "search", "status": "running"}) == "search_papers"
    assert route_after_router({"status": "error"}) == "error_handler"

    # Test route_after_eval_search
    assert route_after_eval_search({"eval_passed": True, "status": "running"}) == "read_paper"
    assert route_after_eval_search({"eval_passed": False, "retry_count": 0, "status": "running"}) == "refine_query"
    # Hard-stop at max retries
    assert route_after_eval_search({"eval_passed": False, "retry_count": config.MAX_RETRIES, "status": "running"}) == "read_paper"

    # Test route_after_write_notes
    dummy_paper = PaperItem(paper_id="p1", title="Test Paper 1")
    assert route_after_write_notes({"selected_papers": [dummy_paper], "intent": "search", "status": "running"}) == "final_report"
    assert route_after_write_notes({"selected_papers": [dummy_paper, dummy_paper], "intent": "search", "status": "running"}) == "compare_benchmark"
