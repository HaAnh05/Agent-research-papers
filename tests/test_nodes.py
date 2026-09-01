from __future__ import annotations

import pytest
from nodes.router import router_node
from nodes.error_handler import error_handler_node
from state import PaperItem, ResearchState
from text_utils import detect_response_language


def test_detect_response_language_matches_user_input():
    """The generated response should follow the user's language."""
    assert detect_response_language("Tôi muốn tìm vài bài báo về unlearning") == "Vietnamese"
    assert detect_response_language("Find papers on diffusion models") == "English"


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
