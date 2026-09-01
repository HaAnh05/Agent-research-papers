from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from config import config
from nodes.router import router_node
from nodes.search import search_papers_node
from nodes.eval_search import eval_search_node
from nodes.refine_query import refine_query_node
from nodes.read_paper import read_paper_node
from nodes.web_enrich import web_enrich_node
from nodes.write_notes import write_notes_node
from nodes.compare_benchmark import compare_benchmark_node
from nodes.final_report import final_report_node
from nodes.error_handler import error_handler_node


# --- CONDITIONAL ROUTING LOGIC ---

def route_after_router(state: ResearchState) -> str:
    """CE1: Nếu có input cụ thể thì đọc ngay; nếu không thì tìm kiếm."""
    if state.get("status") == "error":
        return "error_handler"
    if state.get("intent") in ["direct_read", "direct_compare"]:
        return "read_paper"
    return "search_papers"


def route_after_eval_search(state: ResearchState) -> str:
    """CE2: nếu kết quả đủ tốt thì đọc, nếu chưa thì suy yếu query hoặc dừng."""
    if state.get("status") == "error":
        return "error_handler"

    # 1. Đã đạt yêu cầu -> đi tiếp.
    if state.get("eval_passed", False):
        return "read_paper"

    # 2. Chưa đạt nhưng còn lượt thử -> tinh chỉnh query.
    if state.get("retry_count", 0) < config.MAX_RETRIES:
        return "refine_query"

    # 3. Đến giới hạn retry -> vẫn đọc kết quả tốt nhất thay vì dừng hẳn.
    return "read_paper"


def route_after_write_notes(state: ResearchState) -> str:
    """CE3: nếu có nhiều bài thì so sánh, ngắn hơn thì báo cáo trực tiếp."""
    if state.get("status") == "error":
        return "error_handler"

    selected_count = len(state.get("selected_papers", []))
    if selected_count >= 2 or state.get("intent") == "direct_compare":
        return "compare_benchmark"
    return "final_report"


# --- GRAPH COMPILATION ---

def build_research_graph(checkpointer: MemorySaver | None = None):
    """Builds and compiles the production-grade LangGraph StateGraph workflow."""
    workflow = StateGraph(ResearchState)

    # 1. Add all Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("eval_search", eval_search_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("read_paper", read_paper_node)
    workflow.add_node("web_enrich", web_enrich_node)
    workflow.add_node("write_notes", write_notes_node)
    workflow.add_node("compare_benchmark", compare_benchmark_node)
    workflow.add_node("final_report", final_report_node)
    workflow.add_node("error_handler", error_handler_node)

    # 2. Build Edges & Conditional Edges
    workflow.add_edge(START, "router")

    # CE1: Router Decision
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "read_paper": "read_paper",
            "search_papers": "search_papers",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("search_papers", "eval_search")

    # CE2: Search Quality Evaluation
    workflow.add_conditional_edges(
        "eval_search",
        route_after_eval_search,
        {
            "refine_query": "refine_query",
            "read_paper": "read_paper",
            "error_handler": "error_handler",
        },
    )

    # Loop back from refine to search
    workflow.add_edge("refine_query", "search_papers")

    # Core Pipeline
    workflow.add_edge("read_paper", "web_enrich")
    workflow.add_edge("web_enrich", "write_notes")

    # CE3: Multi-Paper Comparison vs Single Report
    workflow.add_conditional_edges(
        "write_notes",
        route_after_write_notes,
        {
            "compare_benchmark": "compare_benchmark",
            "final_report": "final_report",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("compare_benchmark", "final_report")
    workflow.add_edge("final_report", END)
    workflow.add_edge("error_handler", END)

    # 3. Compile with Checkpointer
    saver = checkpointer or MemorySaver()
    app = workflow.compile(checkpointer=saver)
    return app


# Default compiled instance
research_graph = build_research_graph()
