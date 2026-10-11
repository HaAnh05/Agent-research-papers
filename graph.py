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
from nodes.citation_graph import citation_graph_node
from nodes.final_report import final_report_node
from nodes.error_handler import error_handler_node


# --- CONDITIONAL ROUTING LOGIC ---

def route_after_router(state: ResearchState) -> str:
    """CE1: route supplied papers directly; all text-only requests search."""
    if state.get("status") == "error":
        return "error_handler"
    if state.get("intent") == "direct_read" and state.get("selected_papers"):
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

    # 3. Đến giới hạn retry -> chỉ đọc khi evaluator vẫn giữ được ứng viên.
    # Không chuyển một danh sách rỗng sang PDF parser vì điều đó che mất lỗi
    # search thực sự bằng thông báo "No selected papers" ở node kế tiếp.
    if state.get("selected_papers"):
        return "read_paper"
    return "error_handler"


# Branches that run in parallel after ``read_paper`` and join at ``final_report``.
PARALLEL_BRANCHES = ["write_notes", "citation_graph", "web_enrich"]


def route_after_read_paper(state: ResearchState):
    """CE3: on error go to error_handler, otherwise fan out to the parallel branches."""
    if state.get("status") == "error":
        return "error_handler"
    return list(PARALLEL_BRANCHES)


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
    workflow.add_node("citation_graph", citation_graph_node)
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

    # CE3: after read_paper, fan out to write_notes (PMRL LLM), citation_graph
    # and web_enrich; final_report waits for all three.  Each branch writes its
    # own state keys (see ResearchState.paper_notes / paper_enrichment).
    workflow.add_conditional_edges(
        "read_paper",
        route_after_read_paper,
        {
            "write_notes": "write_notes",
            "citation_graph": "citation_graph",
            "web_enrich": "web_enrich",
            "error_handler": "error_handler",
        },
    )
    workflow.add_edge(PARALLEL_BRANCHES, "final_report")
    workflow.add_edge("final_report", END)
    workflow.add_edge("error_handler", END)

    # 3. Compile with Checkpointer
    saver = checkpointer or MemorySaver()
    app = workflow.compile(checkpointer=saver)
    return app


# Default compiled instance
research_graph = build_research_graph()
