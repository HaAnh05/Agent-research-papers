from __future__ import annotations

from typing import Any, Dict
from state import ResearchState
from config import config
from tools.arxiv_search import arxiv_search


def search_papers_node(state: ResearchState) -> Dict[str, Any]:
    """Queries ArXiv API using search_query and updates search_results."""
    query = state.get("search_query") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)
    
    logs = [f"[search_papers] Lần {retry_count + 1}: '{query}'"]

    try:
        results = arxiv_search(query=query, max_results=config.MAX_SEARCH_RESULTS)
        logs.append(f"[search_papers] Tìm thấy {len(results)} bài báo")
        
        if not results:
            return {
                "search_results": [],
                "trace_logs": logs,
                "error_logs": [f"Không tìm thấy bài báo nào cho truy vấn '{query}'"]
            }
            
        return {
            "search_results": results,
            "trace_logs": logs
        }
    except Exception as exc:
        err = f"Lỗi gọi ArXiv API: {str(exc)}"
        return {
            "search_results": [],
            "status": "error",
            "error_message": err,
            "error_logs": [err],
            "trace_logs": logs + [f"❌ {err}"]
        }
