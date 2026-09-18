from __future__ import annotations

import time
from typing import Any, Dict
from state import ResearchState
from config import config
from tools.arxiv_search import arxiv_search


def search_papers_node(state: ResearchState) -> Dict[str, Any]:
    """Queries ArXiv API using search_query and updates search_results."""
    started = time.perf_counter()
    query = state.get("search_query") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)
    
    logs = [f"[search_papers] Lần {retry_count + 1}: '{query}'"]

    arxiv_started = time.perf_counter()
    try:
        results = arxiv_search(query=query, max_results=config.MAX_SEARCH_RESULTS)
        timing = {"durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                  "arxivMs": max(0, int((time.perf_counter() - arxiv_started) * 1000))}
        logs.append(f"[search_papers] Tìm thấy {len(results)} bài báo")
        
        if not results:
            return {
                "search_results": [],
                "trace_logs": logs,
                "node_timings": {"search_papers": timing},
                "error_logs": [f"Không tìm thấy bài báo nào cho truy vấn '{query}'"]
            }
            
        return {
            "search_results": results,
            "node_timings": {"search_papers": timing},
            "trace_logs": logs
        }
    except Exception as exc:
        timing = {"durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                  "arxivMs": max(0, int((time.perf_counter() - arxiv_started) * 1000))}
        err = f"Lỗi gọi ArXiv API: {str(exc)}"
        return {
            "search_results": [],
            "status": "error",
            "error_message": err,
            "node_timings": {"search_papers": timing},
            "error_logs": [err],
            "trace_logs": logs + [f"❌ {err}"]
        }
