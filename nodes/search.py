from __future__ import annotations

import time
from typing import Any, Dict
from state import ResearchState
from config import config
from tools.arxiv_search import arxiv_search


def _local_index_ready() -> bool:
    """True when the local index flag is on and the index file exists."""
    return bool(config.LOCAL_INDEX_ENABLED) and config.LOCAL_INDEX_PATH.is_file()


def _search_with_local(query: str, logs: list) -> list:
    """Merge local-index hits (first) with arXiv API hits not already present.

    Local hits only cover papers up to ``LOCAL_INDEX_CUTOFF``; newer papers
    come only from the API.  A failing side is logged and skipped as long as
    the other side returns results.
    """
    from tools.local_index import search_local

    local: list = []
    try:
        local = search_local(
            query,
            config.MAX_SEARCH_RESULTS,
            index_path=config.LOCAL_INDEX_PATH,
            date_before=config.LOCAL_INDEX_CUTOFF,
        )
        logs.append(f"[search_papers] Local index: {len(local)} bài báo")
    except Exception as exc:  # local index must never break the API path
        logs.append(f"[search_papers] Local index lỗi, bỏ qua: {exc}")

    try:
        api = arxiv_search(query=query, max_results=config.MAX_SEARCH_RESULTS)
    except Exception as exc:
        if not local:
            raise
        logs.append(f"[search_papers] ArXiv API lỗi, chỉ dùng local index: {exc}")
        return local

    seen = {paper.paper_id for paper in local}
    merged = list(local)
    for paper in api:
        if paper.paper_id not in seen:
            seen.add(paper.paper_id)
            merged.append(paper)
    return merged


def search_papers_node(state: ResearchState) -> Dict[str, Any]:
    """Queries ArXiv API using search_query and updates search_results."""
    started = time.perf_counter()
    query = state.get("search_query") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)
    
    logs = [f"[search_papers] Lần {retry_count + 1}: '{query}'"]

    arxiv_started = time.perf_counter()
    try:
        if _local_index_ready():
            results = _search_with_local(query, logs)
        else:
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
