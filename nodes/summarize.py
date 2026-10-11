from __future__ import annotations

import time
from typing import Any, Dict, List

from config import config
from nodes.final_report import (
    _write_overview,
    build_answer_summary,
    merge_branch_results,
    overview_cache_key,
    overview_from_json,
    render_overview_sections,
    report_title,
)
from state import ResearchState
from text_utils import detect_response_language
from tools.cache_manager import load_json_cache, save_json_cache
from tools.llm_provider import get_llm


def summarize_node(state: ResearchState) -> Dict[str, Any]:
    """Quick summary: the overview LLM call plus ``build_answer_summary`` (code).

    Runs right after ``write_notes``, inside the notes branch, so it does not wait
    for ``citation_graph`` or ``web_enrich``.  It reads the PMRL notes only.  The
    prior works, graph size, BibTeX count and report id are missing from the
    summary it returns; ``final_report`` and the API patch them in when that data
    is ready (``nodes.final_report.patch_answer_summary``).

    Returns ``overview`` (the overview JSON, ``{}`` when the call failed, so
    ``final_report`` does not repeat it) and ``answer_summary``.
    """

    started = time.perf_counter()
    user_query = state.get("user_query") or "Research Analysis"
    papers = merge_branch_results(list(state.get("selected_papers", []) or []), state.get("paper_notes"), None)
    if not papers:
        return {
            "trace_logs": ["[summarize] Không có bài để tóm tắt"],
            "node_timings": {"summarize": {"durationMs": 0, "llmMs": 0, "llmCalls": []}},
        }

    language_mode = detect_response_language(user_query)
    title = report_title(user_query, papers)
    logs: List[str] = []
    llm_calls: List[Dict[str, Any]] = []
    llm_ms = 0
    cache_hit = False
    overview = None
    _, cache_path = overview_cache_key(user_query, title, language_mode, papers)
    cached = load_json_cache(cache_path)
    if isinstance(cached, dict):
        overview = overview_from_json(cached.get("overview"))
        cache_hit = overview is not None
    if cache_hit:
        logs.append("[summarize] Overview cache hit — reused overview")
    else:
        try:
            overview, call_timing = _write_overview(get_llm(), user_query, title, language_mode, papers)
            llm_ms = int(call_timing.get("llmMs") or 0)
            llm_calls = list(call_timing.get("llmCalls") or [])
            missing = [
                key for key in ("summary", "conclusion")
                if not render_overview_sections(overview, papers, language_mode).get(key)
            ]
            if missing:
                logs.append(f"[summarize] Overview degraded (missing sections: {', '.join(missing)})")
            else:
                # A degraded overview is retryable and must not become a cache hit.
                save_json_cache(cache_path, {"overview": overview.model_dump(mode="json")})
        except Exception as exc:
            overview = None
            logs.append(f"[summarize] Overview degraded ({type(exc).__name__})")

    summary = build_answer_summary(
        papers, overview, language_mode,
        citation_graph=state.get("citation_graph"), node_timings=state.get("node_timings"),
        report_id=None, final_ms=0,
    )
    logs.append(f"[summarize] Tóm tắt nhanh cho {len(papers)} bài")
    timing: Dict[str, Any] = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": llm_ms,
        "llmCalls": llm_calls,
    }
    if cache_hit:
        timing["cache"] = True
    return {
        "overview": overview.model_dump(mode="json") if overview is not None else {},
        "answer_summary": summary,
        "trace_logs": logs,
        "node_timings": {"summarize": timing},
    }
