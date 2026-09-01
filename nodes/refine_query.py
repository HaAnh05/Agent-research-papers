from __future__ import annotations

from typing import Any, Dict
from state import ResearchState
from prompts.refine_query_prompt import REFINE_QUERY_PROMPT
from tools.llm_provider import get_llm, extract_text_from_response


def refine_query_node(state: ResearchState) -> Dict[str, Any]:
    """Optimizes the search query based on evaluator feedback and increments retry_count."""
    user_query = state.get("user_query", "")
    current_search_query = state.get("search_query") or user_query
    eval_feedback = state.get("eval_feedback", "Low relevance or empty results.")
    retry_count = state.get("retry_count", 0)

    logs = [f"[refine_query] Bắt đầu tối ưu lại từ khóa tìm kiếm (Lần {retry_count + 1})..."]

    prompt = REFINE_QUERY_PROMPT.format(
        user_query=user_query,
        current_search_query=current_search_query,
        eval_feedback=eval_feedback,
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_text = extract_text_from_response(response.content)
        new_query = raw_text.replace('"', '').replace("'", "").strip()
        
        logs.append(f"[refine_query] Query cũ: '{current_search_query}' -> Query mới: '{new_query}'")
        return {
            "search_query": new_query,
            "retry_count": retry_count + 1,
            "trace_logs": logs
        }
    except Exception as exc:
        # Simple heuristic fallback
        new_query = f"{user_query} deep learning"
        logs.append(f"[refine_query] LLM refine lỗi ({exc}) -> Fallback query: '{new_query}'")
        return {
            "search_query": new_query,
            "retry_count": retry_count + 1,
            "trace_logs": logs
        }
