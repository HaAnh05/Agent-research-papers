from __future__ import annotations

import time
from typing import Any, Dict

from state import ResearchState
from prompts.direct_answer_prompt import DIRECT_ANSWER_PROMPT
from tools.llm_provider import extract_text_from_response, get_llm


def direct_answer_node(state: ResearchState) -> Dict[str, Any]:
    """Answer a general or context-bound follow-up without paper retrieval."""
    started = time.perf_counter()

    user_query = (state.get("user_query") or "").strip()
    context = (state.get("conversation_context") or "").strip()
    context_for_prompt = context[:12000] if context else "(No previous research context was provided.)"
    prompt = DIRECT_ANSWER_PROMPT.format(
        user_query=user_query,
        conversation_context=context_for_prompt,
    )
    logs = [
        "[direct_answer] Trả lời trực tiếp, không chạy pipeline tìm kiếm",
        "[direct_answer] Ràng buộc theo context phiên trước" if context else "[direct_answer] Giải thích khái niệm tổng quát",
    ]

    llm_started = time.perf_counter()
    try:
        response = get_llm().invoke(prompt)
        answer = extract_text_from_response(getattr(response, "content", response))
        if not answer:
            raise ValueError("empty direct answer")
        return {
            "assistant_answer": answer,
            "status": "success",
            "node_timings": {"direct_answer": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "llmMs": max(0, int((time.perf_counter() - llm_started) * 1000)),
            }},
            "trace_logs": logs,
        }
    except Exception:
        # Keep provider details (which can contain sensitive request metadata)
        # out of state/logs.  The API can present this stable public error.
        return {
            "assistant_answer": "Chưa thể tạo câu trả lời trực tiếp lúc này. Vui lòng kiểm tra cấu hình LLM và thử lại.",
            "status": "error",
            "error_message": "Direct answer provider unavailable.",
            "node_timings": {"direct_answer": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "llmMs": max(0, int((time.perf_counter() - llm_started) * 1000)),
            }},
            "error_logs": ["Direct answer provider unavailable."],
            "trace_logs": logs + ["[direct_answer] Provider không khả dụng"],
        }


__all__ = ["direct_answer_node"]
