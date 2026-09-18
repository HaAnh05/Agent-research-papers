from __future__ import annotations

import time
from typing import Any, Dict, List
from state import ResearchState, EvalSearchOutput, PaperItem
from config import config
from prompts.eval_search_prompt import EVAL_SEARCH_PROMPT
from tools.llm_provider import get_llm, invoke_structured_output


def eval_search_node(state: ResearchState) -> Dict[str, Any]:
    """Evaluates search results against the user query, selects best papers or flags for refine."""
    started = time.perf_counter()
    user_query = state.get("user_query", "")
    candidates = state.get("search_results", [])
    retry_count = state.get("retry_count", 0)

    logs = [f"[eval_search] Đánh giá {len(candidates)} bài báo"]

    if not candidates:
        logs.append("[eval_search] Không có kết quả, cần tinh chỉnh query.")
        result: Dict[str, Any] = {
            "eval_passed": False,
            "eval_feedback": "Search returned 0 results from ArXiv. Need broader or different technical keywords.",
            "selected_papers": [],
            "trace_logs": logs,
            "node_timings": {"eval_search": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0}},
        }
        if retry_count >= config.MAX_RETRIES:
            message = (
                f"ArXiv không trả về bài báo nào sau {retry_count + 1} lượt tìm kiếm. "
                "Hãy thử 2-5 từ khóa kỹ thuật bằng tiếng Anh hoặc nhập ArXiv ID trực tiếp."
            )
            result.update({
                "status": "error",
                "error_message": message,
                "error_logs": [message],
            })
        return result

    # Format candidates for LLM prompt
    candidates_text = "\n\n".join([
        f"[{idx}] Title: {p.title}\n"
        f"    ArXiv ID: {p.arxiv_id}\n"
        f"    Published: {p.published}\n"
        f"    Authors: {', '.join(p.authors[:3])}\n"
        f"    Summary: {p.summary[:400]}..."
        for idx, p in enumerate(candidates)
    ])

    prompt = EVAL_SEARCH_PROMPT.format(
        user_query=user_query,
        total_results=len(candidates),
        candidate_papers_text=candidates_text,
        top_k=config.TOP_K_PAPERS,
    )

    llm_started = time.perf_counter()
    try:
        llm = get_llm()
        eval_result: EvalSearchOutput = invoke_structured_output(
            prompt,
            EvalSearchOutput,
            llm=llm,
            provider=config.DEFAULT_PROVIDER,
        )

        # Select the chosen papers
        selected_papers: List[PaperItem] = []
        if eval_result.selected_indices:
            for idx in eval_result.selected_indices:
                if 0 <= idx < len(candidates):
                    selected_papers.append(candidates[idx])
        
        # Fallback if no valid indices returned but passed
        if not selected_papers and candidates:
            selected_papers = candidates[:config.TOP_K_PAPERS]

        logs.append(
            f"[eval_search] passed={eval_result.passed}, score={eval_result.score:.2f}, chọn {len(selected_papers)} bài"
        )

        return {
            "eval_passed": eval_result.passed,
            "eval_feedback": eval_result.feedback,
            "selected_papers": selected_papers,
            "node_timings": {"eval_search": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "llmMs": max(0, int((time.perf_counter() - llm_started) * 1000)),
            }},
            "trace_logs": logs
        }

    except Exception as exc:
        logs.append(f"[eval_search] LLM lỗi ({exc}), dùng top {config.TOP_K_PAPERS} bài")
        return {
            "eval_passed": True,
            "eval_feedback": "Automatic fallback due to evaluation error.",
            "selected_papers": candidates[:config.TOP_K_PAPERS],
            "node_timings": {"eval_search": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "llmMs": max(0, int((time.perf_counter() - llm_started) * 1000)),
            }},
            "trace_logs": logs
        }
