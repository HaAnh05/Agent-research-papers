from __future__ import annotations

from typing import Any, Dict
from state import ResearchState
from prompts.benchmark_prompt import BENCHMARK_MATRIX_PROMPT
from tools.llm_provider import get_llm, extract_text_from_response


def compare_benchmark_node(state: ResearchState) -> Dict[str, Any]:
    """Constructs a Benchmark Comparison Matrix comparing multiple papers based on their PMRL notes."""
    papers = state.get("selected_papers", [])
    logs = [f"[compare_benchmark] So sánh {len(papers)} bài báo"]

    if len(papers) < 2:
        logs.append("[compare_benchmark] Chỉ có 1 bài báo, bỏ qua benchmark.")
        return {
            "benchmark_matrix": None,
            "trace_logs": logs
        }

    # Prepare compact summaries from structured PMRLNotes (strictly token-efficient)
    summaries = []
    for idx, p in enumerate(papers, 1):
        notes = p.notes
        github_url = p.github_repos[0].url if p.github_repos else "N/A"
        github_stars = f"(⭐ {p.github_repos[0].stars})" if p.github_repos else ""
        
        summary_block = (
            f"### Paper {idx}: {p.title} (ArXiv: {p.arxiv_id or 'N/A'})\n"
            f"- **Authors**: {', '.join(p.authors[:4]) if p.authors else 'N/A'}\n"
            f"- **Method / Architecture**: {notes.method if notes else p.summary[:250]}\n"
            f"- **Datasets & Benchmark Results**: {notes.result if notes else 'N/A'}\n"
            f"- **Key Limitations**: {notes.limitation if notes else 'N/A'}\n"
            f"- **GitHub**: {github_url} {github_stars}"
        )
        summaries.append(summary_block)

    prompt = BENCHMARK_MATRIX_PROMPT.format(
        papers_pmrl_summary="\n\n".join(summaries),
        paper_1_name=papers[0].title[:30],
        paper_2_name=papers[1].title[:30] if len(papers) > 1 else "",
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        matrix_content = extract_text_from_response(response.content)
        logs.append("[compare_benchmark] Đã có benchmark matrix.")
        return {
            "benchmark_matrix": matrix_content,
            "trace_logs": logs
        }
    except Exception as exc:
        err_msg = f"Lỗi tạo bảng so sánh: {exc}"
        logs.append(f"⚠️ [compare_benchmark] {err_msg}")
        return {
            "benchmark_matrix": f"*(Không thể tạo ma trận so sánh tự động: {exc})*",
            "trace_logs": logs,
            "error_logs": [err_msg]
        }
