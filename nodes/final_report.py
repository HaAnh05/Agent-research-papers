from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict
from state import ResearchState
from config import config
from prompts.final_report_prompt import FINAL_REPORT_PROMPT
from text_utils import detect_response_language
from tools.llm_provider import get_llm, extract_text_from_response


def final_report_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesizes all gathered information into a publication-grade Research Report."""
    user_query = state.get("user_query") or "Research Analysis"
    papers = state.get("selected_papers", [])
    benchmark_matrix = state.get("benchmark_matrix")
    language_mode = detect_response_language(user_query)
    logs = [f"[final_report] Tổng hợp báo cáo cho '{user_query}'"]

    # Format detailed breakdown for each paper
    breakdowns = []
    for idx, p in enumerate(papers, 1):
        notes = p.notes
        github_info = (
            f"- Repository: [{p.github_repos[0].name}]({p.github_repos[0].url}) (⭐ {p.github_repos[0].stars}, {p.github_repos[0].framework})"
            if p.github_repos
            else "- Repository: No public code repository found."
        )

        block = (
            f"### {idx}. {p.title}\n"
            f"- **ArXiv ID**: {p.arxiv_id or 'N/A'} | **Published**: {p.published or 'N/A'}\n"
            f"- **Authors**: {', '.join(p.authors) if p.authors else 'N/A'}\n"
            f"- **Problem**: {notes.problem if notes else 'N/A'}\n"
            f"- **Methodology**: {notes.method if notes else 'N/A'}\n"
            f"- **Results & Metrics**: {notes.result if notes else 'N/A'}\n"
            f"- **Limitations**: {notes.limitation if notes else 'N/A'}\n"
            f"{github_info}\n\n"
            f"```bibtex\n{p.bibtex or ''}\n```"
        )
        breakdowns.append(block)

    benchmark_section = (
        f"=== BENCHMARK COMPARISON MATRIX & TRADE-OFFS ===\n{benchmark_matrix}"
        if benchmark_matrix
        else ""
    )

    prompt = FINAL_REPORT_PROMPT.format(
        user_query=user_query,
        language_mode=language_mode,
        detailed_papers_breakdown="\n\n".join(breakdowns),
        optional_benchmark_matrix_section=benchmark_section,
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        report_text = extract_text_from_response(response.content)

        # Save report to disk
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in user_query[:30] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        report_path = config.REPORTS_DIR / f"report_{safe_title}_{timestamp}.md"
        report_path.write_text(report_text, encoding="utf-8")

        logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")

        return {
            "final_report": report_text,
            "status": "success",
            "trace_logs": logs
        }
    except Exception as exc:
        err_msg = f"Lỗi tạo báo cáo tổng hợp: {exc}"
        logs.append(f"❌ [final_report] {err_msg}")
        return {
            "final_report": f"# Error Generating Final Report\n\n{err_msg}",
            "status": "error",
            "error_message": err_msg,
            "trace_logs": logs,
            "error_logs": [err_msg]
        }
