from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from state import ResearchState, PaperItem
from tools.pdf_parser import sync_extract_pdf_content


def read_paper_node(state: ResearchState) -> Dict[str, Any]:
    """Reads PDF files / downloads arXiv papers and parses sections in parallel using ThreadPool."""
    target_papers = state.get("selected_papers", [])
    logs = [f"[read_paper] Đọc {len(target_papers)} bài báo"]

    if not target_papers:
        return {
            "status": "error",
            "error_message": "Không có bài báo nào trong danh sách được chọn.",
            "error_logs": ["No selected papers to read."],
            "trace_logs": logs + ["❌ Danh sách bài báo trống."]
        }

    valid_papers: List[PaperItem] = []
    error_messages: List[str] = []

    # Execute PDF extraction concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(target_papers), 4)) as executor:
        future_to_paper = {executor.submit(sync_extract_pdf_content, paper): paper for paper in target_papers}
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                processed_paper = future.result()
                valid_papers.append(processed_paper)
                method_len = len(processed_paper.sections.get("methodology", ""))
                logs.append(f"[read_paper] {processed_paper.title[:45]}... (method {method_len} ký tự)")
            except Exception as exc:
                err = f"Lỗi đọc bài báo '{paper.title}': {str(exc)}"
                error_messages.append(err)
                logs.append(f"❌ {err}")

    if not valid_papers:
        return {
            "status": "error",
            "error_message": "Tất cả các bài báo đều gặp lỗi khi đọc/tải PDF.",
            "error_logs": error_messages,
            "trace_logs": logs
        }

    return {
        "selected_papers": valid_papers,
        "trace_logs": logs,
        "error_logs": error_messages
    }
