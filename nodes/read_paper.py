from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from state import PaperItem, ResearchState
from tools.arxiv_search import fetch_arxiv_metadata
from tools.pdf_parser import sync_extract_pdf_content


def _needs_arxiv_metadata(paper: PaperItem) -> bool:
    if paper.source_type != "arxiv" or not paper.arxiv_id:
        return False
    title = (paper.title or "").strip().lower()
    return (
        not paper.summary.strip()
        or not paper.authors
        or not paper.published.strip()
        or not paper.subjects
        or paper.metadata_status != "complete"
        or title.startswith("arxiv ")
        or title.startswith("paper ")
    )


def _read_one(paper: PaperItem) -> Tuple[PaperItem, Dict[str, Any], str | None]:
    """Read one PDF while resolving direct-input metadata in parallel."""

    result = copy.deepcopy(paper)
    metadata_timing: Dict[str, Any] = {}
    metadata_future = None
    worker_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        pdf_future = executor.submit(sync_extract_pdf_content, result)
        if _needs_arxiv_metadata(result):
            metadata_future = executor.submit(fetch_arxiv_metadata, result.arxiv_id or "", timing=metadata_timing)
        try:
            processed = pdf_future.result()
        except Exception as exc:
            return result, {"durationMs": max(0, int((time.perf_counter() - worker_started) * 1000)), "metadata": metadata_timing}, type(exc).__name__

        if metadata_future is not None:
            try:
                metadata = metadata_future.result()
            except Exception:
                metadata = None
            if metadata:
                # The direct router intentionally starts with a safe label;
                # replace it only with fields returned by ArXiv metadata.
                processed.title = str(metadata.get("title") or processed.title)
                processed.summary = str(metadata.get("summary") or processed.summary)
                processed.authors = list(metadata.get("authors") or processed.authors)
                processed.published = str(metadata.get("published") or processed.published)
                processed.subjects = list(metadata.get("subjects") or processed.subjects)
                processed.arxiv_abs_url = str(metadata.get("abs_url") or processed.arxiv_abs_url or "") or None
                processed.arxiv_html_url = str(metadata.get("html_url") or processed.arxiv_html_url or "") or None
                processed.pdf_url = metadata.get("pdf_url") or processed.pdf_url
                processed.metadata_status = str(metadata.get("metadata_status") or "missing")
            elif processed.metadata_status == "unknown" and processed.source_type == "arxiv":
                # The safe ID label remains usable, but the UI must show that
                # official metadata was unavailable instead of implying it is
                # the paper title.
                processed.metadata_status = "unknown"
        processed.processing_metadata = dict(processed.processing_metadata or {})
        if metadata_timing:
            processed.processing_metadata["arxiv"] = metadata_timing
        timing = {
            "durationMs": max(0, int((time.perf_counter() - worker_started) * 1000)),
            "pdf": processed.processing_metadata.get("pdf", {}),
            "arxiv": metadata_timing,
        }
        return processed, timing, None


def read_paper_node(state: ResearchState) -> Dict[str, Any]:
    """Read PDFs in input order and resolve direct ArXiv metadata concurrently."""

    target_papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[read_paper] Đọc {len(target_papers)} bài báo"]

    if not target_papers:
        return {
            "status": "error",
            "error_message": "Không có bài báo nào trong danh sách được chọn.",
            "error_logs": ["No selected papers to read."],
            "trace_logs": logs + ["❌ Danh sách bài báo trống."],
            "node_timings": {"read_paper": {"durationMs": 0, "workerCount": 0}},
        }

    with ThreadPoolExecutor(max_workers=min(4, len(target_papers))) as executor:
        # map preserves the source order even when metadata or PDF work ends
        # at different times.
        completed = list(executor.map(_read_one, target_papers))

    valid_papers: List[PaperItem] = []
    error_messages: List[str] = []
    paper_timings: Dict[str, Any] = {}
    for original, (processed, timing, error_type) in zip(target_papers, completed):
        paper_timings[original.paper_id] = timing
        if error_type:
            err = f"Lỗi đọc bài báo '{original.title}': {error_type}"
            error_messages.append(err)
            logs.append(f"❌ {err}")
            continue
        valid_papers.append(processed)
        method_len = len(processed.sections.get("methodology", ""))
        logs.append(f"[read_paper] {processed.title[:45]}... (method {method_len} ký tự)")
        coverage = processed.source_quality.get("coverageStatus", processed.source_quality.get("coverage", "unknown"))
        if coverage != "complete":
            warning = processed.source_quality.get("warning") or "Source extraction is incomplete or unknown."
            logs.append(f"⚠️ [read_paper] {processed.paper_id}: {warning}")

    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "workerCount": min(4, len(target_papers)),
        "papers": paper_timings,
    }
    has_incomplete_source = any(
        paper.source_quality.get("coverageStatus", paper.source_quality.get("coverage", "unknown")) != "complete"
        for paper in valid_papers
    )
    if not valid_papers:
        return {
            "status": "error",
            "error_message": "Tất cả các bài báo đều gặp lỗi khi đọc/tải PDF.",
            "error_logs": error_messages,
            "trace_logs": logs,
            "node_timings": {"read_paper": timing},
        }

    return {
        "selected_papers": valid_papers,
        "status": "degraded" if has_incomplete_source else "running",
        "trace_logs": logs,
        "error_logs": error_messages,
        "node_timings": {"read_paper": timing},
    }
