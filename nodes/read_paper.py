from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Tuple

from config import config
from state import PaperItem, ResearchState
from tools.arxiv_search import _extract_arxiv_id, fetch_arxiv_metadata_batch
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


def _read_pdf_one(paper: PaperItem) -> Tuple[PaperItem, int, str | None]:
    """Extract one PDF; return ``(paper, durationMs, error_type)``."""

    result = copy.deepcopy(paper)
    started = time.perf_counter()
    try:
        processed = sync_extract_pdf_content(result)
    except Exception as exc:
        return result, max(0, int((time.perf_counter() - started) * 1000)), type(exc).__name__
    return processed, max(0, int((time.perf_counter() - started) * 1000)), None


def _apply_arxiv_metadata(processed: PaperItem, metadata: Dict[str, Any] | None) -> None:
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


def read_paper_node(state: ResearchState) -> Dict[str, Any]:
    """Read PDFs concurrently; resolve direct ArXiv metadata without blocking.

    Metadata for every paper that needs it is requested ONCE at node start (a
    batched ``id_list`` call).  PDF extraction runs for all papers in parallel.
    After the PDFs finish the node waits at most ``ARXIV_METADATA_GRACE_SEC``
    for the metadata; otherwise it continues on the metadata-failure path with
    ``arxiv.status = "timeout"``.  The background request is never joined, so
    a late success still fills the metadata cache for the next run.
    """

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

    worker_count = max(1, min(int(config.READ_PAPER_MAX_WORKERS), len(target_papers)))

    metadata_ids = [paper.arxiv_id or "" for paper in target_papers if _needs_arxiv_metadata(paper)]
    metadata_timings: Dict[str, Dict[str, Any]] = {}
    metadata_executor: ThreadPoolExecutor | None = None
    metadata_future = None
    if metadata_ids:
        metadata_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arxiv-meta")
        metadata_future = metadata_executor.submit(
            fetch_arxiv_metadata_batch, metadata_ids, timing=metadata_timings
        )

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # map preserves the source order even when PDF work ends at
            # different times.
            pdf_results = list(executor.map(_read_pdf_one, target_papers))

        metadata_by_id: Dict[str, Dict[str, Any] | None] = {}
        metadata_timed_out = False
        if metadata_future is not None:
            try:
                metadata_by_id = metadata_future.result(timeout=max(0.0, float(config.ARXIV_METADATA_GRACE_SEC)))
            except FutureTimeoutError:
                metadata_timed_out = True
            except Exception:
                metadata_by_id = {}
    finally:
        if metadata_executor is not None:
            # Do not join the background request; it may still cache a late
            # success after this node returns.
            metadata_executor.shutdown(wait=False)
    metadata_wait_ms = max(0, int((time.perf_counter() - started) * 1000))

    completed: List[Tuple[PaperItem, Dict[str, Any], str | None]] = []
    for original, (processed, pdf_ms, error_type) in zip(target_papers, pdf_results):
        metadata_timing: Dict[str, Any] = {}
        if _needs_arxiv_metadata(original):
            clean_id = _extract_arxiv_id(original.arxiv_id or "")
            if metadata_timed_out:
                metadata_timing = {"durationMs": metadata_wait_ms, "cache": "miss", "status": "timeout"}
            else:
                metadata_timing = dict(metadata_timings.get(clean_id) or {"durationMs": 0, "cache": "miss", "status": "invalid_id"})
        if error_type:
            completed.append((processed, {"durationMs": pdf_ms, "metadata": metadata_timing}, error_type))
            continue
        if _needs_arxiv_metadata(original):
            _apply_arxiv_metadata(processed, metadata_by_id.get(_extract_arxiv_id(original.arxiv_id or "")))
        processed.processing_metadata = dict(processed.processing_metadata or {})
        if metadata_timing:
            processed.processing_metadata["arxiv"] = metadata_timing
        completed.append((processed, {
            "durationMs": pdf_ms,
            "pdf": processed.processing_metadata.get("pdf", {}),
            "arxiv": metadata_timing,
        }, None))

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
        "workerCount": worker_count,
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
