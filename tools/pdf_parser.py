from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests

from config import config
from state import PaperItem, PaperReference, PaperTable, RepoLink
from tools.cache_manager import (
    canonicalize_paper_key,
    get_cached_pdf_path,
    get_cached_text_data,
    save_cached_text_data,
)
from tools.html_extractor import HTMLExtraction
from tools.html_source import fetch_html_extraction
from tools.rate_limit import host_slot

_EXPERIMENT_HEADING = (
    r"(?:experiments?(?:\s+(?:and|&)\s+results?|\s+on[^\r\n]*)?"
    r"|experimental(?:\s+(?:evaluation|results?|setup))?"
    r"|evaluations?(?:\s+and\s+analysis)?"
    r"|implementation\s+and\s+evaluation"
    r"|quantitative\s+evaluation|results?|benchmarks?)"
)

SECTION_PATTERNS = {
    "abstract": r"(?i)(?:abstract|tóm tắt)\s*[\r\n]+(.*?)(?=(?:(?:1|i)\.?\s+)?introduction|giới thiệu|\Z)",
    "methodology": rf"(?i)(?:(?:\d+|[ivxlcdm]+)[.)]?\s+)?(?:method|methodology|proposed method|proposed approach|architecture|model|approach|framework|system|phương pháp)\s*[\r\n]+(.*?)(?=(?:(?:\d+|[ivxlcdm]+)[.)]?\s+)?{_EXPERIMENT_HEADING}|thực nghiệm|\Z)",
    "experiments": rf"(?i)(?:(?:\d+|[ivxlcdm]+)[.)]?\s+)?(?:{_EXPERIMENT_HEADING}|kết quả)\s*[\r\n]+(.*?)(?=(?:(?:\d+|[ivxlcdm]+)[.)]?\s+)?(?:limitations?|discussion|related work|conclusion)|\Z)",
    "limitations": r"(?i)(?:(?:\d+|[ivxlcdm]+)[.)]?\s+)?(?:limitations?|discussion|future work|giới hạn)\s*[\r\n]+(.*?)(?=(?:references|acknowledgments|tài liệu tham khảo)|\Z)",
}

MAX_REMOTE_PDF_BYTES = 50 * 1024 * 1024
MAX_ARXIV_REDIRECTS = 3
REQUIRED_COVERAGE_SECTIONS = ("abstract", "methodology", "experiments")
# Bump when the stored extraction changes shape or quality: older text-cache
# entries (sections, source_quality and the structured fields) are then ignored.
# The cached PDF files are still reused.
EXTRACTION_CACHE_VERSION = 2


def _validated_arxiv_pdf_url(value: str) -> str:
    """Return a canonical HTTPS ArXiv PDF URL or reject the source."""

    parsed = urlparse(str(value or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme not in {"http", "https"}
        or host not in {"arxiv.org", "www.arxiv.org"}
        or parsed.username
        or parsed.password
        or port not in {None, 80, 443}
        or not re.fullmatch(r"/pdf/\d{4}\.\d{4,5}(?:v\d+)?(?:\.pdf)?/?", parsed.path, flags=re.IGNORECASE)
    ):
        raise ValueError("Remote PDF sources must be public ArXiv PDF URLs")
    return urlunparse(("https", "arxiv.org", parsed.path, "", "", ""))


def _save_pdf_response(response: Any, partial_path: Path) -> None:
    """Stream one PDF response to ``partial_path`` with size and magic checks."""

    response.raise_for_status()
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_REMOTE_PDF_BYTES:
        raise ValueError("Remote PDF exceeds the download size limit")

    total = 0
    prefix = b""
    with partial_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_REMOTE_PDF_BYTES:
                raise ValueError("Remote PDF exceeds the download size limit")
            if len(prefix) < 5:
                prefix += chunk[: 5 - len(prefix)]
            handle.write(chunk)
    if not prefix.startswith(b"%PDF-"):
        raise ValueError("ArXiv response is not a PDF document")


def _download_pdf(pdf_url: str, output_path: Path) -> Path:
    """Download an allowlisted ArXiv PDF with redirects and size bounded.

    Each request (including redirect hops) takes a slot from the shared
    arxiv.org limiter, so downloads and HTML reads respect ``Crawl-delay: 15``.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    current_url = _validated_arxiv_pdf_url(pdf_url)
    try:
        for redirect_count in range(MAX_ARXIV_REDIRECTS + 1):
            next_url = ""
            with host_slot(urlparse(current_url).hostname or ""):
                response = requests.get(
                    current_url,
                    headers={"User-Agent": config.ARXIV_USER_AGENT},
                    timeout=20,
                    stream=True,
                    allow_redirects=False,
                )
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count >= MAX_ARXIV_REDIRECTS:
                            raise ValueError("ArXiv PDF redirect limit exceeded")
                        next_url = _validated_arxiv_pdf_url(urljoin(current_url, location))
                    else:
                        _save_pdf_response(response, partial_path)
                        partial_path.replace(output_path)
                        return output_path
                finally:
                    response.close()
            current_url = next_url
        raise ValueError("ArXiv PDF redirect limit exceeded")
    finally:
        if partial_path.exists():
            partial_path.unlink()


def extract_ordered_blocks_from_pdf(pdf_path: Path, max_pages: int = config.PDF_MAX_PAGES) -> str:
    """Extract text from PDF preserving two-column academic reading order."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page_texts = []

        for page_idx in range(min(len(doc), max_pages)):
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            # Filter regular text blocks (type 0)
            text_blocks = [b for b in blocks if b[6] == 0]

            # Detect two-column layout by splitting across middle X coordinate
            mid_x = page.rect.width / 2.0
            left_col = sorted([b for b in text_blocks if b[0] < mid_x], key=lambda b: b[1])
            right_col = sorted([b for b in text_blocks if b[0] >= mid_x], key=lambda b: b[1])

            ordered_page = "\n\n".join(b[4].strip() for b in left_col + right_col if b[4].strip())
            page_texts.append(ordered_page)

        return "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    except ImportError:
        # Fallback to standard pypdf
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_to_read = min(len(reader.pages), max_pages)
        parts = [reader.pages[i].extract_text() or "" for i in range(pages_to_read)]
        return "\n\n".join(parts)


def _coverage_quality(
    sections: Dict[str, str],
    *,
    heading: str = "unknown",
    parser_status: str = "unknown",
    source: str = "pdf",
    warning: str | None = None,
) -> Dict[str, Any]:
    missing = [section for section in REQUIRED_COVERAGE_SECTIONS if not sections.get(section, "").strip()]
    quality: Dict[str, Any] = {
        "heading": heading,
        "parserStatus": parser_status,
        "source": source,
        "sourceFormat": source,
        "coverage": "complete" if not missing and parser_status == "success" else "incomplete",
        "coverageStatus": "complete" if not missing and parser_status == "success" else "incomplete",
        "missingSections": missing,
        "codeStatus": "unknown",
        "codeUrls": [],
    }
    if warning:
        quality["warning"] = warning
    return quality


def _coverage_incomplete(sections: Dict[str, str], quality: Dict[str, Any]) -> bool:
    return (
        quality.get("parserStatus") != "success"
        or quality.get("coverageStatus", quality.get("coverage")) != "complete"
        or any(not sections.get(section, "").strip() for section in REQUIRED_COVERAGE_SECTIONS)
    )


def parse_structured_sections(raw_text: str, *, metadata: Dict[str, Any] | None = None) -> Dict[str, str]:
    """Parse sections and record whether heading parsing fell back."""
    structured: Dict[str, str] = {}
    matched_sections: set[str] = set()
    for section, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            matched_sections.add(section)
            text = match.group(1).strip()
            max_chars = 2000 if section in ["abstract", "limitations"] else 4000
            structured[section] = " ".join(text.split())[:max_chars]
        else:
            structured[section] = ""

    # Fallback if section headers were not matched
    if not structured.get("methodology") and not structured.get("experiments"):
        structured["abstract"] = raw_text[:1500]
        structured["methodology"] = raw_text[1500:6500]
        structured["experiments"] = raw_text[6500:11500]
        structured["limitations"] = raw_text[11500:14000]
        if metadata is not None:
            metadata.update(_coverage_quality(
                structured,
                heading="fallback",
                parser_status="degraded",
                source="pdf",
                warning="PDF heading extraction fell back to positional text; coverage is incomplete until verified by another source.",
            ))
    elif metadata is not None:
        metadata.update(_coverage_quality(
            structured,
            heading="heading" if matched_sections else "unknown",
            parser_status="success" if matched_sections else "unknown",
            source="pdf",
        ))

    return structured


def _paper_cache_key(paper: PaperItem) -> str:
    """Use a distinct text/PDF cache for an explicitly versioned ArXiv input."""

    arxiv_id = str(paper.arxiv_id or "").strip()
    version = re.search(r"(\d{4}\.\d{4,5}v\d+)", arxiv_id)
    if version:
        return f"arxiv_{version.group(1)}"
    return canonicalize_paper_key(arxiv_id or paper.paper_id or paper.title)


def _unknown_cached_quality() -> Dict[str, Any]:
    return {
        "heading": "unknown",
        "parserStatus": "unknown",
        "source": "unknown",
        "sourceFormat": "unknown",
        "coverage": "unknown",
        "coverageStatus": "unknown",
        "missingSections": [],
        "codeStatus": "unknown",
        "codeUrls": [],
        "warning": "Cached extraction predates source quality metadata; coverage is unknown.",
    }


_SECTION_ORDER = ("abstract", "methodology", "setup", "experiments", "limitations")
_OPTIONAL_FILL_SECTIONS = (*REQUIRED_COVERAGE_SECTIONS, "limitations")


def _compose_text(sections: Dict[str, str]) -> str:
    return "\n\n".join(
        f"=== {key.upper()} ===\n{sections[key]}" for key in _SECTION_ORDER if sections.get(key)
    )


def _html_quality(extraction: HTMLExtraction, source: str, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Source-quality record for sections read from ar5iv / arxiv.org HTML."""

    missing = [key for key in REQUIRED_COVERAGE_SECTIONS if not extraction.sections.get(key, "").strip()]
    complete = not missing
    code_urls = extraction.code_urls
    return {
        "heading": "heading" if extraction.heading_seen else "unknown",
        "parserStatus": "success" if complete else "degraded",
        "source": source,
        "sourceFormat": "html",
        "coverage": "complete" if complete else "incomplete",
        "coverageStatus": "complete" if complete else "incomplete",
        "missingSections": missing,
        "codeStatus": "available" if code_urls else "not_found",
        "codeUrls": code_urls,
        "htmlStatus": "ok",
        "htmlSource": source,
        "htmlAttempts": list(attempts),
    }


def _merge_pdf_into_html(
    html_sections: Dict[str, str],
    html_quality: Dict[str, Any],
    pdf_sections: Dict[str, str],
    pdf_quality: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fill sections the HTML lacked from the PDF; HTML text always wins."""

    merged = dict(html_sections)
    filled: List[str] = []
    for key in _OPTIONAL_FILL_SECTIONS:
        if not merged.get(key, "").strip() and pdf_sections.get(key, "").strip():
            merged[key] = pdf_sections[key]
            filled.append(key)
    missing = [key for key in REQUIRED_COVERAGE_SECTIONS if not merged.get(key, "").strip()]
    complete = not missing and (not filled or pdf_quality.get("parserStatus") == "success")
    quality = dict(html_quality)
    source = "mixed" if filled else html_quality.get("source", "mixed")
    quality.update({
        "source": source,
        "sourceFormat": "mixed" if filled else "html",
        "parserStatus": "success" if complete else "degraded",
        "coverage": "complete" if complete else "incomplete",
        "coverageStatus": "complete" if complete else "incomplete",
        "missingSections": missing,
        "pdfFilledSections": filled,
    })
    if not complete:
        quality["warning"] = (
            "Extraction incomplete: the HTML and the PDF did not expose all key sections. "
            "Missing sections may be outside the readable source window; this is not evidence that the paper lacks results."
        )
    return merged, quality


def _structured_payload(paper: PaperItem) -> Dict[str, Any]:
    return {
        "tables": [item.model_dump(mode="json") for item in paper.tables],
        "references": [item.model_dump(mode="json") for item in paper.references],
        "repo_links": [item.model_dump(mode="json") for item in paper.repo_links],
    }


def _restore_structured(paper: PaperItem, cached: Dict[str, Any]) -> None:
    """Load tables/references/repo_links from a cache entry (missing keys -> empty)."""

    for name, model in (("tables", PaperTable), ("references", PaperReference), ("repo_links", RepoLink)):
        items = []
        for raw in cached.get(name) or []:
            try:
                items.append(model.model_validate(raw))
            except Exception:
                continue
        setattr(paper, name, items)


def sync_extract_pdf_content(paper: PaperItem) -> PaperItem:
    """Read a paper HTML-first (ar5iv, then arxiv.org/html) with the PDF as fallback.

    The PDF is downloaded and parsed only when no HTML source gave all of
    abstract, methodology and experiments (or when the paper is not on arXiv).
    ``source_quality["source"]`` records the winner: ``ar5iv``, ``arxiv_html``,
    ``pdf`` or ``mixed``.
    """

    started = time.perf_counter()
    paper_key = _paper_cache_key(paper)
    cache_started = time.perf_counter()
    cached_data = get_cached_text_data(paper_key)
    if not (isinstance(cached_data, dict) and cached_data.get("extraction_version") == EXTRACTION_CACHE_VERSION):
        cached_data = None  # older extraction format: do not reuse
    cached_sections = cached_data.get("sections") if isinstance(cached_data, dict) else None
    cached_quality_raw = cached_data.get("source_quality") if isinstance(cached_data, dict) else None
    cached_quality = dict(cached_quality_raw) if isinstance(cached_quality_raw, dict) else _unknown_cached_quality()
    cache_quality_known = isinstance(cached_quality_raw, dict)

    if isinstance(cached_sections, dict) and cached_sections:
        paper.sections = {str(key): str(value or "") for key, value in cached_sections.items()}
        paper.extracted_text = str(cached_data.get("extracted_text") or "")
        paper.local_pdf_path = cached_data.get("local_pdf_path")
        paper.source_quality = cached_quality
        _restore_structured(paper, cached_data)
        if cache_quality_known and not _coverage_incomplete(paper.sections, paper.source_quality):
            paper.processing_metadata = dict(paper.processing_metadata or {})
            paper.processing_metadata["pdf"] = {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "cacheMs": max(0, int((time.perf_counter() - cache_started) * 1000)),
                "cache": "hit",
            }
            return paper

    # 1. HTML first: ar5iv, then arxiv.org/html (paced, see tools.rate_limit).
    html_attempts: List[Dict[str, Any]] = []
    html_timing: Dict[str, Any] = {}
    extraction: HTMLExtraction | None = None
    html_source: str | None = None
    if paper.arxiv_id and paper.source_type != "local_pdf":
        extraction, html_source = fetch_html_extraction(paper.arxiv_id, html_attempts, html_timing)
    html_complete = extraction is not None and all(
        extraction.sections.get(key, "").strip() for key in REQUIRED_COVERAGE_SECTIONS
    )

    # 2. PDF only when HTML is missing or incomplete.
    pdf_path = get_cached_pdf_path(paper_key)
    download_started = time.perf_counter()
    download_kind = "skipped"
    pdf_error: Exception | None = None
    if not html_complete:
        download_kind = "cache" if pdf_path else "download"
        if not pdf_path or not pdf_path.exists():
            try:
                if paper.source_type == "local_pdf" and paper.local_pdf_path:
                    local_p = Path(paper.local_pdf_path)
                    if local_p.exists():
                        pdf_path = local_p
                        download_kind = "local"
                    else:
                        raise FileNotFoundError(f"Local PDF file not found: {paper.local_pdf_path}")
                elif paper.pdf_url:
                    pdf_path = _download_pdf(paper.pdf_url, config.PDF_CACHE_DIR / f"{paper_key}.pdf")
                elif paper.arxiv_id:
                    pdf_path = _download_pdf(
                        f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf", config.PDF_CACHE_DIR / f"{paper_key}.pdf"
                    )
                elif cached_sections:
                    pdf_path = None
                    download_kind = "cached_text"
                else:
                    raise ValueError(f"No PDF source available for paper: {paper.title}")
            except (OSError, ValueError, requests.RequestException) as exc:
                if extraction is None:
                    raise
                # Partial HTML is still better than nothing: keep it, flag it.
                pdf_path = None
                pdf_error = exc
                download_kind = "failed"
    download_ms = max(0, int((time.perf_counter() - download_started) * 1000))

    extract_started = time.perf_counter()
    raw_pdf_text = ""
    pdf_sections: Dict[str, str] = {}
    pdf_quality: Dict[str, Any] = {}
    have_pdf_text = False
    if not html_complete:
        if pdf_path is not None:
            raw_pdf_text = extract_ordered_blocks_from_pdf(pdf_path, max_pages=config.PDF_MAX_PAGES)
            pdf_sections = parse_structured_sections(raw_pdf_text, metadata=pdf_quality)
            have_pdf_text = True
        elif extraction is None:
            raw_pdf_text = paper.extracted_text or ""
            pdf_sections = dict(paper.sections or {})
            pdf_quality = dict(paper.source_quality or _unknown_cached_quality())
            pdf_quality.setdefault("coverageStatus", "unknown")
            pdf_quality.setdefault("sourceFormat", "unknown")
            have_pdf_text = True
    extract_ms = max(0, int((time.perf_counter() - extract_started) * 1000))

    if extraction is not None and html_source is not None:
        html_quality = _html_quality(extraction, html_source, html_attempts)
        if html_complete:
            sections, source_quality = dict(extraction.sections), html_quality
            text = _compose_text(sections)
        elif have_pdf_text:
            sections, source_quality = _merge_pdf_into_html(extraction.sections, html_quality, pdf_sections, pdf_quality)
            text = _compose_text(sections)
            if source_quality.get("source") == "mixed" and raw_pdf_text:
                text = f"{text}\n\n{raw_pdf_text}".strip()
        else:
            sections, source_quality = dict(extraction.sections), html_quality
            source_quality["warning"] = (
                "Extraction incomplete: the HTML did not expose all key sections and the PDF was unavailable. "
                "Missing sections must not be interpreted as absent from the paper."
            )
            text = _compose_text(sections)
        paper.tables = list(extraction.tables)
        paper.references = list(extraction.references)
        paper.repo_links = list(extraction.repo_links)
    else:
        sections = pdf_sections
        source_quality = dict(pdf_quality)
        text = raw_pdf_text
        paper.tables, paper.references, paper.repo_links = [], [], []
        if paper.arxiv_id and paper.source_type != "local_pdf":
            source_quality["htmlStatus"] = "unavailable"
            source_quality["htmlAttempts"] = list(html_attempts)
            if _coverage_incomplete(sections, source_quality):
                known = source_quality.get("coverageStatus") != "unknown"
                source_quality.update({
                    "coverageStatus": "incomplete" if known else "unknown",
                    "coverage": "incomplete" if known else "unknown",
                    "warning": (
                        "Extraction incomplete: ArXiv HTML (ar5iv and arxiv.org/html) was unavailable. "
                        "Missing sections must not be interpreted as absent from the paper."
                    ),
                })
        else:
            source_quality.setdefault("htmlStatus", "not_applicable")

    if pdf_error is not None:
        source_quality["pdfError"] = type(pdf_error).__name__

    paper.local_pdf_path = str(pdf_path) if pdf_path is not None else paper.local_pdf_path
    paper.extracted_text = text[:config.PDF_MAX_CHARS]
    paper.sections = sections
    paper.source_quality = source_quality or _unknown_cached_quality()
    paper.processing_metadata = dict(paper.processing_metadata or {})
    paper.processing_metadata["pdf"] = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "cacheMs": max(0, int((time.perf_counter() - cache_started) * 1000)),
        "downloadMs": download_ms,
        "extractMs": extract_ms,
        "htmlMs": int(html_timing.get("durationMs") or 0),
        "cache": download_kind,
    }

    save_cached_text_data(
        paper_key,
        {
            "extraction_version": EXTRACTION_CACHE_VERSION,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "local_pdf_path": paper.local_pdf_path,
            "sections": sections,
            "extracted_text": paper.extracted_text,
            "source_quality": paper.source_quality,
            **_structured_payload(paper),
        },
    )
    return paper
