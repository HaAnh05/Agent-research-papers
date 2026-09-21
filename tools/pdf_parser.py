from __future__ import annotations

from html.parser import HTMLParser
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests

from config import config
from state import PaperItem
from tools.cache_manager import (
    canonicalize_paper_key,
    get_cached_pdf_path,
    get_cached_text_data,
    save_cached_text_data,
)

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
MAX_REMOTE_HTML_BYTES = 20 * 1024 * 1024
REQUIRED_COVERAGE_SECTIONS = ("abstract", "methodology", "experiments")


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


def _validated_arxiv_html_url(arxiv_id: str) -> str:
    """Return the official versioned ArXiv HTML URL for a paper."""

    clean_id = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", str(arxiv_id or ""))
    if not clean_id:
        raise ValueError("HTML fallback requires a valid ArXiv identifier")
    return f"https://arxiv.org/html/{clean_id.group(1)}"


def _download_pdf(pdf_url: str, output_path: Path) -> Path:
    """Download an allowlisted ArXiv PDF with redirects and size bounded."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")
    current_url = _validated_arxiv_pdf_url(pdf_url)
    response = None
    try:
        for redirect_count in range(MAX_ARXIV_REDIRECTS + 1):
            response = requests.get(
                current_url,
                headers={"User-Agent": config.ARXIV_USER_AGENT},
                timeout=20,
                stream=True,
                allow_redirects=False,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("location")
            response.close()
            response = None
            if not location or redirect_count >= MAX_ARXIV_REDIRECTS:
                raise ValueError("ArXiv PDF redirect limit exceeded")
            current_url = _validated_arxiv_pdf_url(urljoin(current_url, location))

        if response is None:
            raise ValueError("ArXiv PDF download did not return a response")
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
        partial_path.replace(output_path)
        return output_path
    finally:
        if response is not None:
            response.close()
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


def _classify_html_heading(value: str) -> str | None:
    normalized = " ".join((value or "").lower().split())
    if not normalized:
        return None
    if re.search(r"\b(abstract|tóm tắt)\b", normalized):
        return "abstract"
    if re.search(r"\b(experiment|experimental|evaluation|benchmark|result|ablation|thực nghiệm|kết quả)\b", normalized):
        return "experiments"
    if re.search(r"\b(method|methodology|approach|architecture|model|framework|proposed|system|formulation|phương pháp)\b", normalized):
        return "methodology"
    if re.search(r"\b(limitation|discussion|future work|giới hạn)\b", normalized):
        return "limitations"
    # Many arXiv HTML papers use descriptive numbered headings such as
    # "III Globally Consistent 2DGS" instead of the word "Method".  Treat
    # those middle sections as methodology unless they are clearly prose or
    # end matter, so the coverage gate can still inspect the actual method.
    if re.match(r"^(?:[ivxlcdm]+|\d+)[.)]?\s+", normalized) and not re.search(
        r"\b(introduction|related work|conclusion|references|acknowledg|discussion)\b",
        normalized,
    ):
        return "methodology"
    return None


class _ArxivHTMLSectionParser(HTMLParser):
    """Extract coarse section text from the official arXiv HTML export.

    The HTML export has stable semantic classes but its exact tag nesting
    varies between papers.  This parser deliberately keeps only section text
    and ignores scripts/styles, equations' presentation markup, and links.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: List[str] = []
        self._ignored_depth = 0
        self._active_section: str | None = None
        self._abstract_depth: int | None = None
        self._heading_tag: str | None = None
        self._heading_parts: List[str] = []
        self._parts: Dict[str, List[str]] = {section: [] for section in (*REQUIRED_COVERAGE_SECTIONS, "limitations")}
        self.code_urls: List[str] = []
        self.heading_seen = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        self._stack.append(tag_name)
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag_name == "a":
            href = attributes.get("href", "").strip()
            if re.fullmatch(r"https://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", href):
                if href not in self.code_urls:
                    self.code_urls.append(href.rstrip("/"))
        if tag_name in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if "ltx_abstract" in classes or "ltx_abstract" in attributes.get("id", ""):
            self._active_section = "abstract"
            self._abstract_depth = len(self._stack)
            return
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"} or "ltx_title_section" in classes:
            self._heading_tag = tag_name
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if self._ignored_depth and tag_name in {"script", "style", "noscript", "svg"}:
            self._ignored_depth -= 1
        if self._heading_tag == tag_name:
            heading = " ".join(" ".join(self._heading_parts).split())
            section = _classify_html_heading(heading)
            if section:
                self._active_section = section
                self.heading_seen = True
            self._heading_tag = None
            self._heading_parts = []
        if self._stack:
            # HTMLParser can receive imperfect markup; remove through the
            # matching tag while keeping the parser usable for later sections.
            try:
                index = len(self._stack) - 1 - self._stack[::-1].index(tag_name)
                del self._stack[index:]
            except ValueError:
                self._stack.pop()
        if self._abstract_depth is not None and len(self._stack) < self._abstract_depth:
            self._abstract_depth = None
            self._active_section = None

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data.strip():
            return
        cleaned = " ".join(data.split())
        if self._heading_tag:
            self._heading_parts.append(cleaned)
        elif self._active_section in self._parts:
            self._parts[self._active_section].append(cleaned)

    def sections(self) -> Dict[str, str]:
        return {
            section: " ".join(parts).strip()
            for section, parts in self._parts.items()
        }


def _parse_arxiv_html(raw_html: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    parser = _ArxivHTMLSectionParser()
    parser.feed(raw_html)
    parser.close()
    sections = parser.sections()
    for section in sections:
        max_chars = 2000 if section in {"abstract", "limitations"} else 4000
        sections[section] = sections[section][:max_chars]
    missing = [section for section in REQUIRED_COVERAGE_SECTIONS if not sections.get(section)]
    quality: Dict[str, Any] = {
        "heading": "heading" if parser.heading_seen else "unknown",
        "parserStatus": "success" if not missing else "degraded",
        "source": "html",
        "sourceFormat": "html",
        "coverage": "complete" if not missing else "incomplete",
        "coverageStatus": "complete" if not missing else "incomplete",
        "missingSections": missing,
        "codeStatus": "available" if parser.code_urls else "not_found",
        "codeUrls": parser.code_urls,
    }
    return sections, quality


def _fetch_arxiv_html_sections(arxiv_id: str, timing: Dict[str, Any] | None = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Fetch and parse only the official versioned ArXiv HTML fallback."""

    started = time.perf_counter()
    quality: Dict[str, Any] = {"source": "html", "sourceFormat": "html", "htmlStatus": "unavailable"}
    response = None
    try:
        response = requests.get(
            _validated_arxiv_html_url(arxiv_id),
            headers={"User-Agent": config.ARXIV_USER_AGENT},
            timeout=20,
            stream=True,
        )
        response.raise_for_status()
        requested_url = _validated_arxiv_html_url(arxiv_id)
        final_url = urlparse(getattr(response, "url", requested_url))
        expected_path = urlparse(requested_url).path
        if (
            final_url.scheme != "https"
            or (final_url.hostname or "").lower() not in {"arxiv.org", "www.arxiv.org"}
            or final_url.path.rstrip("/") != expected_path.rstrip("/")
        ):
            raise ValueError("ArXiv HTML redirected outside the requested paper version")
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_REMOTE_HTML_BYTES:
            raise ValueError("ArXiv HTML exceeds the download size limit")
        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=32768):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_REMOTE_HTML_BYTES:
                raise ValueError("ArXiv HTML exceeds the download size limit")
            chunks.append(chunk)
        raw_html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        sections, parsed_quality = _parse_arxiv_html(raw_html)
        parsed_quality["htmlStatus"] = "ok"
        quality = parsed_quality
        return sections, parsed_quality
    except (OSError, ValueError, requests.RequestException, UnicodeError):
        return {}, quality
    finally:
        if response is not None:
            response.close()
        if timing is not None:
            timing.update({
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "status": quality.get("htmlStatus", "unavailable"),
            })


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


def _needs_html_fallback(sections: Dict[str, str], quality: Dict[str, Any]) -> bool:
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


def _merge_html_fallback(
    sections: Dict[str, str],
    pdf_quality: Dict[str, Any],
    html_sections: Dict[str, str],
    html_quality: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    merged = dict(sections)
    parser_degraded = pdf_quality.get("parserStatus") != "success"
    for section, text in html_sections.items():
        if text and (parser_degraded or not merged.get(section, "").strip()):
            merged[section] = text
    missing = [section for section in REQUIRED_COVERAGE_SECTIONS if not merged.get(section, "").strip()]
    complete = not missing and html_quality.get("htmlStatus") == "ok"
    source_format = "mixed" if any(sections.get(section, "").strip() for section in REQUIRED_COVERAGE_SECTIONS) else "html"
    quality: Dict[str, Any] = {
        "heading": "heading" if html_quality.get("heading") == "heading" or pdf_quality.get("heading") == "heading" else "unknown",
        "parserStatus": "success" if complete else "degraded",
        "source": source_format,
        "sourceFormat": source_format,
        "coverage": "complete" if complete else "incomplete",
        "coverageStatus": "complete" if complete else "incomplete",
        "missingSections": missing,
        "htmlFallback": "used",
        "htmlStatus": html_quality.get("htmlStatus", "unavailable"),
        "codeStatus": html_quality.get("codeStatus", "unknown"),
        "codeUrls": html_quality.get("codeUrls", []),
    }
    if not complete:
        quality["warning"] = (
            "Extraction incomplete: the PDF and official ArXiv HTML did not expose all key sections. "
            "Missing sections may be outside the readable source window; this is not evidence that the paper lacks results."
        )
    return merged, quality


def sync_extract_pdf_content(paper: PaperItem) -> PaperItem:
    """Download/read a paper and use official ArXiv HTML when PDF coverage is incomplete."""

    started = time.perf_counter()
    paper_key = _paper_cache_key(paper)
    cache_started = time.perf_counter()
    cached_data = get_cached_text_data(paper_key)
    cached_sections = cached_data.get("sections") if isinstance(cached_data, dict) else None
    cached_quality_raw = cached_data.get("source_quality") if isinstance(cached_data, dict) else None
    cached_quality = dict(cached_quality_raw) if isinstance(cached_quality_raw, dict) else _unknown_cached_quality()
    cache_quality_known = isinstance(cached_quality_raw, dict)

    if isinstance(cached_sections, dict) and cached_sections:
        paper.sections = {str(key): str(value or "") for key, value in cached_sections.items()}
        paper.extracted_text = str(cached_data.get("extracted_text") or "")
        paper.local_pdf_path = cached_data.get("local_pdf_path")
        paper.source_quality = cached_quality
        if cache_quality_known and not _needs_html_fallback(paper.sections, paper.source_quality):
            paper.processing_metadata = dict(paper.processing_metadata or {})
            paper.processing_metadata["pdf"] = {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "cacheMs": max(0, int((time.perf_counter() - cache_started) * 1000)),
                "cache": "hit",
            }
            return paper

    # Determine PDF path (version-aware cache, local file, or bounded download).
    pdf_path = get_cached_pdf_path(paper_key)
    download_started = time.perf_counter()
    download_kind = "cache" if pdf_path else "download"
    if not pdf_path or not pdf_path.exists():
        if paper.source_type == "local_pdf" and paper.local_pdf_path:
            local_p = Path(paper.local_pdf_path)
            if local_p.exists():
                pdf_path = local_p
                download_kind = "local"
            else:
                raise FileNotFoundError(f"Local PDF file not found: {paper.local_pdf_path}")
        elif paper.pdf_url:
            target_path = config.PDF_CACHE_DIR / f"{paper_key}.pdf"
            pdf_path = _download_pdf(paper.pdf_url, target_path)
        elif paper.arxiv_id:
            target_path = config.PDF_CACHE_DIR / f"{paper_key}.pdf"
            pdf_path = _download_pdf(f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf", target_path)
        elif cached_sections:
            pdf_path = None
            download_kind = "cached_text"
        else:
            raise ValueError(f"No PDF source available for paper: {paper.title}")

    download_ms = max(0, int((time.perf_counter() - download_started) * 1000))
    extract_started = time.perf_counter()
    html_timing: Dict[str, Any] = {}
    if pdf_path is not None:
        raw_ordered_text = extract_ordered_blocks_from_pdf(pdf_path, max_pages=config.PDF_MAX_PAGES)
        source_quality: Dict[str, Any] = {}
        sections = parse_structured_sections(raw_ordered_text, metadata=source_quality)
    else:
        raw_ordered_text = paper.extracted_text or ""
        sections = dict(paper.sections or {})
        source_quality = dict(paper.source_quality or _unknown_cached_quality())
        source_quality.setdefault("coverageStatus", "unknown")
        source_quality.setdefault("sourceFormat", "unknown")
    extract_ms = max(0, int((time.perf_counter() - extract_started) * 1000))

    if paper.arxiv_id and _needs_html_fallback(sections, source_quality):
        html_sections, html_quality = _fetch_arxiv_html_sections(paper.arxiv_id, html_timing)
        if html_sections:
            sections, source_quality = _merge_html_fallback(sections, source_quality, html_sections, html_quality)
            html_text = "\n\n".join(
                f"=== {section.upper()} (OFFICIAL ARXIV HTML) ===\n{text}"
                for section, text in html_sections.items()
                if text
            )
            raw_ordered_text = f"{raw_ordered_text}\n\n{html_text}".strip()
        else:
            source_quality = dict(source_quality)
            source_quality.update({
                "htmlFallback": "unavailable",
                "htmlStatus": html_timing.get("status", "unavailable"),
                "coverageStatus": "incomplete" if source_quality.get("coverageStatus") != "unknown" else "unknown",
                "coverage": "incomplete" if source_quality.get("coverageStatus") != "unknown" else "unknown",
                "warning": (
                    "Extraction incomplete: official ArXiv HTML fallback was unavailable. "
                    "Missing sections must not be interpreted as absent from the paper."
                ),
            })
    else:
        source_quality.setdefault("htmlFallback", "not_needed")

    paper.local_pdf_path = str(pdf_path) if pdf_path is not None else paper.local_pdf_path
    paper.extracted_text = raw_ordered_text[:config.PDF_MAX_CHARS]
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
            "paper_id": paper.paper_id,
            "title": paper.title,
            "local_pdf_path": paper.local_pdf_path,
            "sections": sections,
            "extracted_text": paper.extracted_text,
            "source_quality": paper.source_quality,
        },
    )
    return paper
