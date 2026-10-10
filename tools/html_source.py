"""Fetch LaTeXML HTML for an arXiv paper: ar5iv first, then arxiv.org/html.

Every request goes through the shared per-host limiter in
``tools.rate_limit`` (arxiv.org: ``Crawl-delay: 15``; ar5iv: 3 s).  Only
``/html/<id>`` pages are requested.  ``/e-print`` and ``/src`` are disallowed
by arXiv's robots.txt and are never used.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

from config import config
from tools.html_extractor import HTMLExtraction, extract_html, looks_like_latexml
from tools.rate_limit import host_slot

MAX_REMOTE_HTML_BYTES = 20 * 1024 * 1024
MAX_HTML_REDIRECTS = 3
HTML_TIMEOUT_SEC = 20

# (source name, host, URL prefix), in read order.
HTML_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("ar5iv", "ar5iv.labs.arxiv.org", "https://ar5iv.labs.arxiv.org/html/"),
    ("arxiv_html", "arxiv.org", "https://arxiv.org/html/"),
)
_ALLOWED_HOSTS = {"ar5iv.labs.arxiv.org", "arxiv.org", "www.arxiv.org"}
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_CONTENT_SECTIONS = ("abstract", "methodology", "setup", "experiments", "limitations")


class HTMLUnavailable(Exception):
    """The source has no usable HTML; ``status`` says why."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def clean_arxiv_id(arxiv_id: str) -> str:
    """Return ``2303.04137`` / ``2303.04137v2`` / ``hep-th/9901001`` or raise ValueError."""

    text = str(arxiv_id or "")
    new_style = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", text)
    if new_style:
        return new_style.group(1)
    old_style = re.search(r"([a-z][a-z\-]+/\d{7}(?:v\d+)?)", text, flags=re.IGNORECASE)
    if old_style:
        return old_style.group(1).lower()
    raise ValueError("HTML source requires a valid ArXiv identifier")


def html_url(prefix: str, arxiv_id: str) -> str:
    return f"{prefix}{clean_arxiv_id(arxiv_id)}"


def _expected_path(arxiv_id: str) -> "re.Pattern[str]":
    base = re.sub(r"v\d+$", "", clean_arxiv_id(arxiv_id))
    return re.compile(rf"/html/{re.escape(base)}(?:v\d+)?/?", flags=re.IGNORECASE)


def _check_target(url: str, arxiv_id: str) -> str:
    """Validate a request target and return its host."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme not in {"http", "https"}
        or host not in _ALLOWED_HOSTS
        or parsed.username
        or parsed.password
        or port not in {None, 80, 443}
    ):
        raise HTMLUnavailable("redirected_away")
    if not _expected_path(arxiv_id).fullmatch(parsed.path):
        # For example a redirect to /abs/<id>: no HTML version exists.
        raise HTMLUnavailable("redirected_away")
    return host


def _read_body(response: Any) -> str:
    content_type = str(response.headers.get("content-type") or "")
    if content_type and "html" not in content_type.lower():
        raise HTMLUnavailable("not_html")
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_REMOTE_HTML_BYTES:
        raise HTMLUnavailable("too_large")
    chunks: List[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=32768):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_REMOTE_HTML_BYTES:
            raise HTMLUnavailable("too_large")
        chunks.append(chunk)
    charset = re.search(r"charset=([\w-]+)", content_type, flags=re.IGNORECASE)
    try:
        return b"".join(chunks).decode(charset.group(1) if charset else "utf-8", errors="replace")
    except LookupError:
        return b"".join(chunks).decode("utf-8", errors="replace")


def fetch_html_page(url: str, arxiv_id: str) -> str:
    """GET one LaTeXML page; raise :class:`HTMLUnavailable` when there is none."""

    current = url
    for _ in range(MAX_HTML_REDIRECTS + 1):
        host = _check_target(current, arxiv_id)
        next_url: Optional[str] = None
        with host_slot(host):
            response = requests.get(
                current,
                headers={"User-Agent": config.ARXIV_USER_AGENT},
                timeout=HTML_TIMEOUT_SEC,
                stream=True,
                allow_redirects=False,
            )
            try:
                status = int(response.status_code)
                if status in _REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise HTMLUnavailable("redirect_without_location")
                    next_url = urljoin(current, location)
                elif status == 200:
                    return _read_body(response)
                elif status in {404, 410}:
                    raise HTMLUnavailable("not_found")
                else:
                    raise HTMLUnavailable(f"http_{status}")
            finally:
                response.close()
        current = next_url or current
    raise HTMLUnavailable("redirect_limit")


def fetch_html_extraction(
    arxiv_id: str,
    attempts: List[Dict[str, Any]],
    timing: Dict[str, Any] | None = None,
) -> Tuple[Optional[HTMLExtraction], Optional[str]]:
    """Try ar5iv, then arxiv.org/html.  Returns ``(extraction, source_name)``.

    ``attempts`` receives one ``{"source", "status"}`` entry per source tried.
    """

    started = time.perf_counter()
    try:
        for name, _host, prefix in HTML_SOURCES:
            try:
                url = html_url(prefix, arxiv_id)
            except ValueError:
                attempts.append({"source": name, "status": "invalid_id"})
                break
            try:
                raw_html = fetch_html_page(url, arxiv_id)
            except HTMLUnavailable as exc:
                attempts.append({"source": name, "status": exc.status})
                continue
            except (OSError, ValueError, UnicodeError):
                attempts.append({"source": name, "status": "error"})
                continue
            if not looks_like_latexml(raw_html):
                attempts.append({"source": name, "status": "not_latexml"})
                continue
            extraction = extract_html(raw_html)
            if not any(extraction.sections.get(key) for key in _CONTENT_SECTIONS):
                attempts.append({"source": name, "status": "no_content"})
                continue
            attempts.append({"source": name, "status": "ok"})
            return extraction, name
        return None, None
    finally:
        if timing is not None:
            timing["durationMs"] = max(0, int((time.perf_counter() - started) * 1000))
