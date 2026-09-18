from __future__ import annotations

import random
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests

from config import config
from state import PaperItem
from tools.cache_manager import canonicalize_paper_key

ARXIV_METADATA_CACHE_TTL_SEC = 24 * 60 * 60

_ARXIV_LOCK = threading.Lock()
_LAST_ARXIV_CALL = 0.0

STOPWORDS = {
    "in", "of", "for", "the", "and", "to", "a", "an", "on", "with", "by", "at", "from", "as", "about",
    "paper", "papers", "research", "recent", "latest", "survey", "study",
    "toi", "muon", "tim", "bai", "bao", "trong", "ve", "cac", "nhung", "mot", "vai"
}


def _rate_limit_arxiv(min_interval: float = config.ARXIV_MIN_INTERVAL_SEC) -> None:
    """Thread-safe rate limiter with random jitter to prevent HTTP 429."""
    global _LAST_ARXIV_CALL
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _LAST_ARXIV_CALL
        if elapsed < min_interval:
            jitter = random.uniform(0.1, 0.4)
            time.sleep((min_interval - elapsed) + jitter)
        _LAST_ARXIV_CALL = time.monotonic()


def _arxiv_get(url: str, params: Dict[str, Any] | None = None, max_retries: int = 3) -> requests.Response:
    """Perform robust HTTP GET request to ArXiv with Exponential Backoff."""
    delay = 3.0
    for attempt in range(max_retries):
        _rate_limit_arxiv(delay)
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": config.ARXIV_USER_AGENT},
                timeout=12,
            )
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                delay = delay * 2 + random.uniform(1.0, 2.0)
            else:
                resp.raise_for_status()
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
    raise RuntimeError("ArXiv API unreachable after maximum retries.")


def _extract_title_terms(query: str) -> List[str]:
    """Keep title-like tokens such as 'DeepSeek-MoE' or 'DeepSeek-V2' together."""
    cleaned = re.sub(r"[^A-Za-z0-9\-\s]", " ", (query or "")).strip()
    terms = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", cleaned)
    return [term for term in terms if len(term) > 3]


def _title_match_score(query: str, title: str) -> int:
    """Higher score for exact title phrases; lower score for generic topic keywords."""
    q_text = (query or "").lower()
    t_text = (title or "").lower()
    q_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q_text)]
    if not q_terms:
        return 0

    score = 0
    for term in q_terms:
        if term in t_text:
            score += 5
        if len(term) > 3 and term.replace("-", " ") in t_text.replace("-", " "):
            score += 2
    return score


def _clean_arxiv_query(query: str) -> str:
    """Clean and structure search query for ArXiv API, prioritizing exact paper names."""
    cleaned = " ".join((query or "").split())
    if ":" in cleaned:
        return cleaned

    title_terms = _extract_title_terms(cleaned)
    if len(title_terms) >= 2:
        exact_terms = [term for term in title_terms if term.lower() not in {"deepseek","v2","moe"}][:4]
        if exact_terms:
            return " AND ".join(f'all:"{term}"' for term in exact_terms)

    raw_terms = [term for term in re.findall(r"[A-Za-z0-9_\\-]+", cleaned) if len(term) > 1]
    terms = [t for t in raw_terms if t.lower() not in STOPWORDS]
    if not terms:
        terms = raw_terms[:4]
    return " AND ".join(f"all:{term}" for term in terms[:6]) or cleaned


def _extract_arxiv_id(value: str) -> str:
    """Extract standard ArXiv ID from URL or text."""
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", value or "")
    return match.group(1) if match else ""


def _canonical_arxiv_urls(arxiv_id: str) -> Dict[str, str]:
    """Build version preserving links from a validated ArXiv identifier."""

    clean_id = _extract_arxiv_id(arxiv_id)
    if not clean_id:
        return {}
    return {
        "abs_url": f"https://arxiv.org/abs/{clean_id}",
        "html_url": f"https://arxiv.org/html/{clean_id}",
        "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
    }


def _entry_text(entry: ET.Element, path: str, namespaces: Dict[str, str]) -> str:
    node = entry.find(path, namespaces)
    return (node.text or "").strip() if node is not None and node.text else ""


def _metadata_cache_path(arxiv_id: str):
    # ``canonicalize_paper_key`` intentionally collapses versions for UI
    # identity.  Metadata must keep the version separate: v1 and v2 can have
    # different titles, authors, dates, and subjects.
    clean_id = _extract_arxiv_id(arxiv_id)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", clean_id or str(arxiv_id or "unknown"))
    return config.TEXT_CACHE_DIR / f"arxiv_{safe_id}_metadata.json"


def _metadata_status(metadata: Dict[str, Any]) -> str:
    required = (metadata.get("title"), metadata.get("authors"), metadata.get("published"))
    if all(required) and metadata.get("subjects"):
        return "complete"
    return "missing"


def _normalise_cached_metadata(value: Dict[str, Any], requested_id: str) -> Dict[str, Any]:
    """Keep legacy metadata safe while exposing an explicit quality state."""

    metadata = dict(value)
    clean_requested = _extract_arxiv_id(requested_id)
    cached_id = _extract_arxiv_id(str(metadata.get("arxiv_id") or ""))
    # A cache entry for another explicit version must never satisfy this input.
    requested_version = re.search(r"v\d+$", clean_requested)
    cached_version = re.search(r"v\d+$", cached_id)
    if requested_version and requested_version.group(0) != (cached_version.group(0) if cached_version else ""):
        return {}
    effective_id = clean_requested if requested_version else (cached_id or clean_requested)
    metadata["arxiv_id"] = effective_id
    metadata.setdefault("subjects", [])
    metadata.setdefault("abs_url", _canonical_arxiv_urls(effective_id).get("abs_url", ""))
    metadata.setdefault("html_url", _canonical_arxiv_urls(effective_id).get("html_url", ""))
    metadata.setdefault("pdf_url", _canonical_arxiv_urls(effective_id).get("pdf_url", ""))
    # Recompute this for legacy cache entries instead of trusting a status that
    # predates subjects/URL fields.
    metadata["metadata_status"] = _metadata_status(metadata)
    return metadata


def get_cached_arxiv_metadata(arxiv_id: str) -> Dict[str, Any] | None:
    """Return fresh direct-input metadata without making a network call."""

    try:
        payload = json.loads(_metadata_cache_path(arxiv_id).read_text(encoding="utf-8"))
        if time.time() - float(payload.get("savedAt", 0)) > ARXIV_METADATA_CACHE_TTL_SEC:
            return None
        value = payload.get("metadata")
        if not isinstance(value, dict):
            return None
        normalised = _normalise_cached_metadata(value, arxiv_id)
        return normalised or None
    except (OSError, ValueError, TypeError):
        return None


def fetch_arxiv_metadata(arxiv_id: str, *, timing: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """Fetch and cache one ArXiv record for direct-input title resolution."""

    started = time.perf_counter()
    clean_id = _extract_arxiv_id(arxiv_id)
    if not clean_id:
        if timing is not None:
            timing.update({"durationMs": 0, "cache": "miss", "status": "invalid_id"})
        return None

    cached = get_cached_arxiv_metadata(clean_id)
    if cached is not None:
        if timing is not None:
            timing.update({
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "cache": "hit",
                "status": "ok",
            })
        return cached

    status = "error"
    try:
        response = _arxiv_get(config.ARXIV_API_URL, params={"id_list": clean_id, "max_results": 1})
        root = ET.fromstring(response.text)
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        entry = root.find(".//atom:entry", namespaces)
        if entry is None:
            status = "missing"
            return None
        response_id = _extract_arxiv_id(_entry_text(entry, "./atom:id", namespaces))
        explicit_version = re.search(r"v\d+$", clean_id)
        if explicit_version:
            response_version = re.search(r"v\d+$", response_id)
            if not response_id or not response_version or response_version.group(0) != explicit_version.group(0):
                status = "version_mismatch"
                return None
        effective_id = clean_id if explicit_version else (response_id or clean_id)
        links = entry.findall("./atom:link", namespaces)
        urls = _canonical_arxiv_urls(effective_id)
        subjects = [
            str(category.get("term") or "").strip()
            for category in entry.findall("./atom:category", namespaces)
            if str(category.get("term") or "").strip()
        ]
        metadata = {
            "arxiv_id": effective_id,
            "title": " ".join(_entry_text(entry, "./atom:title", namespaces).split()),
            "summary": " ".join(_entry_text(entry, "./atom:summary", namespaces).split()),
            "authors": [_entry_text(author, "./atom:name", namespaces) for author in entry.findall("./atom:author", namespaces)],
            "published": _entry_text(entry, "./atom:published", namespaces)[:10],
            "subjects": subjects,
            "abs_url": urls.get("abs_url", ""),
            "html_url": urls.get("html_url", ""),
            "pdf_url": urls.get("pdf_url", ""),
        }
        metadata["metadata_status"] = _metadata_status(metadata)
        try:
            path = _metadata_cache_path(effective_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(".json.part")
            temp_path.write_text(json.dumps({"savedAt": time.time(), "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(path)
        except OSError:
            pass
        status = "ok"
        return metadata
    finally:
        if timing is not None:
            timing.update({
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                "cache": "miss",
                "status": status,
            })


def arxiv_search(
    query: str = "",
    max_results: int = config.MAX_SEARCH_RESULTS,
    sort_by: str = "relevance",
    *,
    timing: Dict[str, Any] | None = None,
) -> List[PaperItem]:
    """Query ArXiv API and return a list of structured PaperItem objects."""
    if not query.strip():
        return []

    max_results = max(1, min(int(max_results or 5), 10))
    sort_by = sort_by if sort_by in {"relevance", "lastUpdatedDate", "submittedDate"} else "relevance"
    
    params = {
        "search_query": _clean_arxiv_query(query),
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    started = time.perf_counter()
    resp = _arxiv_get(config.ARXIV_API_URL, params=params)
    root = ET.fromstring(resp.text)
    
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers: List[PaperItem] = []
    for entry in root.findall(".//atom:entry", namespaces):
        abs_url = _entry_text(entry, "./atom:id", namespaces)
        arxiv_id = _extract_arxiv_id(abs_url)
        if not arxiv_id:
            continue

        links = entry.findall("./atom:link", namespaces)
        pdf_url = next(
            (link.get("href") for link in links if link.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )
        summary = _entry_text(entry, "./atom:summary", namespaces).replace("\n", " ")
        title = _entry_text(entry, "./atom:title", namespaces).replace("\n", " ")
        authors = [_entry_text(author, "./atom:name", namespaces) for author in entry.findall("./atom:author", namespaces)]
        published = _entry_text(entry, "./atom:published", namespaces)[:10]
        subjects = [
            str(category.get("term") or "").strip()
            for category in entry.findall("./atom:category", namespaces)
            if str(category.get("term") or "").strip()
        ]
        urls = _canonical_arxiv_urls(arxiv_id)

        paper_key = canonicalize_paper_key(arxiv_id)

        paper = PaperItem(
            paper_id=paper_key,
            title=" ".join(title.split()),
            summary=" ".join(summary.split()),
            authors=authors,
            published=published,
            subjects=subjects,
            source_type="arxiv",
            arxiv_id=arxiv_id,
            arxiv_abs_url=urls.get("abs_url"),
            arxiv_html_url=urls.get("html_url"),
            pdf_url=urls.get("pdf_url") or pdf_url,
            metadata_status=_metadata_status({
                "title": title,
                "authors": authors,
                "published": published,
                "subjects": subjects,
            }),
        )
        papers.append(paper)

    papers.sort(key=lambda p: _title_match_score(query, p.title), reverse=True)
    if timing is not None:
        timing.update({"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "resultCount": len(papers)})
    return papers
