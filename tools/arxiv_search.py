from __future__ import annotations

import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests

from config import config
from state import PaperItem
from tools.cache_manager import canonicalize_paper_key

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


def _entry_text(entry: ET.Element, path: str, namespaces: Dict[str, str]) -> str:
    node = entry.find(path, namespaces)
    return (node.text or "").strip() if node is not None and node.text else ""


def arxiv_search(
    query: str = "",
    max_results: int = config.MAX_SEARCH_RESULTS,
    sort_by: str = "relevance",
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

        paper_key = canonicalize_paper_key(arxiv_id)

        paper = PaperItem(
            paper_id=paper_key,
            title=" ".join(title.split()),
            summary=" ".join(summary.split()),
            authors=authors,
            published=published,
            source_type="arxiv",
            arxiv_id=arxiv_id,
            pdf_url=pdf_url,
        )
        papers.append(paper)

    papers.sort(key=lambda p: _title_match_score(query, p.title), reverse=True)
    return papers
