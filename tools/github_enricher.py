from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from config import config
from state import GitHubRepoInfo
from tools.cache_manager import canonicalize_paper_key

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_CACHE_TTL_SEC = 24 * 60 * 60


def _model_dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        raw = value.model_dump()
    elif hasattr(value, "dict"):
        raw = value.dict()
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}
    return dict(raw) if isinstance(raw, dict) else {}


def _cache_path(title: str, arxiv_id: str) -> Path:
    key = canonicalize_paper_key(arxiv_id or title)
    # The canonical key is already stable for ArXiv IDs.  Hash the fallback
    # too so a long/private title never becomes a path component.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:96]
    digest = hashlib.sha256((arxiv_id or title or key).encode("utf-8")).hexdigest()[:12]
    return config.GITHUB_CACHE_DIR / f"{safe}_{digest}.json"


def _load_cached(title: str, arxiv_id: str) -> List[GitHubRepoInfo] | None:
    path = _cache_path(title, arxiv_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at = float(payload.get("savedAt", 0))
        if time.time() - saved_at > GITHUB_CACHE_TTL_SEC:
            return None
        repos: List[GitHubRepoInfo] = []
        for item in payload.get("repos", []) or []:
            if isinstance(item, dict):
                repos.append(GitHubRepoInfo(**item))
        return repos
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _save_cache(title: str, arxiv_id: str, repos: List[GitHubRepoInfo]) -> None:
    path = _cache_path(title, arxiv_id)
    temp_path = path.with_suffix(".json.part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "savedAt": time.time(),
            "repos": [_model_dump(repo) for repo in repos],
        }
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        # Caching is an optimization.  A read-only or full cache must not make
        # a research run fail.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def search_github_code_multitier(
    title: str,
    arxiv_id: str = "",
    authors: List[str] | None = None,
    max_results: int = 3,
    *,
    timing: Dict[str, Any] | None = None,
) -> List[GitHubRepoInfo]:
    """Search GitHub with three fallbacks and a 24-hour success cache.

    ``timing`` is an optional numeric observability sink used by the node; no
    query, token, response body, or exception text is recorded there.
    Network failures are intentionally not cached.
    """

    started = time.perf_counter()
    metrics: Dict[str, Any] = timing if timing is not None else {}
    cached = _load_cached(title, arxiv_id)
    if cached is not None:
        metrics.update({"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "cache": "hit", "tiers": {}})
        return cached[:max_results]

    headers = {"User-Agent": "Research-Agent/4.0"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"token {config.GITHUB_TOKEN}"

    queries: List[tuple[str, str]] = []
    if arxiv_id:
        clean_arxiv = re.sub(r"v\d+$", "", arxiv_id.strip())
        queries.append(("arxiv_id", f'"{clean_arxiv}"'))

    clean_title = re.sub(r"[^\w\s-]", " ", title or "").strip()
    title_words = [word for word in clean_title.split() if len(word) > 2][:6]
    if title_words:
        queries.append(("title_keywords", f'{" ".join(title_words)} paper'))

    if authors and title_words:
        first_author_last = authors[0].split()[-1]
        queries.append(("author_keywords", f"{first_author_last} {' '.join(title_words[:3])}"))

    found_repos: List[GitHubRepoInfo] = []
    seen_urls: set[str] = set()
    tier_metrics: Dict[str, Any] = {}
    successful_request = False
    had_failure = False
    for tier_name, query in queries:
        tier_started = time.perf_counter()
        tier_status = "error"
        try:
            resp = requests.get(
                GITHUB_SEARCH_URL,
                params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
                headers=headers,
                timeout=6,
            )
            if resp.status_code == 200:
                successful_request = True
                tier_status = "ok"
                try:
                    items = resp.json().get("items", [])
                except (TypeError, ValueError):
                    had_failure = True
                    tier_status = "invalid_response"
                    items = []
                for item in items:
                    url = item.get("html_url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    desc = (item.get("description") or "").lower()
                    is_official = (
                        "official" in desc
                        or (arxiv_id and arxiv_id in desc)
                        or (clean_title and clean_title.lower() in desc)
                    )
                    found_repos.append(
                        GitHubRepoInfo(
                            name=item.get("full_name", ""),
                            url=url,
                            stars=item.get("stargazers_count", 0),
                            framework=item.get("language") or "Python",
                            is_official=is_official,
                            description=(item.get("description") or "")[:200],
                        )
                    )
                if len(found_repos) >= max_results:
                    break
            elif resp.status_code in {401, 403, 404, 422, 429, 500, 502, 503, 504}:
                had_failure = True
                tier_status = f"http_{resp.status_code}"
            else:
                had_failure = True
                tier_status = "http_error"
        except requests.RequestException:
            had_failure = True
            tier_status = "network_error"
        finally:
            tier_metrics[tier_name] = {
                "durationMs": max(0, int((time.perf_counter() - tier_started) * 1000)),
                "status": tier_status,
            }

    result = found_repos[:max_results]
    metrics.update({
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "cache": "miss",
        "tiers": tier_metrics,
    })
    if successful_request and not had_failure:
        _save_cache(title, arxiv_id, result)
    return result
