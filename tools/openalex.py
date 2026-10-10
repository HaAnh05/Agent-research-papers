"""Tiny OpenAlex client for the citation graph.

- API key from ``config.OPENALEX_API_KEY`` (env ``OPENALEX_API_KEY``), sent as the
  ``api_key`` query parameter.  Without a key every call returns ``None`` and
  the client logs one warning (``has_key`` is ``False``).
- Pacing: request starts are at least ``config.OPENALEX_MIN_INTERVAL_SEC`` apart
  (default 0.2 s), also across threads.  Cache hits are not paced.
- Cache: one JSON file per URL under ``config.OPENALEX_CACHE_DIR`` (default
  ``data/cache/openalex/``).  The file name is the hash of the URL WITHOUT the
  key, and the key is never written to disk.  A 404 is cached as a miss.
- Failures (network, HTTP 4xx/5xx other than 404, bad JSON) return ``None``,
  count in ``errors`` and are not cached.  There is no retry.

Endpoints used (OpenAlex docs): ``/works/doi:10.48550/arXiv.<id>`` and
``/works?filter=cites:<W id>&sort=cited_by_count:desc&per_page=50``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

import requests

from config import config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
_WORK_FIELDS = "id,title,publication_year,cited_by_count,ids,doi,locations"
_SIM_FIELDS = "id,doi,ids,title,publication_year,cited_by_count,referenced_works,authorships"
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7}))(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.((?:\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7}))(?:v\d+)?", re.IGNORECASE)


def short_id(openalex_id: Any) -> str:
    """``https://openalex.org/W123`` -> ``W123`` (empty string if absent)."""

    return str(openalex_id or "").rstrip("/").rsplit("/", 1)[-1]


def arxiv_id_of_work(work: Dict[str, Any]) -> Optional[str]:
    """Find an arXiv id in a work's DOI, ``ids`` and ``locations``."""

    candidates: List[str] = []
    for value in (work.get("doi"), (work.get("ids") or {}).get("doi")):
        if isinstance(value, str):
            candidates.append(value)
    locations = list(work.get("locations") or [])
    if isinstance(work.get("primary_location"), dict):
        locations.append(work["primary_location"])
    for location in locations:
        if isinstance(location, dict):
            for key in ("landing_page_url", "pdf_url"):
                if isinstance(location.get(key), str):
                    candidates.append(location[key])
    for text in candidates:
        match = _ARXIV_DOI_RE.search(text) or _ARXIV_URL_RE.search(text)
        if match:
            return match.group(1)
    return None


class OpenAlexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        min_interval: Optional[float] = None,
        cache_dir: Optional[Path] = None,
        http_get: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (config.OPENALEX_API_KEY if api_key is None else api_key).strip()
        self.min_interval = config.OPENALEX_MIN_INTERVAL_SEC if min_interval is None else min_interval
        self.cache_dir = Path(cache_dir) if cache_dir else config.OPENALEX_CACHE_DIR
        self._http_get = http_get or requests.get
        self._clock = clock
        self._sleep = sleep
        self._pace_lock = threading.Lock()
        self._stat_lock = threading.Lock()
        self._last_start: Optional[float] = None
        self._warned = False
        self.requests = 0       # network requests sent
        self.cache_hits = 0
        self.errors = 0

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    # -- public ---------------------------------------------------------------

    def work_by_arxiv(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        data = self._get(f"/works/doi:10.48550/arXiv.{arxiv_id}", {"select": _WORK_FIELDS})
        return data if isinstance(data, dict) and data.get("id") else None

    def citing_works(self, work_id: str, per_page: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Top citing works by citation count; ``None`` when the call did not work."""

        data = self._get(
            "/works",
            {
                "filter": f"cites:{short_id(work_id)}",
                "sort": "cited_by_count:desc",
                "per_page": str(per_page),
                "select": _WORK_FIELDS,
            },
        )
        if not isinstance(data, dict):
            return None
        results = data.get("results")
        return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else None

    # -- batched queries for the similarity graph ---------------------------------

    def _results(self, params: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        data = self._get("/works", params)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return None
        return [item for item in data["results"] if isinstance(item, dict)]

    def works_by_arxiv_ids(
        self, arxiv_ids: List[str], select: str = _SIM_FIELDS
    ) -> Optional[List[Dict[str, Any]]]:
        """Works for up to 50 arXiv ids in one request (DOI ``10.48550/arxiv.<id>``)."""

        dois = "|".join(f"10.48550/arxiv.{i}" for i in arxiv_ids[:50])
        return self._results({"filter": f"doi:{dois}", "per_page": "50", "select": select})

    def works_by_ids(self, work_ids: List[str], select: str = _SIM_FIELDS) -> Optional[List[Dict[str, Any]]]:
        """Works for up to 50 OpenAlex ids in one request."""

        ids = "|".join(short_id(i) for i in work_ids[:50])
        return self._results({"filter": f"openalex_id:{ids}", "per_page": "50", "select": select})

    def citer_ids(self, work_id: str, per_page: int = 100) -> Optional[List[str]]:
        """Short ids of the most cited works citing ``work_id``."""

        items = self._results({
            "filter": f"cites:{short_id(work_id)}", "sort": "cited_by_count:desc",
            "per_page": str(per_page), "select": "id",
        })
        return None if items is None else [short_id(i.get("id")) for i in items if i.get("id")]

    # -- internals ------------------------------------------------------------

    def _url(self, path: str, params: Dict[str, str]) -> str:
        return f"{BASE_URL}{path}?{urlencode(sorted(params.items()), safe=':,|')}"

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".json")

    def _pace(self) -> None:
        with self._pace_lock:
            if self._last_start is not None:
                wait = self._last_start + max(0.0, float(self.min_interval)) - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last_start = self._clock()

    def _get(self, path: str, params: Dict[str, str]) -> Optional[Any]:
        if not self.has_key:
            if not self._warned:
                self._warned = True
                logger.warning("OPENALEX_API_KEY is not set; citation counts and citing papers are skipped")
            return None
        url = self._url(path, params)
        cache_file = self._cache_path(url)
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("url") == url:
                with self._stat_lock:
                    self.cache_hits += 1
                return cached.get("data")
        except (OSError, ValueError):
            pass

        self._pace()
        with self._stat_lock:
            self.requests += 1
        try:
            response = self._http_get(
                url, params={"api_key": self.api_key}, timeout=20,
                headers={"User-Agent": config.ARXIV_USER_AGENT},
            )
            if response.status_code == 404:
                data: Any = None
            elif response.status_code >= 400:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            else:
                data = response.json()
        except Exception as exc:
            with self._stat_lock:
                self.errors += 1
            # Log the exception class only: request errors can embed the URL with the key.
            logger.warning("OpenAlex request failed (%s)", type(exc).__name__)
            return None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".part")
            tmp.write_text(json.dumps({"url": url, "data": data}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_file)
        except OSError:
            pass
        return data
