"""Resolve a bibliography entry (``PaperReference``) to an arXiv id.

Order, first hit wins (the method name is recorded in ``Resolution.method``):

1. ``explicit``    the entry already carries an arXiv id (version suffix removed);
2. ``arxiv_doi``   DOI ``10.48550/arXiv.<id>`` (DataCite DOIs of arXiv papers);
3. ``index_doi``   DOI equals the ``doi`` of one row in the local index;
4. ``index_title`` normalized title equals the normalized title of exactly one
   row in the local index (titles under 4 words or with 2+ matching rows are
   skipped).

The local index (``data/arxiv/index.sqlite``, built by ``tools/local_index.py`` on
branch ``feat/local-index``) is opened READ-ONLY and never modified:

- title lookup uses the FTS5 table: a column-filtered phrase query returns the
  candidates, then Python compares the normalized titles for equality.  No new
  index is needed.
- DOI lookup has no index in ``papers`` (``doi`` is not indexed).  The first DOI
  request loads ``SELECT doi, paper_id FROM papers WHERE doi != ''`` (about 160k
  of 1M rows, 0.5 s with a warm file cache) into an in-process dict that all
  resolvers share.  Nothing is written to disk.  Without a DOI-bearing
  bibliography entry the scan never runs.

A missing or unreadable index file degrades to the first two methods.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from state import PaperReference

logger = logging.getLogger(__name__)

METHODS = ("explicit", "arxiv_doi", "index_doi", "index_title")
MIN_TITLE_WORDS = 4
_MAX_TITLE_CANDIDATES = 2000

_NEW_ID = r"\d{4}\.\d{4,5}"
_OLD_ID = r"[a-z][a-z.\-]*/\d{7}"
_ARXIV_ID_RE = re.compile(rf"^(?:{_NEW_ID}|{_OLD_ID})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_DOI_RE = re.compile(rf"^10\.48550/arxiv\.({_NEW_ID}|{_OLD_ID})(?:v\d+)?$", re.IGNORECASE)
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")


@dataclass(frozen=True)
class Resolution:
    arxiv_id: str
    method: str


def clean_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Return the id without ``arXiv:`` prefix and version, or ``None`` if malformed."""

    text = (value or "").strip()
    text = re.sub(r"^arxiv\s*:\s*", "", text, flags=re.IGNORECASE)
    if not _ARXIV_ID_RE.match(text):
        return None
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)


def normalize_title(title: str) -> str:
    """Lowercase, drop accents, LaTeX commands and punctuation; collapse spaces."""

    text = unicodedata.normalize("NFKD", title or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _LATEX_CMD_RE.sub(" ", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalize_doi(doi: Optional[str]) -> str:
    text = (doi or "").strip().lower()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text)
    return text.rstrip(".,;)")


# DOI map shared by all resolvers: {resolved index path: {doi: paper_id | None}}.
# ``None`` marks a DOI that two rows share (ambiguous).
_doi_maps: Dict[str, Dict[str, Optional[str]]] = {}
_doi_lock = threading.Lock()


class RefResolver:
    """Resolve references against the explicit data and the optional local index."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index_path = Path(index_path) if index_path else None
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self.index_available = False
        self.counts: Dict[str, int] = {method: 0 for method in METHODS}
        self.counts["ambiguous"] = 0
        self.counts["unresolved"] = 0
        if self.index_path is not None and self.index_path.is_file():
            try:
                uri = f"file:{self.index_path.resolve()}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                self._conn.execute("SELECT 1 FROM papers LIMIT 1").fetchone()
                self.index_available = True
            except sqlite3.Error as exc:
                logger.warning("ref_resolver: local index unusable (%s); explicit ids only", exc)
                self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self.index_available = False

    # -- public ---------------------------------------------------------------

    def resolve(self, ref: PaperReference) -> Optional[Resolution]:
        found = self._resolve(ref)
        if found is None:
            self.counts["unresolved"] += 1
        else:
            self.counts[found.method] += 1
        return found

    def lookup_meta(self, arxiv_id: str) -> Optional[Dict[str, str]]:
        """Return ``{"title", "year"}`` for an id in the index (primary-key lookup)."""

        if self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT title, first_version_date FROM papers WHERE paper_id = ?", (arxiv_id,)
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        return {"title": " ".join(str(row[0]).split()), "year": str(row[1] or "")[:4]}

    # -- internals ------------------------------------------------------------

    def _resolve(self, ref: PaperReference) -> Optional[Resolution]:
        explicit = clean_arxiv_id(ref.arxiv_id)
        if explicit:
            return Resolution(explicit, "explicit")

        doi = normalize_doi(ref.doi)
        if doi:
            match = _ARXIV_DOI_RE.match(doi)
            if match:
                return Resolution(clean_arxiv_id(match.group(1)) or match.group(1), "arxiv_doi")
            paper_id = self._doi_lookup(doi)
            if paper_id:
                return Resolution(paper_id, "index_doi")

        paper_id = self._title_lookup(ref.title)
        if paper_id:
            return Resolution(paper_id, "index_title")
        return None

    def _doi_lookup(self, doi: str) -> Optional[str]:
        if self._conn is None or self.index_path is None:
            return None
        key = str(self.index_path.resolve())
        with _doi_lock:
            mapping = _doi_maps.get(key)
            if mapping is None:
                mapping = {}
                try:
                    with self._lock:
                        rows = self._conn.execute(
                            "SELECT doi, paper_id FROM papers WHERE doi != ''"
                        ).fetchall()
                except sqlite3.Error as exc:
                    logger.warning("ref_resolver: DOI scan failed (%s)", exc)
                    rows = []
                for raw_doi, paper_id in rows:
                    norm = normalize_doi(raw_doi)
                    if not norm:
                        continue
                    mapping[norm] = None if norm in mapping and mapping[norm] != paper_id else paper_id
                _doi_maps[key] = mapping
        return mapping.get(doi)

    def _title_lookup(self, title: str) -> Optional[str]:
        if self._conn is None:
            return None
        norm = normalize_title(title)
        words = norm.split()
        if len(words) < MIN_TITLE_WORDS:
            return None
        query = 'title : "' + " ".join(words) + '"'
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT p.paper_id, p.title FROM papers_fts "
                    "JOIN papers p ON p.rowid = papers_fts.rowid "
                    "WHERE papers_fts MATCH ? LIMIT ?",
                    (query, _MAX_TITLE_CANDIDATES),
                ).fetchall()
        except sqlite3.Error:
            return None
        ids = {paper_id for paper_id, row_title in rows if normalize_title(row_title) == norm}
        if len(ids) == 1:
            return next(iter(ids))
        if len(ids) > 1:
            self.counts["ambiguous"] += 1
        return None
