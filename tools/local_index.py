"""Offline full-text index over arXiv metadata (SQLite FTS5, bm25 ranking).

``build_index`` streams the HF ``secemp9/arxiv-complete`` metadata parquet by
row group into an SQLite file.  ``search_local`` queries that file and returns
the same ``PaperItem`` objects as ``tools.arxiv_search.arxiv_search``.

Design:
- ``papers`` holds one row per paper (text stored once).
- ``papers_fts`` is an external-content FTS5 table over ``title`` + ``abstract``
  (``content='papers'``), so the text is not stored twice.
- Ranking: ``bm25(papers_fts, TITLE_WEIGHT, ABSTRACT_WEIGHT)`` (title weighted higher).
- Read connections are per thread (``threading.local``) and read-only.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from state import PaperItem
from tools.arxiv_search import _canonical_arxiv_urls, _metadata_status
from tools.cache_manager import canonicalize_paper_key

TITLE_WEIGHT = 10.0
ABSTRACT_WEIGHT = 1.0
MAX_QUERY_TERMS = 24
MAX_RESULTS_CAP = 1000

# Columns read from the parquet file (the rest of the 21 columns is not needed).
_PARQUET_COLUMNS = [
    "paper_id", "title", "authors", "abstract", "categories",
    "primary_category", "first_version_date", "doi",
]

_SCHEMA = """
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    categories TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    first_version_date TEXT NOT NULL,
    doi TEXT NOT NULL
);
CREATE VIRTUAL TABLE papers_fts USING fts5(
    title, abstract,
    content='papers', content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);
"""

# Small function-word list.  Dropping these keeps the AND query from failing on
# a word that no abstract needs to contain.  Topic words ("survey") stay.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "not", "with",
    "by", "at", "from", "as", "about", "is", "are", "be", "using", "via",
})
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# New-style ids look like 2205.09991; old-style ids look like cs/0101001.
_MODERN_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


# --------------------------------------------------------------------------- build

def _clean_text_column(column: Any) -> list:
    import pyarrow.compute as pc

    cleaned = pc.utf8_trim_whitespace(pc.replace_substring_regex(column, r"\s+", " "))
    return pc.fill_null(cleaned, "").to_pylist()


def _prefix_mask(categories: Sequence[str], prefixes: tuple[str, ...]) -> list[bool]:
    return [
        any(cat.startswith(prefixes) for cat in (value or "").split())
        for value in categories
    ]


def build_index(
    parquet_path: str | os.PathLike,
    index_path: str | os.PathLike,
    *,
    categories_prefix: str | Sequence[str] | None = None,
    batch_rows: int = 50_000,
    optimize: bool = False,
    progress: Callable[[str], None] | None = print,
) -> Dict[str, Any]:
    """Build the SQLite FTS5 index from the metadata parquet file.

    Streams one row group at a time (low memory).  Writes ``<index_path>.tmp``
    and renames it over ``index_path`` when done, so a failed build never
    damages an existing index and a re-run replaces it (idempotent).

    ``categories_prefix`` ("cs." or ["cs.", "stat.ML"]) keeps only papers with at
    least one category starting with a prefix.  Returns build statistics.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    say = progress or (lambda _msg: None)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(index_path.name + ".tmp")
    for stale in (tmp_path, tmp_path.with_name(tmp_path.name + "-journal")):
        if stale.exists():
            stale.unlink()

    prefixes: tuple[str, ...] | None = None
    if categories_prefix:
        prefixes = (categories_prefix,) if isinstance(categories_prefix, str) else tuple(categories_prefix)

    started = time.perf_counter()
    pf = pq.ParquetFile(str(parquet_path))
    total_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    batch_rows = max(1, int(batch_rows))

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=-262144")  # 256 MB
        conn.executescript(_SCHEMA)
        inserted = 0
        scanned = 0
        for group in range(total_groups):
            table = pf.read_row_group(group, columns=_PARQUET_COLUMNS)
            for offset in range(0, table.num_rows, batch_rows):
                chunk = table.slice(offset, batch_rows)
                scanned += chunk.num_rows
                if prefixes:
                    mask = _prefix_mask(pc.fill_null(chunk["categories"], "").to_pylist(), prefixes)
                    chunk = chunk.filter(mask)
                    if chunk.num_rows == 0:
                        continue
                dates = pc.fill_null(pc.strftime(chunk["first_version_date"], "%Y-%m-%d"), "").to_pylist()
                rows = list(zip(
                    pc.fill_null(chunk["paper_id"], "").to_pylist(),
                    _clean_text_column(chunk["title"]),
                    pc.fill_null(chunk["authors"], "").to_pylist(),
                    _clean_text_column(chunk["abstract"]),
                    pc.fill_null(chunk["categories"], "").to_pylist(),
                    pc.fill_null(chunk["primary_category"], "").to_pylist(),
                    dates,
                    pc.fill_null(chunk["doi"], "").to_pylist(),
                ))
                last_rowid = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM papers").fetchone()[0]
                conn.executemany(
                    "INSERT OR IGNORE INTO papers(paper_id, title, authors, abstract, categories,"
                    " primary_category, first_version_date, doi) VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.execute(
                    "INSERT INTO papers_fts(rowid, title, abstract)"
                    " SELECT rowid, title, abstract FROM papers WHERE rowid > ?",
                    (last_rowid,),
                )
                inserted += len(rows)
            elapsed = time.perf_counter() - started
            say(
                f"[build_index] row group {group + 1}/{total_groups}: "
                f"{inserted:,} indexed, {scanned:,}/{total_rows:,} scanned, {elapsed:,.0f}s"
            )
        conn.commit()
        inserted = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        if optimize:
            say("[build_index] optimizing FTS index ...")
            conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('optimize')")
            conn.commit()
        say("[build_index] running ANALYZE ...")
        conn.execute("ANALYZE")
        conn.commit()
    except BaseException:
        conn.close()
        for leftover in (tmp_path, tmp_path.with_name(tmp_path.name + "-journal")):
            if leftover.exists():
                leftover.unlink()
        raise
    conn.close()
    os.replace(tmp_path, index_path)
    _forget_connections(index_path)

    stats = {
        "papers": inserted,
        "scanned": scanned,
        "seconds": round(time.perf_counter() - started, 1),
        "bytes": index_path.stat().st_size,
    }
    say(f"[build_index] done: {stats}")
    return stats


# --------------------------------------------------------------------------- query

_local = threading.local()
_registry_lock = threading.Lock()
_generation = 0  # bumped by build_index so threads reopen a replaced file


def _forget_connections(_path: Path) -> None:
    global _generation
    with _registry_lock:
        _generation += 1


def _connect(index_path: str | os.PathLike) -> sqlite3.Connection:
    """Return this thread's read-only connection to ``index_path``."""
    key = str(Path(index_path).resolve())
    cache: Dict[str, tuple[int, sqlite3.Connection]] = getattr(_local, "conns", None) or {}
    _local.conns = cache
    entry = cache.get(key)
    if entry is not None and entry[0] == _generation:
        return entry[1]
    if entry is not None:
        entry[1].close()
    conn = sqlite3.connect(f"file:{key}?mode=ro", uri=True, check_same_thread=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-262144")
    conn.execute("PRAGMA mmap_size=4294967296")
    conn.row_factory = sqlite3.Row
    cache[key] = (_generation, conn)
    return conn


def _query_terms(query: str) -> List[str]:
    """Split free text into plain lowercase word tokens (no FTS5 syntax)."""
    seen: set[str] = set()
    terms: List[str] = []
    for token in _TOKEN_RE.findall((query or "").lower()):
        if token in _QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:MAX_QUERY_TERMS]


def build_match_query(terms: Iterable[str], operator: str = "AND") -> str:
    """Quote every term, so operators, ``*``, ``-``, ``:`` or quotes in user text stay literal."""
    return f" {operator} ".join(f'"{term}"' for term in terms)


def _split_authors(value: str) -> List[str]:
    text = re.sub(r"\([^)]*\)", " ", value or "")
    text = re.sub(r"\s+and\s+", ", ", text)
    return [name.strip() for name in text.split(",") if name.strip()]


def _row_to_paper(row: sqlite3.Row) -> PaperItem:
    arxiv_id = row["paper_id"]
    urls = _canonical_arxiv_urls(arxiv_id)
    if not urls:  # old-style id such as cs/0101001
        urls = {
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "html_url": f"https://arxiv.org/html/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        }
    paper_key = (
        canonicalize_paper_key(arxiv_id)
        if _MODERN_ID_RE.match(arxiv_id)
        else "arxiv_" + arxiv_id.replace("/", "_")
    )
    authors = _split_authors(row["authors"])
    subjects = (row["categories"] or "").split()
    published = row["first_version_date"] or ""
    return PaperItem(
        paper_id=paper_key,
        title=row["title"],
        summary=row["abstract"],
        authors=authors,
        published=published,
        subjects=subjects,
        source_type="arxiv",
        arxiv_id=arxiv_id,
        arxiv_abs_url=urls.get("abs_url"),
        arxiv_html_url=urls.get("html_url"),
        pdf_url=urls.get("pdf_url"),
        metadata_status=_metadata_status({
            "title": row["title"], "authors": authors,
            "published": published, "subjects": subjects,
        }),
        processing_metadata={
            "source": "local_index",
            "local_bm25": round(-float(row["score"]), 4),  # higher is better
        },
    )


def _category_clause(categories: Sequence[str]) -> tuple[str, List[str]]:
    """SQL for "paper has any of these categories".  A trailing ``*`` means prefix."""
    parts: List[str] = []
    params: List[str] = []
    for raw in categories:
        cat = (raw or "").strip()
        if not cat:
            continue
        pattern = cat[:-1] if cat.endswith("*") else cat + " "
        escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        parts.append("(' ' || p.categories || ' ') LIKE ? ESCAPE '\\'")
        params.append(f"% {escaped}%")
    return (" AND (" + " OR ".join(parts) + ")", params) if parts else ("", [])


def _run_match(
    conn: sqlite3.Connection,
    match: str,
    limit: int,
    *,
    date_before: str | None,
    categories: Sequence[str] | None,
    include_legacy_ids: bool,
) -> List[sqlite3.Row]:
    sql = (
        "SELECT p.*, bm25(papers_fts, ?, ?) AS score FROM papers_fts"
        " JOIN papers p ON p.rowid = papers_fts.rowid WHERE papers_fts MATCH ?"
    )
    params: List[Any] = [TITLE_WEIGHT, ABSTRACT_WEIGHT, match]
    if date_before:
        sql += " AND p.first_version_date <= ?"
        params.append(str(date_before)[:10])
    if categories:
        clause, cat_params = _category_clause(categories)
        sql += clause
        params.extend(cat_params)
    if not include_legacy_ids:
        sql += " AND p.paper_id GLOB '[0-9][0-9][0-9][0-9].*'"
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # Defensive: sanitized queries should never be rejected by FTS5.
        return []


def search_local(
    query: str,
    max_results: int = 5,
    *,
    index_path: str | os.PathLike | None = None,
    date_before: str | None = None,
    categories: Sequence[str] | None = None,
    include_legacy_ids: bool = False,
) -> List[PaperItem]:
    """Search the local index; return ``PaperItem`` objects like ``arxiv_search``.

    - ``date_before``: keep papers with ``first_version_date <= date_before`` (YYYY-MM-DD).
    - ``categories``: keep papers having any listed category (``"cs.AI"``, or ``"cs.*"`` for a prefix).
    - Query text is reduced to quoted word terms, so it can never raise an FTS5 syntax error.
    - Runs an AND query first; when it gives fewer than ``max_results`` hits, an OR query
      fills the rest (AND hits stay first).
    - ``processing_metadata["local_bm25"]`` holds the bm25 score (higher is better).
    - Old-style ids (``cs/0101001``) are skipped unless ``include_legacy_ids`` is true,
      because the rest of the pipeline only parses new-style ids.
    """
    terms = _query_terms(query)
    if not terms:
        return []
    if index_path is None:
        from config import config
        index_path = config.LOCAL_INDEX_PATH
    if not Path(index_path).exists():
        raise FileNotFoundError(f"Local index not found: {index_path}")

    limit = max(1, min(int(max_results or 5), MAX_RESULTS_CAP))
    conn = _connect(index_path)
    options = dict(date_before=date_before, categories=categories, include_legacy_ids=include_legacy_ids)

    rows = _run_match(conn, build_match_query(terms, "AND"), limit, **options)
    if len(rows) < limit and len(terms) > 1:
        have = {row["paper_id"] for row in rows}
        extra = _run_match(conn, build_match_query(terms, "OR"), limit + len(have), **options)
        rows = rows + [row for row in extra if row["paper_id"] not in have][: limit - len(rows)]
    return [_row_to_paper(row) for row in rows]
