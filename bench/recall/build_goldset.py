"""Build the search-recall gold set from survey bibliographies.

For each candidate survey (cs.AI survey papers from the local arXiv metadata
snapshot) the script reads the reference list from ar5iv / arXiv HTML, resolves
every reference to an arXiv id, and keeps the references that were published
before the survey.  The result maps a topic query to the papers that a survey
author found worth citing.

Usage (see ``bench/README.md``):

    .venv/bin/python -I bench/recall/build_goldset.py                 # online, 200 surveys
    .venv/bin/python -I bench/recall/build_goldset.py --html-dir DIR --ids 2205.09991 ...   # offline

Network use is limited to ``tools.html_source.fetch_html_page`` (ar5iv, then
arxiv.org/html, with the shared per-host pacing).  Raw HTML is cached, so a
rerun never refetches.  ``/e-print`` and ``/src`` are never requested.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from state import PaperReference  # noqa: E402
from tools.html_extractor import extract_html, looks_like_latexml  # noqa: E402

DEFAULT_OUT_DIR = config.DATA_DIR / "arxiv" / "recall"
DEFAULT_METADATA = config.DATA_DIR / "arxiv" / "metadata.parquet"
DEFAULT_MANIFEST = ROOT / "bench" / "recall" / "goldset_manifest.json"
DEFAULT_TITLE_REGEX = r"\b(survey|a review|literature review|systematic review)\b"
MIN_TITLE_WORDS = 4
NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
VERSION_RE = re.compile(r"v\d+$")

METHODS = ("arxiv_id", "doi_arxiv", "doi_metadata", "title")


# ---------------------------------------------------------------------------
# Title normalisation and query stripping
# ---------------------------------------------------------------------------

_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
_MATH_RE = re.compile(r"\$[^$]*\$")


def normalize_title(title: str) -> str:
    """Lowercase, drop LaTeX and punctuation, fold accents, collapse spaces."""

    text = _MATH_RE.sub(" ", title or "")
    text = _LATEX_CMD_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def title_is_matchable(norm: str) -> bool:
    return len(norm.split()) >= MIN_TITLE_WORDS


_ADJ = (
    r"(?:comprehensive|systematic|brief|critical|short|recent|detailed|in-depth|extensive|thorough|"
    r"concise|structured|focused|scoping|narrative|bibliometric|state-of-the-art|current|up-to-date|"
    r"quick|holistic|unified|broad)"
)
_SURVEY_CORE = (
    r"(?:systematic\s+literature\s+review|literature\s+review|systematic\s+review|"
    r"systematic\s+mapping(?:\s+study)?|surveys?)"
)
_SURVEY_PHRASE_RE = re.compile(
    r"(?ix)\b(?:"
    rf"(?:(?:an?|the)\s+)?(?:{_ADJ}\s+)*{_SURVEY_CORE}"          # survey / literature review ...
    rf"|(?:an?|the)\s+(?:{_ADJ}\s+)*(?:review|overview)"         # "a review", "a comprehensive overview"
    rf"|(?:{_ADJ}\s+)+(?:review|overview)"                       # "comprehensive review"
    r")\b"
    r"(?:\s+(?:and|&)\s+(?:taxonomy|outlook|perspectives?|challenges|analysis|future\s+directions|"
    r"open\s+problems?|tutorial|roadmap|prospects?|opportunities))*"
    r"(?:\s+(?:on|of|for|in|about|to|into)\b)?"
)
_EDGE_WORDS_RE = re.compile(
    r"(?i)^(?:and|of|on|for|in|the|a|an|with)\b\s*|\s*\b(?:and|of|on|for|in|the|a|an|with)$"
)
_EDGE_CHARS = " ,-\u2013\u2014.&"


def strip_survey_words(title: str) -> str:
    """Remove survey wording ("A Survey on", ": A Review", ...) to get a topic query.

    Colons are removed too: ``arxiv_search`` sends a query with ":" to the API
    unchanged.  Falls back to the colon-free original when nothing useful remains.
    """

    original = " ".join((title or "").split())
    text = _SURVEY_PHRASE_RE.sub(" ", original)
    text = re.sub(r"\s[-\u2013\u2014]+\s", " ", re.sub(r"[:;|]", " ", text))
    text = " ".join(text.split()).strip(_EDGE_CHARS)
    previous = None
    while previous != text:                       # drop dangling connector words
        previous = text
        text = _EDGE_WORDS_RE.sub("", text).strip(_EDGE_CHARS)
    if len(text.split()) < 2:
        return " ".join(re.sub(r"[:;|]", " ", original).split())
    return text


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------

def clean_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Return an id without version suffix, lowercase, or ``None``."""

    if not value:
        return None
    cleaned = VERSION_RE.sub("", value.strip().lower())
    return cleaned or None


def clean_doi(value: Optional[str]) -> Optional[str]:
    return value.strip().lower() if value and value.strip() else None


@dataclass
class MetaIndex:
    """Lookup tables restricted to the ids, DOIs and titles a run asks for."""

    dates: Dict[str, datetime] = field(default_factory=dict)          # paper_id -> first_version_date
    by_doi: Dict[str, str] = field(default_factory=dict)              # lowercase doi -> paper_id
    by_title: Dict[str, Optional[str]] = field(default_factory=dict)  # normalized title -> id; None = ambiguous
    titles: Dict[str, str] = field(default_factory=dict)              # paper_id -> raw title (wanted ids only)

    def add_title(self, norm: str, paper_id: str) -> None:
        if norm in self.by_title and self.by_title[norm] != paper_id:
            self.by_title[norm] = None
        else:
            self.by_title[norm] = paper_id


def resolve_reference(ref: PaperReference, index: MetaIndex) -> Optional[Tuple[str, str]]:
    """Resolve one reference to ``(paper_id, method)`` in the fixed order:

    explicit arXiv id, DOI ``10.48550/arXiv.<id>``, DOI match in metadata,
    exact normalized title (unambiguous, at least 4 words).  A step that finds
    nothing in the metadata falls through to the next step.
    """

    explicit = clean_arxiv_id(ref.arxiv_id)
    if explicit and explicit in index.dates:
        return explicit, "arxiv_id"
    doi = clean_doi(ref.doi)
    if doi:
        match = ARXIV_DOI_RE.match(doi)
        if match:
            doi_id = clean_arxiv_id(match.group(1))
            if doi_id and doi_id in index.dates:
                return doi_id, "doi_arxiv"
        if doi in index.by_doi:
            return index.by_doi[doi], "doi_metadata"
    norm = normalize_title(ref.title)
    if title_is_matchable(norm):
        found = index.by_title.get(norm)
        if found:                                   # None means ambiguous: skip
            return found, "title"
    return None


def wanted_keys(refs: Iterable[PaperReference]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Collect the (ids, dois, normalized titles) to look up in the metadata scan."""

    ids: Set[str] = set()
    dois: Set[str] = set()
    titles: Set[str] = set()
    for ref in refs:
        explicit = clean_arxiv_id(ref.arxiv_id)
        if explicit:
            ids.add(explicit)
        doi = clean_doi(ref.doi)
        if doi:
            dois.add(doi)
            match = ARXIV_DOI_RE.match(doi)
            doi_id = clean_arxiv_id(match.group(1)) if match else None
            if doi_id:
                ids.add(doi_id)
        norm = normalize_title(ref.title)
        if title_is_matchable(norm):
            titles.add(norm)
    return ids, dois, titles


# ---------------------------------------------------------------------------
# Metadata access (streamed by row group; the full table never sits in memory)
# ---------------------------------------------------------------------------

def _iter_row_groups(path: Path, columns: List[str]) -> Iterator[List[Dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(path))
    for i in range(parquet.metadata.num_row_groups):
        yield parquet.read_row_group(i, columns=columns).to_pylist()


def select_surveys(
    metadata_path: Path,
    *,
    category: str = "cs.AI",
    year_from: int = 2021,
    year_to: Optional[int] = None,
    title_regex: str = DEFAULT_TITLE_REGEX,
) -> List[Dict[str, Any]]:
    """Return every survey candidate, sorted by paper_id (deterministic)."""

    pattern = re.compile(title_regex, re.IGNORECASE)
    found: List[Dict[str, Any]] = []
    columns = ["paper_id", "title", "primary_category", "first_version_date"]
    for rows in _iter_row_groups(metadata_path, columns):
        for row in rows:
            pid = row["paper_id"]
            if row["primary_category"] != category or not NEW_ID_RE.match(pid):
                continue
            year = 2000 + int(pid[:2])
            if year < year_from or (year_to is not None and year > year_to):
                continue
            title = " ".join((row["title"] or "").split())
            if pattern.search(title.lower()):
                found.append({"id": pid, "title": title, "date": row["first_version_date"]})
    found.sort(key=lambda item: item["id"])
    return found


def sample_surveys(candidates: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(candidates):
        return list(candidates)
    return sorted(random.Random(seed).sample(candidates, n), key=lambda item: item["id"])


def scan_metadata(metadata_path: Path, ids: Set[str], dois: Set[str], titles: Set[str]) -> MetaIndex:
    """One pass over the metadata; keeps only rows that match a wanted key."""

    index = MetaIndex()
    for rows in _iter_row_groups(metadata_path, ["paper_id", "title", "doi", "first_version_date"]):
        for row in rows:
            pid = row["paper_id"]
            if pid in ids:
                index.dates[pid] = row["first_version_date"]
                index.titles[pid] = row["title"] or ""
            if dois and row.get("doi"):
                for doi in row["doi"].lower().split():
                    if doi in dois:
                        index.by_doi.setdefault(doi, pid)
                        index.dates.setdefault(pid, row["first_version_date"])
            if titles:
                norm = normalize_title(row["title"] or "")
                if norm in titles:
                    index.add_title(norm, pid)
                    index.dates.setdefault(pid, row["first_version_date"])
    return index


# ---------------------------------------------------------------------------
# HTML fetch with cache
# ---------------------------------------------------------------------------

def find_cached_html(html_dir: Path, paper_id: str) -> Optional[Path]:
    for prefix in ("ar5iv_", "arxiv_"):
        path = html_dir / f"{prefix}{paper_id}.html"
        if path.is_file():
            return path
    return None


def fetch_survey_html(paper_id: str, cache_dir: Path) -> Tuple[Optional[str], str]:
    """Return ``(raw_html, status)``; fetch ar5iv then arxiv.org/html on a cache miss."""

    cached = find_cached_html(cache_dir, paper_id)
    if cached is not None:
        return cached.read_text(encoding="utf-8", errors="replace"), "cache"
    missing_marker = cache_dir / f"{paper_id}.missing"
    if missing_marker.is_file():
        return None, "cached_missing"

    from tools.html_source import HTML_SOURCES, HTMLUnavailable, fetch_html_page, html_url

    statuses: List[str] = []
    for name, _host, prefix in HTML_SOURCES:
        try:
            raw = fetch_html_page(html_url(prefix, paper_id), paper_id)
        except HTMLUnavailable as exc:
            statuses.append(f"{name}:{exc.status}")
            continue
        except (OSError, ValueError, UnicodeError):
            statuses.append(f"{name}:error")
            continue
        if not looks_like_latexml(raw):
            statuses.append(f"{name}:not_latexml")
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{name}_{paper_id}.html"
        part = target.with_suffix(".html.part")
        part.write_text(raw, encoding="utf-8")
        part.replace(target)
        return raw, name
    if statuses and all(s.endswith(("not_found", "not_latexml")) for s in statuses):
        cache_dir.mkdir(parents=True, exist_ok=True)       # permanent failure: do not retry
        missing_marker.write_text("\n".join(statuses), encoding="utf-8")
    return None, "unavailable(" + ",".join(statuses) + ")"


# ---------------------------------------------------------------------------
# Gold-set assembly
# ---------------------------------------------------------------------------

def _to_date(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return None


def id_month_date(paper_id: str) -> Optional[datetime]:
    """Fallback survey date for ids absent from the metadata: first day of the id's month."""

    match = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", paper_id)
    if not match or not 1 <= int(match.group(2)) <= 12:
        return None
    return datetime(2000 + int(match.group(1)), int(match.group(2)), 1)


def build_goldset(
    surveys: List[Dict[str, Any]],
    raw_html: Dict[str, str],
    metadata_path: Path,
    *,
    min_refs: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Resolve references for all fetched surveys.  Returns ``(records, stats)``."""

    parsed: Dict[str, List[PaperReference]] = {}
    html_titles: Dict[str, str] = {}
    for survey in surveys:
        html = raw_html.get(survey["id"])
        if html is None:
            continue
        extraction = extract_html(html)
        parsed[survey["id"]] = extraction.references
        html_titles[survey["id"]] = extraction.title

    ids, dois, titles = wanted_keys(ref for refs in parsed.values() for ref in refs)
    ids |= {s["id"] for s in surveys}
    index = scan_metadata(metadata_path, ids, dois, titles)

    records: List[Dict[str, Any]] = []
    per_survey: List[Dict[str, Any]] = []
    totals: Counter = Counter()
    by_method: Counter = Counter()
    for survey in surveys:
        sid = survey["id"]
        row: Dict[str, Any] = {"survey_id": sid, "html": sid in parsed}
        per_survey.append(row)
        if sid not in parsed:
            continue
        title = survey.get("title") or index.titles.get(sid) or html_titles.get(sid) or sid
        survey_date = _to_date(survey.get("date")) or _to_date(index.dates.get(sid))
        date_source = "metadata"
        if survey_date is None:
            survey_date, date_source = id_month_date(sid), "id_month"
        refs = parsed[sid]
        method_counts: Counter = Counter()
        gold: Dict[str, str] = {}
        resolved = 0
        for ref in refs:
            hit = resolve_reference(ref, index)
            if hit is None:
                continue
            pid, method = hit
            resolved += 1
            method_counts[method] += 1
            ref_date = _to_date(index.dates.get(pid))
            if pid == sid or survey_date is None or ref_date is None or ref_date >= survey_date:
                continue
            gold.setdefault(pid, method)
        row.update(
            refs=len(refs), resolved=resolved, by_method=dict(method_counts),
            gold=len(gold), date_source=date_source,
        )
        totals["refs"] += len(refs)
        totals["resolved"] += resolved
        by_method.update(method_counts)
        if len(gold) >= min_refs:
            records.append(
                {
                    "survey_id": sid,
                    "title": title,
                    "query": strip_survey_words(title),
                    "date": survey_date.date().isoformat() if survey_date else None,
                    "date_source": date_source,
                    "refs_parsed": len(refs),
                    "refs_resolved": resolved,
                    "gold": [{"id": pid, "method": method} for pid, method in gold.items()],
                }
            )

    stats = {
        "surveys_selected": len(surveys),
        "surveys_fetched": len(parsed),
        "refs_parsed": totals["refs"],
        "refs_resolved": totals["resolved"],
        "resolved_pct": round(100.0 * totals["resolved"] / totals["refs"], 1) if totals["refs"] else 0.0,
        "by_method": {m: by_method.get(m, 0) for m in METHODS},
        "min_refs": min_refs,
        "surveys_kept": len(records),
        "gold_total": sum(len(r["gold"]) for r in records),
        "per_survey": per_survey,
    }
    return records, stats


def make_manifest(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Committed manifest: ids and queries only, no paper text."""

    return {
        "version": 1,
        "surveys": [
            {
                "survey_id": r["survey_id"],
                "query": r["query"],
                "date": r["date"],
                "gold": [g["id"] for g in r["gold"]],
            }
            for r in records
        ],
    }


def format_stats(stats: Dict[str, Any]) -> str:
    lines = [
        f"surveys selected : {stats['surveys_selected']}",
        f"surveys fetched  : {stats['surveys_fetched']}",
        f"refs parsed      : {stats['refs_parsed']}",
        f"refs resolved    : {stats['refs_resolved']} ({stats['resolved_pct']}%)",
        "by method        : " + ", ".join(f"{m}={n}" for m, n in stats["by_method"].items()),
        f"surveys kept     : {stats['surveys_kept']} (>= {stats['min_refs']} gold refs)",
        f"gold ids total   : {stats['gold_total']}",
    ]
    if len(stats["per_survey"]) <= 20:
        for row in stats["per_survey"]:
            if not row.get("html"):
                lines.append(f"  {row['survey_id']}: no HTML")
                continue
            methods = ", ".join(f"{m}={n}" for m, n in sorted(row["by_method"].items())) or "-"
            lines.append(
                f"  {row['survey_id']}: refs={row['refs']} resolved={row['resolved']} gold={row['gold']} "
                f"[{methods}] date={row['date_source']}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="arXiv metadata parquet")
    p.add_argument("--category", default="cs.AI", help="primary category of candidate surveys")
    p.add_argument("--year-from", type=int, default=2021, help="first year, taken from the id prefix")
    p.add_argument("--year-to", type=int, default=None)
    p.add_argument("--title-regex", default=DEFAULT_TITLE_REGEX)
    p.add_argument("--sample", type=int, default=200, help="number of surveys to sample (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-refs", type=int, default=15, help="minimum gold refs to keep a survey")
    p.add_argument("--ids", nargs="+", help="use these survey ids; skips selection")
    p.add_argument("--html-dir", type=Path, help="offline: read HTML from this dir only, no network")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="goldset.jsonl, stats and html cache")
    p.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest path (default: bench/recall/goldset_manifest.json; <out-dir>/goldset_manifest.json "
        "when --out-dir is set)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (
        DEFAULT_MANIFEST if out_dir == DEFAULT_OUT_DIR else out_dir / "goldset_manifest.json"
    )

    if args.ids:
        surveys: List[Dict[str, Any]] = [{"id": i, "title": None, "date": None} for i in args.ids]
        if args.metadata.is_file():                        # fill title/date when the id is in the snapshot
            index = scan_metadata(args.metadata, set(args.ids), set(), set())
            for s in surveys:
                s["title"], s["date"] = index.titles.get(s["id"]) or None, index.dates.get(s["id"])
    else:
        candidates = select_surveys(
            args.metadata,
            category=args.category,
            year_from=args.year_from,
            year_to=args.year_to,
            title_regex=args.title_regex,
        )
        surveys = sample_surveys(candidates, args.sample, args.seed)
        print(f"candidates: {len(candidates)}; sampled: {len(surveys)} (seed {args.seed})")

    raw_html: Dict[str, str] = {}
    for n, survey in enumerate(surveys, 1):
        sid = survey["id"]
        if args.html_dir:                                  # offline: local files only
            path = find_cached_html(args.html_dir, sid)
            html = path.read_text(encoding="utf-8", errors="replace") if path else None
            status = "local" if path else "missing"
        else:
            html, status = fetch_survey_html(sid, out_dir / "html")
        print(f"[{n}/{len(surveys)}] {sid}: {status}", flush=True)
        if html is not None:
            raw_html[sid] = html

    records, stats = build_goldset(surveys, raw_html, args.metadata, min_refs=args.min_refs)
    goldset_path = out_dir / "goldset.jsonl"
    goldset_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out_dir / "goldset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(make_manifest(records), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_stats(stats))
    print(f"wrote {goldset_path}, goldset_stats.json, {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
