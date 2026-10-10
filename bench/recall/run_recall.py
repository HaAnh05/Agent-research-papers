"""Measure search recall against the survey gold set.

Each system searches every gold-set query once with k = max(--ks).  The survey
itself is removed from the ranking.  Raw results are cached per
(system, query, k[, date_before]) under ``data/arxiv/recall/results/``, so a
rerun costs nothing.  Output: JSON + Markdown table in
``bench/results/recall/<YYYY-MM-DD>/``.

Usage (see ``bench/README.md``):

    .venv/bin/python -I bench/recall/run_recall.py --systems arxiv_api
    .venv/bin/python -I bench/recall/run_recall.py --systems arxiv_api,local_fts

``--systems fake`` is an offline stand-in (title keyword match on the metadata
snapshot) that only checks the plumbing.  No LLM is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:      # works under ``python -I``, which drops the script dir
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

RECALL_DIR = config.DATA_DIR / "arxiv" / "recall"
DEFAULT_GOLDSET = RECALL_DIR / "goldset.jsonl"
DEFAULT_CACHE_DIR = RECALL_DIR / "results"
DEFAULT_METADATA = config.DATA_DIR / "arxiv" / "metadata.parquet"
RESULTS_ROOT = ROOT / "bench" / "results" / "recall"
VERSION_RE = re.compile(r"v\d+$")


class SystemUnavailable(Exception):
    """A registered system cannot run in this checkout (for example no local index)."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def clean_id(value: str) -> str:
    """Normalize a result id: drop an ``arxiv_`` key prefix and the version suffix."""

    text = (value or "").strip().lower()
    if text.startswith("arxiv_"):
        text = text[len("arxiv_"):]
    return VERSION_RE.sub("", text)


def query_metrics(ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    """Per-query metrics for a ranked id list (duplicates keep their first position)."""

    seen: set = set()
    ranking = [x for x in ranked if not (x in seen or seen.add(x))]
    gold_set = set(gold)
    out: Dict[str, float] = {}
    for k in ks:
        hits = sum(1 for x in ranking[:k] if x in gold_set)
        out[f"recall@{k}"] = hits / len(gold_set) if gold_set else 0.0
    hits5 = sum(1 for x in ranking[:5] if x in gold_set)
    out["precision@5"] = hits5 / 5
    out["hit_rate@5"] = 1.0 if hits5 else 0.0
    out["mrr"] = next((1.0 / rank for rank, x in enumerate(ranking, 1) if x in gold_set), 0.0)
    return out


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile; 0.0 for an empty list."""

    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def aggregate(per_query: List[Dict[str, float]], latencies: Sequence[float]) -> Dict[str, float]:
    keys = per_query[0].keys() if per_query else []
    summary = {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
    summary["latency_p50_s"] = percentile(latencies, 50)
    summary["latency_p90_s"] = percentile(latencies, 90)
    return summary


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

@dataclass
class System:
    name: str
    search: Callable[[str, int, Optional[str]], List[Any]]   # (query, k, date_before) -> items
    uses_date: bool = False        # True: date_before changes the result, so it joins the cache key
    min_gap_s: float = 0.0         # pause between uncached calls, so the tool's own pacing never adds latency


def _item_id(item: Any) -> str:
    if isinstance(item, dict):
        return clean_id(str(item.get("arxiv_id") or item.get("paper_id") or ""))
    return clean_id(str(getattr(item, "arxiv_id", None) or getattr(item, "paper_id", "")))


def _make_arxiv_api() -> System:
    from tools.arxiv_search import arxiv_search

    return System(
        "arxiv_api",
        lambda query, k, _date: arxiv_search(query, k),
        min_gap_s=config.ARXIV_MIN_INTERVAL_SEC + 0.5,
    )


def _make_local_fts() -> System:
    try:
        from tools.local_index import search_local          # exists on feat/local-index only
    except ImportError as exc:
        raise SystemUnavailable("tools.local_index is not importable on this branch") from exc

    return System(
        "local_fts",
        lambda query, k, date_before: search_local(query, k, date_before=date_before),
        uses_date=True,
    )


_STOP = {"the", "and", "for", "with", "from", "into", "using", "based", "via", "its", "are", "that"}


def _make_fake(metadata_path: Path) -> System:
    """Offline stand-in: rank titles by how many query keywords they contain."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if not metadata_path.is_file():
        raise SystemUnavailable(f"metadata not found: {metadata_path}")
    table = pq.read_table(str(metadata_path), columns=["paper_id", "title"])
    titles = pc.utf8_lower(table["title"])
    paper_ids = table["paper_id"]

    def search(query: str, k: int, _date: Optional[str]) -> List[Any]:
        words = sorted({w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 3 and w not in _STOP})
        if not words:
            return []
        score = np.zeros(len(titles), dtype=np.int16)
        for word in words:
            score += pc.match_substring(titles, word).fill_null(False).to_numpy(zero_copy_only=False).astype(np.int16)
        top = np.argpartition(-score, k)[:k]
        top = top[np.argsort(-score[top], kind="stable")]
        top = top[score[top] > 0]
        return [{"paper_id": i} for i in paper_ids.take(pa.array(top)).to_pylist()]

    return System("fake", search)


def get_system(name: str, metadata_path: Path) -> System:
    if name == "arxiv_api":
        return _make_arxiv_api()
    if name == "local_fts":
        return _make_local_fts()
    if name == "fake":
        return _make_fake(metadata_path)
    raise SystemUnavailable(f"unknown system: {name}")


# ---------------------------------------------------------------------------
# Cached search
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, system: System, query: str, k: int, date_before: Optional[str]) -> Path:
    key = json.dumps([query, k, date_before if system.uses_date else None], ensure_ascii=False)
    return cache_dir / system.name / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".json")


def search_cached(
    system: System, query: str, k: int, date_before: Optional[str], cache_dir: Path, state: Dict[str, float]
) -> Dict[str, Any]:
    """Return ``{"ids", "latency_s", "cached"}``; errors are returned but never cached."""

    path = cache_path(cache_dir, system, query, k, date_before)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"ids": data["ids"], "latency_s": data["latency_s"], "cached": True}
    wait = system.min_gap_s - (time.monotonic() - state.get(system.name, -1e9))
    if wait > 0:
        time.sleep(wait)
    started = time.perf_counter()
    try:
        items = system.search(query, k, date_before)
    except Exception as exc:                                  # network / API failure: count as empty
        state[system.name] = time.monotonic()
        return {"ids": [], "latency_s": time.perf_counter() - started, "cached": False, "error": repr(exc)[:200]}
    latency = time.perf_counter() - started
    state[system.name] = time.monotonic()
    ids = [i for i in (_item_id(item) for item in items) if i]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"query": query, "k": k, "date_before": date_before, "ids": ids, "latency_s": latency}),
        encoding="utf-8",
    )
    return {"ids": ids, "latency_s": latency, "cached": False}


def evaluate_system(
    system: System, goldset: List[Dict[str, Any]], ks: Sequence[int], cache_dir: Path
) -> Dict[str, Any]:
    k_search = max(ks)
    state: Dict[str, float] = {}
    per_query: List[Dict[str, float]] = []
    latencies: List[float] = []
    details: List[Dict[str, Any]] = []
    errors = 0
    max_returned = 0
    for n, record in enumerate(goldset, 1):
        sid = record["survey_id"]
        result = search_cached(system, record["query"], k_search, record.get("date"), cache_dir, state)
        if result.get("error"):
            errors += 1
        ids = [i for i in result["ids"] if i != clean_id(sid)]       # the survey never counts
        gold = [g["id"] for g in record["gold"]]
        metrics = query_metrics(ids, gold, ks)
        per_query.append(metrics)
        if not result.get("error"):
            latencies.append(result["latency_s"])
        max_returned = max(max_returned, len(ids))
        details.append(
            {"survey_id": sid, "query": record["query"], "n_gold": len(gold), "n_results": len(ids),
             "cached": result["cached"], **({"error": result["error"]} if result.get("error") else {}), **metrics}
        )
        print(f"  [{system.name} {n}/{len(goldset)}] {sid} results={len(ids)} "
              f"R@5={metrics['recall@5']:.2f}{' (cached)' if result['cached'] else ''}", flush=True)
    return {
        "system": system.name,
        "queries": len(goldset),
        "errors": errors,
        "max_results_returned": max_returned,
        "summary": aggregate(per_query, latencies),
        "per_query": details,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any], ks: Sequence[int]) -> str:
    cols = [f"Recall@{k}" for k in ks] + ["Precision@5", "HitRate@5", "MRR", "p50 s", "p90 s", "queries", "errors", "max results"]
    lines = [
        f"# Search recall - {report['date']}",
        "",
        f"Gold set: {report['surveys']} surveys, {report['gold_total']} gold ids "
        f"(mean {report['gold_mean']:.1f} per survey). k searched = {max(ks)}.",
        "",
        "| System | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for res in report["systems"]:
        s = res["summary"]
        cells = [f"{s[f'recall@{k}']:.3f}" for k in ks]
        cells += [f"{s['precision@5']:.3f}", f"{s['hit_rate@5']:.3f}", f"{s['mrr']:.3f}",
                  f"{s['latency_p50_s']:.2f}", f"{s['latency_p90_s']:.2f}",
                  str(res["queries"]), str(res["errors"]), str(res["max_results_returned"])]
        lines.append(f"| {res['system']} | " + " | ".join(cells) + " |")
    lines += ["", "Recall@k = gold ids found in the top k / gold ids. `max results` is the longest result list a system "
              "returned: Recall@k for k above it equals Recall at that length. Latency comes from the first (uncached) run."]
    for res in report["skipped"]:
        lines.append(f"\nSkipped: {res}")
    return "\n".join(lines) + "\n"


def load_goldset(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    p.add_argument("--systems", default="arxiv_api,local_fts", help="comma list: arxiv_api, local_fts, fake")
    p.add_argument("--ks", default="5,20,100", help="comma list of cut-offs")
    p.add_argument("--limit", type=int, default=0, help="use only the first N surveys (0 = all)")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=None, help="default: bench/results/recall/<YYYY-MM-DD>")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="metadata parquet (fake system only)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    if 5 not in ks:
        ks = sorted({*ks, 5})            # Precision@5 and HitRate@5 need the top-5 cut
    goldset = load_goldset(args.goldset)
    if args.limit > 0:
        goldset = goldset[: args.limit]
    if not goldset:
        print("goldset is empty", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name in (n.strip() for n in args.systems.split(",") if n.strip()):
        try:
            system = get_system(name, args.metadata)
        except SystemUnavailable as exc:
            skipped.append(f"{name} ({exc})")
            print(f"skip {name}: {exc}")
            continue
        print(f"system {name}: {len(goldset)} queries, k={max(ks)}")
        results.append(evaluate_system(system, goldset, ks, args.cache_dir))

    gold_sizes = [len(r["gold"]) for r in goldset]
    report = {
        "date": date.today().isoformat(),
        "goldset": str(args.goldset),
        "surveys": len(goldset),
        "gold_total": sum(gold_sizes),
        "gold_mean": sum(gold_sizes) / len(gold_sizes),
        "ks": ks,
        "systems": results,
        "skipped": skipped,
    }
    out_dir = args.out_dir or RESULTS_ROOT / report["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recall.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report, ks)
    (out_dir / "recall.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out_dir / 'recall.json'} and recall.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
