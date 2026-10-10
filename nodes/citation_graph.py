"""Citation graph around the selected papers (no LLM call).

Inputs: ``selected_papers[*].references`` (HTML bibliography), the local arXiv
index (``tools.ref_resolver``) and, when ``OPENALEX_API_KEY`` is set, OpenAlex
(``tools.openalex``).  Output: ``citation_graph`` in the state, a JSON-ready dict

    {"nodes": [{id, arxiv_id, title, year, cited_by_count, selected, role, support}],
     "edges": [{source, target, kind: "cites"}],
     "stats": {refs_total, refs_resolved, refs_ambiguous, by_method, openalex}}

``role`` is ``selected``, ``foundational`` or ``followup``.  ``support`` counts
the selected papers that cite the node (foundational) or that the node cites
(follow-up).  The report section is built from this dict in ``final_report``.

Scoring:
- foundational = resolved reference cited by >= 2 selected papers; sorted by
  support, then ``cited_by_count``, then year.  With ONE selected paper the
  top-cited resolved references are used instead (bibliography order when
  OpenAlex is unavailable).
- follow-up = OpenAlex citing work (top 50 per selected paper) that cites >= 2
  selected papers; if fewer than 3, top-cited single-paper citers fill up to
  ``top_n``.  Selected papers are never candidates.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from config import config
from state import PaperItem, ResearchState
from nodes.similarity_graph import build_similarity_graph
from tools.openalex import OpenAlexClient, arxiv_id_of_work, short_id
from tools.ref_resolver import METHODS, RefResolver, clean_arxiv_id, title_jaccard

MIN_FOLLOWUPS_BEFORE_FILL = 3
MAX_COCITED_LOOKUPS = 12
MAX_SINGLE_PAPER_LOOKUPS = 30
TITLE_MATCH_MIN = 0.5      # token Jaccard below this = OpenAlex record is not the indexed paper
OPENALEX_WORKERS = 4


def year_from_arxiv_id(arxiv_id: str) -> Optional[int]:
    match = re.match(r"^(\d{2})\d{2}\.\d{4,5}$", arxiv_id)
    if match:
        return 2000 + int(match.group(1))
    match = re.match(r"^[a-z][a-z.\-]*/(\d{2})\d{5}$", arxiv_id, re.IGNORECASE)
    if match:
        yy = int(match.group(1))
        return (1900 if yy >= 91 else 2000) + yy
    return None


def _clean_title(value: Any) -> str:
    return " ".join(str(value or "").split())


def _int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _index_meta(resolver: RefResolver, arxiv: Optional[str]) -> Optional[Dict[str, str]]:
    """Local-index ``{"title", "year"}`` for an arXiv id; ``None`` when absent or without title."""

    meta = resolver.lookup_meta(arxiv) if arxiv else None
    return meta if meta and meta.get("title") else None


def _openalex_mismatch(meta: Optional[Dict[str, str]], openalex_title: Any) -> bool:
    """True when the index title and the OpenAlex title are different papers (corrupt record)."""

    title = _clean_title(openalex_title)
    return bool(meta and title and title_jaccard(meta["title"], title) < TITLE_MATCH_MIN)


def _sort_key(cand: Dict[str, Any]) -> Tuple[int, int, int, int]:
    count = cand.get("cited_by_count")          # unknown count ranks below any known count
    return (-cand["support"], -(count if count is not None else -1), -(cand.get("year") or 0), cand["order"])


def build_citation_graph(
    papers: List[PaperItem],
    resolver: RefResolver,
    client: Optional[OpenAlexClient] = None,
    *,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Build the graph dict.  ``client=None`` or a client without key = no OpenAlex."""

    use_openalex = client is not None and client.has_key
    n_sel = len(papers)

    # --- selected papers -----------------------------------------------------
    sel_ids: List[str] = []
    for paper in papers:
        sel_ids.append(clean_arxiv_id(paper.arxiv_id) or paper.paper_id)
    sel_arxiv = {sid for sid, paper in zip(sel_ids, papers) if clean_arxiv_id(paper.arxiv_id)}
    sel_set = set(sel_ids)

    # --- outgoing: resolve references ----------------------------------------
    refs_total = 0
    refs_resolved = 0
    by_method = {method: 0 for method in METHODS}
    cited: Dict[str, Dict[str, Any]] = {}    # arxiv id -> candidate (support = selected papers citing it)
    sel_cites_sel: List[Tuple[str, str]] = []
    order = 0
    for sid, paper in zip(sel_ids, papers):
        seen_in_paper: set = set()
        for ref in paper.references:
            refs_total += 1
            found = resolver.resolve(ref)
            if found is None:
                continue
            refs_resolved += 1
            by_method[found.method] += 1
            target = found.arxiv_id
            if target == sid or target in seen_in_paper:
                continue
            seen_in_paper.add(target)
            if target in sel_set:
                sel_cites_sel.append((sid, target))
                continue
            cand = cited.get(target)
            if cand is None:
                order += 1
                cand = {
                    "id": target, "arxiv_id": target, "title": _clean_title(ref.title),
                    "year": None, "cited_by_count": None, "support": 0, "order": order, "sources": [],
                }
                cited[target] = cand
            cand["support"] += 1
            cand["sources"].append(sid)
            if not cand["title"]:
                cand["title"] = _clean_title(ref.title)

    # --- OpenAlex ------------------------------------------------------------
    sel_work: Dict[str, Dict[str, Any]] = {}
    sel_count_fix: Dict[str, Optional[int]] = {}
    citers: Dict[str, Dict[str, Any]] = {}   # W id -> candidate
    ok_calls = 0
    mismatched: set = set()                  # ids whose OpenAlex record is a different paper
    fallback_used: set = set()               # mismatches repaired by a title search

    def title_fallback(key: str, meta: Dict[str, str]) -> Optional[int]:
        """Count from a title search with the index title (one request); ``None`` = no match."""

        assert client is not None
        found = client.work_by_title(meta["title"])
        if not found:
            return None
        fallback_used.add(key)
        return _int(found.get("cited_by_count"))
    if use_openalex:
        assert client is not None
        with ThreadPoolExecutor(max_workers=OPENALEX_WORKERS) as pool:
            sel_list = [sid for sid in sel_ids if sid in sel_arxiv]
            works = list(pool.map(client.work_by_arxiv, sel_list))
            for sid, work in zip(sel_list, works):
                if work:
                    sel_work[sid] = work
                    meta = _index_meta(resolver, sid)
                    if _openalex_mismatch(meta, work.get("title")):
                        mismatched.add(sid)
                        sel_count_fix[sid] = title_fallback(sid, meta)       # type: ignore[arg-type]
            with_work = [sid for sid in sel_list if sid in sel_work]
            citing = list(pool.map(lambda sid: client.citing_works(sel_work[sid]["id"]), with_work))
            for sid, items in zip(with_work, citing):
                for item in items or []:
                    wid = short_id(item.get("id"))
                    if not wid:
                        continue
                    arxiv = clean_arxiv_id(arxiv_id_of_work(item))
                    if wid == short_id(sel_work[sid]["id"]):
                        continue
                    cand = citers.get(wid)
                    if cand is None:
                        cand = {
                            "id": arxiv or wid, "arxiv_id": arxiv, "wid": wid,
                            "title": _clean_title(item.get("title")),
                            "year": _int(item.get("publication_year")),
                            "cited_by_count": _int(item.get("cited_by_count")),
                            "support": 0, "order": len(citers), "sources": [],
                        }
                        citers[wid] = cand
                    if sid not in cand["sources"]:
                        cand["support"] += 1
                        cand["sources"].append(sid)

            # Citers with an index entry: index title/year; a wrong OpenAlex record loses its count.
            for cand in citers.values():
                meta = _index_meta(resolver, cand.get("arxiv_id"))
                if meta is None:
                    continue
                if _openalex_mismatch(meta, cand["title"]):
                    mismatched.add(cand["id"])
                    cand["cited_by_count"] = None
                cand["title"] = meta["title"]
                cand["year"] = _int(meta["year"]) or cand["year"]

            # Candidate counts, years and titles for the foundational list.
            if n_sel >= 2:
                pool_cands = sorted((c for c in cited.values() if c["support"] >= 2), key=_sort_key)
                lookup = pool_cands[:MAX_COCITED_LOOKUPS]
            else:
                lookup = sorted(cited.values(), key=lambda c: c["order"])[:MAX_SINGLE_PAPER_LOOKUPS]
            for cand, work in zip(lookup, pool.map(lambda c: client.work_by_arxiv(c["arxiv_id"]), lookup)):
                if not work:
                    continue
                meta = _index_meta(resolver, cand["arxiv_id"])
                if meta is None:                 # not in the index: OpenAlex is the only source
                    cand["cited_by_count"] = _int(work.get("cited_by_count"))
                    cand["year"] = _int(work.get("publication_year")) or cand["year"]
                    cand["title"] = _clean_title(work.get("title")) or cand["title"]
                elif _openalex_mismatch(meta, work.get("title")):
                    mismatched.add(cand["id"])   # wrong record: title search for the count; title/year from the index
                    cand["cited_by_count"] = title_fallback(cand["id"], meta)
                else:
                    cand["cited_by_count"] = _int(work.get("cited_by_count"))
        ok_calls = client.requests - client.errors + client.cache_hits

    # --- foundational ----------------------------------------------------------
    if n_sel >= 2:
        foundational = sorted((c for c in cited.values() if c["support"] >= 2), key=_sort_key)[:top_n]
    else:
        pool_cands = list(cited.values())
        if any(c["cited_by_count"] is not None for c in pool_cands):
            foundational = sorted(pool_cands, key=_sort_key)[:top_n]
        else:
            foundational = sorted(pool_cands, key=lambda c: c["order"])[:top_n]

    # --- follow-up -------------------------------------------------------------
    followups: List[Dict[str, Any]] = []
    if citers:
        sel_wids = {short_id(work.get("id")) for work in sel_work.values()}
        eligible = [
            c for c in citers.values()
            if c.get("arxiv_id") not in sel_set and c["id"] not in sel_set and c["wid"] not in sel_wids
        ]
        multi = sorted((c for c in eligible if c["support"] >= 2), key=_sort_key)
        singles = sorted((c for c in eligible if c["support"] < 2), key=_sort_key)
        followups = multi[:top_n]
        if len(multi) < MIN_FOLLOWUPS_BEFORE_FILL:
            followups = (multi + singles)[:top_n]

    # --- title / year: the local index always wins over OpenAlex ------------------
    for cand in foundational + followups:
        arxiv = cand.get("arxiv_id")
        meta = _index_meta(resolver, arxiv)
        if meta:
            if cand.get("wid") and cand["id"] in mismatched and cand["cited_by_count"] is None:   # citer: one title search
                cand["cited_by_count"] = title_fallback(cand["id"], meta)
            cand["title"] = meta["title"]
            cand["year"] = _int(meta["year"]) or cand["year"]
        if arxiv and cand["year"] is None:
            cand["year"] = year_from_arxiv_id(arxiv)

    # --- assemble ----------------------------------------------------------------
    nodes: List[Dict[str, Any]] = []
    node_ids: Dict[str, int] = {}

    def add_node(node: Dict[str, Any]) -> None:
        if node["id"] not in node_ids:
            node_ids[node["id"]] = len(nodes)
            nodes.append(node)

    for sid, paper in zip(sel_ids, papers):
        work = sel_work.get(sid) or {}
        meta = _index_meta(resolver, sid) if sid in sel_arxiv else None
        oa_year = None if sid in mismatched else _int(work.get("publication_year"))
        year = (_int(meta["year"]) if meta else None) or oa_year or _int((paper.published or "")[:4]) \
            or (year_from_arxiv_id(sid) if sid in sel_arxiv else None)
        add_node({
            "id": sid, "arxiv_id": sid if sid in sel_arxiv else None,
            "title": _clean_title(paper.title), "year": year,
            "cited_by_count": sel_count_fix[sid] if sid in sel_count_fix else _int(work.get("cited_by_count")),
            "selected": True,
            "role": "selected", "support": 0,
        })
    edges: List[Dict[str, str]] = []

    def add_edge(source: str, target: str) -> None:
        edge = {"source": source, "target": target, "kind": "cites"}
        if edge not in edges:
            edges.append(edge)

    for source, target in sel_cites_sel:
        add_edge(source, target)
    for role, group in (("foundational", foundational), ("followup", followups)):
        for cand in group:
            add_node({
                "id": cand["id"], "arxiv_id": cand.get("arxiv_id"), "title": cand["title"] or cand["id"],
                "year": cand["year"], "cited_by_count": cand["cited_by_count"], "selected": False,
                "role": role, "support": cand["support"],
            })
            for sid in cand["sources"]:
                if role == "foundational":
                    add_edge(sid, cand["id"])
                else:
                    add_edge(cand["id"], sid)

    if not use_openalex:
        status = "no_key"
    elif client is not None and client.errors and ok_calls <= 0:
        status = "error"
    else:
        status = "ok"
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "refs_total": refs_total,
            "refs_resolved": refs_resolved,
            "refs_ambiguous": resolver.counts.get("ambiguous", 0),
            "by_method": by_method,
            "openalex": status,
            "openalex_mismatch": len(mismatched),
            "openalex_title_fallback": len(fallback_used),
        },
    }


def _empty_similarity(status: str) -> Dict[str, Any]:
    return {
        "nodes": [], "edges": [], "prior_works": [], "derivative_works": [],
        "stats": {"seeds": 0, "pool": 0, "requests": 0, "cache_hits": 0, "unverified_dropped": 0, "duration_ms": 0, "status": status},
    }


def citation_graph_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node.  Never raises: a failure yields an empty graph and a log line."""

    started = time.perf_counter()
    papers = list(state.get("selected_papers", []) or [])
    logs = [f"[citation_graph] Dựng đồ thị trích dẫn cho {len(papers)} bài"]
    resolver = RefResolver(config.CITATION_INDEX_PATH)
    client = OpenAlexClient()
    try:
        graph = build_citation_graph(papers, resolver, client, top_n=config.CITATION_TOP_N)
    except Exception as exc:  # noqa: BLE001 - optional enrichment must not fail the run
        logs.append(f"[citation_graph] Bỏ qua do lỗi: {type(exc).__name__}")
        graph = {
            "nodes": [], "edges": [],
            "stats": {"refs_total": 0, "refs_resolved": 0, "refs_ambiguous": 0,
                      "by_method": {m: 0 for m in METHODS}, "openalex": "error" if client.has_key else "no_key",
                      "openalex_mismatch": 0, "openalex_title_fallback": 0},
        }
    stats = graph["stats"]
    logs.append(
        f"[citation_graph] refs {stats['refs_resolved']}/{stats['refs_total']} resolved, "
        f"{len(graph['nodes'])} nodes, {len(graph['edges'])} edges, openalex={stats['openalex']}"
    )
    if config.CITATION_SIMILARITY_ENABLED and papers:
        try:
            graph["similarity"] = build_similarity_graph(papers, client, resolver)
        except Exception as exc:  # noqa: BLE001 - optional enrichment must not fail the run
            logs.append(f"[citation_graph] Bỏ qua đồ thị liên quan do lỗi: {type(exc).__name__}")
            graph["similarity"] = _empty_similarity("error")
        sim = graph["similarity"]["stats"]
        logs.append(
            f"[citation_graph] similarity {len(graph['similarity']['nodes'])} nodes, "
            f"{len(graph['similarity']['edges'])} edges, seeds={sim['seeds']}, pool={sim['pool']}, status={sim['status']}"
        )
    resolver.close()
    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": 0,
        "openalexRequests": client.requests,
        "openalexCacheHits": client.cache_hits,
    }
    return {"citation_graph": graph, "trace_logs": logs, "node_timings": {"citation_graph": timing}}
