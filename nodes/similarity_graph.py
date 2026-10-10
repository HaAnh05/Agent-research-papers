"""Connected-Papers-style similarity graph around the selected papers (no LLM call).

``build_similarity_graph`` returns (stored in ``citation_graph["similarity"]``)

    {"nodes": [{id, arxiv_id, title, year, cited_by_count, first_author, abstract,
                similarity, role: "origin"|"similar", url}],
     "edges": [{source, target, weight}],            # undirected, each pair once
     "prior_works": [node + support], "derivative_works": [node + support],
     "stats": {seeds, pool, requests, cache_hits, duration_ms, status}}

``status`` is ``ok``, ``no_key``, ``error`` (nothing usable) or ``partial``
(wall-clock budget reached or some request failed; the graph uses what exists).

Method (OpenAlex ``referenced_works`` is often empty for preprints, so the
references of the selected papers come from the HTML bibliography):

1. seeds = HTML-resolved reference arXiv ids -> OpenAlex ids (batched DOI
   filter) + ``referenced_works`` of the selected works;
2. pool = citers of every seed and of the selected works; keep the ``pool_size``
   works that cite the most seeds; fetch their details (with ``referenced_works``);
3. score = cosine bibliographic coupling ``|refs & seeds| / sqrt(|refs| |seeds|)``
   + ``CITATION_SIMILARITY_DIRECT_BONUS`` when the work cites a selected work
   + ``CITATION_SIMILARITY_INDEX_BONUS`` when the work is in the local cs.* index
   (arXiv id from the seed DOI map, the OpenAlex DOI, or an exact title match);
   candidates sharing fewer than ``CITATION_SIMILARITY_MIN_SHARED`` seeds are dropped
   unless they cite a selected work.  Title, year and abstract come from the local
   index when the paper is in it (OpenAlex titles of arXiv preprints can be wrong),
   else from the HTML bibliography title of a seed, else from OpenAlex;
   ``title_source`` records which one;
4. edges = cosine of reference sets between graph nodes (origin = its seeds);
   each node keeps its ``edges_per_node`` strongest edges >= ``EDGE_MIN_WEIGHT``;
5. prior works = most referenced by the graph nodes; derivative works = pool works
   citing the most graph nodes (from the lists fetched in step 2).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from config import config
from state import PaperItem
from tools.openalex import OpenAlexClient, arxiv_id_of_work, short_id
from tools.ref_resolver import RefResolver, clean_arxiv_id

EDGE_MIN_WEIGHT = 0.05
LIST_SIZE = 10
BATCH = 50
WORKERS = 4


def _cosine(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _year(work: Dict[str, Any]) -> Optional[int]:
    try:
        return int(work["publication_year"]) if work.get("publication_year") is not None else None
    except (TypeError, ValueError):
        return None


def _count(work: Dict[str, Any]) -> Optional[int]:
    try:
        return int(work["cited_by_count"]) if work.get("cited_by_count") is not None else None
    except (TypeError, ValueError):
        return None


def _first_author(work: Dict[str, Any]) -> str:
    for auth in work.get("authorships") or []:
        if isinstance(auth, dict):
            name = (auth.get("author") or {}).get("display_name") if isinstance(auth.get("author"), dict) else None
            if name:
                return " ".join(str(name).split())
    return ""


def _refs_of(work: Dict[str, Any]) -> FrozenSet[str]:
    return frozenset(short_id(r) for r in (work.get("referenced_works") or []) if r)


class _Budget:
    """Wall-clock budget and request counter shared by the worker threads."""

    def __init__(self, seconds: float, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.deadline = clock() + max(0.0, seconds)
        self.hit = False
        self.failed = 0
        self.calls = 0
        self._lock = threading.Lock()

    def call(self, fn: Callable[[], Optional[Any]]) -> Optional[Any]:
        """Run one OpenAlex call unless the budget is spent; ``None`` = no data."""

        with self._lock:
            if self.clock() >= self.deadline:
                self.hit = True
                return None
            self.calls += 1
        result = fn()
        if result is None:
            with self._lock:
                self.failed += 1
        return result


def _empty(status: str, started: float, **stats: Any) -> Dict[str, Any]:
    return {
        "nodes": [], "edges": [], "prior_works": [], "derivative_works": [],
        "stats": {"seeds": 0, "pool": 0, "requests": 0, "cache_hits": 0,
                  "duration_ms": max(0, int((time.perf_counter() - started) * 1000)), "status": status, **stats},
    }


def build_similarity_graph(
    selected_papers: List[PaperItem],
    client: Optional[OpenAlexClient],
    resolver: RefResolver,
    *,
    max_nodes: Optional[int] = None,
    pool_size: Optional[int] = None,
    edges_per_node: Optional[int] = None,
    budget_sec: Optional[float] = None,
    index_bonus: Optional[float] = None,
    direct_bonus: Optional[float] = None,
    min_shared: Optional[int] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    started = time.perf_counter()
    max_nodes = config.CITATION_SIMILARITY_NODES if max_nodes is None else max_nodes
    pool_size = config.CITATION_SIMILARITY_POOL if pool_size is None else pool_size
    edges_per_node = config.CITATION_SIMILARITY_EDGES_PER_NODE if edges_per_node is None else edges_per_node
    budget_sec = config.CITATION_SIMILARITY_BUDGET_SEC if budget_sec is None else budget_sec
    index_bonus = config.CITATION_SIMILARITY_INDEX_BONUS if index_bonus is None else index_bonus
    direct_bonus = config.CITATION_SIMILARITY_DIRECT_BONUS if direct_bonus is None else direct_bonus
    min_shared = config.CITATION_SIMILARITY_MIN_SHARED if min_shared is None else min_shared
    if client is None or not client.has_key:
        return _empty("no_key", started)

    budget = _Budget(budget_sec, clock)
    hits_before = getattr(client, "cache_hits", 0)

    # --- 1. selected works and seeds ------------------------------------------------
    sel_ids = [clean_arxiv_id(p.arxiv_id) for p in selected_papers]
    sel_arxiv = [i for i in dict.fromkeys(sel_ids) if i]
    sel_works: Dict[str, Dict[str, Any]] = {}          # arxiv id -> OpenAlex work
    for i in range(0, len(sel_arxiv), BATCH):
        chunk = sel_arxiv[i:i + BATCH]
        for work in budget.call(lambda c=chunk: client.works_by_arxiv_ids(c)) or []:
            arxiv = clean_arxiv_id(arxiv_id_of_work(work))
            if arxiv in chunk:
                sel_works[arxiv] = work
    sel_wids = {short_id(w.get("id")) for w in sel_works.values()}

    ref_arxiv: Dict[int, List[str]] = {}               # paper index -> resolved reference arXiv ids
    all_ref_arxiv: List[str] = []
    ref_titles: Dict[str, str] = {}                    # arxiv id -> title in the HTML bibliography
    for idx, paper in enumerate(selected_papers):
        found: List[str] = []
        for ref in paper.references:
            res = resolver.resolve(ref)
            if res is not None and res.arxiv_id not in found and res.arxiv_id != sel_ids[idx]:
                found.append(res.arxiv_id)
                if ref.title and res.arxiv_id not in ref_titles:
                    ref_titles[res.arxiv_id] = " ".join(ref.title.split())
        ref_arxiv[idx] = found
        all_ref_arxiv.extend(a for a in found if a not in all_ref_arxiv)
    arxiv_to_w: Dict[str, str] = {}
    for i in range(0, len(all_ref_arxiv), BATCH):
        chunk = all_ref_arxiv[i:i + BATCH]
        for work in budget.call(lambda c=chunk: client.works_by_arxiv_ids(c, "id,doi,ids")) or []:
            arxiv = clean_arxiv_id(arxiv_id_of_work(work))
            if arxiv and work.get("id"):
                arxiv_to_w[arxiv] = short_id(work["id"])

    w_arxiv: Dict[str, str] = {short_id(w.get("id")): a for a, w in sel_works.items()}
    w_arxiv.update({w: a for a, w in arxiv_to_w.items()})     # W id -> arXiv id for selected works and seeds

    origins: List[Dict[str, Any]] = []                 # one per selected paper
    seen_origin: Set[str] = set()
    for idx, paper in enumerate(selected_papers):
        work = sel_works.get(sel_ids[idx]) or {}
        refs = {arxiv_to_w[a] for a in ref_arxiv[idx] if a in arxiv_to_w} | set(_refs_of(work))
        key = short_id(work.get("id")) or f"arxiv:{sel_ids[idx]}"
        if key in seen_origin:
            continue
        seen_origin.add(key)
        origins.append({"paper": paper, "arxiv_id": sel_ids[idx], "work": work, "key": key, "refs": frozenset(refs)})
    seeds: FrozenSet[str] = frozenset().union(*(o["refs"] for o in origins)) - sel_wids if origins else frozenset()

    # --- 2. pool of citers ---------------------------------------------------------------
    seed_list = sorted(seeds)
    sources = seed_list + sorted(sel_wids)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        citer_lists = list(ex.map(lambda w: budget.call(lambda: client.citer_ids(w)), sources))
    freq: Counter = Counter()
    direct: Set[str] = set()
    for source, ids in zip(sources, citer_lists):
        for wid in dict.fromkeys(ids or []):
            freq[wid] += 1
            if source in sel_wids:
                direct.add(wid)
    for wid in list(freq):
        if wid in sel_wids:
            del freq[wid]
    pool_ids = [wid for wid, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:pool_size]]

    # --- 3. details of the pool --------------------------------------------------------
    details: Dict[str, Dict[str, Any]] = {}
    batches = [pool_ids[i:i + BATCH] for i in range(0, len(pool_ids), BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(lambda b: budget.call(lambda: client.works_by_ids(b)), batches))
    for works in results:
        for work in works or []:
            wid = short_id(work.get("id"))
            if wid and wid not in sel_wids:
                details[wid] = work

    # --- 4. identity (arXiv id, index record) and scoring --------------------------------
    ident_cache: Dict[str, Tuple[Optional[str], Optional[Dict[str, str]]]] = {}

    def identify(wid: str, work: Dict[str, Any], hint: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        if wid not in ident_cache:
            arxiv = hint or w_arxiv.get(wid) or clean_arxiv_id(arxiv_id_of_work(work))
            record = resolver.lookup_record(arxiv) if arxiv else None
            if arxiv is None:
                arxiv = resolver.resolve_title(str(work.get("title") or ""))
                record = resolver.lookup_record(arxiv) if arxiv else None
            ident_cache[wid] = (arxiv, record)
        return ident_cache[wid]

    sel_arxiv_set = set(sel_arxiv)
    scored: List[Tuple[float, float, str]] = []
    for wid, work in details.items():
        refs = _refs_of(work)
        arxiv, record = identify(wid, work)
        if arxiv in sel_arxiv_set or not refs:
            continue
        shared = len(refs & seeds)
        is_direct = wid in direct
        if shared < max(1, min_shared) and not is_direct:
            continue
        cos = _cosine(refs, seeds)
        score = cos + (direct_bonus if is_direct else 0.0) + (index_bonus if record else 0.0)
        scored.append((score, cos, wid))
    scored.sort(key=lambda t: (-t[0], -(_count(details[t[2]]) or 0), t[2]))
    top = scored[:max(0, max_nodes)]

    # --- 5. nodes ----------------------------------------------------------------------------
    def node_dict(wid: str, work: Dict[str, Any], role: str, similarity: float,
                  paper: Optional[PaperItem] = None, arxiv_hint: Optional[str] = None) -> Dict[str, Any]:
        arxiv, record = identify(wid, work, arxiv_hint)
        openalex_title = " ".join(str(work.get("title") or "").split())
        if record and record["title"]:
            title, title_source = record["title"], "index"
        elif arxiv and ref_titles.get(arxiv):
            title, title_source = ref_titles[arxiv], "reference"
        elif paper and paper.title:
            title, title_source = paper.title, "paper"
        else:
            title, title_source = openalex_title or wid, "openalex"
        year = int(record["year"]) if record and record["year"].isdigit() else None
        year = year or _year(work)
        if year is None and paper and paper.published[:4].isdigit():
            year = int(paper.published[:4])
        if arxiv:
            url = f"https://arxiv.org/abs/{arxiv}"
        else:
            url = str(work.get("doi") or "") or (f"https://openalex.org/{wid}" if wid.startswith("W") else "")
        return {
            "id": wid, "arxiv_id": arxiv, "title": title, "title_source": title_source, "year": year, "cited_by_count": _count(work),
            "first_author": _first_author(work) or (paper.authors[0] if paper and paper.authors else ""),
            "abstract": (record["abstract"] if record else "") or (paper.summary if paper else ""),
            "similarity": round(similarity, 4), "role": role, "url": url,
        }

    nodes: List[Dict[str, Any]] = []
    refsets: Dict[str, FrozenSet[str]] = {}
    for o in origins:
        node = node_dict(o["key"], o["work"], "origin", 1.0, paper=o["paper"], arxiv_hint=o["arxiv_id"])
        nodes.append(node)
        refsets[node["id"]] = o["refs"]
    for _score, cos, wid in top:
        nodes.append(node_dict(wid, details[wid], "similar", cos))
        refsets[wid] = _refs_of(details[wid])

    # --- 6. edges ------------------------------------------------------------------------------
    ids = [n["id"] for n in nodes]
    neighbours: Dict[str, List[Tuple[float, str]]] = {i: [] for i in ids}
    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1:]:
            w = _cosine(refsets[a], refsets[b])
            if w >= EDGE_MIN_WEIGHT:
                neighbours[a].append((w, b))
                neighbours[b].append((w, a))
    chosen: Dict[Tuple[str, str], float] = {}
    for a, lst in neighbours.items():
        for w, b in sorted(lst, key=lambda t: (-t[0], t[1]))[:max(0, edges_per_node)]:
            chosen[tuple(sorted((a, b)))] = w          # type: ignore[index]
    edges = [{"source": a, "target": b, "weight": round(w, 4)}
             for (a, b), w in sorted(chosen.items(), key=lambda kv: (-kv[1], kv[0]))]

    # --- 7. prior and derivative works -------------------------------------------------------------
    node_set = set(ids)
    prior_count: Counter = Counter()
    for refs in refsets.values():
        prior_count.update(refs)
    prior_ranked = [w for w, _ in sorted(prior_count.items(), key=lambda kv: (-kv[1], kv[0]))
                    if w not in sel_wids and w not in node_set][:LIST_SIZE]
    missing = [w for w in prior_ranked if w not in details]
    for i in range(0, len(missing), BATCH):
        chunk = missing[i:i + BATCH]
        for work in budget.call(lambda c=chunk: client.works_by_ids(c)) or []:
            wid = short_id(work.get("id"))
            if wid:
                details[wid] = work
    prior_works = []
    for wid in prior_ranked:
        if wid in details:
            item = node_dict(wid, details[wid], "prior", 0.0)
            item.update(similarity=None, support=prior_count[wid])
            prior_works.append(item)

    derivative_ranked = sorted(
        ((len(_refs_of(w) & node_set), wid) for wid, w in details.items()
         if wid not in node_set and wid not in sel_wids),
        key=lambda t: (-t[0], -(_count(details[t[1]]) or 0), t[1]),
    )
    derivative_works = []
    for support, wid in derivative_ranked[:LIST_SIZE]:
        if support <= 0:
            break
        item = node_dict(wid, details[wid], "derivative", 0.0)
        item.update(similarity=None, support=support)
        derivative_works.append(item)

    # --- 8. status ------------------------------------------------------------------------------------
    usable = bool(details or sel_works)
    if budget.failed and not usable and not budget.hit:
        status = "error"
    elif budget.hit or budget.failed:
        status = "partial"
    else:
        status = "ok"
    return {
        "nodes": nodes, "edges": edges, "prior_works": prior_works, "derivative_works": derivative_works,
        "stats": {
            "seeds": len(seeds), "pool": len(pool_ids), "requests": budget.calls,
            "cache_hits": getattr(client, "cache_hits", 0) - hits_before,
            "duration_ms": max(0, int((time.perf_counter() - started) * 1000)), "status": status,
        },
    }
