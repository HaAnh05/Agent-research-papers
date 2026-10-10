"""Similarity graph: scoring, edges, budget and no-key behaviour (fake OpenAlex, no network)."""

from __future__ import annotations

from typing import Any, Dict, List

import sqlite3
from pathlib import Path

from nodes.similarity_graph import build_similarity_graph, _cosine
from state import PaperItem, PaperReference
from tools.openalex import arxiv_id_of_work, short_id
from tools.ref_resolver import RefResolver

SEEDS = {f"S{i}": f"1111.0000{i}" for i in range(1, 5)}      # seed W id -> arXiv id


def _work(wid: str, refs: List[str], arxiv: str = "", count: int = 10) -> Dict[str, Any]:
    return {
        "id": f"https://openalex.org/{wid}", "title": f"Title {wid}", "publication_year": 2024,
        "cited_by_count": count, "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "doi": f"https://doi.org/10.48550/arxiv.{arxiv}" if arxiv else None,
        "authorships": [{"author": {"display_name": f"Author {wid}"}}],
    }


class FakeOpenAlex:
    has_key = True

    def __init__(self, citers: Dict[str, List[str]], works: List[Dict[str, Any]], key: bool = True) -> None:
        self.has_key = key
        self.citers = citers
        self.works = {short_id(w["id"]): w for w in works}
        self.requests = 0
        self.cache_hits = 0

    def works_by_arxiv_ids(self, arxiv_ids: List[str], select: str = "") -> List[Dict[str, Any]]:
        return [w for w in self.works.values() if arxiv_id_of_work(w) in arxiv_ids]

    def works_by_ids(self, work_ids: List[str], select: str = "") -> List[Dict[str, Any]]:
        return [self.works[short_id(i)] for i in work_ids if short_id(i) in self.works]

    def citer_ids(self, work_id: str, per_page: int = 100) -> List[str]:
        return list(self.citers.get(short_id(work_id), []))


def _fixture():
    works = [_work(w, [], arxiv=a) for w, a in SEEDS.items()]
    works += [
        _work("W0", [], arxiv="2001.00001"),                                   # selected paper
        _work("WA", ["S1", "S2", "S3", "S4"]),                                 # cos 1.0
        _work("WC", ["S2", "S3"]),                                             # cos 0.707
        _work("WB", ["S1"] + [f"X{i}" for i in range(8)]),                     # cos 0.167
        _work("WD", ["S4", "WA", "WC"] + [f"Y{i}" for i in range(5)]),         # cites 2 graph nodes
    ]
    citers = {
        "S1": ["W0", "WA", "WB"], "S2": ["WA", "WC"], "S3": ["WA", "WC"], "S4": ["WA", "WD"],
    }
    paper = PaperItem(
        paper_id="arxiv_2001.00001", title="Origin", arxiv_id="2001.00001",
        references=[PaperReference(index=i, arxiv_id=a, title=f"ref {i}") for i, a in enumerate(SEEDS.values())],
    )
    return paper, FakeOpenAlex(citers, works)


def test_ranking_drops_origin_and_lists_prior_derivative() -> None:
    paper, client = _fixture()
    graph = build_similarity_graph([paper], client, RefResolver(None), max_nodes=2, budget_sec=60)
    assert [n["id"] for n in graph["nodes"]] == ["W0", "WA", "WC"]          # origin once, best match first
    assert graph["nodes"][0]["role"] == "origin" and graph["nodes"][1]["similarity"] == 1.0
    assert graph["nodes"][1]["first_author"] == "Author WA" and graph["nodes"][0]["arxiv_id"] == "2001.00001"
    assert [w["id"] for w in graph["derivative_works"]] == ["WD"] and graph["derivative_works"][0]["support"] == 2
    assert {w["id"] for w in graph["prior_works"]} == {"S1", "S2", "S3", "S4"}
    assert graph["stats"]["status"] == "ok" and graph["stats"]["seeds"] == 4


def test_edges_are_top_k_and_undirected() -> None:
    paper, client = _fixture()
    graph = build_similarity_graph([paper], client, RefResolver(None), max_nodes=3, edges_per_node=1, budget_sec=60)
    pairs = [frozenset((e["source"], e["target"])) for e in graph["edges"]]
    assert len(pairs) == len(set(pairs))                                    # each pair stored once
    assert all(e["weight"] > 0 for e in graph["edges"])
    degree = {n["id"]: sum(n["id"] in (e["source"], e["target"]) for e in graph["edges"]) for n in graph["nodes"]}
    assert all(d >= 1 for d in degree.values())                              # each node has its best neighbour
    top = next(e for e in graph["edges"] if e["weight"] == 1.0)
    assert {top["source"], top["target"]} == {"W0", "WA"}                    # origin refs == WA refs
    assert _cosine(frozenset("ab"), frozenset("bc")) == _cosine(frozenset("bc"), frozenset("ab")) == 0.5


def test_budget_exceeded_returns_partial() -> None:
    paper, client = _fixture()
    ticks = iter(range(1000))
    graph = build_similarity_graph([paper], client, RefResolver(None), budget_sec=3, clock=lambda: float(next(ticks)))
    assert graph["stats"]["status"] == "partial"
    assert isinstance(graph["nodes"], list) and graph["stats"]["requests"] <= 3


def test_no_key_status() -> None:
    paper, _ = _fixture()
    for client in (FakeOpenAlex({}, [], key=False), None):
        graph = build_similarity_graph([paper], client, RefResolver(None))
        assert graph["stats"]["status"] == "no_key" and graph["nodes"] == []


def test_identity_from_index_fixes_openalex_titles(tmp_path: Path) -> None:
    from tests.test_citation_graph import _make_index

    index = _make_index(tmp_path / "index.sqlite")
    conn = sqlite3.connect(index)
    conn.execute("INSERT INTO papers VALUES ('1111.00001','Real Seed Paper On Reasoning','A','Seed abstract','cs.AI','cs.AI','2011-01-01','')")
    conn.execute("INSERT INTO papers_fts(rowid, title, abstract) SELECT rowid, title, abstract FROM papers WHERE paper_id='1111.00001'")
    conn.commit()
    conn.close()
    paper, client = _fixture()
    client.works["S1"]["title"] = "Corrupted unrelated OpenAlex title"           # wrong title on a seed work
    client.works["WC"]["title"] = "Attention Is All You Need"                    # no DOI: found by exact title
    resolver = RefResolver(index)
    graph = build_similarity_graph([paper], client, resolver, max_nodes=3, budget_sec=60)
    prior = {w["id"]: w for w in graph["prior_works"]}
    assert prior["S1"]["arxiv_id"] == "1111.00001" and prior["S1"]["title"] == "Real Seed Paper On Reasoning"
    assert prior["S1"]["title_source"] == "index" and prior["S1"]["abstract"] == "Seed abstract"
    assert prior["S2"]["title_source"] == "reference"                             # not in the index: bibliography title
    wc = next(n for n in graph["nodes"] if n["id"] == "WC")
    assert wc["arxiv_id"] == "1706.03762" and wc["title_source"] == "index" and wc["abstract"] == "abs"
    resolver.close()
