"""Citation graph: reference resolver, scoring, no-key report section, OpenAlex cache.

No network: the index is a tiny SQLite file in tmp_path and OpenAlex is faked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from config import config
from nodes import citation_graph as cg
from nodes import final_report
from state import PaperItem, PaperReference
from tools.openalex import OpenAlexClient
from tools.ref_resolver import RefResolver, normalize_title


def _make_index(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE papers (paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT NOT NULL,
            abstract TEXT NOT NULL, categories TEXT NOT NULL, primary_category TEXT NOT NULL,
            first_version_date TEXT NOT NULL, doi TEXT NOT NULL);
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract, content='papers', content_rowid='rowid',
            tokenize='porter unicode61 remove_diacritics 2');
        """
    )
    rows = [
        ("1706.03762", "Attention Is All You Need", "2017-06-12", ""),
        ("2001.00001", "A Study of Sparse Graph Methods", "2020-01-01", "10.1000/XYZ.1"),
        ("2002.00002", "Sparse graph methods: a study", "2020-02-01", ""),
        ("2003.00003", "A study of sparse graph methods!", "2020-03-01", ""),   # same normalized title as 2001
        ("2004.00004", "Unique Title For The Doi Test", "2020-04-01", "10.1000/doi.only"),
        ("2005.00005", "Deep Learning", "2020-05-01", ""),
    ]
    for paper_id, title, date, doi in rows:
        conn.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)", (paper_id, title, "A", "abs", "cs.AI", "cs.AI", date, doi))
    conn.execute("INSERT INTO papers_fts(rowid, title, abstract) SELECT rowid, title, abstract FROM papers")
    conn.commit()
    conn.close()
    return path


def _ref(index: int, title: str = "", arxiv_id: Optional[str] = None, doi: Optional[str] = None) -> PaperReference:
    return PaperReference(index=index, text=title, title=title, arxiv_id=arxiv_id, doi=doi)


def test_resolver_order_and_ambiguous_skip(tmp_path: Path) -> None:
    resolver = RefResolver(_make_index(tmp_path / "index.sqlite"))
    # Explicit id wins over everything and loses the version suffix.
    got = resolver.resolve(_ref(1, "Attention Is All You Need", arxiv_id="0000.00000v2", doi="10.1000/doi.only"))
    assert (got.arxiv_id, got.method) == ("0000.00000", "explicit")
    # arXiv DOI comes before an index DOI and the title.
    got = resolver.resolve(_ref(2, "Attention Is All You Need", doi="https://doi.org/10.48550/arXiv.1234.56789v3"))
    assert (got.arxiv_id, got.method) == ("1234.56789", "arxiv_doi")
    # Index DOI (case-insensitive) comes before the title.
    got = resolver.resolve(_ref(3, "Attention Is All You Need", doi="10.1000/DOI.ONLY"))
    assert (got.arxiv_id, got.method) == ("2004.00004", "index_doi")
    # Normalized-title exact match: case, punctuation and LaTeX are ignored.
    got = resolver.resolve(_ref(4, "attention is all you need."))
    assert (got.arxiv_id, got.method) == ("1706.03762", "index_title")
    assert normalize_title(r"\textbf{Attention}: is  ALL you-need") == "attention is all you need"
    # Under 4 words: skipped even though one row matches exactly.
    assert resolver.resolve(_ref(5, "Deep Learning")) is None
    # Two rows share the normalized title: skipped as ambiguous (not a guess).
    assert resolver.resolve(_ref(6, "A study of sparse graph methods")) is None
    assert resolver.counts["ambiguous"] == 1
    # Substring of a longer title is not a match.
    assert resolver.resolve(_ref(7, "Attention Is All You Need Really")) is None
    resolver.close()

    # No index file: explicit ids and arXiv DOIs still work, titles do not.
    offline = RefResolver(tmp_path / "missing.sqlite")
    assert not offline.index_available
    assert offline.resolve(_ref(1, "Attention Is All You Need")) is None
    assert offline.resolve(_ref(2, "x", arxiv_id="1706.03762")).method == "explicit"


class FakeOpenAlex:
    """Stands in for OpenAlexClient: works by arXiv id and citing works by W id."""

    has_key = True
    requests = 0
    cache_hits = 0
    errors = 0

    def __init__(self, works: Dict[str, Dict[str, Any]], citers: Dict[str, List[Dict[str, Any]]]) -> None:
        self.works, self.citers = works, citers

    def work_by_arxiv(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        return self.works.get(arxiv_id)

    titles: Dict[str, Dict[str, Any]] = {}      # normalized-index-title lookups for work_by_title

    def work_by_title(self, title: str, min_jaccard: float = 0.8) -> Optional[Dict[str, Any]]:
        return self.titles.get(title)

    def citing_works(self, work_id: str, per_page: int = 50) -> List[Dict[str, Any]]:
        return self.citers.get(work_id.rsplit("/", 1)[-1], [])


def _paper(arxiv_id: str, cited: List[str]) -> PaperItem:
    return PaperItem(
        paper_id=f"arxiv_{arxiv_id}", title=f"Paper {arxiv_id}", arxiv_id=arxiv_id,
        references=[_ref(i, f"ref {c}", arxiv_id=c) for i, c in enumerate(cited, 1)],
    )


def _work(wid: str, title: str, year: int, cites: int, arxiv_id: Optional[str] = None) -> Dict[str, Any]:
    work: Dict[str, Any] = {"id": f"https://openalex.org/{wid}", "title": title, "publication_year": year, "cited_by_count": cites}
    if arxiv_id:
        work["locations"] = [{"landing_page_url": f"https://arxiv.org/abs/{arxiv_id}"}]
    return work


def test_scoring_roles_on_toy_graph(tmp_path: Path) -> None:
    papers = [
        _paper("2301.00001", ["1111.00001", "1111.00002", "1111.00003", "2301.00002", "1111.00009"]),
        _paper("2301.00002", ["1111.00001", "1111.00002", "1111.00004"]),
        _paper("2301.00003", ["1111.00001", "1111.00003", "1111.00005"]),
    ]
    works = {
        "2301.00001": _work("W1", "P1", 2023, 10),
        "2301.00002": _work("W2", "P2", 2023, 11),
        "2301.00003": _work("W3", "P3", 2023, 12),
        "1111.00001": _work("W11", "Old A", 2011, 100),
        "1111.00002": _work("W12", "Old B", 2012, 900),
        "1111.00003": _work("W13", "Old C", 2013, 900),
    }
    citers = {
        "W1": [_work("W90", "Both", 2024, 5, "2401.00090"), _work("W91", "Single big", 2024, 500, "2401.00091"),
               _work("W2", "Selected P2", 2023, 11, "2301.00002")],
        "W2": [_work("W90", "Both", 2024, 5, "2401.00090"), _work("W92", "Single small", 2025, 1)],
        "W3": [_work("W93", "Single mid", 2025, 50)],
    }
    resolver = RefResolver(tmp_path / "missing.sqlite")
    graph = cg.build_citation_graph(papers, resolver, FakeOpenAlex(works, citers), top_n=5)

    roles: Dict[str, List[str]] = {}
    for node in graph["nodes"]:
        roles.setdefault(node["role"], []).append(node["id"])
    assert roles["selected"] == ["2301.00001", "2301.00002", "2301.00003"]
    # support 3 first; then support 2 by cited_by_count (900 before 100... C and B tie at 900 -> year desc: C).
    assert roles["foundational"] == ["1111.00001", "1111.00003", "1111.00002"]
    # One multi-citer (< 3), so top-cited singles fill; selected P2 (a citer of P1) is excluded.
    assert roles["followup"] == ["2401.00090", "2401.00091", "W93", "W92"]
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["1111.00001"]["support"] == 3 and by_id["1111.00002"]["cited_by_count"] == 900
    assert by_id["2401.00090"]["support"] == 2 and by_id["W92"]["arxiv_id"] is None
    assert {"source": "2301.00001", "target": "2301.00002", "kind": "cites"} in graph["edges"]
    assert {"source": "2401.00090", "target": "2301.00002", "kind": "cites"} in graph["edges"]
    stats = graph["stats"]
    assert (stats["refs_total"], stats["refs_resolved"], stats["openalex"]) == (11, 11, "ok")
    assert stats["by_method"]["explicit"] == 11


def test_mismatched_record_uses_title_search_else_keeps_node_without_count(tmp_path: Path) -> None:
    resolver = RefResolver(_make_index(tmp_path / "index.sqlite"))
    papers = [_paper("2301.00001", ["1706.03762", "2001.00001", "2005.00005"]),
              _paper("2301.00002", ["1706.03762", "2001.00001", "2005.00005"])]
    works = {
        "2301.00001": _work("W1", "P1", 2023, 1), "2301.00002": _work("W2", "P2", 2023, 1),
        "1706.03762": _work("W10", "Totally Unrelated Corrupt Record", 1999, 777),         # wrong, title search finds it
        "2001.00001": _work("W11", "A study of sparse graph methods", 2021, 42),           # right record
        "2005.00005": _work("W12", "Another Corrupt Title Here", 1990, 999),               # wrong, no title match
    }
    client = FakeOpenAlex(works, {})
    client.titles = {"Attention Is All You Need": _work("W99", "Attention is all you need", 2017, 120000)}
    graph = cg.build_citation_graph(papers, resolver, client, top_n=5)
    by_id = {n["id"]: n for n in graph["nodes"]}
    fixed = by_id["1706.03762"]
    assert (fixed["title"], fixed["year"], fixed["cited_by_count"]) == ("Attention Is All You Need", 2017, 120000)
    ok = by_id["2001.00001"]                      # index title/year kept; OpenAlex count used
    assert (ok["title"], ok["year"], ok["cited_by_count"]) == ("A Study of Sparse Graph Methods", 2020, 42)
    unknown = by_id["2005.00005"]                 # kept, count unknown, ranked last
    assert (unknown["title"], unknown["year"], unknown["cited_by_count"]) == ("Deep Learning", 2020, None)
    assert [n["id"] for n in graph["nodes"] if n["role"] == "foundational"] == ["1706.03762", "2001.00001", "2005.00005"]
    assert graph["stats"]["openalex_mismatch"] == 2 and graph["stats"]["openalex_title_fallback"] == 1
    resolver.close()


def test_no_key_mode_still_yields_section_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENALEX_API_KEY", "")
    monkeypatch.setattr(config, "CITATION_INDEX_PATH", tmp_path / "missing.sqlite")
    papers = [_paper("2301.00001", ["1111.00001", "1111.00002"]), _paper("2301.00002", ["1111.00001", "1111.00003"])]
    update = cg.citation_graph_node({"selected_papers": papers})
    graph = update["citation_graph"]
    assert graph["stats"]["openalex"] == "no_key"
    assert [n["id"] for n in graph["nodes"] if n["role"] == "foundational"] == ["1111.00001"]
    assert "durationMs" in update["node_timings"]["citation_graph"]

    report = final_report._assemble_parallel_report(
        "T", "Vietnamese", papers, ["### A"], {"summary": "s", "conclusion": "c"},
        final_report._context_section(graph, "Vietnamese"),
    )
    heads = [line for line in report.split("\n") if line.startswith("## ")]
    assert heads[2] == "## 3. Bối cảnh nghiên cứu" and heads[3] == "## 4. Kết luận"
    section = report.split("## 3. Bối cảnh nghiên cứu")[1].split("## 4.")[0]
    assert "**Bài nền tảng**" in section and "](https://arxiv.org/abs/1111.00001)" in section
    assert "được 2/2 bài đã chọn trích dẫn" in section
    assert "**Bài kế thừa**" in section and "OPENALEX_API_KEY" in section
    english = final_report._context_section(graph, "English")
    assert "cited by 2/2 selected papers" in english and "OpenAlex API key" in english


def test_openalex_client_caches_without_key_and_is_quiet_without_one(tmp_path: Path) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_get(url: str, params: Dict[str, str], timeout: int, headers: Dict[str, str]) -> Any:
        calls.append({"url": url, "params": params})
        if "9999.99999" in url:
            return SimpleNamespace(status_code=404, json=lambda: {})
        if url.startswith("https://api.openalex.org/works?"):
            return SimpleNamespace(status_code=200, json=lambda: {"results": [{"id": "https://openalex.org/W5"}]})
        return SimpleNamespace(status_code=200, json=lambda: {"id": "https://openalex.org/W1", "cited_by_count": 7})

    sleeps: List[float] = []
    client = OpenAlexClient("sekret-key", min_interval=0.2, cache_dir=tmp_path, http_get=fake_get,
                            clock=lambda: 0.0, sleep=sleeps.append)
    assert client.work_by_arxiv("1706.03762")["cited_by_count"] == 7
    assert client.work_by_arxiv("1706.03762")["cited_by_count"] == 7      # served from disk
    assert client.citing_works("https://openalex.org/W1")[0]["id"].endswith("W5")
    assert client.work_by_arxiv("9999.99999") is None and client.work_by_arxiv("9999.99999") is None
    assert len(calls) == 3 and client.cache_hits == 2 and client.errors == 0
    assert sleeps == [0.2, 0.2]                                           # starts paced; cache hits are not
    assert calls[0]["url"] == "https://api.openalex.org/works/doi:10.48550/arXiv.1706.03762?select=" \
        "id,title,publication_year,cited_by_count,ids,doi,locations"
    assert calls[0]["params"] == {"api_key": "sekret-key"}
    assert "filter=cites:W1" in calls[1]["url"] and "per_page=50" in calls[1]["url"]
    assert not any("sekret-key" in p.read_text() for p in tmp_path.glob("*.json"))   # key is never cached

    quiet = OpenAlexClient("", cache_dir=tmp_path / "none", http_get=fake_get)
    assert quiet.has_key is False and quiet.work_by_arxiv("1706.03762") is None and quiet.citing_works("W1") is None
    assert len(calls) == 3


def test_api_exposes_citation_graph_and_node_label() -> None:
    import api

    graph = {"nodes": [{"id": "1111.00001", "arxiv_id": "1111.00001", "title": "T", "year": 2011, "cited_by_count": None,
                        "selected": False, "role": "foundational", "support": 2}],
             "edges": [], "stats": {"refs_total": 3, "refs_resolved": 1, "by_method": {"explicit": 1}, "openalex": "no_key"}}
    result = api._result_from_state({"citation_graph": graph, "selected_papers": []})
    dumped = result.model_dump(by_alias=True)
    assert dumped["citationGraph"]["nodes"][0]["id"] == "1111.00001" and dumped["citationGraph"]["stats"]["openalex"] == "no_key"
    assert api._result_from_state({"selected_papers": []}).model_dump(by_alias=True)["citationGraph"] is None
    assert "citation_graph" in api.NODE_LABELS and "citation_graph" in api.NODE_KINDS
