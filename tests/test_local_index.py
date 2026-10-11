"""Offline tests for the local FTS5 index (tools/local_index.py) and its search-node wiring."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nodes.search as search_node
from config import config
from state import PaperItem
from tools import local_index
from tools.cache_manager import canonicalize_paper_key
from tools.local_index import build_index, search_local


def _row(pid, title, abstract, cats, date, authors="Ada Lovelace and Alan Turing"):
    return {
        "paper_id": pid, "title": title, "authors": authors, "abstract": abstract,
        "categories": cats, "primary_category": cats.split()[0],
        "first_version_date": date, "doi": None,
    }


ROWS = [
    _row("2401.00001", "Diffusion policy for robot manipulation", "We study visuomotor control.", "cs.RO cs.LG", dt.datetime(2024, 1, 5)),
    _row("2401.00002", "A survey of robots", "Mentions diffusion policy once in the abstract body only.", "cs.RO", dt.datetime(2024, 1, 9)),
    _row("2305.00003", "Graph neural networks for molecules", "Message passing on molecular graphs.", "cs.LG q-bio.BM", dt.datetime(2023, 5, 3)),
    _row("2305.00004", "Retrieval augmented generation survey", "A review of RAG systems.", "cs.CL", dt.datetime(2023, 5, 4)),
    _row("2305.00005", "Language models", "We use retrieval and generation in graph settings.", "cs.CL", dt.datetime(2023, 5, 8)),
    _row("2610.00006", "Newest paper on diffusion policy", "Posted after the snapshot cutoff.", "cs.RO", dt.datetime(2026, 9, 30)),
    _row("hep-th/0310283", "Decay of massive closed superstring states", "Exact results for string decay.", "hep-th", dt.datetime(2003, 10, 30)),
    _row("2201.00008", "C++ templates and the foo-bar protocol", "Notes on baz and unbalanced quotes.", "cs.PL", dt.datetime(2022, 1, 2)),
    _row("2201.00009", "Quantum error correction codes", "Surface code thresholds.", "quant-ph", dt.datetime(2022, 1, 3)),
    _row("2201.00010", "Protein folding with transformers", "AlphaFold style structure prediction.", "q-bio.BM cs.LG", dt.datetime(2022, 1, 4)),
]
# Pad to ~20 rows with unrelated filler papers.
for n in range(11, 21):
    ROWS.append(_row(f"2202.{n:05d}", f"Filler topic number {n}", f"Unrelated content about subject {n}.", "math.CO", dt.datetime(2022, 2, n % 28 + 1)))


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("meta") / "metadata.parquet"
    table = pa.Table.from_pylist(ROWS, schema=pa.schema([
        ("paper_id", pa.string()), ("title", pa.string()), ("authors", pa.string()),
        ("abstract", pa.string()), ("categories", pa.string()), ("primary_category", pa.string()),
        ("first_version_date", pa.timestamp("ms")), ("doi", pa.string()),
    ]))
    pq.write_table(table, path, row_group_size=7)  # several row groups
    return path


@pytest.fixture(scope="module")
def index_path(parquet_path, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "index.sqlite"
    stats = build_index(parquet_path, path, batch_rows=4, progress=None)
    assert stats["papers"] == len(ROWS)
    return path


def test_ranking_title_match_first_and_or_fallback(index_path):
    papers = search_local("diffusion policy", 5, index_path=index_path, date_before="2025-12-31")
    assert [p.arxiv_id for p in papers[:2]] == ["2401.00001", "2401.00002"]
    assert papers[0].processing_metadata["local_bm25"] > papers[1].processing_metadata["local_bm25"]
    # Porter stemming: "manipulations" matches "manipulation".
    assert search_local("robot manipulations", 3, index_path=index_path)[0].arxiv_id == "2401.00001"
    # No paper has both words, so OR fills the slots after the (empty) AND result.
    both = {p.arxiv_id for p in search_local("quantum protein", 3, index_path=index_path)}
    assert {"2201.00009", "2201.00010"} <= both


def test_hostile_queries_do_not_raise(index_path):
    for hostile in ["C++", '"unbalanced', "AND OR NOT", "foo-bar: baz*", "NEAR(a b)", "title:x", "(((", "", "'; DROP TABLE papers;--"]:
        assert isinstance(search_local(hostile, 5, index_path=index_path), list)
    assert search_local("foo-bar: baz*", 5, index_path=index_path)[0].arxiv_id == "2201.00008"


def test_date_category_and_legacy_filters(index_path):
    everything = {p.arxiv_id for p in search_local("diffusion policy", 10, index_path=index_path)}
    cut = {p.arxiv_id for p in search_local("diffusion policy", 10, index_path=index_path, date_before="2026-08-27")}
    assert "2610.00006" in everything and "2610.00006" not in cut
    cl = search_local("survey graph retrieval", 10, index_path=index_path, categories=["cs.CL"])
    assert cl and all("cs.CL" in p.subjects for p in cl)
    assert search_local("graph molecules", 10, index_path=index_path, categories=["cs.L"]) == []  # exact token
    assert search_local("superstring decay", 5, index_path=index_path) == []  # old-style id skipped
    assert search_local("superstring decay", 5, index_path=index_path, include_legacy_ids=True)[0].arxiv_id == "hep-th/0310283"


def test_output_shape_matches_arxiv_search(index_path):
    paper = search_local("diffusion policy robot", 1, index_path=index_path)[0]
    # arxiv_search builds a PaperItem with exactly these fields set.
    expected = PaperItem(
        paper_id="arxiv_2401.00001", title="t", summary="s", authors=["a"], published="2024-01-05",
        subjects=["cs.RO"], source_type="arxiv", arxiv_id="2401.00001",
        arxiv_abs_url="https://arxiv.org/abs/2401.00001",
        arxiv_html_url="https://arxiv.org/html/2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf", metadata_status="complete",
    )
    assert isinstance(paper, PaperItem) and set(paper.model_dump()) == set(expected.model_dump())
    assert (paper.paper_id, paper.arxiv_id, paper.published) == ("arxiv_2401.00001", "2401.00001", "2024-01-05")
    assert paper.subjects == ["cs.RO", "cs.LG"] and paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.arxiv_abs_url == expected.arxiv_abs_url and paper.pdf_url == expected.pdf_url
    assert paper.metadata_status == "complete"


def _api_paper(arxiv_id: str, title: str = "api") -> PaperItem:
    return PaperItem(paper_id=canonicalize_paper_key(arxiv_id), title=title, arxiv_id=arxiv_id)


def test_search_node_flag_off_unchanged_and_flag_on_merges(monkeypatch, index_path):
    assert config.LOCAL_INDEX_ENABLED is False  # default off
    calls = []

    def fake_arxiv(query, max_results):
        calls.append((query, max_results))
        return [_api_paper("2401.00001v2", "dup of local hit"), _api_paper("2610.00006", "newer than cutoff")]

    monkeypatch.setattr(config, "LOCAL_INDEX_PATH", index_path)
    monkeypatch.setattr(search_node, "arxiv_search", fake_arxiv)
    monkeypatch.setattr(local_index, "search_local", lambda *a, **k: (_ for _ in ()).throw(AssertionError("flag off")))

    off = search_node.search_papers_node({"search_query": "diffusion policy"})
    assert calls == [("diffusion policy", config.MAX_SEARCH_RESULTS)]
    assert [p.arxiv_id for p in off["search_results"]] == ["2401.00001v2", "2610.00006"]
    assert off["trace_logs"] == ["[search_papers] Lần 1: 'diffusion policy'", "[search_papers] Tìm thấy 2 bài báo"]

    monkeypatch.undo()  # restore real search_local; then switch the flag on
    monkeypatch.setattr(config, "LOCAL_INDEX_ENABLED", True)
    monkeypatch.setattr(config, "LOCAL_INDEX_PATH", index_path)
    monkeypatch.setattr(search_node, "arxiv_search", fake_arxiv)
    on = search_node.search_papers_node({"search_query": "diffusion policy"})
    ids = [p.arxiv_id for p in on["search_results"]]
    assert ids[0] == "2401.00001" and "2401.00001v2" not in ids  # local first, API duplicate dropped
    assert ids[-1] == "2610.00006"  # newer than the cutoff: API only
