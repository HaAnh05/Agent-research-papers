"""Offline tests for the search-recall bench (bench/recall).  No network, no metadata file."""

from __future__ import annotations

import pytest

from bench.recall.build_goldset import (
    MetaIndex,
    normalize_title,
    resolve_reference,
    strip_survey_words,
)
from bench.recall.run_recall import query_metrics
from state import PaperReference


def _index() -> MetaIndex:
    index = MetaIndex()
    index.dates = {"2001.00001": 1, "2002.00002": 1, "2003.00003": 1, "2004.00004": 1, "2005.00005": 1}
    index.by_doi = {"10.1000/real": "2003.00003"}
    index.add_title("attention is all you need", "2001.00001")
    index.add_title("deep residual learning for image recognition", "2004.00004")
    index.add_title("deep residual learning for image recognition", "2005.00005")   # same title twice: ambiguous
    index.add_title("a b c", "2002.00002")                                           # only 3 words
    return index


def test_normalize_title_and_title_match_rules():
    assert normalize_title(r"Attention \textbf{Is} All You Need!") == "attention is all you need"
    assert normalize_title("Café: $\\alpha$-Learning  for   Agents") == "cafe learning for agents"

    index = _index()
    hit = resolve_reference(PaperReference(title="Attention is all you need."), index)
    assert hit == ("2001.00001", "title")
    assert resolve_reference(PaperReference(title="Deep Residual Learning for Image Recognition"), index) is None
    assert resolve_reference(PaperReference(title="A B C"), index) is None          # fewer than 4 words


def test_resolution_order_arxiv_id_then_doi_then_title():
    index = _index()
    full = PaperReference(
        title="Attention is all you need",
        arxiv_id="2002.00002v3",
        doi="10.48550/arXiv.2003.00003",
    )
    assert resolve_reference(full, index) == ("2002.00002", "arxiv_id")             # id beats DOI and title

    full.arxiv_id = None
    assert resolve_reference(full, index) == ("2003.00003", "doi_arxiv")            # arXiv DOI beats title

    full.doi = "10.1000/REAL"
    assert resolve_reference(full, index) == ("2003.00003", "doi_metadata")         # metadata DOI beats title

    full.doi = "10.1000/unknown"
    assert resolve_reference(full, index) == ("2001.00001", "title")                # unknown DOI falls through
    assert resolve_reference(PaperReference(arxiv_id="9999.99999"), index) is None  # id not in metadata


def test_strip_survey_words():
    cases = {
        "A Survey on Large Language Model based Autonomous Agents": "Large Language Model based Autonomous Agents",
        "A Comprehensive Survey of Graph Neural Networks": "Graph Neural Networks",
        "Explainable AI: A Systematic Literature Review": "Explainable AI",
        "Large Language Models for Planning: A Review": "Large Language Models for Planning",
        "Agents: A Survey on Planning and Memory": "Agents Planning and Memory",
        "Peer Review with LLMs": "Peer Review with LLMs",                              # bare "review" stays
    }
    for title, expected in cases.items():
        assert strip_survey_words(title) == expected
    assert ":" not in strip_survey_words("A Survey: Foo Bar Baz")


def test_metric_math_on_toy_ranking():
    gold = ["a", "b", "c", "d"]
    ranked = ["x", "a", "y", "a", "b", "z", "c"]            # duplicate "a" counts once
    m = query_metrics(ranked, gold, [5, 20, 100])
    # unique ranking: x a y b z c -> top 5 holds a, b
    assert m["recall@5"] == pytest.approx(2 / 4)
    assert m["recall@20"] == pytest.approx(3 / 4)
    assert m["recall@100"] == pytest.approx(3 / 4)
    assert m["precision@5"] == pytest.approx(2 / 5)
    assert m["hit_rate@5"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 2)

    miss = query_metrics(["x", "y"], gold, [5])
    assert miss["recall@5"] == 0.0 and miss["hit_rate@5"] == 0.0 and miss["mrr"] == 0.0
