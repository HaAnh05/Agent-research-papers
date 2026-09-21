from __future__ import annotations

import pytest
import json
from tools.cache_manager import canonicalize_paper_key
from tools.bibtex_generator import generate_bibtex
import tools.pdf_parser as pdf_parser
import tools.github_enricher as github_enricher
from tools.pdf_parser import parse_structured_sections
from state import GitHubRepoInfo


def test_canonicalize_paper_key():
    """Test deterministic ArXiv ID and identifier normalization."""
    assert canonicalize_paper_key("https://arxiv.org/abs/1706.03762") == "arxiv_1706.03762"
    assert canonicalize_paper_key("1706.03762v2") == "arxiv_1706.03762"
    assert canonicalize_paper_key("2312.00752") == "arxiv_2312.00752"
    assert canonicalize_paper_key("https://arxiv.org/pdf/2401.12345.pdf") == "arxiv_2401.12345"


def test_generate_bibtex():
    """Test standard BibTeX generation."""
    bib = generate_bibtex(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        published="2017-06-12",
        arxiv_id="1706.03762",
    )
    assert "@article{" in bib
    assert "vaswani2017attention" in bib.lower()
    assert "Attention Is All You Need" in bib
    assert "1706.03762" in bib


def test_parse_structured_sections():
    """Test Regex section splitting from academic text."""
    sample_text = """
    Abstract
    We introduce a novel transformer architecture with linear attention.
    
    1. Introduction
    Transformers have revolutionized natural language processing.
    
    2. Proposed Method
    Our architecture replaces quadratic softmax with linear kernel attention.
    
    4. Experiments and Results
    We evaluate on ImageNet and achieve 88.5% top-1 accuracy.
    
    5. Limitations
    Memory bandwidth remains a bottleneck on older GPUs.
    
    References
    [1] Vaswani et al. 2017.
    """
    sections = parse_structured_sections(sample_text)
    assert "novel transformer" in sections.get("abstract", "").lower()
    assert "linear kernel" in sections.get("methodology", "").lower()
    assert "88.5%" in sections.get("experiments", "").lower()
    assert "memory bandwidth" in sections.get("limitations", "").lower()


def test_parse_structured_sections_reports_heading_or_fallback_quality():
    matched = {}
    parse_structured_sections("Abstract\nA.\nMethod\nM.\nExperiments\nE.", metadata=matched)
    assert matched["heading"] == "heading"
    assert matched["parserStatus"] == "success"

    fallback = {}
    parse_structured_sections("unstructured paper text", metadata=fallback)
    assert fallback["heading"] == "fallback"
    assert fallback["parserStatus"] == "degraded"


@pytest.mark.parametrize(
    "heading",
    [
        "Experimental Results",
        "Experimental Setup",
        "Experiments and Results",
        "Evaluation and Analysis",
        "Implementation and Evaluation",
        "Quantitative Evaluation",
        "Experiments on ImageNet",
    ],
)
def test_parse_structured_sections_accepts_common_experiment_headings(heading):
    text = f"Abstract\nA supported abstract.\n2. Proposed Approach\nA supported method.\n4. {heading}\nA measured result.\n5. Conclusion\nDone."
    sections = parse_structured_sections(text)
    assert "supported method" in sections["methodology"].lower()
    assert "measured result" in sections["experiments"].lower()


def test_github_success_cache_skips_second_request_and_does_not_cache_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(github_enricher.config, "GITHUB_CACHE_DIR", tmp_path)
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {"items": [{"html_url": "https://github.com/example/repo", "full_name": "example/repo", "stargazers_count": 1, "language": "Python", "description": "official"}]}

    monkeypatch.setattr(github_enricher.requests, "get", lambda *args, **kwargs: (calls.append(1) or Response()))
    first = github_enricher.search_github_code_multitier("A Paper", arxiv_id="1706.03762")
    first_call_count = len(calls)
    second = github_enricher.search_github_code_multitier("A Paper", arxiv_id="1706.03762")
    assert len(first) == len(second) == 1
    assert len(calls) == first_call_count

    error_calls = []

    def flaky_get(*args, **kwargs):
        error_calls.append(1)
        if len(error_calls) == 1:
            return Response()
        raise github_enricher.requests.RequestException("network")

    monkeypatch.setattr(github_enricher.requests, "get", flaky_get)
    github_enricher.search_github_code_multitier("Another Paper", arxiv_id="2005.14165")
    cache_files = list(tmp_path.glob("*2005.14165*"))
    assert cache_files == []


def test_pdf_download_refuses_non_arxiv_hosts(tmp_path, monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("unsafe URL reached requests.get")

    monkeypatch.setattr(pdf_parser.requests, "get", unexpected_request)
    with pytest.raises(ValueError, match="ArXiv"):
        pdf_parser._download_pdf("http://127.0.0.1/private.pdf", tmp_path / "paper.pdf")


def test_pdf_download_enforces_stream_size_and_removes_partial_file(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"%PDF-1.7\n"
            yield b"x" * 32

        def close(self):
            return None

    monkeypatch.setattr(pdf_parser, "MAX_REMOTE_PDF_BYTES", 16)
    monkeypatch.setattr(pdf_parser.requests, "get", lambda *args, **kwargs: Response())
    output = tmp_path / "paper.pdf"
    with pytest.raises(ValueError, match="size limit"):
        pdf_parser._download_pdf("https://arxiv.org/pdf/1706.03762.pdf", output)
    assert not output.exists()
    assert not output.with_suffix(".pdf.part").exists()
