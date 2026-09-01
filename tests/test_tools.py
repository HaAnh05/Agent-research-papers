from __future__ import annotations

import pytest
from tools.cache_manager import canonicalize_paper_key
from tools.bibtex_generator import generate_bibtex
from tools.pdf_parser import parse_structured_sections


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
