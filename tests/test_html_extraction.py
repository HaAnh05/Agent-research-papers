"""Offline tests for HTML-first paper reading (PR 2: extraction).

All HTML below is small, synthetic LaTeXML-style markup.  No test touches the
network or really sleeps: HTTP is mocked and the host limiters use a fake clock.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

import pytest

import tools.pdf_parser as pdf_parser
from config import config
from state import PaperItem
from tools import rate_limit
from tools.html_extractor import (
    classify_sub_heading,
    classify_top_heading,
    extract_html,
    find_arxiv_id,
    find_doi,
    looks_like_latexml,
    tex_to_text,
)

# ---------------------------------------------------------------------------
# Synthetic LaTeXML documents
# ---------------------------------------------------------------------------

PAPER_HTML = """<!DOCTYPE html><html><head><title>Fake Paper</title><script>var x = "Hardware 9999";</script></head>
<body><div class="ltx_page_main"><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Fast Diffusion Planning</h1>
<div class="ltx_authors"><span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">1</sup>
<span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>
Videos at <a class="ltx_ref ltx_href" href="https://fastplan.github.io/">fastplan.github.io</a>.</span></span></span></div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">We plan with diffusion. Code is available at <a href="https://github.com/acme/fastplan" class="ltx_ref ltx_url">github.com/acme/fastplan</a>.</p></div>
<section id="S1" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
<div class="ltx_para"><p class="ltx_p">INTRO_ONLY_TEXT planning is hard.</p></div></section>
<section id="S2" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">2 </span>Diffusion Planner Formulation</h2>
<div class="ltx_para"><p class="ltx_p">We model trajectories as <math id="m1" class="ltx_Math" alttext="p_{\\theta}(\\tau)" display="inline"><semantics><mi>MATHML_NOISE</mi><annotation encoding="application/x-tex">p_{\\theta}(\\tau)</annotation></semantics></math> with a U-Net.
The model was trained on 8 NVIDIA A100 GPUs for two days.</p></div></section>
<section id="S3" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">3 </span>Experiments</h2>
<div class="ltx_para"><p class="ltx_p">We compare against CQL. We use the official implementation of CQL from
<a href="https://github.com/young-geng/cql" class="ltx_ref ltx_url">github.com/young-geng/cql</a> as a baseline.</p></div>
<section id="S3.SS1" class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection"><span class="ltx_tag ltx_tag_subsection">3.1 </span>Implementation Details</h3>
<div class="ltx_para"><p class="ltx_p">SETUP_BODY We use Adam with learning rate 3e-4.</p></div></section>
<section id="S3.SS2" class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection"><span class="ltx_tag ltx_tag_subsection">3.2 </span>Main Results</h3>
<figure id="S3.T2" class="ltx_table"><figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_table">Table 2: </span>Average return on <math alttext="\\text{Maze2D}" class="ltx_Math"><semantics><mi>x</mi></semantics></math> tasks.</figcaption>
<table id="S3.T2.1" class="ltx_tabular ltx_centering"><thead class="ltx_thead">
<tr class="ltx_tr"><th class="ltx_td ltx_th ltx_th_row" rowspan="2">Env</th><th class="ltx_td ltx_th" colspan="2">Return</th></tr>
<tr class="ltx_tr"><th class="ltx_td ltx_th">CQL</th><th class="ltx_td ltx_th">Ours</th></tr></thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr"><td class="ltx_td">U-Maze</td><td class="ltx_td">5.7</td><td class="ltx_td"><math alttext="113.9\\pm 3.1" class="ltx_Math"><semantics><mn>113.9</mn></semantics></math></td></tr>
<tr class="ltx_tr"><td class="ltx_td">Large</td><td class="ltx_td">1.0</td><td class="ltx_td"><span class="ltx_text ltx_font_bold">121.5</span></td></tr>
</tbody></table></figure>
<table class="ltx_equation ltx_eqn_table"><tbody><tr class="ltx_equation"><td class="ltx_eqn_cell">EQUATION_CELL</td></tr></tbody></table>
</section></section>
<section id="S4" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">4 </span>Limitations and Future Work</h2>
<div class="ltx_para"><p class="ltx_p">LIMITS_TEXT slow sampling.</p></div></section>
<section id="S5" class="ltx_section"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">5 </span>Conclusion</h2>
<div class="ltx_para"><p class="ltx_p">CONCLUSION_ONLY. Our implementation is released at
<a href="https://github.com/acme/fastplan/tree/main" class="ltx_ref ltx_url">our repository</a>.</p></div></section>
<section id="bib" class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2>
<ul class="ltx_biblist">
<li id="bib.bib1" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Ho et al. (2020)</span>
<span class="ltx_bibblock">Ho, J., Jain, A., and Abbeel, P.</span>
<span class="ltx_bibblock">Denoising diffusion probabilistic models.</span>
<span class="ltx_bibblock">arXiv preprint arXiv:2006.11239v2, 2020.</span></li>
<li id="bib.bib2" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Smith (2019)</span>
<span class="ltx_bibblock">Smith, A.</span>
<span class="ltx_bibblock">A study of things.</span>
<span class="ltx_bibblock">Journal of Things, 2019. doi: <a href="https://doi.org/10.1000/xyz123" class="ltx_ref">10.1000/xyz123</a>.</span></li>
<li id="bib.bib3" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Old (2001)</span>
<span class="ltx_bibblock">Old, B.</span>
<span class="ltx_bibblock">Strings.</span>
<span class="ltx_bibblock">arXiv:hep-th/9901001.</span></li>
<li id="bib.bib4" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Young (2021)</span>
<span class="ltx_bibblock">Young, G.</span>
<span class="ltx_bibblock">CQL implementation. URL <a href="https://github.com/young-geng/cql" class="ltx_ref ltx_url">https://github.com/young-geng/cql</a>.</span></li>
</ul></section>
<section id="A1" class="ltx_appendix"><h2 class="ltx_title ltx_title_appendix"><span class="ltx_tag ltx_tag_appendix">Appendix A </span>Training Details</h2>
<div class="ltx_para"><p class="ltx_p">APPENDIX_SETUP batch size 256.</p></div></section>
<section id="A2" class="ltx_appendix"><h2 class="ltx_title ltx_title_appendix"><span class="ltx_tag ltx_tag_appendix">Appendix B </span>Proofs</h2>
<div class="ltx_para"><p class="ltx_p">PROOF_TEXT_DROPPED lemma.</p></div></section>
</article>
<footer class="ltx_page_footer"><a class="ar5iv-text-button" href="https://github.com/dginev/ar5iv/issues/new">Improve</a></footer>
</div></body></html>"""

NO_HTML_PAGE = "<html><head><title>Not found</title></head><body><h1>HTML not available</h1></body></html>"


def _minimal_latexml(abstract: bool = True, method: bool = True, experiments: bool = True) -> str:
    parts = ['<html><body><article class="ltx_document"><h1 class="ltx_title ltx_title_document">T</h1>']
    if abstract:
        parts.append('<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6><p class="ltx_p">ABS words.</p></div>')
    if method:
        parts.append('<section class="ltx_section"><h2 class="ltx_title ltx_title_section">2 Method</h2><div class="ltx_para"><p class="ltx_p">METH words.</p></div></section>')
    if experiments:
        parts.append('<section class="ltx_section"><h2 class="ltx_title ltx_title_section">3 Experiments</h2><div class="ltx_para"><p class="ltx_p">EXP words.</p></div></section>')
    parts.append("</article></body></html>")
    return "".join(parts)


@pytest.fixture(scope="module")
def parsed():
    return extract_html(PAPER_HTML)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_sections_are_routed_and_appendix_setup_is_kept(parsed):
    sections = parsed.sections
    assert set(sections) == {"abstract", "methodology", "setup", "experiments", "limitations"}
    assert "We plan with diffusion" in sections["abstract"]
    assert "Introduction" not in sections["methodology"] and "INTRO_ONLY_TEXT" not in "".join(sections.values())
    assert "We model trajectories as p_θ(τ)" in sections["methodology"]
    assert "MATHML_NOISE" not in "".join(sections.values())
    assert "We compare against CQL" in sections["experiments"]
    assert "SETUP_BODY" in sections["setup"] and "SETUP_BODY" not in sections["experiments"]
    assert "APPENDIX_SETUP" in sections["setup"]
    assert "LIMITS_TEXT" in sections["limitations"]
    assert "CONCLUSION_ONLY" not in "".join(sections.values())
    assert "PROOF_TEXT_DROPPED" not in "".join(sections.values())
    assert "Hardware 9999" not in "".join(sections.values())
    assert "EQUATION_CELL" in sections["experiments"]  # equation tables are text, not data tables


def test_hardware_sentence_is_copied_into_setup(parsed):
    assert "8 NVIDIA A100 GPUs" in parsed.sections["setup"]
    # the method section keeps its own text
    assert "8 NVIDIA A100 GPUs" in parsed.sections["methodology"]


def test_section_caps_come_from_config(monkeypatch):
    assert config.HTML_EXPERIMENTS_MAX_CHARS == 12000
    assert config.HTML_METHODOLOGY_MAX_CHARS == 8000
    assert config.HTML_SETUP_MAX_CHARS == 6000
    assert config.HTML_ABSTRACT_MAX_CHARS == 3000
    assert config.HTML_LIMITATIONS_MAX_CHARS == 3000
    monkeypatch.setattr(config, "HTML_EXPERIMENTS_MAX_CHARS", 20)
    assert len(extract_html(PAPER_HTML).sections["experiments"]) == 20


@pytest.mark.parametrize(
    "title, appendix, seen_exp, expected",
    [
        ("2 Diffusion Policy Formulation", False, False, "methodology"),
        ("3 Key Design Decisions", False, False, "methodology"),
        ("4 Intriguing Properties of Diffusion Policy", False, False, "methodology"),
        ("7 Realworld Bimanual Tasks", False, True, "experiments"),
        ("5 Evaluation", False, False, "experiments"),
        ("5 Experimental Setup", False, False, "setup"),
        ("6 Implementation Details", False, True, "setup"),
        ("Appendix A Diffusion Policy Implementation Details", True, True, "setup"),
        ("Appendix D Realworld Setup Details", True, True, "setup"),
        ("Appendix C Realworld Task Details", True, True, "setup"),
        ("Appendix B Additional Ablation Results", True, True, "experiments"),
        ("Appendix B Proofs", True, True, None),
        ("8 Related Work", False, True, None),
        ("1 Introduction", False, False, None),
        ("10 Conclusion", False, True, None),
        ("9 Limitations and Future Work", False, True, "limitations"),
        ("Results and Discussion", False, True, "experiments"),
        ("References", False, True, "references"),
    ],
)
def test_top_heading_classification(title, appendix, seen_exp, expected):
    assert classify_top_heading(title, appendix=appendix, seen_experiments=seen_exp) == expected


@pytest.mark.parametrize(
    "title, expected",
    [
        ("A.4 Hyperparameters", "setup"),
        ("Training Details", "setup"),
        ("Hardware and Compute", "setup"),
        ("A.3 Experimental setup", "setup"),
        ("A.2. Configuration", "setup"),
        ("2.1 Problem Setting", None),
        ("5.2 Evaluation Methodology", None),
        ("4.3 Discussion", None),
        ("Limitations", "limitations"),
    ],
)
def test_sub_heading_classification(title, expected):
    assert classify_sub_heading(title) == expected


def test_sub_heading_under_unclassified_appendix_still_becomes_setup():
    html = (
        '<article class="ltx_document"><div class="ltx_abstract"><p class="ltx_p">A.</p></div>'
        '<section class="ltx_appendix"><h2 class="ltx_title ltx_title_appendix">Appendix A Appendix</h2>'
        '<h3 class="ltx_title ltx_title_subsection">A.1 Proof of Lemma</h3><p class="ltx_p">PROOF</p>'
        '<h3 class="ltx_title ltx_title_subsection">A.2 NeRF Details and Training Hyperparameters</h3>'
        '<p class="ltx_p">OPT Adam.</p><h4 class="ltx_title ltx_title_paragraph">Optimizer.</h4><p class="ltx_p">OPT2 lr.</p>'
        '<h3 class="ltx_title ltx_title_subsection">A.3 Derivation</h3><p class="ltx_p">DERIV</p></section></article>'
    )
    sections = extract_html(html).sections
    assert "OPT Adam." in sections["setup"] and "OPT2 lr." in sections["setup"]
    assert "PROOF" not in sections["setup"] and "DERIV" not in sections["setup"]


def test_math_alttext_is_converted_to_readable_text():
    assert tex_to_text(r"113.9\pm 3.1") == "113.9± 3.1"
    assert tex_to_text(r"p_{\theta}(\bm{\tau}^{i-1}\!\mid\!\bm{\tau}^{i})") == "p_θ(τ^(i-1)|τ^i)"
    assert tex_to_text(r"\frac{a}{b}") == "(a)/(b)"
    assert tex_to_text(r"\mathbf{x}\in\mathbb{R}^{n}") == "x∈R^n"
    assert tex_to_text("33.2") == "33.2"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_table_with_colspan_and_rowspan(parsed):
    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.label == "Table 2"
    assert table.caption == "Average return on Maze2D tasks."
    assert table.section == "3.2 Main Results"
    assert table.rows == [
        ["Env", "Return", "Return"],      # colspan repeats the text
        ["Env", "CQL", "Ours"],           # rowspan repeats the text on the next row
        ["U-Maze", "5.7", "113.9± 3.1"],  # math alttext, not MathML noise
        ["Large", "1.0", "121.5"],
    ]
    # the section text keeps a compact version so numbers stay visible
    assert "Table 2" in parsed.sections["experiments"] and "U-Maze | 5.7 | 113.9± 3.1" in parsed.sections["experiments"]


def test_layout_only_tabular_is_dropped():
    html = (
        '<article class="ltx_document"><section class="ltx_section"><h2 class="ltx_title ltx_title_section">3 Experiments</h2>'
        '<table class="ltx_tabular"><tr class="ltx_tr"><td class="ltx_td"><img src="a.png"></td></tr></table></section></article>'
    )
    assert extract_html(html).tables == []


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_references_parse_ids_titles_and_urls(parsed):
    refs = {ref.index: ref for ref in parsed.references}
    assert len(refs) == 4
    assert refs[1].arxiv_id == "2006.11239"
    assert refs[1].title == "Denoising diffusion probabilistic models"
    assert refs[1].text.startswith("Ho, J., Jain, A., and Abbeel, P.")
    assert "Ho et al." not in refs[1].text  # the citation tag is not part of the text
    assert refs[2].doi == "10.1000/xyz123" and refs[2].arxiv_id is None
    assert refs[2].urls == ["https://doi.org/10.1000/xyz123"]
    assert refs[3].arxiv_id == "hep-th/9901001"
    assert refs[4].urls == ["https://github.com/young-geng/cql"]


@pytest.mark.parametrize(
    "text, urls, expected",
    [
        ("arXiv preprint arXiv:1706.03762v5, 2017.", None, "1706.03762"),
        ("arXiv:hep-th/9901001", None, "hep-th/9901001"),
        ("CoRR, abs/2303.04137, 2023", None, "2303.04137"),
        ("", ["https://arxiv.org/abs/2303.04137v2"], "2303.04137"),
        ("", ["https://arxiv.org/pdf/math.GT/0309136"], "math.gt/0309136"),
        ("Journal 2020, pp. 1234.5678", None, None),
        ("Nature 2015", None, None),
    ],
)
def test_find_arxiv_id(text, urls, expected):
    assert find_arxiv_id(text, urls) == expected


def test_find_doi_from_text_and_href():
    assert find_doi("see doi:10.1145/3295222.3295349.") == "10.1145/3295222.3295349"
    assert find_doi("", ["https://doi.org/10.48550/arXiv.1706.03762"]) == "10.48550/arXiv.1706.03762"
    assert find_doi("no doi here") is None


# ---------------------------------------------------------------------------
# Repository links
# ---------------------------------------------------------------------------


def test_repo_links_primary_versus_baseline(parsed):
    links = {link.url: link for link in parsed.repo_links}

    project = links["https://fastplan.github.io"]
    assert project.kind == "project_page" and project.is_primary_candidate and not project.in_bibliography
    assert "Videos at" in project.context

    own = links["https://github.com/acme/fastplan"]  # abstract + conclusion "tree/main" are merged
    assert own.kind == "github" and own.is_primary_candidate and not own.in_bibliography
    assert len(own.context) <= 200

    baseline = links["https://github.com/young-geng/cql"]  # experiments text and bibliography
    assert baseline.kind == "github"
    assert not baseline.is_primary_candidate
    assert not baseline.in_bibliography  # it also appears in the body

    assert "https://github.com/dginev/ar5iv/issues/new" not in links  # site chrome
    assert not any("ar5iv" in url for url in links)
    assert all(len(link.context) <= 200 for link in parsed.repo_links)
    assert parsed.code_urls == ["https://github.com/acme/fastplan"]


def test_bibliography_only_repo_is_flagged_and_never_primary():
    html = (
        '<article class="ltx_document"><div class="ltx_abstract"><p class="ltx_p">Abs.</p></div>'
        '<section class="ltx_bibliography"><h2 class="ltx_title ltx_title_bibliography">References</h2><ul>'
        '<li class="ltx_bibitem"><span class="ltx_bibblock">A.</span><span class="ltx_bibblock">Code release.</span>'
        '<span class="ltx_bibblock"><a href="https://github.com/o/r">https://github.com/o/r</a></span></li></ul></section></article>'
    )
    (link,) = extract_html(html).repo_links
    assert link.in_bibliography and not link.is_primary_candidate and link.kind == "github"


def test_schemeless_url_and_first_footnote_are_primary():
    html = (
        '<article class="ltx_document"><span class="ltx_note ltx_role_footnote"><span class="ltx_note_content">'
        'Homepage: <a class="ltx_ref ltx_url" href="proj.example.edu">proj.example.edu</a></span></span>'
        '<section class="ltx_section"><h2 class="ltx_title ltx_title_section">1 Introduction</h2></section></article>'
    )
    (link,) = extract_html(html).repo_links
    assert link.url == "https://proj.example.edu" and link.kind == "project_page" and link.is_primary_candidate


# ---------------------------------------------------------------------------
# LaTeXML detection
# ---------------------------------------------------------------------------


def test_no_html_page_is_not_latexml():
    assert not looks_like_latexml(NO_HTML_PAGE)
    assert not looks_like_latexml("")
    assert looks_like_latexml(PAPER_HTML)


@pytest.mark.parametrize(
    "markup",
    [
        "",
        "plain text",
        '</div></p><td>x</td><table class="ltx_tabular"><tr><td colspan="x" rowspan="-3">a</td>',
        '<a href="http://[bad">x</a>',
        '<math alttext="\\frac{">x</math>',
        '<li class="ltx_bibitem">',
        '<h2 class="ltx_title ltx_title_section">',
        PAPER_HTML[: len(PAPER_HTML) // 2],
    ],
)
def test_malformed_markup_never_raises(markup):
    result = extract_html(markup)
    assert set(result.sections) == {"abstract", "methodology", "setup", "experiments", "limitations"}


# ---------------------------------------------------------------------------
# Rate limiter (fake clock, no sleeping)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: List[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    clock = FakeClock()
    rate_limit.configure(clock=clock.time, sleep=clock.sleep)
    return clock


def test_arxiv_limiter_spaces_request_starts_by_crawl_delay(fake_clock):
    with rate_limit.host_slot("arxiv.org"):
        pass
    assert fake_clock.sleeps == []  # first request goes straight through
    fake_clock.now += 4.0           # 4 s of "request time" count toward the gap
    with rate_limit.host_slot("arxiv.org"):
        pass
    assert fake_clock.sleeps == [pytest.approx(11.0)]
    fake_clock.now += 20.0          # already past the delay: no wait
    with rate_limit.host_slot("www.arxiv.org"):  # same host as arxiv.org
        pass
    assert fake_clock.sleeps == [pytest.approx(11.0)]


def test_ar5iv_and_arxiv_have_independent_limiters(fake_clock):
    with rate_limit.host_slot("ar5iv.labs.arxiv.org"):
        pass
    with rate_limit.host_slot("arxiv.org"):
        pass
    with rate_limit.host_slot("ar5iv.labs.arxiv.org"):
        pass
    assert fake_clock.sleeps == [pytest.approx(3.0)]  # only the second ar5iv call waits


def test_limiter_delays_are_config_values(fake_clock, monkeypatch):
    assert config.ARXIV_HOST_MIN_INTERVAL_SEC == 15 and config.AR5IV_MIN_INTERVAL_SEC == 3
    monkeypatch.setattr(config, "ARXIV_HOST_MIN_INTERVAL_SEC", 20.0)
    with rate_limit.host_slot("arxiv.org"):
        pass
    with rate_limit.host_slot("arxiv.org"):
        pass
    assert fake_clock.sleeps == [pytest.approx(20.0)]


def test_limiter_allows_one_connection_at_a_time(fake_clock):
    limiter = rate_limit.limiter_for_host("arxiv.org")
    assert limiter is not None and rate_limit.limiter_for_host("example.com") is None
    entered = threading.Event()
    release = threading.Event()
    order: List[str] = []

    def first() -> None:
        with limiter.slot():
            order.append("first-in")
            entered.set()
            release.wait(timeout=5)
            order.append("first-out")

    def second() -> None:
        entered.wait(timeout=5)
        with limiter.slot():
            order.append("second-in")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    entered.wait(timeout=5)
    release.set()
    for thread in threads:
        thread.join(timeout=5)
    assert order == ["first-in", "first-out", "second-in"]
    assert fake_clock.sleeps == [pytest.approx(15.0)]


# ---------------------------------------------------------------------------
# Read order with mocked HTTP
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes = b"", headers: Dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = {"content-type": "text/html; charset=utf-8", **(headers or {})}
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int = 1024):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self) -> None:
        return None


ID = "2303.04137"
AR5IV = f"https://ar5iv.labs.arxiv.org/html/{ID}"
ARXIV_HTML = f"https://arxiv.org/html/{ID}"
ARXIV_PDF = f"https://arxiv.org/pdf/{ID}.pdf"
PDF_TEXT = (
    "Abstract\nPDF abstract text.\nIntroduction\nintro\n"
    "2 Method\nPDF method text.\n3 Experiments\nPDF experiment text.\nLimitations\nPDF limits.\nReferences\n"
)


class Web:
    """Mock for ``requests.get`` keyed by URL; records the call order."""

    def __init__(self, routes: Dict[str, FakeResponse]) -> None:
        self.routes = routes
        self.calls: List[str] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        assert kwargs.get("allow_redirects") is False
        self.calls.append(url)
        assert "/e-print" not in url and "/src/" not in url, "robots.txt disallows these paths"
        response = self.routes.get(url)
        if response is None:
            raise AssertionError(f"unexpected request: {url}")
        return response


@pytest.fixture
def env(tmp_path, monkeypatch, fake_clock):
    monkeypatch.setattr(config, "PDF_CACHE_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(config, "TEXT_CACHE_DIR", tmp_path / "text")
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "text").mkdir()
    monkeypatch.setattr(pdf_parser, "extract_ordered_blocks_from_pdf", lambda path, max_pages=8: PDF_TEXT)

    def install(routes: Dict[str, FakeResponse]) -> Web:
        web = Web(routes)
        monkeypatch.setattr(pdf_parser.requests, "get", web)
        return web

    class Env:
        pass

    result = Env()
    result.install = install
    result.clock = fake_clock
    result.text_dir = tmp_path / "text"
    return result


def _paper() -> PaperItem:
    return PaperItem(paper_id=f"arxiv_{ID}", title="Diffusion Policy", arxiv_id=ID)


def _html(body: str = "") -> FakeResponse:
    return FakeResponse(200, (body or PAPER_HTML).encode("utf-8"))


def test_ar5iv_wins_and_pdf_is_not_downloaded(env):
    web = env.install({AR5IV: _html()})
    paper = pdf_parser.sync_extract_pdf_content(_paper())
    assert web.calls == [AR5IV]
    assert paper.source_quality["source"] == "ar5iv"
    assert paper.source_quality["sourceFormat"] == "html"
    assert paper.source_quality["coverageStatus"] == "complete"
    assert paper.source_quality["codeUrls"] == ["https://github.com/acme/fastplan"]
    assert paper.sections["setup"] and "SETUP_BODY" in paper.sections["setup"]
    assert len(paper.tables) == 1 and len(paper.references) == 4 and paper.repo_links
    assert paper.processing_metadata["pdf"]["cache"] == "skipped"
    assert paper.extracted_text and "=== METHODOLOGY ===" in paper.extracted_text


def test_falls_through_to_arxiv_html_when_ar5iv_has_nothing(env):
    web = env.install({AR5IV: FakeResponse(404, b"nope"), ARXIV_HTML: _html()})
    paper = pdf_parser.sync_extract_pdf_content(_paper())
    assert web.calls == [AR5IV, ARXIV_HTML]
    assert paper.source_quality["source"] == "arxiv_html"
    assert paper.source_quality["htmlAttempts"] == [
        {"source": "ar5iv", "status": "not_found"},
        {"source": "arxiv_html", "status": "ok"},
    ]
    assert env.clock.sleeps == []  # different hosts, first request each


def test_redirect_to_abs_and_non_latexml_page_fall_through_to_pdf(env):
    web = env.install({
        AR5IV: FakeResponse(302, b"", {"location": f"https://arxiv.org/abs/{ID}"}),
        ARXIV_HTML: _html(NO_HTML_PAGE),
        ARXIV_PDF: FakeResponse(200, b"%PDF-1.7 fake", {"content-type": "application/pdf"}),
    })
    paper = pdf_parser.sync_extract_pdf_content(_paper())
    assert web.calls == [AR5IV, ARXIV_HTML, ARXIV_PDF]  # the /abs redirect target is never requested
    assert paper.source_quality["source"] == "pdf"
    assert paper.source_quality["sourceFormat"] == "pdf"
    assert [a["status"] for a in paper.source_quality["htmlAttempts"]] == ["redirected_away", "not_latexml"]
    assert "PDF method text." in paper.sections["methodology"]
    assert paper.tables == [] and paper.references == [] and paper.repo_links == []
    # arxiv.org/html then arxiv.org/pdf share one limiter: 15 s between request starts
    assert env.clock.sleeps == [pytest.approx(15.0)]


def test_incomplete_html_is_completed_from_pdf_and_marked_mixed(env):
    web = env.install({
        AR5IV: _html(_minimal_latexml(experiments=False)),
        ARXIV_PDF: FakeResponse(200, b"%PDF-1.7 fake", {"content-type": "application/pdf"}),
    })
    paper = pdf_parser.sync_extract_pdf_content(_paper())
    assert web.calls == [AR5IV, ARXIV_PDF]
    assert paper.source_quality["source"] == "mixed"
    assert paper.sections["methodology"].startswith("2 Method") and "METH words." in paper.sections["methodology"]
    assert "PDF experiment text." in paper.sections["experiments"]
    assert paper.source_quality["pdfFilledSections"] == ["experiments", "limitations"]
    assert paper.source_quality["coverageStatus"] == "complete"


def test_html_failure_and_pdf_failure_raise_but_partial_html_survives(env):
    env.install({AR5IV: _html(_minimal_latexml(experiments=False)), ARXIV_PDF: FakeResponse(503, b"")})
    paper = pdf_parser.sync_extract_pdf_content(_paper())
    assert paper.source_quality["source"] == "ar5iv"
    assert paper.source_quality["coverageStatus"] == "incomplete"
    assert paper.source_quality["missingSections"] == ["experiments"]
    assert paper.source_quality["pdfError"]

    env.install({AR5IV: FakeResponse(404), ARXIV_HTML: FakeResponse(404), ARXIV_PDF: FakeResponse(503, b"")})
    with pytest.raises(Exception):
        pdf_parser.sync_extract_pdf_content(PaperItem(paper_id="arxiv_9999.00001", title="x", arxiv_id="9999.00001"))


def test_extraction_is_cached_and_old_cache_versions_are_ignored(env):
    web = env.install({AR5IV: _html()})
    pdf_parser.sync_extract_pdf_content(_paper())
    cache_file = env.text_dir / f"arxiv_{ID}.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["extraction_version"] == pdf_parser.EXTRACTION_CACHE_VERSION
    assert data["tables"][0]["label"] == "Table 2" and data["references"] and data["repo_links"]

    # second read: served from cache, structured fields restored, no request at all
    cached_web = env.install({})
    again = pdf_parser.sync_extract_pdf_content(_paper())
    assert cached_web.calls == []
    assert again.processing_metadata["pdf"]["cache"] == "hit"
    assert again.tables[0].rows[2] == ["U-Maze", "5.7", "113.9± 3.1"]
    assert again.repo_links and again.references[0].arxiv_id == "2006.11239"

    # an entry written before this change (no version key, no structured fields) is not reused
    data.pop("extraction_version")
    for key in ("tables", "references", "repo_links"):
        data.pop(key)
    data["sections"] = {"abstract": "old", "methodology": "old", "experiments": "old"}
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    fresh_web = env.install({AR5IV: _html()})
    refreshed = pdf_parser.sync_extract_pdf_content(_paper())
    assert fresh_web.calls == [AR5IV]
    assert refreshed.sections["abstract"] != "old"
    assert web.calls == [AR5IV]


def test_paper_item_accepts_cached_json_without_structured_fields():
    paper = PaperItem.model_validate({"paper_id": "arxiv_1", "title": "t", "sections": {"abstract": "a"}})
    assert paper.tables == [] and paper.references == [] and paper.repo_links == []
    dumped = PaperItem(paper_id="x", title="y", repo_links=[{"url": "https://github.com/a/b", "kind": "github"}]).model_dump(mode="json")
    assert json.loads(json.dumps(dumped))["repo_links"][0]["in_bibliography"] is False


def test_local_pdf_never_requests_html(env, tmp_path):
    pdf = tmp_path / "local.pdf"
    pdf.write_bytes(b"%PDF-1.7 fake" + b"x" * 2000)
    web = env.install({})
    paper = pdf_parser.sync_extract_pdf_content(
        PaperItem(paper_id="local_x", title="L", source_type="local_pdf", local_pdf_path=str(pdf))
    )
    assert web.calls == []
    assert paper.source_quality["source"] == "pdf"


def test_pdf_download_goes_through_the_arxiv_limiter(env, tmp_path):
    env.install({
        ARXIV_PDF: FakeResponse(302, b"", {"location": f"https://arxiv.org/pdf/{ID}v2"}),
        f"https://arxiv.org/pdf/{ID}v2": FakeResponse(200, b"%PDF-1.7 ok", {"content-type": "application/pdf"}),
    })
    out = pdf_parser._download_pdf(ARXIV_PDF, tmp_path / "x.pdf")
    assert out.read_bytes().startswith(b"%PDF-")
    assert env.clock.sleeps == [pytest.approx(15.0)]  # the redirect hop is a second request start


def test_html_fetch_refuses_hosts_and_paths_outside_the_allowlist(env):
    from tools import html_source

    with pytest.raises(html_source.HTMLUnavailable):
        html_source.fetch_html_page("http://127.0.0.1/html/2303.04137", ID)
    with pytest.raises(html_source.HTMLUnavailable):
        html_source.fetch_html_page(f"https://arxiv.org/e-print/{ID}", ID)
