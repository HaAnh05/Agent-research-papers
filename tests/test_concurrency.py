"""Concurrency tests: metadata grace wait, PMRL pools, parallel report sections.

All LLM, network and PDF work is faked; no provider or arXiv call is made.
"""
from __future__ import annotations

import copy
import importlib
import threading
import time
from types import SimpleNamespace

from config import config
from nodes import final_report, read_paper as read_paper_module, web_enrich as web_enrich_module
from nodes import write_notes as write_notes_module
from state import GitHubRepoInfo, PMRLNotes, PaperItem, SummaryCards

arxiv_search = importlib.import_module("tools.arxiv_search")  # ``tools`` re-exports a function with this name


# --- arXiv batched metadata ---------------------------------------------------

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>"""
_ENTRY = """<entry>
  <id>http://arxiv.org/abs/{aid}v1</id>
  <title>Title {aid}</title>
  <summary>Summary {aid}</summary>
  <published>2023-03-08T00:00:00Z</published>
  <author><name>Ann Author</name></author>
  <category term="cs.LG"/>
</entry>"""


def test_batch_metadata_uses_one_short_request_and_per_id_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TEXT_CACHE_DIR", tmp_path)
    requests_seen: list = []

    def fake_get(url, params=None, max_retries=3, timeout=12):
        requests_seen.append({"params": params, "max_retries": max_retries, "timeout": timeout})
        entries = "\n".join(_ENTRY.format(aid=i) for i in params["id_list"].split(","))
        return SimpleNamespace(text=_ATOM.format(entries=entries))

    monkeypatch.setattr(arxiv_search, "_arxiv_get", fake_get)
    timing: dict = {}
    result = arxiv_search.fetch_arxiv_metadata_batch(["2303.04137", "2205.09991"], timing=timing)

    assert len(requests_seen) == 1
    assert requests_seen[0]["params"]["id_list"] == "2303.04137,2205.09991"
    assert requests_seen[0]["max_retries"] == config.ARXIV_METADATA_MAX_ATTEMPTS == 1
    assert requests_seen[0]["timeout"] == config.ARXIV_METADATA_TIMEOUT_SEC == 6
    assert result["2303.04137"]["title"] == "Title 2303.04137"
    assert timing["2205.09991"]["status"] == "ok" and timing["2205.09991"]["cache"] == "miss"
    assert arxiv_search.get_cached_arxiv_metadata("2205.09991") is not None

    # Cached ids are served without HTTP; only the new id is requested.
    requests_seen.clear()
    timing.clear()
    result = arxiv_search.fetch_arxiv_metadata_batch(["2303.04137", "2401.00001"], timing=timing)
    assert [r["params"]["id_list"] for r in requests_seen] == ["2401.00001"]
    assert timing["2303.04137"]["cache"] == "hit"
    assert result["2401.00001"]["title"] == "Title 2401.00001"

    requests_seen.clear()
    arxiv_search.fetch_arxiv_metadata_batch(["2303.04137", "2401.00001"])
    assert requests_seen == []


def test_batch_metadata_failure_returns_none_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TEXT_CACHE_DIR", tmp_path)

    def failing_get(*args, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(arxiv_search, "_arxiv_get", failing_get)
    timing: dict = {}
    result = arxiv_search.fetch_arxiv_metadata_batch(["2303.04137"], timing=timing)
    assert result == {"2303.04137": None}
    assert timing["2303.04137"]["status"] == "error"


# --- read_paper ---------------------------------------------------------------

def _arxiv_paper(index: int) -> PaperItem:
    return PaperItem(
        paper_id=f"arxiv_2303.0000{index}",
        title=f"arXiv 2303.0000{index}",
        arxiv_id=f"2303.0000{index}",
        source_type="arxiv",
    )


def _fake_pdf(paper: PaperItem) -> PaperItem:
    time.sleep(0.2)
    paper.processing_metadata = {"pdf": {"durationMs": 200}}
    paper.source_quality = {"coverageStatus": "complete"}
    return paper


def test_read_paper_does_not_wait_for_slow_metadata_beyond_grace(monkeypatch):
    release = threading.Event()

    def slow_batch(ids, *, timing=None):
        release.wait(5)
        return {}

    monkeypatch.setattr(config, "ARXIV_METADATA_GRACE_SEC", 0.3)
    monkeypatch.setattr(read_paper_module, "fetch_arxiv_metadata_batch", slow_batch)
    monkeypatch.setattr(read_paper_module, "sync_extract_pdf_content", _fake_pdf)
    papers = [_arxiv_paper(i) for i in range(1, 4)]
    try:
        started = time.perf_counter()
        result = read_paper_module.read_paper_node({"selected_papers": papers})
        elapsed = time.perf_counter() - started
    finally:
        release.set()

    # 3 PDFs ran concurrently (0.2 s, not 0.6 s) plus at most the 0.3 s grace.
    assert elapsed < 0.9
    timing = result["node_timings"]["read_paper"]
    assert timing["workerCount"] == 3
    assert [p.paper_id for p in result["selected_papers"]] == [p.paper_id for p in papers]
    for paper in papers:
        assert timing["papers"][paper.paper_id]["arxiv"]["status"] == "timeout"
        assert "pdf" in timing["papers"][paper.paper_id]
    # Same fields as the metadata-failure path: safe label, unknown status.
    assert result["selected_papers"][0].title == "arXiv 2303.00001"
    assert result["selected_papers"][0].metadata_status == "unknown"


def test_read_paper_applies_metadata_that_arrives_within_grace(monkeypatch):
    def batch(ids, *, timing=None):
        for aid in ids:
            timing[aid] = {"durationMs": 5, "cache": "miss", "status": "ok"}
        return {
            aid: {"title": f"Real {aid}", "summary": "s", "authors": ["A"], "published": "2023-03-08",
                  "subjects": ["cs.LG"], "metadata_status": "complete"}
            for aid in ids
        }

    monkeypatch.setattr(read_paper_module, "fetch_arxiv_metadata_batch", batch)
    monkeypatch.setattr(read_paper_module, "sync_extract_pdf_content", _fake_pdf)
    result = read_paper_module.read_paper_node({"selected_papers": [_arxiv_paper(1)]})
    paper = result["selected_papers"][0]
    assert paper.title == "Real 2303.00001"
    assert paper.metadata_status == "complete"
    assert result["node_timings"]["read_paper"]["papers"][paper.paper_id]["arxiv"]["status"] == "ok"


# --- PMRL concurrency ---------------------------------------------------------

def _pmrl_ok(paper, *, llm=None):
    result = copy.deepcopy(paper)
    result.notes = PMRLNotes(problem="p", method="m", result="r", limitation="l", brief_summary="C.\n- a\n- b\n- c")
    result.notes_quality = "complete"
    return result, "ok", {"durationMs": 1, "llmMs": 1}


def test_write_notes_fallback_runs_more_than_two_papers_concurrently(monkeypatch):
    barrier = threading.Barrier(5)

    def fake(paper, *, llm=None):
        barrier.wait(timeout=3)  # BrokenBarrierError unless 5 calls overlap
        return _pmrl_ok(paper)

    monkeypatch.setattr(write_notes_module, "analyze_paper_notes", fake)
    papers = [PaperItem(paper_id=f"p{i}", title=f"Paper {i}") for i in range(5)]
    analyzed, _, errors, _ = write_notes_module._run_notes_in_parallel(papers, llm=None)
    assert errors == []
    assert [p.paper_id for p in analyzed] == [f"p{i}" for i in range(5)]


def test_web_enrich_runs_all_pmrl_jobs_in_one_wave(monkeypatch):
    barrier = threading.Barrier(3)

    def fake_pmrl(paper, *, llm=None):
        barrier.wait(timeout=3)
        return _pmrl_ok(paper)

    def fake_github(paper):
        return [], {"durationMs": 1, "tiers": {}}, False

    monkeypatch.setattr(web_enrich_module, "get_llm", lambda: object())
    monkeypatch.setattr(web_enrich_module, "_github_one", fake_github)
    monkeypatch.setattr(web_enrich_module, "analyze_paper_notes", fake_pmrl)
    papers = [PaperItem(paper_id=f"p{i}", title=f"Paper {i}") for i in range(3)]
    result = web_enrich_module.web_enrich_node({"selected_papers": papers})
    assert [p.paper_id for p in result["selected_papers"]] == ["p0", "p1", "p2"]
    assert result["error_logs"] == []
    assert result["node_timings"]["web_enrich"]["workerCount"] == 3


# --- parallel report ----------------------------------------------------------

def _paper(pid: str, title: str, arxiv_id: str, repo: str | None = None) -> PaperItem:
    return PaperItem(
        paper_id=pid,
        title=title,
        arxiv_id=arxiv_id,
        authors=["Ann Author"],
        published="2023-03-08",
        summary="alpha beta gamma delta epsilon",
        sections={"abstract": "alpha beta gamma delta epsilon " * 200, "methodology": "alpha beta gamma delta epsilon " * 200},
        source_quality={"coverageStatus": "complete"},
        notes=PMRLNotes(problem=f"problem of {title}", method="m", result="r", limitation="l",
                        brief_summary="Conclusion.\n- a\n- b\n- c"),
        github_repos=[GitHubRepoInfo(name="o/r", url=repo, stars=3, framework="PyTorch")] if repo else [],
        bibtex=f"@article{{{pid}}}",
    )


class SectionLLM:
    """Fake provider: answers per prompt kind; can fail selected calls."""

    def __init__(self, *, fail_title: str = "", fail_overview: bool = False, barrier: threading.Barrier | None = None):
        self.fail_title = fail_title
        self.fail_overview = fail_overview
        self.barrier = barrier
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def invoke(self, prompt):
        prompt = prompt if isinstance(prompt, str) else "\n".join(str(m.content) for m in prompt)
        with self.lock:
            self.prompts.append(prompt)
        if self.barrier is not None:
            self.barrier.wait(timeout=3)  # BrokenBarrierError unless the calls overlap
        if "ONE paper block" in prompt:
            if self.fail_title and self.fail_title in prompt:
                raise RuntimeError("provider failure")
            return SimpleNamespace(content=(
                "## 2. leaked heading\n### 7. LLM block\n- **Vấn đề**: written by llm\n"
                "- **Mã nguồn**: invented https://github.com/fake/fake\n"
                "- **Liên kết nguồn**: https://evil.example/x\n"
                "- **Trích dẫn**:\n\n```bibtex\n@article{llm_invented}\n```\n"
            ))
        if "summary and conclusion of a research report" in prompt:
            if self.fail_overview:
                raise RuntimeError("provider failure")
            return SimpleNamespace(content="## 1. Tóm tắt\n- overview text\n\n## 3. Kết luận\n- done\n")
        return SimpleNamespace(content="# Legacy report\n\nlegacy body\n")


def _setup(monkeypatch, tmp_path, llm):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path / "summaries")
    (tmp_path / "summaries").mkdir()
    monkeypatch.setattr(final_report, "get_llm", lambda: llm)
    def card(word: str, count: int) -> str:
        return ((word + " ") * count).strip() + "."

    cards = SummaryCards(
        tldr=card("alpha", 25),
        problem=card("beta", 40),
        method=card("gamma", 40),
        key_results=card("delta", 40),
        why_it_matters=card("epsilon", 40),
    )
    monkeypatch.setattr(final_report, "invoke_structured_output", lambda *a, **k: cards)


def _state(papers):
    return {"user_query": "so sánh các bài báo", "selected_papers": papers}


def test_parallel_report_assembles_in_order_with_deterministic_parts(monkeypatch, tmp_path):
    # 2 section calls + 1 overview call must overlap, or the barrier breaks.
    llm = SectionLLM(barrier=threading.Barrier(3))
    _setup(monkeypatch, tmp_path, llm)
    papers = [
        _paper("p1", "First Paper", "2303.04137", repo="https://github.com/o/r"),
        _paper("p2", "Second Paper", "2205.09991"),
    ]
    result = final_report.final_report_node(_state(papers))
    report = result["final_report"]

    assert report.startswith("# Research Report:")
    heads = [line for line in report.split("\n") if line.startswith("## ")]
    assert heads == ["## 1. Tóm tắt", "## 2. Phân tích bài báo", "## 3. Kết luận", "## 4. Tài liệu tham khảo"]
    assert "So sánh" not in report
    paper_section = report.split("## 2. Phân tích bài báo")[1].split("## 3. Kết luận")[0]
    assert paper_section.count("### ") == 2
    assert "leaked heading" not in report
    assert "### 7." not in report and "### LLM block" in paper_section
    # Tail bullets come from code only, exactly once per paper; the LLM's own
    # (invented) code/link/BibTeX lines are stripped.
    assert paper_section.count("```bibtex") == 2
    assert paper_section.count("- **Mã nguồn**") == 2 and paper_section.count("- **Liên kết nguồn**") == 2
    assert paper_section.count("- **Trích dẫn**") == 2
    assert "llm_invented" not in report and "evil.example" not in report and "fake/fake" not in report
    assert "[o/r](https://github.com/o/r) (⭐ 3, PyTorch)" in paper_section
    assert "đừng kết luận bài báo không có code" in paper_section  # paper 2 has no repo
    assert "@article{p1}" in paper_section
    assert "https://arxiv.org/abs/2303.04137" in paper_section
    refs = report.split("## 4. Tài liệu tham khảo")[1]
    assert refs.index("First Paper") < refs.index("Second Paper")
    assert "[ArXiv abstract](https://arxiv.org/abs/2303.04137) | [PDF](https://arxiv.org/pdf/2303.04137.pdf)" in refs
    assert "[Mã nguồn](https://github.com/o/r)" in refs
    # Per-paper prompts carry only that paper's breakdown.
    section_prompts = [p for p in llm.prompts if "ONE paper block" in p]
    assert len(section_prompts) == 2
    assert all(("First Paper" in p) != ("Second Paper" in p) for p in section_prompts)
    assert result["status"] == "success"
    timing = result["node_timings"]["final_report"]
    assert set(timing["sectionPapers"]) == {"p1", "p2"}
    assert {"durationMs", "llmMs"} <= set(timing["sectionPapers"]["p1"])
    assert "overviewMs" in timing and set(timing["summaryPapers"]) == {"p1", "p2"}
    assert result["selected_papers"][0].notes.summary_quality == "complete"


def test_parallel_report_degrades_failed_section_from_notes_and_skips_cache(monkeypatch, tmp_path):
    llm = SectionLLM(fail_title="Second Paper")
    _setup(monkeypatch, tmp_path, llm)
    papers = [_paper("p1", "First Paper", "2303.04137"), _paper("p2", "Second Paper", "2205.09991")]
    result = final_report.final_report_node(_state(papers))

    report = result["final_report"]
    assert result["status"] == "degraded"
    assert "### Second Paper" in report and "problem of Second Paper" in report
    assert "written by llm" in report  # the other section is still LLM-written
    assert len([p for p in llm.prompts if "ONE paper block" in p]) == 2  # no legacy retry
    assert not any("Synthesize a clear" in p for p in llm.prompts)
    # A degraded report is retryable: a rerun calls the provider again.
    llm.prompts.clear()
    final_report.final_report_node(_state(papers))
    assert llm.prompts


def test_parallel_report_overview_failure_uses_minimal_text(monkeypatch, tmp_path):
    llm = SectionLLM(fail_overview=True)
    _setup(monkeypatch, tmp_path, llm)
    papers = [_paper("p1", "First Paper", "2303.04137")]
    result = final_report.final_report_node(_state(papers))
    report = result["final_report"]
    assert result["status"] == "degraded"
    assert "## 1. Tóm tắt" in report and "**First Paper** (2303.04137)" in report
    assert "## 3. Kết luận" in report


def test_parallel_report_caches_good_reports_separately_from_legacy(monkeypatch, tmp_path):
    llm = SectionLLM()
    _setup(monkeypatch, tmp_path, llm)
    papers = [_paper("p1", "First Paper", "2303.04137")]
    first = final_report.final_report_node(_state(papers))
    llm.prompts.clear()
    second = final_report.final_report_node(_state(papers))
    assert llm.prompts == [] and second["node_timings"]["final_report"]["cache"] is True
    assert second["final_report"] == first["final_report"]

    # Legacy mode must miss that entry and use the single synthesis call.
    monkeypatch.setattr(config, "REPORT_PARALLEL_SECTIONS", False)
    legacy = final_report.final_report_node(_state(papers))
    assert legacy["node_timings"]["final_report"].get("cache") is not True
    assert len(llm.prompts) == 1 and "Synthesize a clear" in llm.prompts[0]


def test_legacy_flag_keeps_single_synthesis_call(monkeypatch, tmp_path):
    llm = SectionLLM()
    _setup(monkeypatch, tmp_path, llm)
    monkeypatch.setattr(config, "REPORT_PARALLEL_SECTIONS", False)
    result = final_report.final_report_node(_state([_paper("p1", "First Paper", "2303.04137")]))
    assert len(llm.prompts) == 1
    assert result["final_report"].startswith("# Research Report: First Paper")
    assert "legacy body" in result["final_report"]
    assert "sectionPapers" not in result["node_timings"]["final_report"]


def test_section_prompt_has_no_code_link_or_bibtex_input(monkeypatch, tmp_path):
    llm = SectionLLM()
    _setup(monkeypatch, tmp_path, llm)
    papers = [_paper("p1", "First Paper", "2303.04137", repo="https://github.com/o/r"), _paper("p2", "Second Paper", "2205.09991")]
    final_report.final_report_node(_state(papers))
    for prompt in (p for p in llm.prompts if "ONE paper block" in p):
        assert "@article" not in prompt and "github.com" not in prompt and "arxiv.org" not in prompt
        assert "- **Mã nguồn**" not in prompt and "- **Trích dẫn**" not in prompt


def test_single_paper_overview_has_two_sections_and_no_compare(monkeypatch, tmp_path):
    llm = SectionLLM()
    _setup(monkeypatch, tmp_path, llm)
    result = final_report.final_report_node(_state([_paper("p1", "First Paper", "2303.04137")]))
    overview_prompts = [p for p in llm.prompts if "summary and conclusion of a research report" in p]
    assert len(overview_prompts) == 1
    assert "So sánh" not in overview_prompts[0] and "<benchmark_matrix>" not in overview_prompts[0]
    report = result["final_report"]
    assert "So sánh" not in report
    assert [l for l in report.split("\n") if l.startswith("## ")] == [
        "## 1. Tóm tắt", "## 2. Phân tích bài báo", "## 3. Kết luận", "## 4. Tài liệu tham khảo",
    ]
    assert result["status"] == "success"


def test_single_paper_english_mode_headings(monkeypatch, tmp_path):
    llm = SectionLLM()
    _setup(monkeypatch, tmp_path, llm)
    result = final_report.final_report_node({
        "user_query": "summarize this paper", "selected_papers": [_paper("p1", "First Paper", "2303.04137")],
    })
    report = result["final_report"]
    assert [l for l in report.split("\n") if l.startswith("## ")] == [
        "## 1. Summary", "## 2. Paper Analysis", "## 3. Conclusion", "## 4. References",
    ]
    assert "Comparison" not in report


def test_parallel_cache_key_includes_version(monkeypatch):
    paper = _paper("p1", "First Paper", "2303.04137")
    assert final_report.PARALLEL_REPORT_VERSION == "p7"
    first, _ = final_report._report_cache_key("q", "T", "English", "b", [paper], parallel=True)
    monkeypatch.setattr(final_report, "PARALLEL_REPORT_VERSION", "next")
    second, _ = final_report._report_cache_key("q", "T", "English", "b", [paper], parallel=True)
    legacy1, _ = final_report._report_cache_key("q", "T", "English", "b", [paper])
    assert first != second and legacy1 not in {first, second}
