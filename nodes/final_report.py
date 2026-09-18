from __future__ import annotations

import copy
import datetime
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from config import config
from prompts.final_report_prompt import FINAL_REPORT_PROMPT
from prompts.results_summary_prompt import RESULTS_SUMMARY_PROMPT
from state import PaperItem, ResearchState, SummaryCards
from text_utils import detect_response_language
from tools.llm_provider import extract_text_from_response, get_llm, invoke_structured_output
from nodes.write_notes import SUMMARY_CARD_FIELDS, validate_summary_cards


_ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$", re.IGNORECASE)
_RAW_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_INTERNAL_REPORT_LINE_RE = re.compile(
    r"^\s*(?:synthesis requested|analysis scope:|language mode:|pmrl quality:|"
    r"delivered in .* per language mode|agent (?:status|self-review))\b",
    re.IGNORECASE,
)


def _is_internal_report_line(line: str) -> bool:
    normalized = line.lstrip(" -*•>_\t").strip()
    if _INTERNAL_REPORT_LINE_RE.match(normalized):
        return True
    return bool(re.search(r"\b(?:PMRL\s+(?:quality|status)|language[- ]mode)\b", normalized, re.IGNORECASE))


def _canonical_arxiv_links(paper: PaperItem) -> str:
    arxiv_id = (paper.arxiv_id or "").strip().removeprefix("arxiv:")
    if not arxiv_id or not _ARXIV_ID_RE.fullmatch(arxiv_id):
        return ""
    return f"- **Sources**: [ArXiv abstract](https://arxiv.org/abs/{arxiv_id}) | [PDF](https://arxiv.org/pdf/{arxiv_id}.pdf)"


def _safe_github_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"https://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", text):
        return None
    return text


def _usable_paper_title(value: Any) -> str:
    title = " ".join(str(value or "").split()).strip()
    if not title or _RAW_URL_RE.search(title) or re.fullmatch(r"(?:arxiv\s*)?\d{4}\.\d{4,5}(?:v\d+)?", title, re.IGNORECASE):
        return ""
    return title


def _topic_label(user_query: str) -> str:
    """Turn a multi-paper request into a short readable report topic."""

    text = _RAW_URL_RE.sub("", " ".join((user_query or "").split())).strip(" -:;,.")
    text = re.sub(r"\b(?:https?://|arxiv:)\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:compare|comparison|versus|vs\.?|so sánh|so sanh|đối chiếu)\b", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" -:;,.")
    return text[:100].rstrip(" -:;,.")


def report_title(user_query: str, papers: List[PaperItem]) -> str:
    """Return a metadata-first title safe for the Markdown report heading."""

    if len(papers) == 1:
        title = _usable_paper_title(papers[0].title)
        if title:
            return f"Research Report: {title}"
        arxiv_id = str(papers[0].arxiv_id or "").strip().removeprefix("arxiv:")
        if _ARXIV_ID_RE.fullmatch(arxiv_id):
            return f"Research Report: arXiv {arxiv_id}"
    topic = _topic_label(user_query)
    if topic.casefold() in {"tổng hợp đi", "tổng hợp", "summarize", "summary", "research analysis", "direct paper analysis"}:
        topic = ""
    if not topic:
        titles = [_usable_paper_title(paper.title) for paper in papers]
        titles = [title for title in titles if title]
        if titles:
            topic = " and ".join(titles[:2])
            if len(titles) > 2:
                topic += " and more"
    return f"Research Report: {(topic or 'Selected Papers')[:100].rstrip()}"


def _source_quality_complete(paper: PaperItem) -> bool:
    quality = paper.source_quality if isinstance(paper.source_quality, dict) else {}
    coverage = str(
        quality.get("coverage_status")
        or quality.get("coverageStatus")
        or quality.get("status")
        or ""
    ).strip().casefold()
    if coverage in {"complete", "completed", "success", "ok"}:
        return True
    if coverage in {"incomplete", "degraded", "missing", "unknown", "error"}:
        return False
    section_coverage = quality.get("sections") or quality.get("coverage")
    if isinstance(section_coverage, dict):
        required = ("abstract", "methodology", "experiments")
        return all(bool(section_coverage.get(key)) for key in required)
    # A legacy cache has no measured coverage metadata.  It must not be
    # advertised as final until a current parser records its coverage.
    return False


def _sanitize_report_text(value: Any, title: str) -> str:
    """Remove internal orchestration chatter and normalize the first heading."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return f"# {title}\n"
    lines = [line for line in text.split("\n") if not _is_internal_report_line(line)]
    first_heading = next((index for index, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if first_heading is None:
        lines.insert(0, f"# {title}")
    else:
        heading = lines[first_heading].lstrip().lstrip("#").strip()
        # The report heading is deterministic and never carries a URL or a
        # provider-generated variant of the title.
        if first_heading == 0 or _RAW_URL_RE.search(heading) or heading.lower().startswith(("comprehensive research report", "research report")):
            lines[first_heading] = f"# {title}"
    return "\n".join(lines).strip() + "\n"


def _paper_source_text(paper: PaperItem) -> str:
    if paper.sections:
        return (
            f"=== ABSTRACT ===\n{paper.sections.get('abstract', '')}\n\n"
            f"=== METHODOLOGY / ARCHITECTURE ===\n{paper.sections.get('methodology', '')}\n\n"
            f"=== EXPERIMENTS / BENCHMARKS / RESULTS ===\n{paper.sections.get('experiments', '')}\n\n"
            f"=== LIMITATIONS / FUTURE WORK ===\n{paper.sections.get('limitations', '')}"
        )
    return paper.extracted_text or paper.summary or ""


def _summary_prompt(paper: PaperItem, report_text: str, language_mode: str) -> str:
    notes = paper.notes
    pmrl_text = (
        f"Problem: {notes.problem}\n"
        f"Method: {notes.method}\n"
        f"Result: {notes.result}\n"
        f"Limitation: {notes.limitation}"
        if notes
        else "No detailed PMRL notes were available."
    )
    return RESULTS_SUMMARY_PROMPT.format(
        title=paper.title,
        arxiv_id=paper.arxiv_id or "N/A",
        authors=", ".join(paper.authors) if paper.authors else "N/A",
        published=paper.published or "N/A",
        language_mode=language_mode,
        pmrl_notes=pmrl_text[:9000],
        final_report=report_text[:12000],
        source_text=f"{paper.title}\n{paper.summary}\n{_paper_source_text(paper)}"[:14000],
    )


def _mark_summary_status(paper: PaperItem, status: str) -> PaperItem:
    result = copy.deepcopy(paper)
    if result.notes is not None:
        result.notes.summary_cards = SummaryCards()
        result.notes.summary_quality = status  # type: ignore[assignment]
    result.processing_metadata = dict(result.processing_metadata or {})
    result.processing_metadata["summary"] = {"status": status, "wordCount": 0}
    return result


def _summarize_one(
    paper: PaperItem,
    report_text: str,
    language_mode: str,
    llm: Any,
) -> tuple[PaperItem, str, Dict[str, Any]]:
    started = time.perf_counter()
    result = copy.deepcopy(paper)
    try:
        llm_started = time.perf_counter()
        cards = invoke_structured_output(
            _summary_prompt(result, report_text, language_mode),
            SummaryCards,
            llm=llm,
            provider=config.DEFAULT_PROVIDER,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        validated, status = validate_summary_cards(
            cards,
            source_text=f"{result.title}\n{result.summary}\n{_paper_source_text(result)}",
        )
        if result.notes is None:
            return result, "missing", {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms}
        result.notes.summary_cards = validated
        result.notes.summary_quality = status  # type: ignore[assignment]
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {
            "status": status,
            "wordCount": sum(len(str(getattr(validated, field, "") or "").split()) for field in SUMMARY_CARD_FIELDS),
        }
        return result, status, {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
        }
    except Exception:
        result = _mark_summary_status(result, "invalid")
        return result, "invalid", {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": 0,
        }


def _post_report_summary(
    papers: List[PaperItem],
    report_text: str,
    language_mode: str,
    llm: Any,
) -> tuple[List[PaperItem], List[str], Dict[str, Dict[str, Any]], bool]:
    """Generate validated cards after report synthesis, preserving paper order."""

    if not papers:
        return [], [], {}, False

    with ThreadPoolExecutor(max_workers=min(2, len(papers))) as executor:
        completed = list(executor.map(lambda paper: _summarize_one(paper, report_text, language_mode, llm), papers))

    results: List[PaperItem] = []
    logs: List[str] = []
    timings: Dict[str, Dict[str, Any]] = {}
    failed = False
    for paper, status, timing in completed:
        results.append(paper)
        timings[paper.paper_id] = timing
        if status == "complete":
            logs.append(f"[final_report] Summary cards ready: {paper.paper_id}")
        else:
            failed = True
            logs.append(f"[final_report] Summary cards unavailable: {paper.paper_id}")
    return results, logs, timings, failed


def _paper_breakdown(paper: PaperItem, index: int) -> str:
    notes = paper.notes
    repositories: List[str] = []
    seen_urls: set[str] = set()
    for repo in paper.github_repos[:4]:
        url = _safe_github_url(repo.url)
        if url and url not in seen_urls:
            repositories.append(f"[{repo.name}]({url}) (⭐ {repo.stars}, {repo.framework})")
            seen_urls.add(url)
    quality = paper.source_quality if isinstance(paper.source_quality, dict) else {}
    verified_code_urls = quality.get("codeUrls") or quality.get("code_urls") or []
    if isinstance(verified_code_urls, list):
        for candidate in verified_code_urls[:4]:
            url = _safe_github_url(candidate)
            if url and url not in seen_urls:
                repositories.append(f"[Code linked from official arXiv HTML]({url})")
                seen_urls.add(url)
    github_info = "- Repository: " + ", ".join(repositories) if repositories else "- Repository: No repository was verified by the enrichment step."
    source_links = _canonical_arxiv_links(paper)
    extraction_warning = ""
    if not _source_quality_complete(paper):
        extraction_warning = "- **Extraction warning**: Important source sections were not fully verified; absent evidence is unverified.\n"
    return (
        f"### {index}. {paper.title}\n"
        f"- **ArXiv ID**: {paper.arxiv_id or 'N/A'} | **Published**: {paper.published or 'N/A'}\n"
        f"- **Authors**: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
        f"- **Problem**: {notes.problem if notes else 'N/A'}\n"
        f"- **Methodology**: {notes.method if notes else 'N/A'}\n"
        f"- **Results & Metrics**: {notes.result if notes else 'N/A'}\n"
        f"- **Limitations**: {notes.limitation if notes else 'N/A'}\n"
        f"{extraction_warning}"
        f"{github_info}\n"
        f"{source_links}\n\n"
        f"```bibtex\n{paper.bibtex or ''}\n```"
    )


def final_report_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesize and persist the report while preserving canonical links."""

    started = time.perf_counter()
    user_query = state.get("user_query") or "Research Analysis"
    papers = list(state.get("selected_papers", []) or [])
    benchmark_matrix = state.get("benchmark_matrix")
    language_mode = detect_response_language(user_query)
    title = report_title(user_query, papers)
    logs = [f"[final_report] Tổng hợp báo cáo cho '{user_query}'"]
    incomplete_sources = [paper.paper_id for paper in papers if not _source_quality_complete(paper)]
    if incomplete_sources:
        logs.append(f"[final_report] Cảnh báo trích xuất chưa hoàn chỉnh cho {len(incomplete_sources)} bài")

    benchmark_section = (
        f"=== BENCHMARK COMPARISON MATRIX & TRADE-OFFS ===\n{benchmark_matrix}"
        if benchmark_matrix
        else ""
    )
    prompt = FINAL_REPORT_PROMPT.format(
        user_query=user_query,
        report_title=title,
        language_mode=language_mode,
        detailed_papers_breakdown="\n\n".join(_paper_breakdown(paper, index) for index, paper in enumerate(papers, 1)),
        optional_benchmark_matrix_section=benchmark_section,
    )

    llm_ms = 0
    try:
        llm = get_llm()
        llm_started = time.perf_counter()
        response = llm.invoke(prompt)
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        raw_report = extract_text_from_response(response.content)
        if not raw_report.strip():
            raise ValueError("Empty report response")
        report_text = _sanitize_report_text(raw_report, title)

        summary_started = time.perf_counter()
        summary_logs: List[str] = []
        summary_timings: Dict[str, Dict[str, Any]] = {}
        summary_failed = False
        if incomplete_sources:
            summarized_papers = [_mark_summary_status(paper, "missing") for paper in papers]
            summary_logs.append("[final_report] Bỏ qua Summary cards vì trích xuất nguồn chưa hoàn chỉnh")
            summary_failed = True
        else:
            summarized_papers, summary_logs, summary_timings, summary_failed = _post_report_summary(
                papers,
                report_text,
                language_mode,
                llm,
            )
        logs.extend(summary_logs)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        filename_seed = title.removeprefix("Research Report:").strip() if papers else user_query
        safe_title = "".join(c for c in filename_seed[:80] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        safe_title = safe_title or "research_report"
        report_path = config.REPORTS_DIR / f"report_{safe_title}_{timestamp}_{unique_suffix}.md"
        report_path.write_text(report_text, encoding="utf-8")

        logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")
        timing = {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
            "paperCount": len(papers),
            "summaryMs": max(0, int((time.perf_counter() - summary_started) * 1000)),
            "summaryPapers": summary_timings,
        }
        return {
            "final_report": report_text,
            "selected_papers": summarized_papers,
            "status": "degraded" if incomplete_sources or summary_failed else "success",
            "trace_logs": logs,
            "node_timings": {"final_report": timing},
        }
    except Exception as exc:
        err_msg = f"Lỗi tạo báo cáo tổng hợp: {type(exc).__name__}"
        logs.append(f"❌ [final_report] {err_msg}")
        timing = {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
            "paperCount": len(papers),
        }
        return {
            "final_report": f"# {title}\n\n{err_msg}",
            "status": "error",
            "error_message": err_msg,
            "trace_logs": logs,
            "error_logs": [err_msg],
            "node_timings": {"final_report": timing},
        }
