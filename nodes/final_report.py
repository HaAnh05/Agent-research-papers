from __future__ import annotations

import copy
import datetime
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from config import config
from nodes.write_notes import SUMMARY_CARD_FIELDS, validate_summary_cards
from prompts.final_report_prompt import (
    FINAL_REPORT_PROMPT,
    PAPER_SECTION_SYSTEM,
    PAPER_SECTION_USER,
    REPORT_OVERVIEW_SINGLE_SYSTEM,
    REPORT_OVERVIEW_SINGLE_USER,
    REPORT_OVERVIEW_SYSTEM,
    REPORT_OVERVIEW_USER,
)
from prompts.results_summary_prompt import RESULTS_SUMMARY_SYSTEM, RESULTS_SUMMARY_USER
from state import PaperItem, ResearchState, SummaryCards
from text_utils import detect_response_language
from tools.cache_manager import load_json_cache, prompt_fingerprint, save_json_cache, summary_cache_path
from tools.llm_provider import extract_text_from_response, get_llm, invoke_structured_output, invoke_text


SUMMARY_PIPELINE_VERSION = "v4"
# Bump when the parallel report prompts or assembly change, so cached parallel
# reports from an older layout are not reused.  The legacy key is unaffected.
PARALLEL_REPORT_VERSION = "p7"


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


def _summary_source_evidence(paper: PaperItem) -> str:
    """Return claim-bearing evidence; metadata such as the title is not evidence."""

    if paper.sections:
        body = "\n".join(
            str(paper.sections.get(section) or "")
            for section in ("abstract", "methodology", "experiments", "limitations")
        )
    else:
        body = paper.extracted_text or ""
    return f"{paper.summary}\n{body}".strip()


def _summary_source_quality(paper: PaperItem) -> str:
    quality = paper.source_quality if isinstance(paper.source_quality, dict) else {}
    coverage = str(quality.get("coverageStatus") or quality.get("coverage_status") or "unknown")
    missing = quality.get("missingSections") or quality.get("missing_sections") or []
    missing_text = ", ".join(str(item) for item in missing if str(item).strip()) if isinstance(missing, list) else ""
    warning = str(quality.get("warning") or "").strip()
    return (
        f"Coverage status: {coverage}\n"
        f"Missing verified sections: {missing_text or 'none reported'}\n"
        f"Parser warning: {warning or 'none'}"
    )


def build_cards_messages(paper: PaperItem, language_mode: str) -> tuple[str, str]:
    """Return the pure ``(system, user)`` pair for one summary-card call."""

    notes = paper.notes
    pmrl_text = (
        f"Problem: {notes.problem}\n"
        f"Method: {notes.method}\n"
        f"Result: {notes.result}\n"
        f"Limitation: {notes.limitation}"
        if notes
        else "No detailed PMRL notes were available."
    )
    user = RESULTS_SUMMARY_USER.format(
        title=paper.title,
        arxiv_id=paper.arxiv_id or "N/A",
        authors=", ".join(paper.authors) if paper.authors else "N/A",
        published=paper.published or "N/A",
        language_mode=language_mode,
        pmrl_notes=pmrl_text[:9000],
        source_quality=_summary_source_quality(paper),
        source_text=f"{paper.summary}\n{_paper_source_text(paper)}"[:14000],
    )
    return RESULTS_SUMMARY_SYSTEM, user


def _mark_summary_status(paper: PaperItem, status: str, reason: str = "") -> PaperItem:
    result = copy.deepcopy(paper)
    if result.notes is not None:
        result.notes.summary_cards = SummaryCards()
        result.notes.summary_quality = status  # type: ignore[assignment]
        result.notes.summary_card_statuses = {
            field: status if status in {"invalid", "missing"} else "unknown"
            for field in SUMMARY_CARD_FIELDS
        }
        result.notes.summary_card_reasons = {
            field: reason for field in SUMMARY_CARD_FIELDS if reason
        }
    result.processing_metadata = dict(result.processing_metadata or {})
    result.processing_metadata["summary"] = {"status": status, "wordCount": 0, "reason": reason}
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
        # ``report_text`` is kept in the signature for callers; the cards are
        # grounded in the notes and the source evidence only.
        system_text, user_text = build_cards_messages(result, language_mode)
        cards = invoke_structured_output(
            user_text,
            SummaryCards,
            llm=llm,
            provider=config.DEFAULT_PROVIDER,
            system=system_text,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        validated, status, reason, card_statuses, card_reasons = validate_summary_cards(
            cards,
            source_text=_summary_source_evidence(result),
        )
        if result.notes is None:
            return result, "missing", {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms, "reason": reason or "no_pmrl_notes"}
        result.notes.summary_cards = validated
        result.notes.summary_quality = status  # type: ignore[assignment]
        result.notes.summary_card_statuses = card_statuses  # type: ignore[assignment]
        result.notes.summary_card_reasons = card_reasons
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {
            "status": status,
            "wordCount": sum(len(str(getattr(validated, field, "") or "").split()) for field in SUMMARY_CARD_FIELDS),
            "reason": reason,
            "cardReasons": card_reasons,
        }
        return result, status, {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
            "reason": reason,
        }
    except Exception as exc:
        result = _mark_summary_status(result, "invalid", f"llm_error:{type(exc).__name__}")
        return result, "invalid", {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": 0,
            "reason": f"llm_error:{type(exc).__name__}",
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
    return _collect_summary_results(completed)


def _collect_summary_results(
    completed: List[tuple[PaperItem, str, Dict[str, Any]]],
) -> tuple[List[PaperItem], List[str], Dict[str, Dict[str, Any]], bool]:
    """Turn ordered ``_summarize_one`` results into papers, logs, timings, failed."""

    results: List[PaperItem] = []
    logs: List[str] = []
    timings: Dict[str, Dict[str, Any]] = {}
    failed = False
    for paper, status, timing in completed:
        results.append(paper)
        timings[paper.paper_id] = timing
        if status == "complete":
            logs.append(f"[final_report] Summary cards ready: {paper.paper_id}")
        elif status == "partial":
            failed = True
            reason = str((timing or {}).get("reason") or status)
            logs.append(f"[final_report] Summary cards partially ready: {paper.paper_id} ({reason})")
        else:
            failed = True
            reason = str((timing or {}).get("reason") or status)
            logs.append(f"[final_report] Summary cards unavailable: {paper.paper_id} ({reason})")
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


def _report_cache_key(
    user_query: str,
    title: str,
    language_mode: str,
    breakdown: str,
    papers: List[PaperItem],
    *,
    parallel: bool = False,
) -> tuple[str, Any]:
    """Resolve the report cache entry for the exact synthesis input (fail-open).

    ``parallel`` selects the parallel-sections mode.  It is part of the key so
    the two modes never share an entry; the legacy key is left unchanged.
    """

    summary_inputs = "\n\n".join(
        "\n".join(build_cards_messages(paper, language_mode)) for paper in papers
    )
    key = (
        "report_"
        + prompt_fingerprint(
            config.DEFAULT_PROVIDER,
            config.DEFAULT_MODEL,
            FINAL_REPORT_PROMPT,
            user_query,
            title,
            language_mode,
            breakdown,
            RESULTS_SUMMARY_SYSTEM,
            SUMMARY_PIPELINE_VERSION,
            summary_inputs,
            *(
                ("parallel_sections", PARALLEL_REPORT_VERSION, PAPER_SECTION_SYSTEM, PAPER_SECTION_USER, REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER, REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER)
                if parallel
                else ()
            ),
        )
    )
    return key, summary_cache_path(key)


def _restore_report_cache(papers: List[PaperItem], cached: Any) -> tuple[str, List[PaperItem]] | None:
    """Rebuild the report text and per-paper cards from a cache entry."""

    if not isinstance(cached, dict):
        return None
    report_text = cached.get("report_text")
    cards_map = cached.get("cards")
    if not isinstance(report_text, str) or not report_text.strip() or not isinstance(cards_map, dict):
        return None
    restored: List[PaperItem] = []
    for paper in papers:
        entry = cards_map.get(paper.paper_id)
        if not isinstance(entry, dict):
            return None
        try:
            cards = SummaryCards.model_validate(entry.get("cards") or {})
        except Exception:
            return None
        quality = str(entry.get("quality") or "")
        # Failed summaries are retryable and must not become durable cache
        # hits. Complete and partial entries passed per-card validation.
        if quality not in {"complete", "partial"}:
            return None
        result = copy.deepcopy(paper)
        if result.notes is None:
            return None
        result.notes.summary_cards = cards
        result.notes.summary_quality = quality  # type: ignore[assignment]
        statuses = entry.get("card_statuses")
        if isinstance(statuses, dict):
            result.notes.summary_card_statuses = {
                str(name): str(status)  # type: ignore[dict-item]
                for name, status in statuses.items()
                if name in SUMMARY_CARD_FIELDS and status in {"complete", "unsupported", "invalid", "missing", "unknown"}
            }
        reasons = entry.get("card_reasons")
        if isinstance(reasons, dict):
            result.notes.summary_card_reasons = {
                str(name): str(reason)
                for name, reason in reasons.items()
                if name in SUMMARY_CARD_FIELDS and isinstance(reason, str) and reason
            }
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {
            "status": quality,
            "wordCount": sum(len(str(getattr(cards, field, "") or "").split()) for field in SUMMARY_CARD_FIELDS),
            "reason": "cache",
        }
        restored.append(result)
    return report_text, restored


def _synthesize_report(llm: Any, prompt: str, title: str) -> tuple[str, int]:
    """Run the free-form synthesis call in a worker thread."""

    llm_started = time.perf_counter()
    response = llm.invoke(prompt)
    llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
    raw_report = extract_text_from_response(response.content)
    if not raw_report.strip():
        raise ValueError("Empty report response")
    return _sanitize_report_text(raw_report, title), llm_ms


# --- Parallel report synthesis (REPORT_PARALLEL_SECTIONS) -------------------

_REPORT_HEADINGS: Dict[str, Dict[str, str]] = {
    "Vietnamese": {
        "summary": "1. Tóm tắt",
        "papers": "2. Phân tích bài báo",
        "conclusion": "3. Kết luận",
        "references": "4. Tài liệu tham khảo",
    },
    "English": {
        "summary": "1. Summary",
        "papers": "2. Paper Analysis",
        "conclusion": "3. Conclusion",
        "references": "4. References",
    },
}
_SECTION_LABELS: Dict[str, Dict[str, str]] = {
    "Vietnamese": {
        "problem": "Vấn đề",
        "method": "Phương pháp",
        "result": "Kết quả chính",
        "limitation": "Hạn chế",
        "code": "Mã nguồn",
        "extraction": "Trích xuất",
        "links": "Liên kết nguồn",
        "citation": "Trích dẫn",
    },
    "English": {
        "problem": "Problem",
        "method": "Methodology",
        "result": "Key results",
        "limitation": "Limitations",
        "code": "Source code",
        "extraction": "Extraction",
        "links": "Source links",
        "citation": "Citation",
    },
}
_FALLBACK_TEXT: Dict[str, Dict[str, str]] = {
    "Vietnamese": {
        "no_repo": "Không tìm thấy repository được xác minh trong lần chạy này; đừng kết luận bài báo không có code.",
        "extraction": "Trích xuất nguồn chưa đầy đủ; nội dung thiếu được coi là chưa xác minh.",
        "conclusion": "Xem phân tích từng bài ở mục 2. Các số liệu từ bộ dữ liệu và đơn vị khác nhau không so sánh trực tiếp được.",
        "no_summary": "Chưa có tóm tắt chi tiết.",
    },
    "English": {
        "no_repo": "No verified repository was found in this run; do not conclude that the paper has no code.",
        "extraction": "Source extraction is incomplete; missing content is unverified.",
        "conclusion": "See the per-paper analysis in section 2. Metrics on different datasets or units are not directly comparable.",
        "no_summary": "No detailed summary is available.",
    },
}


def _report_headings(language_mode: str) -> Dict[str, str]:
    return _REPORT_HEADINGS.get(language_mode, _REPORT_HEADINGS["English"])


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _verified_repo_urls(paper: PaperItem) -> List[str]:
    """Return GitHub URLs that passed ``_safe_github_url``, deduplicated in order."""

    urls: List[str] = []
    for repo in paper.github_repos[:4]:
        url = _safe_github_url(repo.url)
        if url and url not in urls:
            urls.append(url)
    quality = paper.source_quality if isinstance(paper.source_quality, dict) else {}
    code_urls = quality.get("codeUrls") or quality.get("code_urls") or []
    if isinstance(code_urls, list):
        for candidate in code_urls[:4]:
            url = _safe_github_url(candidate)
            if url and url not in urls:
                urls.append(url)
    return urls


def _bibtex_block(paper: PaperItem) -> str:
    return f"```bibtex\n{paper.bibtex or ''}\n```"


def _paper_tail(paper: PaperItem, language_mode: str) -> List[str]:
    """Deterministic bullets: source code, extraction warning, links, BibTeX."""

    labels = _SECTION_LABELS.get(language_mode, _SECTION_LABELS["English"])
    texts = _FALLBACK_TEXT.get(language_mode, _FALLBACK_TEXT["English"])
    repositories: List[str] = []
    for repo in paper.github_repos[:4]:
        url = _safe_github_url(repo.url)
        if url and all(url not in item for item in repositories):
            repositories.append(f"[{repo.name}]({url}) (⭐ {repo.stars}, {repo.framework})")
    for url in _verified_repo_urls(paper):
        if all(url not in item for item in repositories):
            repositories.append(f"[GitHub]({url})")
    lines = [f"- **{labels['code']}**: {', '.join(repositories) if repositories else texts['no_repo']}"]
    if not _source_quality_complete(paper):
        lines.append(f"- **{labels['extraction']}**: {texts['extraction']}")
    source_links = _canonical_arxiv_links(paper).removeprefix("- **Sources**: ")
    if source_links:
        lines.append(f"- **{labels['links']}**: {source_links}")
    lines.append(f"- **{labels['citation']}**:\n\n{_bibtex_block(paper)}")
    return lines


def _deterministic_paper_section(paper: PaperItem, language_mode: str) -> str:
    """Notes-only paper block used when the section call fails."""

    labels = _SECTION_LABELS.get(language_mode, _SECTION_LABELS["English"])
    notes = paper.notes
    lines = [
        f"### {paper.title}",
        f"- **{labels['problem']}**: {notes.problem if notes else 'N/A'}",
        f"- **{labels['method']}**: {notes.method if notes else 'N/A'}",
        f"- **{labels['result']}**: {notes.result if notes else 'N/A'}",
        f"- **{labels['limitation']}**: {notes.limitation if notes else 'N/A'}",
    ]
    return "\n".join(lines + _paper_tail(paper, language_mode))


def _paper_section_input(paper: PaperItem, index: int = 0) -> str:
    """Slim input for the section call: PMRL notes only.

    Repositories, links and BibTeX are appended by code, so the model neither
    reads nor re-emits them.  ``index`` is unused (the order is positional).
    """

    notes = paper.notes
    extraction = (
        ""
        if _source_quality_complete(paper)
        else "Extraction warning: important source sections were not fully verified; absent evidence is unverified.\n"
    )
    return (
        f"Title: {paper.title}\n"
        f"ArXiv ID: {paper.arxiv_id or 'N/A'} | Published: {paper.published or 'N/A'}\n"
        f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
        f"Problem: {notes.problem if notes else 'N/A'}\n"
        f"Methodology: {notes.method if notes else 'N/A'}\n"
        f"Results & Metrics: {notes.result if notes else 'N/A'}\n"
        f"Limitations: {notes.limitation if notes else 'N/A'}\n"
        f"{extraction}"
    ).rstrip()


def build_section_messages(
    paper: PaperItem,
    index: int,
    user_query: str,
    language_mode: str,
) -> tuple[str, str]:
    """Return the pure ``(system, user)`` pair for one paper-section call."""

    labels = _SECTION_LABELS.get(language_mode, _SECTION_LABELS["English"])
    user = PAPER_SECTION_USER.format(
        user_query=user_query,
        language_mode=language_mode,
        paper_breakdown=_paper_section_input(paper, index),
        label_problem=labels["problem"],
        label_method=labels["method"],
        label_result=labels["result"],
        label_limitation=labels["limitation"],
    )
    return PAPER_SECTION_SYSTEM, user


_BIBTEX_FENCE_RE = re.compile(r"```bibtex.*?```", re.DOTALL | re.IGNORECASE)


def _tail_labels() -> set[str]:
    return {
        _SECTION_LABELS[mode][key].casefold()
        for mode in _SECTION_LABELS
        for key in ("code", "extraction", "links", "citation")
    } | {"sources", "source code", "repository", "bibtex", "citation", "mã nguồn", "trích xuất", "liên kết nguồn", "trích dẫn"}


def _normalize_paper_section(text: str, paper: PaperItem, language_mode: str) -> str:
    """Keep one ``###`` heading and the four analysis bullets; append the tail.

    Any LLM line carrying a source-code, extraction, link or citation label is
    removed, so the deterministic tail appears exactly once.
    """

    cleaned = _BIBTEX_FENCE_RE.sub("", str(text or "").replace("\r\n", "\n"))
    tail_labels = _tail_labels()
    kept: List[str] = []
    skipping = False
    for line in cleaned.split("\n"):
        label = re.match(r"^\s*[-*•]\s*\**([^*:\n]{1,40}?)\**\s*:", line)
        if label and label.group(1).strip().casefold() in tail_labels:
            skipping = True
            continue
        if skipping and (not line.strip() or re.match(r"^\s{2,}\S", line)):
            continue
        skipping = False
        if re.match(r"^\s{0,3}#{1,2}\s", line) or _is_internal_report_line(line):
            continue
        kept.append(line)
    while kept and not kept[0].strip():
        kept.pop(0)
    if kept and re.match(r"^\s{0,3}###\s+\S", kept[0]):
        # Drop a numeric prefix such as "### 1. Title"; order is positional.
        kept[0] = re.sub(r"^(\s{0,3}###\s+)\d+\.\s*", r"\1", kept[0]).strip()
    else:
        kept.insert(0, f"### {paper.title}")
    body = "\n".join(kept).strip()
    return body + "\n" + "\n".join(_paper_tail(paper, language_mode))


def _write_paper_section(
    llm: Any,
    paper: PaperItem,
    index: int,
    user_query: str,
    language_mode: str,
) -> tuple[str, Dict[str, int]]:
    """One LLM call writing one paper block from that paper's breakdown only."""

    started = time.perf_counter()
    system_text, user_text = build_section_messages(paper, index, user_query, language_mode)
    llm_started = time.perf_counter()
    response = invoke_text(llm, system_text, user_text)
    llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
    raw = extract_text_from_response(response.content)
    if not raw.strip():
        raise ValueError("Empty paper section response")
    block = _normalize_paper_section(raw, paper, language_mode)
    return block, {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms}


def _papers_context(papers: List[PaperItem]) -> str:
    """Compact per-paper context for the overview call (no full breakdown)."""

    blocks: List[str] = []
    for index, paper in enumerate(papers, 1):
        notes = paper.notes
        blocks.append(
            f"[{index}] {paper.title} (arXiv: {paper.arxiv_id or 'N/A'})\n"
            f"- Problem excerpt: {_clip(notes.problem if notes else '', 500) or 'N/A'}\n"
            f"- Method excerpt: {_clip(notes.method if notes else '', 600) or 'N/A'}\n"
            f"- Key result excerpt: {_clip(notes.result if notes else '', 1200) or 'N/A'}\n"
            f"- Limitation excerpt: {_clip(notes.limitation if notes else '', 700) or 'N/A'}"
            + ("" if _source_quality_complete(paper) else "\n- Source extraction: incomplete; absent evidence is unverified")
        )
    return "\n\n".join(blocks)


def build_overview_messages(
    papers: List[PaperItem],
    user_query: str,
    title: str,
    language_mode: str,
) -> tuple[str, str]:
    """Return the pure ``(system, user)`` pair for the overview call.

    Both variants write the summary + conclusion sections; the single-paper
    variant keeps them shorter.
    """

    headings = _report_headings(language_mode)
    if len(papers) == 1:
        return REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER.format(
            user_query=user_query,
            report_title=title,
            language_mode=language_mode,
            papers_context=_papers_context(papers),
            heading_summary=headings["summary"],
            heading_conclusion=headings["conclusion"],
        )
    return REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER.format(
        user_query=user_query,
        report_title=title,
        language_mode=language_mode,
        papers_context=_papers_context(papers),
        heading_summary=headings["summary"],
        heading_conclusion=headings["conclusion"],
    )


_OVERVIEW_KEYS = {"1": "summary", "3": "conclusion"}


def _parse_overview(text: str) -> Dict[str, str]:
    """Split the overview output into its bodies by the numbered ``##`` headings."""

    sections: Dict[str, List[str]] = {}
    current: str | None = None
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        heading = re.match(r"^\s{0,3}(#{1,6})\s*(?:(\d+)\s*[.)]?)?", line)
        if heading and heading.group(1) == "##":
            current = _OVERVIEW_KEYS.get(heading.group(2) or "")
            if current is not None:
                sections.setdefault(current, [])
            continue
        if heading and heading.group(1) == "#":
            current = None
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items() if "\n".join(lines).strip()}


def _write_overview(
    llm: Any,
    user_query: str,
    title: str,
    language_mode: str,
    papers: List[PaperItem],
) -> tuple[Dict[str, str], Dict[str, int]]:
    """One LLM call writing the summary and conclusion sections."""

    started = time.perf_counter()
    system_text, user_text = build_overview_messages(papers, user_query, title, language_mode)
    llm_started = time.perf_counter()
    response = invoke_text(llm, system_text, user_text)
    llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
    raw = extract_text_from_response(response.content)
    if not raw.strip():
        raise ValueError("Empty overview response")
    return _parse_overview(raw), {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms}


def _fallback_overview_part(
    key: str,
    papers: List[PaperItem],
    language_mode: str,
) -> str:
    texts = _FALLBACK_TEXT.get(language_mode, _FALLBACK_TEXT["English"])
    if key == "summary":
        lines: List[str] = []
        for paper in papers:
            notes = paper.notes
            first = ((notes.brief_summary if notes else "") or "").strip().split("\n")[0]
            detail = _clip(first or (notes.result if notes else ""), 300) or texts["no_summary"]
            lines.append(f"- **{paper.title}** ({paper.arxiv_id or 'N/A'}): {detail}")
        return "\n".join(lines) or texts["no_summary"]
    return texts["conclusion"]


def _references_section(papers: List[PaperItem], language_mode: str) -> str:
    """Deterministic reference list from metadata and verified links only."""

    labels = _SECTION_LABELS.get(language_mode, _SECTION_LABELS["English"])
    lines: List[str] = []
    for index, paper in enumerate(papers, 1):
        meta = ", ".join(
            part
            for part in (
                ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "") if paper.authors else "",
                paper.published or "",
            )
            if part
        )
        parts = [_canonical_arxiv_links(paper).removeprefix("- **Sources**: ")]
        repo_urls = _verified_repo_urls(paper)
        if repo_urls:
            parts.append(f"[{labels['code']}]({repo_urls[0]})")
        links = " | ".join(part for part in parts if part)
        entry = f"{index}. **{paper.title}**"
        if meta:
            entry += f" ({meta})"
        if links:
            entry += f" — {links}"
        lines.append(entry)
    return "\n".join(lines)


def _assemble_parallel_report(
    title: str,
    language_mode: str,
    papers: List[PaperItem],
    paper_blocks: List[str],
    overview: Dict[str, str],
) -> str:
    headings = _report_headings(language_mode)
    parts = [
        f"# {title}",
        f"## {headings['summary']}\n\n{overview.get('summary') or _fallback_overview_part('summary', papers, language_mode)}",
        f"## {headings['papers']}\n\n" + "\n\n".join(paper_blocks),
        f"## {headings['conclusion']}\n\n{overview.get('conclusion') or _fallback_overview_part('conclusion', papers, language_mode)}",
        f"## {headings['references']}\n\n{_references_section(papers, language_mode)}",
    ]
    return _sanitize_report_text("\n\n".join(parts), title)


def _run_parallel_report(
    papers: List[PaperItem],
    *,
    llm_synth: Any,
    llm_summary: Any,
    user_query: str,
    title: str,
    language_mode: str,
    logs: List[str],
) -> Dict[str, Any]:
    """Run every section, overview and summary-card call in ONE pool.

    A failed section or overview call degrades to deterministic text built
    from the PMRL notes; the legacy single call is never used as a fallback
    because it would double the latency.
    """

    pool_size = max(1, min(len(papers) * 2 + 1, int(config.LLM_MAX_CONCURRENCY) * 2))
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        section_futures = [
            executor.submit(_write_paper_section, llm_synth, paper, index, user_query, language_mode)
            for index, paper in enumerate(papers, 1)
        ]
        overview_future = executor.submit(
            _write_overview, llm_synth, user_query, title, language_mode, papers
        )
        summary_futures = [
            executor.submit(_summarize_one, paper, "", language_mode, llm_summary) for paper in papers
        ]

        degraded = False
        paper_blocks: List[str] = []
        section_timings: Dict[str, Dict[str, int]] = {}
        for paper, future in zip(papers, section_futures):
            try:
                block, section_timing = future.result()
            except Exception as exc:
                degraded = True
                block = _deterministic_paper_section(paper, language_mode)
                section_timing = {"durationMs": 0, "llmMs": 0}
                logs.append(f"[final_report] Paper section degraded: {paper.paper_id} ({type(exc).__name__})")
            paper_blocks.append(block)
            section_timings[paper.paper_id] = section_timing

        overview: Dict[str, str] = {}
        overview_timing: Dict[str, int] = {"durationMs": 0, "llmMs": 0}
        try:
            overview, overview_timing = overview_future.result()
        except Exception as exc:
            degraded = True
            logs.append(f"[final_report] Overview degraded ({type(exc).__name__})")
        else:
            missing = [key for key in ("summary", "conclusion") if not overview.get(key)]
            if missing:
                degraded = True
                logs.append(f"[final_report] Overview degraded (missing sections: {', '.join(missing)})")

        completed = [future.result() for future in summary_futures]

    summarized_papers, summary_logs, summary_timings, summary_failed = _collect_summary_results(completed)
    report_text = _assemble_parallel_report(
        title, language_mode, papers, paper_blocks, overview
    )
    llm_ms = max(
        [timing.get("llmMs", 0) for timing in section_timings.values()] + [overview_timing.get("llmMs", 0)]
    )
    return {
        "report_text": report_text,
        "summarized_papers": summarized_papers,
        "summary_logs": summary_logs,
        "summary_timings": summary_timings,
        "summary_failed": summary_failed,
        "degraded": degraded,
        "llm_ms": llm_ms,
        "extra_timing": {
            "sectionPapers": section_timings,
            "overviewMs": overview_timing.get("durationMs", 0),
        },
    }


def final_report_node(state: ResearchState) -> Dict[str, Any]:
    """Synthesize and persist the report while preserving canonical links."""

    started = time.perf_counter()
    user_query = state.get("user_query") or "Research Analysis"
    papers = list(state.get("selected_papers", []) or [])
    language_mode = detect_response_language(user_query)
    title = report_title(user_query, papers)
    logs = [f"[final_report] Tổng hợp báo cáo cho '{user_query}'"]
    incomplete_sources = [paper.paper_id for paper in papers if not _source_quality_complete(paper)]
    if incomplete_sources:
        logs.append(f"[final_report] Cảnh báo trích xuất chưa hoàn chỉnh cho {len(incomplete_sources)} bài")

    breakdown = "\n\n".join(_paper_breakdown(paper, index) for index, paper in enumerate(papers, 1))
    prompt = FINAL_REPORT_PROMPT.format(
        user_query=user_query,
        report_title=title,
        language_mode=language_mode,
        detailed_papers_breakdown=breakdown,
    )

    def write_report_file(report_text: str) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        filename_seed = title.removeprefix("Research Report:").strip() if papers else user_query
        safe_title = "".join(c for c in filename_seed[:80] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        safe_title = safe_title or "research_report"
        report_path = config.REPORTS_DIR / f"report_{safe_title}_{timestamp}_{unique_suffix}.md"
        report_path.write_text(report_text, encoding="utf-8")
        return report_path

    llm_ms = 0
    cache_path: Path | None = None
    # Parallel sections need at least one paper; an empty run keeps the legacy
    # single call and its unique per-run artifact.
    use_parallel = bool(config.REPORT_PARALLEL_SECTIONS) and bool(papers)
    try:
        # Report-level cache: identical synthesis inputs reuse the finalized
        # report and cards without any LLM call.  Skipped for empty runs so
        # degenerate states keep unique per-run artifacts.
        if papers:
            _, cache_path = _report_cache_key(
                user_query, title, language_mode, breakdown, papers,
                parallel=use_parallel,
            )
            assert cache_path is not None
            restored = _restore_report_cache(papers, load_json_cache(cache_path))
            if restored is not None:
                report_text, summarized_papers = restored
                report_path = write_report_file(report_text)
                logs.append(f"[final_report] Report cache hit — reused report and {len(summarized_papers)} summary card set(s)")
                logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")
                timing = {
                    "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                    "llmMs": 0,
                    "paperCount": len(papers),
                    "summaryMs": 0,
                    "summaryPapers": {},
                    "cache": True,
                }
                return {
                    "final_report": report_text,
                    "selected_papers": summarized_papers,
                    "status": "degraded" if incomplete_sources else "success",
                    "trace_logs": logs,
                    "node_timings": {"final_report": timing},
                }

        llm_synth = get_llm()
        llm_summary = get_llm()
        summary_started = time.perf_counter()
        summary_logs: List[str] = []
        summary_timings: Dict[str, Dict[str, Any]] = {}
        summary_failed = False
        report_degraded = False
        extra_timing: Dict[str, Any] = {}
        if use_parallel:
            # One pool runs every paper section, the overview and the summary
            # cards at once; the critical path is the slowest single call.
            outcome = _run_parallel_report(
                papers,
                llm_synth=llm_synth,
                llm_summary=llm_summary,
                user_query=user_query,
                title=title,
                language_mode=language_mode,
                logs=logs,
            )
            report_text = outcome["report_text"]
            summarized_papers = outcome["summarized_papers"]
            summary_logs = outcome["summary_logs"]
            summary_timings = outcome["summary_timings"]
            summary_failed = outcome["summary_failed"]
            report_degraded = outcome["degraded"]
            llm_ms = outcome["llm_ms"]
            extra_timing = outcome["extra_timing"]
        else:
            # Synthesis and summary cards run in parallel workers.  The summary
            # step no longer waits for the report text: cards are grounded in the
            # PMRL notes plus the extracted source, which is sufficient evidence
            # and removes ~40s of sequential LLM latency.
            with ThreadPoolExecutor(max_workers=2) as executor:
                report_future = executor.submit(_synthesize_report, llm_synth, prompt, title)
                summary_future = executor.submit(
                    _post_report_summary, papers, "", language_mode, llm_summary
                )
                report_text, llm_ms = report_future.result()
                summarized_papers, summary_logs, summary_timings, summary_failed = summary_future.result()
        logs.extend(summary_logs)

        cacheable_summaries = all(
            paper.notes is not None and paper.notes.summary_quality in {"complete", "partial"}
            for paper in summarized_papers
        )
        # A degraded parallel report is retryable and must not become a cache hit.
        if papers and cacheable_summaries and not report_degraded:
            assert cache_path is not None
            save_json_cache(cache_path, {
                "report_text": report_text,
                "cards": {
                    paper.paper_id: {
                        "cards": paper.notes.summary_cards.model_dump(mode="json") if paper.notes else {},
                        "quality": paper.notes.summary_quality if paper.notes else "missing",
                        "card_statuses": paper.notes.summary_card_statuses if paper.notes else {},
                        "card_reasons": paper.notes.summary_card_reasons if paper.notes else {},
                    }
                    for paper in summarized_papers
                },
            })

        report_path = write_report_file(report_text)

        logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")
        timing = {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
            "paperCount": len(papers),
            "summaryMs": max(0, int((time.perf_counter() - summary_started) * 1000)),
            "summaryPapers": summary_timings,
            **extra_timing,
        }
        return {
            "final_report": report_text,
            "selected_papers": summarized_papers,
            "status": "degraded" if incomplete_sources or summary_failed or report_degraded else "success",
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
