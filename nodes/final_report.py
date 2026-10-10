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


SUMMARY_PIPELINE_VERSION = "v5"
# Bump when the parallel report prompts or assembly change, so cached parallel
# reports from an older layout are not reused.  The legacy key is unaffected.
PARALLEL_REPORT_VERSION = "p10"


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
    context: str = "",
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
            # The research-context section is part of the finished report text.
            context,
            *(
                (
                    "parallel_sections", PARALLEL_REPORT_VERSION,
                    PAPER_SECTION_SYSTEM, PAPER_SECTION_USER, REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER,
                    REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER,
                    # Insights, results rows and setup reach the section and overview calls.
                    *(_paper_section_input(paper) for paper in papers), _papers_context(papers),
                )
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
        "context": "3. Bối cảnh nghiên cứu",
        "conclusion": "4. Kết luận",
        "references": "5. Tài liệu tham khảo",
    },
    "English": {
        "summary": "1. Summary",
        "papers": "2. Paper Analysis",
        "context": "3. Research Context",
        "conclusion": "4. Conclusion",
        "references": "5. References",
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


# --- Research context section (deterministic; built from state["citation_graph"]) ---

_CONTEXT_TEXT: Dict[str, Dict[str, str]] = {
    "Vietnamese": {
        "foundational": "Bài nền tảng",
        "followup": "Bài kế thừa",
        "cited_many": "được {n}/{total} bài đã chọn trích dẫn",
        "cited_one": "được bài đã chọn trích dẫn",
        "cites_many": "trích dẫn {n}/{total} bài đã chọn",
        "cites_one": "trích dẫn bài đã chọn",
        "count": "{c} lượt trích dẫn",
        "none_many": "Chưa tìm thấy bài nào được ít nhất 2 bài đã chọn cùng trích dẫn.",
        "none_one": "Chưa xác định được tài liệu tham khảo nào có mã arXiv.",
        "none_followup": "Chưa tìm thấy bài kế thừa.",
        "no_key": "Cần OpenAlex API key (OPENALEX_API_KEY) để lấy các bài kế thừa.",
        "error": "OpenAlex không phản hồi trong lần chạy này nên chưa có bài kế thừa.",
        "resolved": "Đã xác định mã arXiv cho {r}/{t} mục tham khảo.",
        "no_graph": "Chưa có dữ liệu trích dẫn trong lần chạy này.",
    },
    "English": {
        "foundational": "Foundational papers",
        "followup": "Follow-up papers",
        "cited_many": "cited by {n}/{total} selected papers",
        "cited_one": "cited by the selected paper",
        "cites_many": "cites {n}/{total} selected papers",
        "cites_one": "cites the selected paper",
        "count": "{c} citations",
        "none_many": "No paper is cited by at least 2 of the selected papers.",
        "none_one": "No reference with an arXiv id was identified.",
        "none_followup": "No follow-up paper was found.",
        "no_key": "An OpenAlex API key (OPENALEX_API_KEY) is needed to list follow-up papers.",
        "error": "OpenAlex did not respond in this run, so there are no follow-up papers.",
        "resolved": "Resolved an arXiv id for {r}/{t} references.",
        "no_graph": "No citation data is available for this run.",
    },
}
_ARXIV_ID_SAFE_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})$", re.IGNORECASE)


def _context_item(node: Dict[str, Any], reason: str, texts: Dict[str, str]) -> str:
    title = " ".join(str(node.get("title") or node.get("id") or "").split())[:200]
    title = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    arxiv_id = str(node.get("arxiv_id") or "")
    head = f"[{title}](https://arxiv.org/abs/{arxiv_id})" if _ARXIV_ID_SAFE_RE.match(arxiv_id) else title
    year = node.get("year")
    if isinstance(year, int) and year > 0:
        head += f" ({year})"
    parts = [head]
    count = node.get("cited_by_count")
    if isinstance(count, int) and count >= 0:
        parts.append(texts["count"].format(c=count))
    parts.append(reason)
    return "- " + " — ".join(parts)


def _context_section(graph: Any, language_mode: str) -> str:
    """Body of the research-context section; plain text when no graph exists."""

    texts = _CONTEXT_TEXT.get(language_mode, _CONTEXT_TEXT["English"])
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return texts["no_graph"]
    nodes = [node for node in graph["nodes"] if isinstance(node, dict)]
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else {}
    total = sum(1 for node in nodes if node.get("selected")) or 1

    def reasons(role: str) -> List[str]:
        group = [node for node in nodes if node.get("role") == role]
        lines: List[str] = []
        for node in group:
            support = node.get("support") if isinstance(node.get("support"), int) else 0
            key = ("cited" if role == "foundational" else "cites") + ("_many" if total > 1 and support >= 1 else "_one")
            lines.append(_context_item(node, texts[key].format(n=support, total=total), texts))
        return lines

    foundational = reasons("foundational")
    followup = reasons("followup")
    openalex = stats.get("openalex")
    out = [f"**{texts['foundational']}**", ""]
    out.extend(foundational or [f"- {texts['none_many'] if total > 1 else texts['none_one']}"])
    out.extend(["", f"**{texts['followup']}**", ""])
    if followup:
        out.extend(followup)
    elif openalex == "no_key":
        out.append(f"- {texts['no_key']}")
    elif openalex == "error":
        out.append(f"- {texts['error']}")
    else:
        out.append(f"- {texts['none_followup']}")
    resolved, refs_total = stats.get("refs_resolved"), stats.get("refs_total")
    if isinstance(resolved, int) and isinstance(refs_total, int) and refs_total > 0:
        out.extend(["", f"_{texts['resolved'].format(r=resolved, t=refs_total)}_"])
    return "\n".join(out)


def _insert_context_section(report_text: str, context: str, language_mode: str) -> str:
    """Legacy single-call path: insert the section before the model's ``## 4.`` heading."""

    heading = f"## {_report_headings(language_mode)['context']}\n\n{context}\n\n"
    match = re.search(r"(?m)^##\s+4\s*[.)]", report_text)
    if match:
        return report_text[: match.start()] + heading + report_text[match.start():]
    return report_text.rstrip("\n") + "\n\n" + heading.rstrip("\n") + "\n"


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


# Sub-bullets of "Key results" and the text of the code-built results table.
_RESULT_SUB_LABELS: Dict[str, Dict[str, str]] = {
    "Vietnamese": {"insight": "Insight", "metrics": "Đại lượng đo", "setup": "Thiết lập", "remarks": "Nhận xét"},
    "English": {"insight": "Insight", "metrics": "Metrics", "setup": "Setup", "remarks": "Remarks"},
}
_RESULT_TEXT: Dict[str, Dict[str, str]] = {
    "Vietnamese": {
        "benchmark": "Benchmark",
        "metric": "Metric",
        "baseline": "Baseline tốt nhất",
        "source": "Nguồn: {labels} trong paper",
        "no_hardware": "Hardware: paper không nêu",
        "own_method": "Phương pháp của bài",
        "higher": "càng cao càng tốt",
        "lower": "càng thấp càng tốt",
        "datasets": "Datasets",
        "training": "Cấu hình huấn luyện",
        "compute": "Compute",
    },
    "English": {
        "benchmark": "Benchmark",
        "metric": "Metric",
        "baseline": "Best baseline",
        "source": "Source: {labels} in the paper",
        "no_hardware": "Hardware: not stated in the paper",
        "own_method": "This paper",
        "higher": "higher is better",
        "lower": "lower is better",
        "datasets": "Datasets",
        "training": "Training",
        "compute": "Compute",
    },
}
_RESULT_SUB_KEYS = ("insight", "metrics", "setup", "remarks")
_HARDWARE_WORD_RE = re.compile(r"hardware|phần cứng", re.IGNORECASE)


def _result_texts(language_mode: str) -> Dict[str, str]:
    return _RESULT_TEXT.get(language_mode, _RESULT_TEXT["English"])


def _method_column_name(paper: PaperItem, language_mode: str) -> str:
    """Short method name for the table header: notes name, else a short title prefix, else a generic label."""

    notes = paper.notes
    name = " ".join((notes.method_name if notes else "").split())
    if name:
        return name[:40]
    head = (paper.title or "").split(":", 1)[0].strip()
    if ":" in (paper.title or "") and 1 <= len(head.split()) <= 4:
        return head[:40]
    return _result_texts(language_mode)["own_method"]


def _cell(value: Any) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|") or "—"


def _results_table_lines(paper: PaperItem, language_mode: str) -> List[str]:
    """Markdown table built from the grounded ``eval_settings`` (no LLM text).

    Empty list when there are no grounded rows.  Lines are indented by two
    spaces so the table stays inside the "Key results" list item.
    """

    rows = list(paper.notes.eval_settings) if paper.notes else []
    if not rows:
        return []
    texts = _result_texts(language_mode)
    lines = [
        f"| {texts['benchmark']} | {texts['metric']} | {_cell(_method_column_name(paper, language_mode))} | {texts['baseline']} |",
        "|---|---|---|---|",
    ]
    labels: List[str] = []
    for row in rows:
        metric = row.metric.strip()
        unit = row.unit.strip()
        if unit and unit.casefold() not in metric.casefold() and unit.casefold() not in row.method_value.casefold():
            metric += f" ({unit})"
        if row.higher_is_better is True:
            metric += " ↑"
        elif row.higher_is_better is False:
            metric += " ↓"
        baseline = row.baseline_value.strip()
        if baseline and row.baseline_name.strip():
            baseline += f" ({row.baseline_name.strip()})"
        lines.append(f"| {_cell(row.benchmark)} | {_cell(metric)} | {_cell(row.method_value)} | {_cell(baseline)} |")
        label = " ".join(row.table_label.split())
        if label and label not in labels:
            labels.append(label)
    out = ["  " + line for line in lines]
    if labels:
        out += ["", "  *" + texts["source"].format(labels=", ".join(labels)) + "*"]
    return out


def _sub_label_re(key: str) -> re.Pattern[str]:
    names = {
        "insight": ("insight", "insights"),
        "metrics": ("đại lượng đo", "metrics", "metric"),
        "setup": ("thiết lập", "setup"),
        "remarks": ("nhận xét", "remarks", "remark", "caveats"),
    }[key]
    return re.compile(r"^\s*[-*•]\s*\**\s*(?:" + "|".join(re.escape(n) for n in names) + r")\s*\**\s*:", re.IGNORECASE)


_MAIN_LABEL_KEYS = ("problem", "method", "result", "limitation")


def _main_label_re(key: str) -> re.Pattern[str]:
    names = {_SECTION_LABELS[mode][key].casefold() for mode in _SECTION_LABELS} | {
        "problem": {"problem"}, "method": {"methodology", "method"}, "result": {"key results", "results"}, "limitation": {"limitations", "limitation"},
    }[key]
    return re.compile(r"^[-*•]\s*\**\s*(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\s*\**\s*:", re.IGNORECASE)


def _find_line(lines: List[str], pattern: re.Pattern[str], start: int = 0, end: int | None = None) -> int:
    for index in range(start, len(lines) if end is None else end):
        if pattern.match(lines[index]):
            return index
    return -1


def _inject_results(lines: List[str], paper: PaperItem, language_mode: str) -> List[str]:
    """Add the code-built parts of "Key results" to an LLM-written (or fallback) block.

    * If ``setup.hardware`` is empty and the Setup sub-bullet has no hardware
      statement, append the localized "Hardware: not stated" line.
    * Insert the results table before the Remarks sub-bullet, or at the end of
      the Key results block when that sub-bullet is missing.
    """

    notes = paper.notes
    texts = _result_texts(language_mode)
    labels = _SECTION_LABELS.get(language_mode, _SECTION_LABELS["English"])
    lines = list(lines)
    table = _results_table_lines(paper, language_mode)
    need_hardware = notes is not None and not notes.setup.hardware.strip()
    if not table and not need_hardware:
        return lines

    main_res = [_main_label_re(key) for key in _MAIN_LABEL_KEYS]
    start = _find_line(lines, main_res[2])
    if start < 0:
        # No Key results bullet: create an empty one before Limitations (or at the end).
        limitation = _find_line(lines, main_res[3])
        start = limitation if limitation >= 0 else len(lines)
        lines.insert(start, f"- **{labels['result']}**:")

    def block_end() -> int:
        for index in range(start + 1, len(lines)):
            if any(pattern.match(lines[index]) for pattern in main_res) or re.match(r"^\s{0,3}#", lines[index]):
                return index
        return len(lines)

    def trim(end: int) -> int:
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        return end

    remarks_re = _sub_label_re("remarks")
    if need_hardware:
        setup_at = _find_line(lines, _sub_label_re("setup"), start + 1, block_end())
        if setup_at < 0:
            remarks = _find_line(lines, remarks_re, start + 1, block_end())
            at = remarks if remarks >= 0 else trim(block_end())
            sub = _RESULT_SUB_LABELS.get(language_mode, _RESULT_SUB_LABELS["English"])
            lines.insert(at, f"  - **{sub['setup']}**: {texts['no_hardware']}")
        else:
            next_sub = next(
                (i for i in range(setup_at + 1, block_end()) if re.match(r"^\s*[-*•]\s*\*\*[^*\n]+\*\*\s*:", lines[i])),
                block_end(),
            )
            last = next_sub - 1
            while last > setup_at and not lines[last].strip():
                last -= 1
            if not _HARDWARE_WORD_RE.search("\n".join(lines[setup_at:last + 1])):
                if last == setup_at and not lines[setup_at].rstrip().endswith(":"):
                    lines[setup_at] = lines[setup_at].rstrip().rstrip(".;") + f"; {texts['no_hardware']}"
                else:
                    lines.insert(last + 1, f"    - {texts['no_hardware']}")

    if table:
        remarks = _find_line(lines, remarks_re, start + 1, block_end())
        at = remarks if remarks >= 0 else trim(block_end())
        before = [""] if at > 0 and lines[at - 1].strip() else []
        after = [""] if at < len(lines) and lines[at].strip() else []
        lines[at:at] = before + table + after
    return lines


def _results_sub_lines(paper: PaperItem, language_mode: str) -> List[str]:
    """Notes-only "Key results" sub-bullets for the deterministic fallback."""

    notes = paper.notes
    sub = _RESULT_SUB_LABELS.get(language_mode, _RESULT_SUB_LABELS["English"])
    texts = _result_texts(language_mode)
    lines: List[str] = []
    insights = [item for item in (notes.insights if notes else []) if item.finding.strip()]
    if insights:
        lines.append(f"  - **{sub['insight']}**:")
        for number, item in enumerate(insights[:3], 1):
            why = f" {item.why_it_matters.strip()}" if item.why_it_matters.strip() else ""
            lines.append(f"    {number}. {item.finding.strip()}{why}")
    else:
        lines.append(f"  - **{sub['insight']}**: N/A")
    metric_lines: List[str] = []
    seen: set[str] = set()
    for row in (notes.eval_settings if notes else []):
        if row.metric.casefold() in seen:
            continue
        seen.add(row.metric.casefold())
        detail = "; ".join(part for part in (
            row.metric_meaning.strip(),
            row.unit.strip(),
            texts["higher"] if row.higher_is_better is True else texts["lower"] if row.higher_is_better is False else "",
        ) if part)
        metric_lines.append(f"    - {row.metric.strip()}" + (f": {detail}" if detail else ""))
    lines.append(f"  - **{sub['metrics']}**:" if metric_lines else f"  - **{sub['metrics']}**: N/A")
    lines += metric_lines
    setup = notes.setup if notes else None
    parts = [
        f"{texts['datasets']}: {setup.datasets.strip()}" if setup and setup.datasets.strip() else "",
        f"{texts['training']}: {setup.training_config.strip()}" if setup and setup.training_config.strip() else "",
        f"Hardware: {setup.hardware.strip()}" if setup and setup.hardware.strip() else "",
        f"{texts['compute']}: {setup.compute.strip()}" if setup and setup.compute.strip() else "",
    ]
    lines.append(f"  - **{sub['setup']}**: " + ("; ".join(part for part in parts if part) or "N/A"))
    return lines


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
        *_results_sub_lines(paper, language_mode),
        f"- **{labels['limitation']}**: {notes.limitation if notes else 'N/A'}",
    ]
    return "\n".join(_inject_results(lines, paper, language_mode) + _paper_tail(paper, language_mode))


def _structured_notes_text(paper: PaperItem) -> str:
    """Insights, metric rows and setup of the PMRL notes, as plain text for the section call."""

    notes = paper.notes
    if notes is None:
        return ""
    lines: List[str] = []
    if notes.method_name.strip():
        lines.append(f"Method name: {notes.method_name.strip()}")
    lines.append("Insights:")
    insights = [item for item in notes.insights if item.finding.strip()]
    lines += [
        f"{number}. {item.finding.strip()}" + (f" (why it matters: {item.why_it_matters.strip()})" if item.why_it_matters.strip() else "")
        for number, item in enumerate(insights, 1)
    ] or ["none"]
    lines.append("Metrics and verified result rows (code adds the results table itself; do not write a table):")
    for row in notes.eval_settings:
        direction = {True: "higher is better", False: "lower is better"}.get(row.higher_is_better, "direction unclear")
        baseline = f"; best baseline {row.baseline_name.strip() or '?'} = {row.baseline_value.strip()}" if row.baseline_value.strip() else ""
        lines.append(
            f"- {row.benchmark.strip()} | {row.metric.strip()}: {row.metric_meaning.strip() or 'meaning not given'}; "
            f"unit {row.unit.strip() or 'none'}; {direction}; method = {row.method_value.strip()}{baseline}"
            + (f"; {row.table_label.strip()}" if row.table_label.strip() else "")
        )
    if not notes.eval_settings:
        lines.append("none")
    setup = notes.setup
    lines += [
        "Setup:",
        f"- Datasets: {setup.datasets.strip() or 'not stated'}",
        f"- Training: {setup.training_config.strip() or 'not stated'}",
        f"- Hardware: {setup.hardware.strip() or 'NOT STATED IN THE PAPER'}",
        f"- Compute: {setup.compute.strip() or 'not stated'}",
    ]
    return "\n".join(lines)


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
    structured = _structured_notes_text(paper)
    return (
        f"Title: {paper.title}\n"
        f"ArXiv ID: {paper.arxiv_id or 'N/A'} | Published: {paper.published or 'N/A'}\n"
        f"Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}\n"
        f"Problem: {notes.problem if notes else 'N/A'}\n"
        f"Methodology: {notes.method if notes else 'N/A'}\n"
        f"Results & Metrics: {notes.result if notes else 'N/A'}\n"
        f"Limitations: {notes.limitation if notes else 'N/A'}\n"
        f"{structured + chr(10) if structured else ''}"
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


def _analysis_labels() -> set[str]:
    return {
        _SECTION_LABELS[mode][key].casefold()
        for mode in _SECTION_LABELS
        for key in ("problem", "method", "result", "limitation")
    }


def _normalize_paper_section(text: str, paper: PaperItem, language_mode: str) -> str:
    """Keep one ``###`` heading and the analysis bullets; add the results table and hardware line; append the tail.

    Any LLM line carrying a source-code, extraction, link or citation label is
    removed, so the deterministic tail appears exactly once.  A top-level bullet
    whose label is not an analysis label (Problem, Methodology, Key results,
    Limitations, or the Vietnamese equivalents) is removed with its sub-lines:
    the user's broader question is answered only in the overview.
    """

    cleaned = _BIBTEX_FENCE_RE.sub("", str(text or "").replace("\r\n", "\n"))
    tail_labels = _tail_labels()
    known_labels = _analysis_labels()
    kept: List[str] = []
    skipping = False
    for line in cleaned.split("\n"):
        label = re.match(r"^\s*[-*•]\s*\**([^*:\n]{1,40}?)\**\s*:", line)
        if label:
            name = label.group(1).strip().casefold()
            top_level = re.match(r"^ {0,1}[-*•]", line) is not None
            if name in tail_labels or (top_level and name not in known_labels):
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
    body = "\n".join(_inject_results(kept, paper, language_mode)).strip()
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


def _insights_context(paper: PaperItem) -> str:
    items = [item for item in (paper.notes.insights if paper.notes else []) if item.finding.strip()]
    return " ".join(
        f"({number}) {_clip(item.finding, 300)}" + (f" Why it matters: {_clip(item.why_it_matters, 250)}" if item.why_it_matters.strip() else "")
        for number, item in enumerate(items[:3], 1)
    ) or "none"


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
            f"- Insights: {_insights_context(paper)}\n"
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


# The prompt asks for "4. <conclusion>" (section 3 is the code-built research
# context); "3" stays accepted for outputs that follow the older numbering.
_OVERVIEW_KEYS = {"1": "summary", "3": "conclusion", "4": "conclusion"}


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
            insight = next((item.finding for item in (notes.insights if notes else []) if item.finding.strip()), "")
            first = insight or ((notes.brief_summary if notes else "") or "").strip().split("\n")[0]
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
    context: str = "",
) -> str:
    headings = _report_headings(language_mode)
    parts = [
        f"# {title}",
        f"## {headings['summary']}\n\n{overview.get('summary') or _fallback_overview_part('summary', papers, language_mode)}",
        f"## {headings['papers']}\n\n" + "\n\n".join(paper_blocks),
        f"## {headings['context']}\n\n{context or _context_section(None, language_mode)}",
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
    context: str = "",
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
        title, language_mode, papers, paper_blocks, overview, context
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

    context = _context_section(state.get("citation_graph"), language_mode)
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
                context=context,
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
                context=context,
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
                report_text = _insert_context_section(report_text, context, language_mode)
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
