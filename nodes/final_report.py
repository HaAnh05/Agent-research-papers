from __future__ import annotations

import copy
import datetime
import json
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from config import config
from prompts.final_report_prompt import (
    PAPER_SECTION_SYSTEM,
    PAPER_SECTION_USER,
    REPORT_OVERVIEW_SINGLE_SYSTEM,
    REPORT_OVERVIEW_SINGLE_USER,
    REPORT_OVERVIEW_SYSTEM,
    REPORT_OVERVIEW_USER,
)
from state import OverviewMulti, OverviewSingle, PaperItem, ResearchState
from text_utils import detect_response_language
from tools.cache_manager import load_json_cache, prompt_fingerprint, save_json_cache, summary_cache_path
from tools.llm_provider import extract_text_from_response, get_llm, invoke_structured_output, invoke_text, track_llm_call


# Bump when the report prompts or assembly change, so cached reports from an
# older layout are not reused.
REPORT_VERSION = "p13"


_ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$", re.IGNORECASE)
_RAW_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_INTERNAL_REPORT_LINE_RE = re.compile(
    r"^\s*(?:synthesis requested|analysis scope:|language mode:|pmrl quality:|"
    r"delivered in .* per language mode|agent (?:status|self-review))\b",
    re.IGNORECASE,
)


def merge_branch_results(
    papers: List[PaperItem],
    paper_notes: Any,
    paper_enrichment: Any,
) -> List[PaperItem]:
    """Join the parallel branches' per-paper results into copies of ``papers``.

    ``write_notes`` and ``web_enrich`` run in parallel with ``citation_graph``
    and write to separate state keys instead of ``selected_papers``, so no two
    branches overwrite each other.  A paper with no entry keeps its fields.
    """

    notes_map = paper_notes if isinstance(paper_notes, dict) else {}
    enrich_map = paper_enrichment if isinstance(paper_enrichment, dict) else {}
    merged: List[PaperItem] = []
    for paper in papers:
        result = copy.deepcopy(paper)
        result.processing_metadata = dict(result.processing_metadata or {})
        notes_entry = notes_map.get(paper.paper_id)
        if isinstance(notes_entry, dict):
            result.notes = notes_entry.get("notes", result.notes)
            result.notes_quality = notes_entry.get("notes_quality", result.notes_quality)
            result.processing_metadata.update(notes_entry.get("processing_metadata") or {})
        enrich_entry = enrich_map.get(paper.paper_id)
        if isinstance(enrich_entry, dict):
            result.github_repos = list(enrich_entry.get("github_repos") or [])
            result.bibtex = enrich_entry.get("bibtex", result.bibtex)
            result.processing_metadata.update(enrich_entry.get("processing_metadata") or {})
        merged.append(result)
    return merged


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
    context: str = "",
    overview_json: str = "",
) -> tuple[str, Any]:
    """Resolve the report cache entry for the exact synthesis input (fail-open).

    ``overview_json`` is the overview made by the ``summarize`` node: the report
    text embeds it, so it is part of the key.
    """

    key = (
        "report_"
        + prompt_fingerprint(
            config.DEFAULT_PROVIDER,
            config.DEFAULT_MODEL,
            user_query,
            title,
            language_mode,
            breakdown,
            # The research-context section is part of the finished report text.
            context,
            overview_json,
            REPORT_VERSION,
            PAPER_SECTION_SYSTEM, PAPER_SECTION_USER, REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER,
            REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER,
            # Insights, results rows and setup reach the section and overview calls.
            *(_paper_section_input(paper) for paper in papers), _papers_context(papers),
        )
    )
    return key, summary_cache_path(key)


def _restore_report_cache(cached: Any) -> tuple[str, OverviewSingle | OverviewMulti | None] | None:
    """Return ``(report_text, overview)`` of a cache entry, or ``None`` when it is unusable."""

    if not isinstance(cached, dict):
        return None
    report_text = cached.get("report_text")
    if not isinstance(report_text, str) or not report_text.strip():
        return None
    return report_text, overview_from_json(cached.get("overview"))


def overview_from_json(raw: Any) -> OverviewSingle | OverviewMulti | None:
    """Rebuild the overview model from its JSON dump (``None`` when empty or invalid)."""

    if not isinstance(raw, dict) or not raw:
        return None
    schema = OverviewMulti if "insights" in raw else OverviewSingle
    try:
        return schema.model_validate(raw)
    except Exception:
        return None


def overview_cache_key(
    user_query: str,
    title: str,
    language_mode: str,
    papers: List[PaperItem],
) -> tuple[str, Any]:
    """Cache entry of the overview call alone (made by ``summarize``, before the graph is ready)."""

    key = (
        "overview_"
        + prompt_fingerprint(
            config.DEFAULT_PROVIDER,
            config.DEFAULT_MODEL,
            user_query,
            title,
            language_mode,
            REPORT_VERSION,
            REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER,
            REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER,
            _papers_context(papers), _terms_text(papers),
        )
    )
    return key, summary_cache_path(key)


# --- Parallel report synthesis ----------------------------------------------

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
        "intuition": "Ý tưởng cốt lõi",
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
        "intuition": "Core idea",
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
    parts.append(texts["count"].format(c=count) if isinstance(count, int) and count >= 0 else texts["count"].format(c="—"))
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
        *([f"  - **{labels['intuition']}**: {notes.intuition.strip()}"] if notes and notes.intuition.strip() else []),
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
    if notes.intuition.strip():
        lines.append(f"Intuition: {notes.intuition.strip()}")
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
        terms=_terms_text([paper]),
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
) -> tuple[str, Dict[str, Any]]:
    """One LLM call writing one paper block from that paper's breakdown only.

    The timing dict carries ``llmCalls`` (duration and token counts of the call).
    """

    started = time.perf_counter()
    system_text, user_text = build_section_messages(paper, index, user_query, language_mode)
    llm_started = time.perf_counter()
    llm_calls: List[Dict[str, Any]] = []
    with track_llm_call("section", llm_calls, paperId=paper.paper_id):
        response = invoke_text(llm, system_text, user_text)
    llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
    raw = extract_text_from_response(response.content)
    if not raw.strip():
        raise ValueError("Empty paper section response")
    block = _normalize_paper_section(raw, paper, language_mode)
    return block, {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": llm_ms,
        "llmCalls": llm_calls,
    }


def _insights_context(paper: PaperItem) -> str:
    items = [item for item in (paper.notes.insights if paper.notes else []) if item.finding.strip()]
    return " ".join(
        f"({number}) {_clip(item.finding, 300)}" + (f" Why it matters: {_clip(item.why_it_matters, 250)}" if item.why_it_matters.strip() else "")
        for number, item in enumerate(items[:3], 1)
    ) or "none"


_TERM_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z]{2,}[A-Za-z0-9]*)(?:-[A-Za-z0-9]+)*\b")
_TERM_STOPWORDS = {"ARXIV", "PDF", "HTML", "NOT", "STATED", "THE", "AND", "FOR", "OF", "IN", "WITH"}


def _terms_text(papers: List[PaperItem]) -> str:
    """The fixed terms for the writing-style rule: names found in the notes.

    Method name, metric names, benchmark names and component names (CamelCase or
    ALL-CAPS tokens that recur in the notes).  Code builds it; the model must
    use these terms exactly.
    """

    methods: List[str] = []
    metrics: List[str] = []
    benchmarks: List[str] = []
    components: Dict[str, int] = {}
    for paper in papers:
        notes = paper.notes
        if notes is None:
            continue
        if notes.method_name.strip() and notes.method_name.strip() not in methods:
            methods.append(notes.method_name.strip())
        for row in notes.eval_settings:
            if row.metric.strip() and row.metric.strip() not in metrics:
                metrics.append(row.metric.strip())
            if row.benchmark.strip() and row.benchmark.strip() not in benchmarks:
                benchmarks.append(row.benchmark.strip())
        text = " ".join([notes.problem, notes.method, notes.result, notes.limitation, notes.intuition])
        for token in _TERM_TOKEN_RE.findall(text):
            if token.upper() not in _TERM_STOPWORDS:
                components[token] = components.get(token, 0) + 1
    known = {item.casefold() for item in methods + metrics + benchmarks}
    component_names = [
        name for name, count in sorted(components.items(), key=lambda pair: (-pair[1], pair[0]))
        if count >= 2 and name.casefold() not in known
    ][:12]
    lines = []
    for label, values in (
        ("Method", methods), ("Metrics", metrics), ("Benchmarks and datasets", benchmarks), ("Components", component_names),
    ):
        if values:
            lines.append(f"- {label}: " + "; ".join(values[:12]))
    return "\n".join(lines) or "- (none found in the notes)"


def _papers_context(papers: List[PaperItem]) -> str:
    """Compact per-paper context for the overview call (no full breakdown)."""

    blocks: List[str] = []
    for index, paper in enumerate(papers, 1):
        notes = paper.notes
        setup = notes.setup if notes else None
        setup_line = (
            f"datasets {_clip(setup.datasets, 200) or 'not stated'}; training {_clip(setup.training_config, 200) or 'not stated'}; "
            f"hardware {_clip(setup.hardware, 120) or 'NOT STATED IN THE PAPER'}"
            if setup else "not available"
        )
        blocks.append(
            f"[{index}] paperId: {paper.paper_id} | {paper.title} (arXiv: {paper.arxiv_id or 'N/A'})\n"
            f"- Problem excerpt: {_clip(notes.problem if notes else '', 500) or 'N/A'}\n"
            f"- Method excerpt: {_clip(notes.method if notes else '', 600) or 'N/A'}\n"
            f"- Core idea: {_clip(notes.intuition if notes else '', 300) or 'N/A'}\n"
            f"- Key result excerpt: {_clip(notes.result if notes else '', 1200) or 'N/A'}\n"
            f"- Insights: {_insights_context(paper)}\n"
            f"- Setup: {setup_line}\n"
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
    """Return the pure ``(system, user)`` pair for the overview call (JSON output)."""

    system, template = (
        (REPORT_OVERVIEW_SINGLE_SYSTEM, REPORT_OVERVIEW_SINGLE_USER)
        if len(papers) == 1
        else (REPORT_OVERVIEW_SYSTEM, REPORT_OVERVIEW_USER)
    )
    return system, template.format(
        user_query=user_query,
        report_title=title,
        language_mode=language_mode,
        papers_context=_papers_context(papers),
        terms=_terms_text(papers),
    )


def _write_overview(
    llm: Any,
    user_query: str,
    title: str,
    language_mode: str,
    papers: List[PaperItem],
) -> tuple[OverviewSingle | OverviewMulti, Dict[str, Any]]:
    """One structured LLM call returning the overview JSON (single or multi schema).

    The timing dict carries ``llmCalls`` (duration and token counts of the call).
    """

    started = time.perf_counter()
    system_text, user_text = build_overview_messages(papers, user_query, title, language_mode)
    schema = OverviewSingle if len(papers) == 1 else OverviewMulti
    llm_started = time.perf_counter()
    llm_calls: List[Dict[str, Any]] = []
    with track_llm_call("overview", llm_calls):
        overview = invoke_structured_output(
            user_text, schema, llm=llm, provider=config.DEFAULT_PROVIDER, system=system_text,
        )
    llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
    if not overview.tldr.strip():
        raise ValueError("Empty overview response")
    return overview, {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": llm_ms,
        "llmCalls": llm_calls,
    }


_OVERVIEW_TEXT: Dict[str, Dict[str, str]] = {
    "Vietnamese": {"caveats": "Cần lưu ý", "takeaway": "Khuyến nghị", "why": "Ý nghĩa"},
    "English": {"caveats": "Caveats", "takeaway": "Takeaway", "why": "Why it matters"},
}


def _bullet(text: str) -> str:
    return "- " + " ".join(str(text).split())


def render_overview_sections(
    overview: OverviewSingle | OverviewMulti | None,
    papers: List[PaperItem],
    language_mode: str,
) -> Dict[str, str]:
    """Markdown bodies of the "Tóm tắt" and "Kết luận" sections, built from the overview JSON.

    An empty part is omitted so the assembler falls back to deterministic text.
    """

    if overview is None:
        return {}
    texts = _OVERVIEW_TEXT.get(language_mode, _OVERVIEW_TEXT["English"])
    summary: List[str] = []
    conclusion: List[str] = []
    if overview.tldr.strip():
        summary.append(_bullet(overview.tldr))
    if isinstance(overview, OverviewSingle):
        summary += [_bullet(item) for item in overview.keyResults if item.strip()]
        if overview.caveats:
            conclusion.append(f"- **{texts['caveats']}**:")
            conclusion += ["  " + _bullet(item) for item in overview.caveats if item.strip()]
    else:
        for item in overview.insights:
            if item.finding.strip():
                why = f" {' '.join(item.why.split())}" if item.why.strip() else ""
                summary.append(f"- **{' '.join(item.finding.split())}**{why}")
    if overview.takeaway.strip():
        conclusion.append(_bullet(overview.takeaway))
    return {
        key: "\n".join(lines)
        for key, lines in (("summary", summary), ("conclusion", conclusion))
        if lines
    }


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
            detail = _clip(insight or (notes.result if notes else ""), 300) or texts["no_summary"]
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


# --- Chat answer (answerSummary), built by code ------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_number(value: str) -> Decimal | None:
    text = str(value or "").replace("\u2212", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)   # thousands separator
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)            # decimal comma
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _single_number(value: str) -> Decimal | None:
    """The number of a plain value ("46.9", "12%", "46.9 ± 3.1"); ``None`` for compound values such as "1.00/0.87"."""

    text = re.sub(r"\s*(?:±|\+/-|\+-)\s*[\d.,]+\s*%?", "", str(value or ""))
    if len(re.findall(r"\d+(?:[.,]\d+)?", text)) != 1:
        return None
    return _parse_number(text)


def _delta_text(method: str, baseline: str) -> str:
    """Signed difference ``method - baseline`` when both are plain numbers, else empty."""

    left, right = _single_number(method), _single_number(baseline)
    if left is None or right is None:
        return ""
    places = max(-left.as_tuple().exponent, -right.as_tuple().exponent, 0)  # type: ignore[operator]
    diff = left - right
    return f"{diff:+.{places}f}"


def _results_rows(paper: PaperItem) -> tuple[List[Dict[str, Any]], List[str]]:
    """Data rows of the grounded results table (same rows as ``_results_table_lines``)."""

    rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    for row in (paper.notes.eval_settings if paper.notes else []):
        metric = row.metric.strip()
        unit = row.unit.strip()
        if unit and unit.casefold() not in metric.casefold() and unit.casefold() not in row.method_value.casefold():
            metric += f" ({unit})"
        baseline_value = row.baseline_value.strip()
        baseline = f"{baseline_value} {row.baseline_name.strip()}".strip() if baseline_value else ""
        rows.append({
            "setting": " ".join(row.benchmark.split()),
            "metric": metric,
            "method": row.method_value.strip(),
            "baseline": baseline,
            "delta": _delta_text(row.method_value, baseline_value) if baseline_value else "",
            "higherIsBetter": row.higher_is_better,
        })
        label = " ".join(row.table_label.split())
        if label and label not in labels:
            labels.append(label)
    return rows, labels


def _foundation_titles(graph: Any, limit: int = 6) -> List[str]:
    """Titles of the prior works: the similarity graph's list, else foundational citation nodes."""

    if not isinstance(graph, dict):
        return []
    similarity = graph.get("similarity")
    items: List[Any] = []
    if isinstance(similarity, dict) and isinstance(similarity.get("prior_works"), list):
        items = list(similarity["prior_works"])
    if not items and isinstance(graph.get("nodes"), list):
        items = sorted(
            (node for node in graph["nodes"] if isinstance(node, dict) and node.get("role") == "foundational"),
            key=lambda node: (-(node.get("support") or 0), -(node.get("cited_by_count") or 0)),
        )
    titles: List[str] = []
    for item in items:
        title = " ".join(str((item or {}).get("title") or "").split())[:140] if isinstance(item, dict) else ""
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _graph_node_count(citation_graph: Any) -> int:
    if not isinstance(citation_graph, dict):
        return 0
    similarity = citation_graph.get("similarity")
    shown = similarity if isinstance(similarity, dict) and similarity.get("nodes") else citation_graph
    return len(shown.get("nodes") or [])


def patch_answer_summary(
    summary: Dict[str, Any],
    papers: List[PaperItem],
    citation_graph: Any,
    *,
    report_id: str | None = None,
    node_timings: Any = None,
) -> Dict[str, Any]:
    """Return ``summary`` with the data that arrives after the quick summary.

    Code only, idempotent.  Sets the prior works (``foundations`` block), the
    related-papers count, the BibTeX count and, once the report exists, its id.
    The prose of the summary is untouched.  ``summarize`` builds the summary
    before ``citation_graph`` and ``web_enrich`` finish; ``final_report`` and the
    API call this when their data is ready.
    """

    patched = copy.deepcopy(summary)
    blocks = [block for block in (patched.get("blocks") or []) if not (isinstance(block, dict) and block.get("type") == "foundations")]
    foundations = _foundation_titles(citation_graph)
    if foundations:
        # Same position as build_answer_summary: before the takeaway.
        index = next((i for i, block in enumerate(blocks) if isinstance(block, dict) and block.get("type") == "takeaway"), len(blocks))
        blocks.insert(index, {"type": "foundations", "items": foundations})
    patched["blocks"] = blocks
    actions = dict(patched.get("actions") or {})
    actions["graphNodes"] = _graph_node_count(citation_graph)
    actions["bibtexCount"] = sum(1 for paper in papers if paper.bibtex)
    if report_id:
        actions["reportId"] = report_id
    patched["actions"] = actions
    if isinstance(node_timings, dict):
        patched["durationSec"] = _estimate_duration_sec(node_timings)
    return patched


def _published_label(value: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})", str(value or ""))
    return f"{match.group(2)}/{match.group(1)}" if match else str(value or "")


def _estimate_duration_sec(node_timings: Any) -> int:
    """Wall-clock estimate: sequential nodes plus the slowest parallel branch."""

    timings = node_timings if isinstance(node_timings, dict) else {}

    def ms(name: str) -> int:
        entry = timings.get(name)
        value = entry.get("durationMs") if isinstance(entry, dict) else 0
        return int(value) if isinstance(value, (int, float)) and value > 0 else 0

    sequential = sum(ms(name) for name in ("router", "search_papers", "eval_search", "refine_query", "read_paper", "final_report"))
    # Notes and the quick summary form one chain; the graph and enrichment run beside it.
    parallel = max(ms("write_notes") + ms("summarize"), ms("citation_graph"), ms("web_enrich"))
    return int(round((sequential + parallel) / 1000))


def build_answer_summary(
    papers: List[PaperItem],
    overview: OverviewSingle | OverviewMulti | None,
    language_mode: str,
    *,
    citation_graph: Any = None,
    node_timings: Any = None,
    report_id: str | None = None,
    final_ms: int = 0,
) -> Dict[str, Any]:
    """The structured chat answer.  Prose comes from the overview JSON; numbers from grounded notes."""

    graph_nodes = _graph_node_count(citation_graph)
    timings = dict(node_timings) if isinstance(node_timings, dict) else {}
    timings.setdefault("final_report", {"durationMs": final_ms})
    actions = {
        "reportId": report_id,
        "bibtexCount": sum(1 for paper in papers if paper.bibtex),
        "graphNodes": graph_nodes,
    }
    foundations = _foundation_titles(citation_graph)
    blocks: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"kind": "multi" if len(papers) > 1 else "single"}

    if len(papers) == 1:
        paper = papers[0]
        notes = paper.notes
        over = overview if isinstance(overview, OverviewSingle) else None
        summary["paper"] = {
            "paperId": paper.paper_id,
            "arxivId": paper.arxiv_id or "",
            "title": paper.title,
            "published": _published_label(paper.published),
        }
        lead = (over.tldr.strip() if over else "") or next(
            (item.finding.strip() for item in (notes.insights if notes else []) if item.finding.strip()), ""
        ) or _clip(notes.result if notes else "", 240)
        if over and over.problem.strip():
            blocks.append({"type": "problem", "text": over.problem.strip()})
        intuition = (over.intuition.strip() if over else "") or (notes.intuition.strip() if notes else "")
        if intuition:
            blocks.append({"type": "intuition", "text": intuition})
        # No insights block for one paper: PMRL insights are English, and the
        # overview JSON (report language) holds keyResults and intuition instead.
        if over and any(item.strip() for item in over.keyResults):
            blocks.append({"type": "keyResults", "items": [" ".join(item.split()) for item in over.keyResults if item.strip()]})
        metrics = (
            [{"name": m.name.strip(), "meaning": m.meaning.strip(), "higherIsBetter": m.higherIsBetter} for m in over.metrics if m.name.strip()]
            if over else []
        )
        if not metrics and notes:
            seen: set[str] = set()
            for row in notes.eval_settings:
                key = row.metric.strip().casefold()
                if key and key not in seen:
                    seen.add(key)
                    metrics.append({"name": row.metric.strip(), "meaning": row.metric_meaning.strip(), "higherIsBetter": row.higher_is_better})
        if metrics:
            blocks.append({"type": "metrics", "items": metrics})
        hardware = notes.setup.hardware.strip() if notes else ""
        setup_items = [" ".join(item.split()) for item in (over.setup if over else []) if item.strip()]
        if not setup_items and notes:
            setup_items = [
                f"{label}: {value.strip()}"
                for label, value in (
                    ("Datasets", notes.setup.datasets), ("Training", notes.setup.training_config),
                    ("Hardware", notes.setup.hardware), ("Compute", notes.setup.compute),
                ) if value.strip()
            ]
        blocks.append({
            "type": "setup", "items": setup_items, "hardware": hardware,
            # UI shows "<name>: paper không nêu" for each missing item.
            "missing": [] if hardware else ["Hardware"],
        })
        rows, labels = _results_rows(paper)
        if rows:
            blocks.append({
                "type": "results", "grounded": True, "source": labels, "rows": rows,
                "methodName": _method_column_name(paper, language_mode),
            })
        caveats = [" ".join(item.split()) for item in (over.caveats if over else []) if item.strip()]
        if caveats:
            blocks.append({"type": "caveats", "items": caveats})
        if foundations:
            blocks.append({"type": "foundations", "items": foundations})
        if over and over.takeaway.strip():
            blocks.append({"type": "takeaway", "text": over.takeaway.strip()})
    else:
        over_multi = overview if isinstance(overview, OverviewMulti) else None
        summary["papers"] = [
            {"paperId": p.paper_id, "arxivId": p.arxiv_id or "", "title": p.title, "published": _published_label(p.published)}
            for p in papers
        ]
        lead = (over_multi.tldr.strip() if over_multi else "") or " ".join(
            f"{p.title}." for p in papers[:3]
        )
        findings = [
            {"finding": " ".join(item.finding.split()), "why": " ".join(item.why.split())}
            for item in (over_multi.insights if over_multi else []) if item.finding.strip()
        ]
        if findings:
            blocks.append({"type": "insights", "items": findings})
        by_id = {entry.paperId.strip(): entry for entry in (over_multi.papers if over_multi else [])}
        cards = []
        for index, paper in enumerate(papers):
            entry = by_id.get(paper.paper_id)
            if entry is None and over_multi and index < len(over_multi.papers) and not over_multi.papers[index].paperId.strip():
                entry = over_multi.papers[index]
            notes = paper.notes
            first_insight = next((i.finding.strip() for i in (notes.insights if notes else []) if i.finding.strip()), "")
            cards.append({
                "paperId": paper.paper_id,
                "title": paper.title,
                "year": (paper.published or "")[:4],
                "intuition": (entry.intuition.strip() if entry else "") or (notes.intuition.strip() if notes else ""),
                "keyResult": (entry.keyResult.strip() if entry else "") or first_insight or _clip(notes.result if notes else "", 200),
            })
        blocks.append({"type": "paperCards", "items": cards})
        if foundations:
            blocks.append({"type": "foundations", "items": foundations})
        if over_multi and over_multi.takeaway.strip():
            blocks.append({"type": "takeaway", "text": over_multi.takeaway.strip()})

    summary.update({
        "lead": lead,
        "tldr": lead,
        "blocks": blocks,
        "actions": actions,
        "durationSec": _estimate_duration_sec(timings),
    })
    return summary


def _run_parallel_report(
    papers: List[PaperItem],
    *,
    llm: Any,
    user_query: str,
    title: str,
    language_mode: str,
    logs: List[str],
    context: str = "",
    overview_given: bool = False,
    overview_data: OverviewSingle | OverviewMulti | None = None,
) -> Dict[str, Any]:
    """Run every paper-section call in ONE pool, plus the overview call when it is not given.

    ``overview_given`` is true when the ``summarize`` node already made the
    overview (``overview_data`` is then that overview, or ``None`` when its call
    failed): no overview call runs here.  A failed section or overview degrades
    to deterministic text built from the PMRL notes.  With no papers, no LLM
    call runs.
    """

    pool_size = max(1, min(len(papers) + (0 if overview_given else 1), int(config.LLM_MAX_CONCURRENCY) * 2))
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        section_futures = [
            executor.submit(_write_paper_section, llm, paper, index, user_query, language_mode)
            for index, paper in enumerate(papers, 1)
        ]
        overview_future = (
            executor.submit(_write_overview, llm, user_query, title, language_mode, papers)
            if papers and not overview_given
            else None
        )

        degraded = False
        paper_blocks: List[str] = []
        section_timings: Dict[str, Dict[str, int]] = {}
        llm_calls: List[Dict[str, Any]] = []
        for paper, future in zip(papers, section_futures):
            try:
                block, section_timing = future.result()
            except Exception as exc:
                degraded = True
                block = _deterministic_paper_section(paper, language_mode)
                section_timing = {"durationMs": 0, "llmMs": 0}
                logs.append(f"[final_report] Paper section degraded: {paper.paper_id} ({type(exc).__name__})")
            llm_calls += section_timing.pop("llmCalls", [])
            paper_blocks.append(block)
            section_timings[paper.paper_id] = section_timing

        overview_timing: Dict[str, Any] = {"durationMs": 0, "llmMs": 0}
        if overview_future is not None:
            try:
                overview_data, overview_timing = overview_future.result()
            except Exception as exc:
                degraded = True
                logs.append(f"[final_report] Overview degraded ({type(exc).__name__})")
            else:
                llm_calls += overview_timing.pop("llmCalls", [])
        if papers and (overview_future is not None or overview_given):
            if overview_data is None:
                degraded = True
                if overview_given:
                    logs.append("[final_report] Overview degraded (summarize had no overview)")
            else:
                missing = [
                    key for key in ("summary", "conclusion")
                    if not render_overview_sections(overview_data, papers, language_mode).get(key)
                ]
                if missing:
                    degraded = True
                    logs.append(f"[final_report] Overview degraded (missing sections: {', '.join(missing)})")

    overview = render_overview_sections(overview_data, papers, language_mode)
    report_text = _assemble_parallel_report(
        title, language_mode, papers, paper_blocks, overview, context
    )
    llm_ms = max(
        [timing.get("llmMs", 0) for timing in section_timings.values()] + [overview_timing.get("llmMs", 0)]
    )
    return {
        "report_text": report_text,
        "overview": overview_data,
        "degraded": degraded,
        "llm_ms": llm_ms,
        "llm_calls": llm_calls,
        "extra_timing": {
            "sectionPapers": section_timings,
            "overviewMs": overview_timing.get("durationMs", 0),
        },
    }


def final_report_node(state: ResearchState) -> Dict[str, Any]:
    """Write the paper sections, assemble ``report.md`` and finish the chat answer.

    ``summarize`` already made the overview and the early ``answer_summary``
    (``state["overview"]``; ``{}`` when its call failed).  This node reuses that
    overview, so it makes one LLM call per paper only, and patches the summary
    with the data that came later (prior works, graph size, BibTeX count, report
    id).  A state without ``overview`` (a direct call, no ``summarize``) makes the
    overview call here, as before.
    """

    started = time.perf_counter()
    user_query = state.get("user_query") or "Research Analysis"
    papers = merge_branch_results(
        list(state.get("selected_papers", []) or []),
        state.get("paper_notes"),
        state.get("paper_enrichment"),
    )
    language_mode = detect_response_language(user_query)
    title = report_title(user_query, papers)
    logs = [f"[final_report] Tổng hợp báo cáo cho '{user_query}'"]
    incomplete_sources = [paper.paper_id for paper in papers if not _source_quality_complete(paper)]
    if incomplete_sources:
        logs.append(f"[final_report] Cảnh báo trích xuất chưa hoàn chỉnh cho {len(incomplete_sources)} bài")

    context = _context_section(state.get("citation_graph"), language_mode)
    breakdown = "\n\n".join(_paper_breakdown(paper, index) for index, paper in enumerate(papers, 1))
    raw_overview = state.get("overview")
    overview_given = isinstance(raw_overview, dict)
    given_overview = overview_from_json(raw_overview) if overview_given else None

    def finish_summary(overview: OverviewSingle | OverviewMulti | None, report_id: str, final_ms: int) -> Dict[str, Any] | None:
        """The early summary patched with the late data; built from the overview when missing."""

        if not papers:
            return None
        timings = dict(state.get("node_timings") or {})
        timings["final_report"] = {"durationMs": final_ms}
        early = state.get("answer_summary")
        if not isinstance(early, dict) or not early:
            early = build_answer_summary(
                papers, overview, language_mode,
                citation_graph=state.get("citation_graph"), node_timings=timings,
                report_id=report_id, final_ms=final_ms,
            )
        return patch_answer_summary(
            early, papers, state.get("citation_graph"), report_id=report_id, node_timings=timings,
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
    try:
        # Report-level cache: identical synthesis inputs reuse the finalized
        # report without any LLM call.  Skipped for empty runs so degenerate
        # states keep unique per-run artifacts.
        if papers:
            _, cache_path = _report_cache_key(
                user_query, title, language_mode, breakdown, papers,
                context=context,
                overview_json=json.dumps(raw_overview, sort_keys=True, ensure_ascii=False) if overview_given else "",
            )
            assert cache_path is not None
            restored = _restore_report_cache(load_json_cache(cache_path))
            if restored is not None:
                cached_report, cached_overview = restored
                report_path = write_report_file(cached_report)
                logs.append("[final_report] Report cache hit — reused report")
                logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")
                timing = {
                    "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
                    "llmMs": 0,
                    "paperCount": len(papers),
                    "cache": True,
                    "llmCalls": [],
                }
                return {
                    "final_report": cached_report,
                    "report_id": report_path.name,
                    "answer_summary": finish_summary(
                        given_overview if overview_given else cached_overview, report_path.name, timing["durationMs"],
                    ),
                    "selected_papers": papers,
                    "status": "degraded" if incomplete_sources else "success",
                    "trace_logs": logs,
                    "node_timings": {"final_report": timing},
                }

        # One pool runs every paper section (and the overview, when ``summarize``
        # did not make it) at once; the critical path is the slowest single call.
        outcome = _run_parallel_report(
            papers,
            llm=get_llm(),
            user_query=user_query,
            title=title,
            language_mode=language_mode,
            logs=logs,
            context=context,
            overview_given=overview_given,
            overview_data=given_overview,
        )
        report_text = outcome["report_text"]
        report_degraded = outcome["degraded"]
        llm_ms = outcome["llm_ms"]

        # A degraded report is retryable and must not become a cache hit.
        if papers and not report_degraded:
            assert cache_path is not None
            overview_data = outcome["overview"]
            save_json_cache(cache_path, {
                "report_text": report_text,
                "overview": overview_data.model_dump(mode="json") if overview_data is not None else None,
            })

        report_path = write_report_file(report_text)

        logs.append(f"[final_report] Báo cáo lưu: {report_path.name}")
        timing = {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": llm_ms,
            "paperCount": len(papers),
            "llmCalls": outcome["llm_calls"],
            **outcome["extra_timing"],
        }
        return {
            "final_report": report_text,
            "report_id": report_path.name,
            "answer_summary": finish_summary(outcome["overview"], report_path.name, timing["durationMs"]),
            "selected_papers": papers,
            "status": "degraded" if incomplete_sources or report_degraded else "success",
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
