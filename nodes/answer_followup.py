"""Follow-up answers from a thread's existing report (one streamed LLM call, no stages).

Input is ``state["followup_context"]``, built by the API from the thread's latest
finished run: report id and Markdown, per-paper structured notes and tables, and
the related-papers map.  The node picks only the parts the question needs, so
the prompt stays within ``CONTEXT_BUDGET`` characters.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from prompts.followup_prompt import FOLLOWUP_SYSTEM, FOLLOWUP_USER
from state import ResearchState
from tools.llm_provider import extract_text_from_response, get_llm, invoke_text, track_llm_call

# Character budgets of the prompt parts.  Worst case: 3 report sections x 2.5k
# + 2 paper blocks x 5k + 2 notes x 3.5k + 4 tables x 2k + 3 related x 0.9k
# = about 35k characters (under 9k tokens).
SUMMARY_SECTION_CHARS = 2_500
PAPER_BLOCK_CHARS = 5_000
NOTES_CHARS = 3_500
TABLE_CHARS = 2_000
MAX_TABLES = 4
RELATED_CHARS = 900
MAX_RELATED = 3
MAX_SELECTED_PAPERS = 2

_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_TABLE_REF_RE = re.compile(r"(?:table|bảng)\s*(\d+[a-z]?)", re.IGNORECASE)
_ORDINAL_RE = re.compile(r"(?:bài|paper|báo|article)\s*(?:số|thứ|no\.?|#)?\s*(\d{1,2})", re.IGNORECASE)
_STOPWORDS = {"the", "and", "for", "with", "what", "why", "how", "does", "this", "that", "are", "bài", "báo", "của", "và", "các", "cho", "với", "trong", "là", "sao", "vì", "thế", "nào", "paper", "papers"}


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall((text or "").casefold()) if word not in _STOPWORDS}


def report_language(report_text: str) -> str:
    return "Vietnamese" if "Tóm tắt" in (report_text or "") else "English"


def split_report(report_text: str, language_mode: str) -> List[Tuple[str, str, str]]:
    """Report units as ``(source_name, title, text)``.

    ``## n.`` sections become ``Báo cáo §n``; the paper-analysis section is
    split further at ``###`` into ``Báo cáo §2: <paper title>``.
    """

    prefix = "Báo cáo" if language_mode == "Vietnamese" else "Report"
    units: List[Tuple[str, str, str]] = []
    sections = re.split(r"(?m)^(?=##\s+\d+\s*[.)])", report_text or "")
    for chunk in sections:
        head = re.match(r"##\s+(\d+)\s*[.)]\s*([^\n]*)", chunk)
        if not head:
            continue
        number, title = head.group(1), head.group(2).strip()
        if number == "2":
            blocks = re.split(r"(?m)^(?=###\s)", chunk)
            for block in blocks[1:]:
                paper_title = re.match(r"###\s*([^\n]*)", block).group(1).strip()  # type: ignore[union-attr]
                units.append((f"{prefix} §2: {paper_title}", paper_title, block.strip()))
        else:
            units.append((f"{prefix} §{number}", title, chunk.strip()))
    return units


def _paper_relevance(question: str, paper: Dict[str, Any], index: int, total: int) -> float:
    q_tokens = _tokens(question)
    overlap = len(q_tokens & _tokens(" ".join([str(paper.get("title") or ""), str(paper.get("methodName") or "")])))
    score = float(overlap)
    arxiv_id = str(paper.get("arxivId") or "")
    if arxiv_id and arxiv_id in question:
        score += 5
    for ordinal in _ORDINAL_RE.findall(question):
        if int(ordinal) == index + 1:
            score += 5
    if total == 1:
        score += 1
    return score


def select_papers(question: str, papers: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """The papers the question is about: ranked by title/ordinal/id match, all when none matches."""

    scored = [(_paper_relevance(question, paper, index, len(papers)), index, paper) for index, paper in enumerate(papers)]
    best = max((score for score, _, _ in scored), default=0)
    if best <= 0:
        return [(index, paper) for _, index, paper in scored][:MAX_SELECTED_PAPERS]
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    return [(index, paper) for score, index, paper in ranked if score > 0][:MAX_SELECTED_PAPERS]


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _notes_text(paper: Dict[str, Any]) -> str:
    notes = paper.get("notes") if isinstance(paper.get("notes"), dict) else {}
    lines = [f"{paper.get('title') or paper.get('paperId')} ({paper.get('paperId')})"]
    for label, key in (("Problem", "problem"), ("Method", "method"), ("Intuition", "intuition"), ("Result", "result"), ("Limitation", "limitation")):
        if notes.get(key):
            lines.append(f"{label}: {notes[key]}")
    for number, item in enumerate(notes.get("insights") or [], 1):
        if isinstance(item, dict) and item.get("finding"):
            lines.append(f"Insight {number}: {item['finding']} (why: {item.get('why_it_matters') or item.get('whyItMatters') or ''})")
    for row in notes.get("eval_settings") or []:
        if isinstance(row, dict):
            lines.append(
                f"Result row: {row.get('benchmark')} | {row.get('metric')}: method {row.get('method_value')}, "
                f"baseline {row.get('baseline_name')} {row.get('baseline_value')} ({row.get('table_label') or 'prose'})"
            )
    setup = notes.get("setup") if isinstance(notes.get("setup"), dict) else {}
    for label, key in (("Datasets", "datasets"), ("Training", "training_config"), ("Hardware", "hardware"), ("Compute", "compute")):
        lines.append(f"{label}: {setup.get(key) or 'not stated in the paper'}")
    return _clip("\n".join(lines), NOTES_CHARS)


def _table_text(table: Dict[str, Any]) -> str:
    rows = [" | ".join(str(cell) for cell in row) for row in (table.get("rows") or []) if isinstance(row, list)]
    return _clip(f"{table.get('label') or 'Table'}: {table.get('caption') or ''}\n" + "\n".join(rows), TABLE_CHARS)


def select_tables(question: str, papers: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[str, str]]:
    """Tables named in the question ("Table 17", "Bảng 3") as ``(source_name, text)``."""

    wanted = {f"table {number}".casefold() for number in _TABLE_REF_RE.findall(question)}
    found: List[Tuple[str, str]] = []
    for _, paper in papers:
        for table in paper.get("tables") or []:
            if isinstance(table, dict) and str(table.get("label") or "").casefold() in wanted:
                found.append((str(table.get("label")), _table_text(table)))
    return found[:MAX_TABLES]


def select_related(question: str, related: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q_tokens = _tokens(question)
    chosen: List[Dict[str, Any]] = []
    for item in related:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        arxiv_id = str(item.get("arxivId") or "")
        if (arxiv_id and arxiv_id in question) or (title and title.casefold() in question.casefold()) or len(q_tokens & _tokens(title)) >= 2:
            chosen.append(item)
        if len(chosen) >= MAX_RELATED:
            break
    return chosen


def build_followup_messages(question: str, context: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Pure ``(system, user, sources)`` for the follow-up call."""

    report_text = str(context.get("reportText") or "")
    language = report_language(report_text)
    papers = [paper for paper in (context.get("papers") or []) if isinstance(paper, dict)]
    selected = select_papers(question, papers)
    units = split_report(report_text, language)
    sources: List[str] = []
    report_parts: List[str] = []
    selected_titles = {str(paper.get("title") or "").strip() for _, paper in selected}
    for source, title, text in units:
        is_paper_block = "§2:" in source
        if is_paper_block and title not in selected_titles:
            continue
        if not is_paper_block and not (source.endswith("§1") or source.endswith("§4") or source.endswith("§3")):
            continue
        limit = PAPER_BLOCK_CHARS if is_paper_block else SUMMARY_SECTION_CHARS
        report_parts.append(f"[{source}]\n{_clip(text, limit)}")
        sources.append(source)
    notes_parts = []
    for _, paper in selected:
        notes_parts.append(_notes_text(paper))
        sources.append(f"Notes: {paper.get('paperId')}")
    tables = select_tables(question, selected)
    sources += [label for label, _ in tables]
    related = select_related(question, [item for item in (context.get("related") or [])])
    related_parts = []
    for item in related:
        related_parts.append(
            f"{item.get('title')} ({item.get('year') or '?'}, {item.get('firstAuthor') or ''})\n{_clip(item.get('abstract'), RELATED_CHARS)}"
        )
        sources.append(f"Related: {item.get('title')}")
    # Table captions of the selected papers: lets the answer point at a table it cannot show.
    captions = [
        f"{table.get('label')}: {_clip(table.get('caption'), 120)}"
        for _, paper in selected for table in (paper.get("tables") or [])
        if isinstance(table, dict) and table.get("label")
    ][:30]
    user = FOLLOWUP_USER.format(
        language_mode=language,
        question=question,
        sources="\n".join(f"- {name}" for name in sources) or "- (none)",
        report_sections="\n\n".join(report_parts) or "(the report has no matching section)",
        paper_notes="\n\n".join(notes_parts) or "(none)",
        tables="\n\n".join([text for _, text in tables] + (["Table captions: " + "; ".join(captions)] if captions and not tables else [])) or "(none)",
        related_papers="\n\n".join(related_parts) or "(none)",
    )
    return FOLLOWUP_SYSTEM, user, sources


def _generate(llm: Any, system: str, user: str) -> str:
    """Stream the answer when the model supports it (LangGraph forwards these tokens), else one call."""

    stream = getattr(llm, "stream", None)
    if callable(stream):
        try:
            text = "".join(
                extract_text_from_response(getattr(chunk, "content", chunk))
                for chunk in stream([SystemMessage(content=system), HumanMessage(content=user)])
            )
        except NotImplementedError:
            text = ""
        if text.strip():
            return text
    return extract_text_from_response(invoke_text(llm, system, user).content)


def answer_followup_node(state: ResearchState) -> Dict[str, Any]:
    started = time.perf_counter()
    question = (state.get("user_query") or "").strip()
    context = state.get("followup_context") or {}
    logs = ["[answer_followup] Trả lời từ báo cáo đã có"]
    llm_calls: list = []
    try:
        system, user, sources = build_followup_messages(question, context)
        llm_started = time.perf_counter()
        with track_llm_call("followup", llm_calls):
            answer = _generate(get_llm(), system, user).strip()
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        if not answer:
            raise ValueError("Empty follow-up answer")
        return {
            "followup_answer": answer,
            "followup_sources": sources,
            "status": "success",
            "trace_logs": logs,
            "node_timings": {"answer_followup": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": llm_ms,
                "llmCalls": llm_calls,
            }},
        }
    except Exception as exc:
        message = f"Không trả lời được câu hỏi tiếp theo: {type(exc).__name__}"
        logs.append(f"❌ [answer_followup] {message}")
        return {
            "status": "error",
            "error_message": message,
            "error_logs": [message],
            "trace_logs": logs,
            "node_timings": {"answer_followup": {
                "durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0,
                "llmCalls": llm_calls,
            }},
        }
