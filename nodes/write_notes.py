from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from config import config
from prompts.pmrl_notes_prompt import PMRL_NOTES_PROMPT
from state import PMRLGeneration, PMRLNotes, PaperItem, ResearchState, SummaryCards
from tools.cache_manager import load_json_cache, prompt_fingerprint, save_json_cache, summary_cache_path
from tools.llm_provider import get_llm, invoke_structured_output


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")

SUMMARY_CARD_FIELDS = ("tldr", "problem", "method", "key_results", "why_it_matters")

_SUMMARY_GENERIC_WORDS = {
    "about", "after", "also", "because", "between", "could", "from", "into", "more",
    "paper", "research", "shows", "than", "that", "their", "there", "these", "this",
    "using", "with", "within", "would", "method", "problem", "results", "approach",
}


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value or ""))


def _normalized_tokens(value: str) -> list[str]:
    return re.findall(r"\w+", (value or "").casefold(), flags=re.UNICODE)


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalized_tokens(value)
        if len(token) >= 4 and token not in _SUMMARY_GENERIC_WORDS
    }


def _number_tokens(value: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", value or "")
    }


def _near_duplicate(left: str, right: str) -> bool:
    """Reject copied cards while allowing shared scientific vocabulary."""

    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if len(left_tokens) < 8 or len(right_tokens) < 8:
        return False
    left_text = " ".join(left_tokens)
    right_text = " ".join(right_tokens)
    if left_text in right_text or right_text in left_text:
        return True
    ratio = SequenceMatcher(None, left_text, right_text).ratio()
    overlap = len(set(left_tokens) & set(right_tokens)) / max(1, min(len(set(left_tokens)), len(set(right_tokens))))
    return ratio >= 0.84 or overlap >= 0.86


def validate_summary_cards(
    value: Any,
    *,
    source_text: str = "",
) -> Tuple[SummaryCards, str, str]:
    """Validate compact cards without truncating or inventing replacement text.

    The return status is ``complete`` only when all five cards are present,
    are sufficiently distinct, and use vocabulary/numbers supported by the
    extracted source.  Near-duplicate cards are blanked individually rather
    than rejecting the whole set.  The third return value is a short
    failure reason (``""`` when complete).
    """

    if isinstance(value, SummaryCards):
        cards = value
    elif isinstance(value, dict):
        try:
            if hasattr(SummaryCards, "model_validate"):
                cards = SummaryCards.model_validate(value)
            else:  # pragma: no cover - Pydantic v1 compatibility
                cards = SummaryCards.parse_obj(value)
        except Exception:
            return SummaryCards(), "invalid", "schema_mismatch"
    else:
        return SummaryCards(), "missing", "no_structured_output"

    fields = {name: str(getattr(cards, name, "") or "").strip() for name in SUMMARY_CARD_FIELDS}
    for name in SUMMARY_CARD_FIELDS:
        if not fields[name]:
            return SummaryCards(), "missing", f"empty_field:{name}"

    for index, left_name in enumerate(SUMMARY_CARD_FIELDS):
        for right_name in SUMMARY_CARD_FIELDS[index + 1:]:
            if _near_duplicate(fields[left_name], fields[right_name]):
                fields[right_name] = ""

    source = (source_text or "").strip()
    if not source:
        return SummaryCards(), "invalid", "no_source_text"
    source_words = _meaningful_tokens(source)
    source_numbers = _number_tokens(source)
    for name, field in fields.items():
        if not field:
            continue
        if not (_meaningful_tokens(field) & source_words):
            return SummaryCards(), "invalid", f"no_source_overlap:{name}"
        unsupported = _number_tokens(field) - source_numbers
        if unsupported:
            return SummaryCards(), "invalid", f"number_not_in_source:{name}={sorted(unsupported)[0]}"

    normalized = SummaryCards(**fields)
    return normalized, "complete", ""


def validate_brief_summary(value: Any, *, max_words: int = 150) -> str:
    """Accept only a conclusion plus 3 or 4 short evidence bullets."""

    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    # Providers sometimes emit bullets on one physical line.
    text = re.sub(r"\s+(?=[-*•]\s+)", "\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_positions = [index for index, line in enumerate(lines) if _BULLET_RE.match(line)]
    if len(bullet_positions) not in {3, 4}:
        return ""
    first_bullet = bullet_positions[0]
    # Exactly one conclusion line must precede a contiguous bullet list.
    if first_bullet != 1 or any(index != first_bullet + offset for offset, index in enumerate(bullet_positions)):
        return ""
    conclusion = lines[0]
    if not conclusion or conclusion.startswith(("-", "*", "•")):
        return ""
    bullets = [_BULLET_RE.match(line).group(1).strip() for line in lines[1:]]
    if any(not bullet for bullet in bullets):
        return ""
    if len(re.findall(r"\S+", text)) > max_words:
        return ""
    return "\n".join([conclusion, *[f"- {bullet}" for bullet in bullets]])


def _paper_content(paper: PaperItem) -> str:
    if paper.sections:
        return (
            f"=== ABSTRACT ===\n{paper.sections.get('abstract', '')}\n\n"
            f"=== METHODOLOGY / ARCHITECTURE ===\n{paper.sections.get('methodology', '')}\n\n"
            f"=== EXPERIMENTS / BENCHMARKS / RESULTS ===\n{paper.sections.get('experiments', '')}\n\n"
            f"=== LIMITATIONS / FUTURE WORK ===\n{paper.sections.get('limitations', '')}"
        )
    return paper.extracted_text or paper.summary


def _fallback_notes(paper: PaperItem) -> PMRLNotes:
    return PMRLNotes(
        problem=paper.summary[:300] or "Not extracted (PMRL generation unavailable).",
        method="Not extracted (PMRL generation unavailable).",
        result="Not extracted (PMRL generation unavailable).",
        limitation="Not extracted (PMRL generation unavailable).",
        summary_cards=SummaryCards(),
        summary_quality="invalid",
        brief_summary="",
    )


def _notes_prompt(paper: PaperItem) -> str:
    return PMRL_NOTES_PROMPT.format(
        title=paper.title,
        arxiv_id=paper.arxiv_id or "N/A",
        authors=", ".join(paper.authors) if paper.authors else "N/A",
        published=paper.published or "N/A",
        extracted_content=_paper_content(paper)[:14000],
    )


def _pmrl_cache_entry(prompt_text: str) -> tuple[str, Any]:
    """Resolve the PMRL cache path for the exact LLM input (fail-open)."""

    key = f"pmrl_{prompt_fingerprint(prompt_text, config.DEFAULT_PROVIDER, config.DEFAULT_MODEL)}"
    return key, summary_cache_path(key)


def _cached_pmrl_notes(cached: Any) -> PMRLNotes | None:
    """Rebuild validated PMRL notes from a cache entry; None when unusable."""

    if not isinstance(cached, dict):
        return None
    try:
        notes = PMRLNotes(
            problem=str(cached.get("problem") or ""),
            method=str(cached.get("method") or ""),
            result=str(cached.get("result") or ""),
            limitation=str(cached.get("limitation") or ""),
            brief_summary=validate_brief_summary(cached.get("brief_summary")),
            summary_cards=SummaryCards(),
            summary_quality="missing",
        )
    except Exception:
        return None
    if not all([notes.problem.strip(), notes.method.strip(), notes.result.strip(), notes.limitation.strip()]):
        return None
    return notes


def analyze_paper_notes(
    paper: PaperItem,
    *,
    llm: Any | None = None,
) -> Tuple[PaperItem, str, Dict[str, Any]]:
    """Run one PMRL call and return an isolated paper result."""

    result = copy.deepcopy(paper)
    started = time.perf_counter()
    prompt_text = _notes_prompt(result)
    _, cache_path = _pmrl_cache_entry(prompt_text)
    cached_notes = _cached_pmrl_notes(load_json_cache(cache_path))
    if cached_notes is not None:
        result.notes = cached_notes
        result.notes_quality = "complete"
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {
            "status": "pending",
            "wordCount": 0,
        }
        total_ms = max(0, int((time.perf_counter() - started) * 1000))
        timing = {"durationMs": total_ms, "llmMs": 0, "cache": "hit"}
        result.processing_metadata["pmrl"] = timing
        return result, "ok:cache", timing
    try:
        model = llm or get_llm()
        llm_started = time.perf_counter()
        generated: PMRLGeneration = invoke_structured_output(
            prompt_text,
            PMRLGeneration,
            llm=model,
            provider=config.DEFAULT_PROVIDER,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        generated.brief_summary = validate_brief_summary(generated.brief_summary)
        notes = PMRLNotes(
            problem=generated.problem,
            method=generated.method,
            result=generated.result,
            limitation=generated.limitation,
            brief_summary=generated.brief_summary,
            summary_cards=SummaryCards(),
            summary_quality="missing",
        )
        save_json_cache(cache_path, {
            "problem": notes.problem,
            "method": notes.method,
            "result": notes.result,
            "limitation": notes.limitation,
            "brief_summary": notes.brief_summary,
        })
        result.notes = notes
        result.notes_quality = "complete"
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {
            "status": "pending",
            "wordCount": 0,
        }
        status = "ok"
    except Exception as exc:
        llm_ms = max(0, int((time.perf_counter() - started) * 1000))
        result.notes = _fallback_notes(result)
        result.notes_quality = "invalid"
        result.processing_metadata = dict(result.processing_metadata or {})
        result.processing_metadata["summary"] = {"status": "invalid", "wordCount": 0}
        status = f"fallback:{type(exc).__name__}"

    total_ms = max(0, int((time.perf_counter() - started) * 1000))
    timing = {"durationMs": total_ms, "llmMs": llm_ms}
    result.processing_metadata = dict(result.processing_metadata or {})
    result.processing_metadata["pmrl"] = timing
    return result, status, timing


def _run_notes_in_parallel(
    papers: List[PaperItem],
    *,
    llm: Any | None,
) -> Tuple[List[PaperItem], List[str], List[str], Dict[str, Dict[str, Any]]]:
    """Analyze papers with at most two workers while preserving input order."""

    if not papers:
        return [], [], [], {}

    def worker(paper: PaperItem) -> Tuple[PaperItem, str, Dict[str, Any]]:
        return analyze_paper_notes(paper, llm=llm)

    with ThreadPoolExecutor(max_workers=min(2, len(papers))) as executor:
        completed = list(executor.map(worker, papers))

    analyzed: List[PaperItem] = []
    logs: List[str] = []
    errors: List[str] = []
    timings: Dict[str, Dict[str, Any]] = {}
    for paper, status, timing in completed:
        analyzed.append(paper)
        timings[paper.paper_id] = timing
        if status.startswith("ok"):
            suffix = "; cache hit" if status == "ok:cache" else ""
            logs.append(f"[write_notes] {paper.paper_id}: PMRL ok{suffix}")
        else:
            logs.append(f"[write_notes] {paper.paper_id}: PMRL fallback")
            errors.append(f"PMRL fallback for {paper.paper_id}")
    return analyzed, logs, errors, timings


def write_notes_node(state: ResearchState) -> Dict[str, Any]:
    """Create or finalize detailed PMRL notes; cards are added after the report."""

    papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[write_notes] Tạo PMRL cho {len(papers)} bài"]

    # web_enrich performs independent GitHub + PMRL work in its bounded pool;
    # retain this node in the graph and consume those prepared notes.
    precomputed = [
        paper
        for paper in papers
        if paper.notes is not None and paper.processing_metadata.get("pmrl_precomputed")
    ]
    if len(precomputed) == len(papers):
        analyzed = [copy.deepcopy(paper) for paper in papers]
        for paper in analyzed:
            if paper.notes is not None:
                paper.notes.brief_summary = validate_brief_summary(paper.notes.brief_summary)
            if paper.notes_quality == "invalid":
                logs.append(f"[write_notes] {paper.paper_id}: PMRL fallback")
            else:
                logs.append(f"[write_notes] {paper.paper_id}: PMRL ready")
        timing = {
            "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
            "llmMs": 0,
            "reused": True,
            "papers": {paper.paper_id: paper.processing_metadata.get("pmrl", {}) for paper in analyzed},
        }
        return {"selected_papers": analyzed, "trace_logs": logs, "node_timings": {"write_notes": timing}}

    llm: Any | None = None
    try:
        llm = get_llm()
    except Exception:
        logs.append("[write_notes] Provider unavailable; using per-paper fallback")

    analyzed, worker_logs, errors, paper_timings = _run_notes_in_parallel(papers, llm=llm)
    logs.extend(worker_logs)
    total_llm_ms = sum(int(item.get("llmMs") or 0) for item in paper_timings.values())
    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": total_llm_ms,
        "papers": paper_timings,
    }
    return {
        "selected_papers": analyzed,
        "trace_logs": logs,
        "error_logs": errors,
        "node_timings": {"write_notes": timing},
    }
