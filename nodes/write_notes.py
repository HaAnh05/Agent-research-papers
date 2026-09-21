from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Tuple

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


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass(frozen=True)
class _NumberMention:
    display: str
    value: Decimal
    unit: Literal["plain", "percent"]
    start: int
    end: int
    decimal_places: int
    context_id: int


_NUMBER_CONTEXT_BOUNDARY_RE = re.compile(
    r"[!?;]|(?<!\d)\.(?!\d)|(?<=\d)\.(?=\s|$)|\n\s*\n"
)


def _number_mentions(value: str) -> list[_NumberMention]:
    """Parse numeric claims while retaining units and source position."""

    mentions: list[_NumberMention] = []
    pattern = re.compile(
        r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?))(\s*%|percent\b)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(value or ""):
        raw_number = match.group(1)
        unsigned = raw_number.lstrip("+-")
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", unsigned):
            parsed_number = raw_number.replace(",", "")
            decimal_places = 0
        else:
            parsed_number = raw_number.replace(",", ".")
            decimal_places = len(parsed_number.partition(".")[2])
        try:
            number = Decimal(parsed_number)
        except InvalidOperation:
            continue
        mentions.append(
            _NumberMention(
                display=match.group(0).strip(),
                value=number,
                unit="percent" if match.group(2) else "plain",
                start=match.start(),
                end=match.end(),
                decimal_places=decimal_places,
                context_id=sum(
                    1 for _ in _NUMBER_CONTEXT_BOUNDARY_RE.finditer((value or "")[:match.start()])
                ),
            )
        )
    return mentions


def _equivalent_ratio(mention: _NumberMention) -> Decimal:
    return mention.value / Decimal(100) if mention.unit == "percent" else mention.value


def _within_claim_precision(actual: Decimal, claim: _NumberMention) -> bool:
    tolerance = Decimal("0.5") * (Decimal(10) ** -claim.decimal_places)
    return abs(actual - claim.value) <= tolerance


def _classify_number_claim(
    claim: _NumberMention,
    source_mentions: list[_NumberMention],
    *,
    allow_arithmetic: bool,
) -> Literal["reported", "derived", "unsupported"]:
    """Classify a number as directly reported, equivalent, or calculated.

    Unit conversion (for example ``0.054`` to ``5.4%``) is derived evidence.
    For Key Results only, a percentage may also be derived from two nearby
    reported values using relative change.  The proximity bound prevents
    unrelated numbers in distant sections from being combined.
    """

    if any(item.unit == claim.unit and item.value == claim.value for item in source_mentions):
        return "reported"
    if any(
        item.unit != claim.unit and _equivalent_ratio(item) == _equivalent_ratio(claim)
        for item in source_mentions
    ):
        return "derived"
    if not allow_arithmetic or claim.unit != "percent":
        return "unsupported"

    plain = [item for item in source_mentions if item.unit == "plain"]
    for index, left in enumerate(plain):
        for right in plain[index + 1:]:
            if left.context_id != right.context_id:
                continue
            if max(left.start, right.start) - min(left.end, right.end) > 160:
                continue
            for baseline in (left.value, right.value):
                if baseline == 0:
                    continue
                relative_change = abs(left.value - right.value) / abs(baseline) * Decimal(100)
                if _within_claim_precision(relative_change, claim):
                    return "derived"
    return "unsupported"


def _number_reason_value(mention: _NumberMention) -> str:
    suffix = "%" if mention.unit == "percent" else ""
    return f"{_decimal_text(mention.value)}{suffix}"


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
) -> Tuple[SummaryCards, str, str, Dict[str, str], Dict[str, str]]:
    """Validate compact cards without truncating or inventing replacement text.

    Each card is validated independently. Supported cards survive when a
    sibling is missing, duplicated, or ungrounded; the aggregate status is
    then ``partial``. The final mappings record per-card status and a stable
    rejection reason for API/UI consumers.
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
            statuses = {name: "invalid" for name in SUMMARY_CARD_FIELDS}
            return SummaryCards(), "invalid", "schema_mismatch", statuses, {
                name: "schema_mismatch" for name in SUMMARY_CARD_FIELDS
            }
    else:
        statuses = {name: "missing" for name in SUMMARY_CARD_FIELDS}
        return SummaryCards(), "missing", "no_structured_output", statuses, {
            name: "no_structured_output" for name in SUMMARY_CARD_FIELDS
        }

    fields = {name: str(getattr(cards, name, "") or "").strip() for name in SUMMARY_CARD_FIELDS}
    statuses: Dict[str, str] = {
        name: "complete" if fields[name] else "missing" for name in SUMMARY_CARD_FIELDS
    }
    reasons = [f"empty_field:{name}" for name in SUMMARY_CARD_FIELDS if not fields[name]]
    card_reasons: Dict[str, str] = {
        name: "empty_field" for name in SUMMARY_CARD_FIELDS if not fields[name]
    }

    source = (source_text or "").strip()
    source_words = _meaningful_tokens(source)
    if len(source_words) < 5:
        statuses = {name: "invalid" for name in SUMMARY_CARD_FIELDS}
        return SummaryCards(), "invalid", "insufficient_source_text", statuses, {
            name: "insufficient_source_text" for name in SUMMARY_CARD_FIELDS
        }
    source_numbers = _number_mentions(source)

    for name in SUMMARY_CARD_FIELDS:
        field = fields[name]
        if not field:
            continue
        # Why-it-matters is commonly translated. Lexical overlap between an
        # English source and Vietnamese summary is not a grounding signal.
        if name != "why_it_matters" and not (_meaningful_tokens(field) & source_words):
            fields[name] = ""
            statuses[name] = "unsupported"
            reasons.append(f"no_source_overlap:{name}")
            card_reasons[name] = "no_source_overlap"
            continue
        unsupported = next(
            (
                mention
                for mention in _number_mentions(field)
                if _classify_number_claim(
                    mention,
                    source_numbers,
                    allow_arithmetic=name == "key_results",
                ) == "unsupported"
            ),
            None,
        )
        if unsupported is not None:
            reason_value = _number_reason_value(unsupported)
            fields[name] = ""
            statuses[name] = "unsupported"
            reasons.append(f"number_not_in_source:{name}={reason_value}")
            card_reasons[name] = f"number_not_in_source:{reason_value}"

    for index, left_name in enumerate(SUMMARY_CARD_FIELDS):
        if not fields[left_name]:
            continue
        for right_name in SUMMARY_CARD_FIELDS[index + 1:]:
            if fields[right_name] and _near_duplicate(fields[left_name], fields[right_name]):
                fields[right_name] = ""
                statuses[right_name] = "invalid"
                reasons.append(f"near_duplicate:{right_name}")
                card_reasons[right_name] = f"near_duplicate:{left_name}"

    normalized = SummaryCards(**fields)
    valid_count = sum(bool(fields[name]) for name in SUMMARY_CARD_FIELDS)
    if valid_count == len(SUMMARY_CARD_FIELDS):
        aggregate = "complete"
    elif valid_count:
        aggregate = "partial"
    elif any(status in {"invalid", "unsupported"} for status in statuses.values()):
        aggregate = "invalid"
    else:
        aggregate = "missing"
    return normalized, aggregate, ";".join(reasons), statuses, card_reasons


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
