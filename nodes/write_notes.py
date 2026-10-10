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
from prompts.pmrl_notes_prompt import PMRL_NOTES_SYSTEM, PMRL_NOTES_USER
from pydantic import BaseModel, Field
from state import EvalSetting, Insight, PMRLNotes, PaperItem, PaperTable, ResearchState, Setup, SummaryCards
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


# Bump when the PMRL schema or its grounding rules change: old cache entries
# then miss instead of loading notes without the structured fields.
PMRL_NOTES_VERSION = "n2"

_TABLE_MAX_ROWS = 40
_TABLE_MAX_CHARS = 2500


def _table_text(table: PaperTable, *, max_chars: int | None = _TABLE_MAX_CHARS) -> str:
    """Compact, labelled text of one parsed table: caption line, header, rows."""

    head = f"[{table.label or 'Table'}] {' '.join((table.caption or '').split())}".rstrip()
    if table.section:
        head += f" (section: {table.section})"
    rows = [" | ".join(cell.strip() for cell in row) for row in table.rows[: _TABLE_MAX_ROWS if max_chars else None]]
    text = "\n".join([head, *rows])
    return text if max_chars is None or len(text) <= max_chars else text[:max_chars].rstrip() + " ..."


def _paper_content(paper: PaperItem) -> str:
    """Paper text for the PMRL call.

    Order matters because the caller trims the tail to ``PMRL_INPUT_MAX_CHARS``:
    prose sections first, labelled tables last (they are also present, in
    compact form, inside the experiments text).
    """

    if paper.sections:
        sections = paper.sections
        parts = [
            f"=== ABSTRACT ===\n{sections.get('abstract', '')}",
            f"=== METHODOLOGY / ARCHITECTURE ===\n{sections.get('methodology', '')}",
        ]
        if (sections.get("setup") or "").strip():
            parts.append(f"=== EXPERIMENTAL SETUP / TRAINING DETAILS / HARDWARE ===\n{sections['setup']}")
        parts.append(f"=== EXPERIMENTS / BENCHMARKS / RESULTS ===\n{sections.get('experiments', '')}")
        parts.append(f"=== LIMITATIONS / FUTURE WORK ===\n{sections.get('limitations', '')}")
        if paper.tables:
            parts.append("=== TABLES (header row first) ===\n" + "\n\n".join(_table_text(t) for t in paper.tables))
        return "\n\n".join(parts)
    return paper.extracted_text or paper.summary


# --- Grounding of the structured fields ---------------------------------------

_DECIMAL_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<!\w)\d+(?:[.,]\d+)?|(?<![\w.\d])\.\d+")


def _decimals(text: str) -> set[Decimal]:
    """Unsigned numeric values in ``text`` (units and signs ignored)."""

    values: set[Decimal] = set()
    for match in _DECIMAL_RE.finditer(text or ""):
        raw = match.group(0)
        raw = raw.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw) else raw.replace(",", ".")
        try:
            values.add(Decimal(raw))
        except InvalidOperation:
            continue
    return values


def _flat(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _source_corpus(paper: PaperItem) -> str:
    """Everything the paper source offers: sections, raw text, abstract, table cells."""

    parts = [str(value) for value in (paper.sections or {}).values() if isinstance(value, str)]
    parts += [paper.extracted_text or "", paper.summary or ""]
    parts += [_table_text(table, max_chars=None) for table in paper.tables]
    return "\n".join(parts)


_HARDWARE_STOPWORDS = {
    "the", "and", "with", "for", "using", "used", "our", "all", "per", "each", "from", "this", "that",
    "trained", "training", "train", "experiments", "experiment", "models", "model",
}


def _match_tokens(text: str) -> list[str]:
    joined = re.sub(r"(\d)\s+([a-z])", r"\1\2", (text or "").casefold())
    tokens = re.findall(r"[a-z0-9]+", joined)
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not any(c.isdigit() for c in t) else t for t in tokens]


def _text_in_source(text: str, source_tokens: set[str]) -> bool:
    """Simple token check: every token with a digit and half of the other words occur in the source."""

    tokens = [t for t in _match_tokens(text) if t not in _HARDWARE_STOPWORDS]
    if not tokens:
        return False
    digit_tokens = [t for t in tokens if any(c.isdigit() for c in t)]
    word_tokens = [t for t in tokens if t not in digit_tokens and len(t) >= 3]
    if any(t not in source_tokens for t in digit_tokens):
        return False
    if word_tokens and sum(t in source_tokens for t in word_tokens) < len(word_tokens) / 2:
        return False
    return True


def ground_structured_notes(
    paper: PaperItem,
    eval_settings: List[EvalSetting],
    setup: Setup,
) -> Tuple[List[EvalSetting], Setup, Dict[str, Any]]:
    """Keep only structured claims that the paper source supports.

    * An ``EvalSetting`` row is kept only when every number in ``method_value``
      (and in ``baseline_value`` when it has any) occurs in the source text or
      table cells.  ``table_label`` survives only when that labelled table holds
      all the row's numbers; ``source_quote`` survives only when the quote
      occurs in the source.
    * ``setup.hardware`` and ``setup.compute`` are cleared when their text does
      not occur in the source (token check).
    * Insights are not touched here.
    """

    corpus = _source_corpus(paper)
    source_values = _decimals(corpus)
    flat_source = _flat(corpus)
    source_tokens = set(_match_tokens(corpus))
    tables_by_label: Dict[str, List[set[Decimal]]] = {}
    for table in paper.tables:
        values = _decimals(_table_text(table, max_chars=None))
        tables_by_label.setdefault(_flat(table.label), []).append(values)

    kept: List[EvalSetting] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in eval_settings:
        method_numbers = _decimals(row.method_value)
        if not (row.benchmark.strip() and row.metric.strip() and method_numbers):
            continue
        baseline_numbers = _decimals(row.baseline_value)
        claims = method_numbers | baseline_numbers
        if not claims <= source_values:
            continue
        key = (_flat(row.benchmark), _flat(row.metric), _flat(row.method_value), _flat(row.baseline_value))
        if key in seen:
            continue
        seen.add(key)
        update: Dict[str, Any] = {}
        if row.baseline_value.strip() and not baseline_numbers:
            update.update(baseline_value="", baseline_name="")
        label = _flat(row.table_label)
        if label and not any(claims <= values for values in tables_by_label.get(label, [])):
            update["table_label"] = ""
        if row.source_quote.strip() and _flat(row.source_quote) not in flat_source:
            update["source_quote"] = ""
        kept.append(row.model_copy(update=update) if update else row)

    hardware_dropped = bool(setup.hardware.strip()) and not _text_in_source(setup.hardware, source_tokens)
    compute_dropped = bool(setup.compute.strip()) and not _text_in_source(setup.compute, source_tokens)
    grounded_setup = setup.model_copy(update={
        **({"hardware": ""} if hardware_dropped else {}),
        **({"compute": ""} if compute_dropped else {}),
    })
    stats = {
        "proposed": len(eval_settings),
        "kept": len(kept),
        "dropped": len(eval_settings) - len(kept),
        "hardwareDropped": hardware_dropped,
        "computeDropped": compute_dropped,
    }
    return kept, grounded_setup, stats


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


# Provider schema for the PMRL call: the four UI-facing text fields plus the
# structured evidence fields (all with defaults, so a response without them
# still validates).  A comment, not a docstring: pydantic would copy a
# docstring into the JSON Schema that is sent to the provider.  The legacy
# ``brief_summary`` is no longer requested; ``PMRLNotes`` keeps the field
# (empty) so stored records and API consumers are unaffected.  Word budgets
# live once in ``PMRL_NOTES_SYSTEM``.
class PMRLCoreGeneration(BaseModel):
    problem: str = Field(description="Core problem and baseline limitations.")
    method: str = Field(description="Core method: architecture, algorithm, formulation.")
    result: str = Field(description="Key empirical results with datasets, metrics and numbers.")
    limitation: str = Field(description="Limitations, trade-offs, assumptions and future work.")
    method_name: str = Field(default="", description="Short name the paper gives its own method; empty if none.")
    insights: List[Insight] = Field(default_factory=list, description="Findings a reader cannot get from the abstract alone.")
    eval_settings: List[EvalSetting] = Field(default_factory=list, description="Key benchmark/metric rows: own method vs strongest baseline.")
    setup: Setup = Field(default_factory=Setup, description="Datasets, key training config, hardware and compute as stated in the paper.")


def build_pmrl_messages(paper: PaperItem) -> tuple[str, str]:
    """Return the pure ``(system, user)`` pair for one PMRL call (no cache access)."""

    user = PMRL_NOTES_USER.format(
        title=paper.title,
        arxiv_id=paper.arxiv_id or "N/A",
        authors=", ".join(paper.authors) if paper.authors else "N/A",
        published=paper.published or "N/A",
        extracted_content=_paper_content(paper)[: int(config.PMRL_INPUT_MAX_CHARS)],
    )
    return PMRL_NOTES_SYSTEM, user


def _pmrl_cache_entry(prompt_text: str) -> tuple[str, Any]:
    """Resolve the PMRL cache path for the exact LLM input (fail-open)."""

    key = f"pmrl_{prompt_fingerprint(prompt_text, config.DEFAULT_PROVIDER, config.DEFAULT_MODEL, PMRL_NOTES_VERSION)}"
    return key, summary_cache_path(key)


def _structured_fields(source: Any) -> Dict[str, Any]:
    """Structured PMRL fields from a cache entry or a generation; empty defaults when absent or invalid.

    Entries written before the structured fields existed carry none of them.
    """

    get = source.get if isinstance(source, dict) else lambda name, default=None: getattr(source, name, default)
    fields: Dict[str, Any] = {"method_name": str(get("method_name", "") or "")}
    for name, model in (("insights", Insight), ("eval_settings", EvalSetting)):
        try:
            fields[name] = [model.model_validate(item) for item in (get(name, None) or [])]
        except Exception:
            fields[name] = []
    try:
        fields["setup"] = Setup.model_validate(get("setup", None) or {})
    except Exception:
        fields["setup"] = Setup()
    return fields


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
            **_structured_fields(cached),
        )
    except Exception:
        return None
    if not all([notes.problem.strip(), notes.method.strip(), notes.result.strip(), notes.limitation.strip()]):
        return None
    return notes


def _apply_grounding(result: PaperItem) -> None:
    """Ground the structured fields of ``result.notes`` against its source and record counts."""

    if result.notes is None:
        return
    rows, setup, stats = ground_structured_notes(result, result.notes.eval_settings, result.notes.setup)
    result.notes.eval_settings = rows
    result.notes.setup = setup
    name = result.notes.method_name.strip()
    if name and _flat(name) not in _flat(f"{result.title}\n{_source_corpus(result)}"):
        result.notes.method_name = ""
    result.processing_metadata = dict(result.processing_metadata or {})
    result.processing_metadata["evalSettings"] = stats


def analyze_paper_notes(
    paper: PaperItem,
    *,
    llm: Any | None = None,
) -> Tuple[PaperItem, str, Dict[str, Any]]:
    """Run one PMRL call and return an isolated paper result."""

    result = copy.deepcopy(paper)
    started = time.perf_counter()
    system_text, user_text = build_pmrl_messages(result)
    _, cache_path = _pmrl_cache_entry(f"{system_text}\n\n{user_text}")
    cached_notes = _cached_pmrl_notes(load_json_cache(cache_path))
    if cached_notes is not None:
        result.notes = cached_notes
        result.notes_quality = "complete"
        result.processing_metadata = dict(result.processing_metadata or {})
        # The cache holds the raw model output; grounding runs on every load.
        _apply_grounding(result)
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
        generated: PMRLCoreGeneration = invoke_structured_output(
            user_text,
            PMRLCoreGeneration,
            llm=model,
            provider=config.DEFAULT_PROVIDER,
            system=system_text,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        notes = PMRLNotes(
            problem=generated.problem,
            method=generated.method,
            result=generated.result,
            limitation=generated.limitation,
            brief_summary="",
            summary_cards=SummaryCards(),
            summary_quality="missing",
            **_structured_fields(generated),
        )
        save_json_cache(cache_path, {
            "problem": notes.problem,
            "method": notes.method,
            "result": notes.result,
            "limitation": notes.limitation,
            "brief_summary": notes.brief_summary,
            "method_name": notes.method_name,
            "insights": [item.model_dump() for item in notes.insights],
            "eval_settings": [item.model_dump() for item in notes.eval_settings],
            "setup": notes.setup.model_dump(),
        })
        result.notes = notes
        result.notes_quality = "complete"
        result.processing_metadata = dict(result.processing_metadata or {})
        _apply_grounding(result)
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
    """Analyze papers with ``min(n, LLM_MAX_CONCURRENCY)`` workers, keeping input order."""

    if not papers:
        return [], [], [], {}

    def worker(paper: PaperItem) -> Tuple[PaperItem, str, Dict[str, Any]]:
        return analyze_paper_notes(paper, llm=llm)

    with ThreadPoolExecutor(max_workers=max(1, min(len(papers), int(config.LLM_MAX_CONCURRENCY)))) as executor:
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
