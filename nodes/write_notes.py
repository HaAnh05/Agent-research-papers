from __future__ import annotations

import copy
import re
from decimal import Decimal, InvalidOperation
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from config import config
from prompts.pmrl_notes_prompt import PMRL_NOTES_SYSTEM, PMRL_NOTES_USER
from pydantic import BaseModel, Field
from state import EvalSetting, Insight, PMRLNotes, PaperItem, PaperTable, ResearchState, Setup
from tools.cache_manager import load_json_cache, prompt_fingerprint, save_json_cache, summary_cache_path
from tools.llm_provider import get_llm, invoke_structured_output, track_llm_call


# Bump when the PMRL schema or its grounding rules change: old cache entries
# then miss instead of loading notes without the structured fields.
PMRL_NOTES_VERSION = "n3"

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
    )


# Provider schema for the PMRL call: the four UI-facing text fields plus the
# structured evidence fields (all with defaults, so a response without them
# still validates).  A comment, not a docstring: pydantic would copy a
# docstring into the JSON Schema that is sent to the provider.  Word budgets
# live once in ``PMRL_NOTES_SYSTEM``.
class PMRLCoreGeneration(BaseModel):
    problem: str = Field(description="Core problem and baseline limitations.")
    method: str = Field(description="Core method: architecture, algorithm, formulation.")
    result: str = Field(description="Key empirical results with datasets, metrics and numbers.")
    limitation: str = Field(description="Limitations, trade-offs, assumptions and future work.")
    method_name: str = Field(default="", description="Short name the paper gives its own method; empty if none.")
    intuition: str = Field(default="", description="The core idea in plain words and why it should work; empty if the paper gives no basis.")
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
    fields: Dict[str, Any] = {
        "method_name": str(get("method_name", "") or ""),
        "intuition": str(get("intuition", "") or ""),
    }
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
        total_ms = max(0, int((time.perf_counter() - started) * 1000))
        timing = {"durationMs": total_ms, "llmMs": 0, "cache": "hit"}
        result.processing_metadata["pmrl"] = timing
        return result, "ok:cache", timing
    llm_calls: List[Dict[str, Any]] = []
    try:
        model = llm or get_llm()
        llm_started = time.perf_counter()
        with track_llm_call("pmrl", llm_calls, paperId=paper.paper_id):
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
            **_structured_fields(generated),
        )
        save_json_cache(cache_path, {
            "problem": notes.problem,
            "method": notes.method,
            "result": notes.result,
            "limitation": notes.limitation,
            "method_name": notes.method_name,
            "intuition": notes.intuition,
            "insights": [item.model_dump() for item in notes.insights],
            "eval_settings": [item.model_dump() for item in notes.eval_settings],
            "setup": notes.setup.model_dump(),
        })
        result.notes = notes
        result.notes_quality = "complete"
        result.processing_metadata = dict(result.processing_metadata or {})
        _apply_grounding(result)
        status = "ok"
    except Exception as exc:
        llm_ms = max(0, int((time.perf_counter() - started) * 1000))
        result.notes = _fallback_notes(result)
        result.notes_quality = "invalid"
        result.processing_metadata = dict(result.processing_metadata or {})
        status = f"fallback:{type(exc).__name__}"

    total_ms = max(0, int((time.perf_counter() - started) * 1000))
    timing = {"durationMs": total_ms, "llmMs": llm_ms}
    result.processing_metadata = dict(result.processing_metadata or {})
    result.processing_metadata["pmrl"] = dict(timing)
    # ``llmCalls`` (duration and token counts of the PMRL call) goes to node_timings only.
    return result, status, {**timing, "llmCalls": llm_calls}


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
    """Create detailed PMRL notes (one LLM call per paper, run in parallel).

    This node runs next to ``citation_graph`` and ``web_enrich`` after
    ``read_paper``.  It does not return ``selected_papers``: the notes go to
    ``paper_notes`` (keyed by paper id) and ``final_report`` joins them.
    """

    papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[write_notes] Tạo PMRL cho {len(papers)} bài"]

    llm: Any | None = None
    try:
        llm = get_llm()
    except Exception:
        logs.append("[write_notes] Provider unavailable; using per-paper fallback")

    analyzed, worker_logs, errors, paper_timings = _run_notes_in_parallel(papers, llm=llm)
    logs.extend(worker_logs)
    total_llm_ms = sum(int(item.get("llmMs") or 0) for item in paper_timings.values())
    llm_calls = [call for item in paper_timings.values() for call in item.get("llmCalls", [])]
    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": total_llm_ms,
        "papers": {
            paper_id: {key: value for key, value in item.items() if key != "llmCalls"}
            for paper_id, item in paper_timings.items()
        },
        "llmCalls": llm_calls,
    }
    paper_notes: Dict[str, Dict[str, Any]] = {}
    for paper in analyzed:
        metadata = paper.processing_metadata or {}
        paper_notes[paper.paper_id] = {
            "notes": paper.notes,
            "notes_quality": paper.notes_quality,
            "processing_metadata": {
                key: metadata[key] for key in ("pmrl", "evalSettings") if key in metadata
            },
        }
    return {
        "paper_notes": paper_notes,
        "trace_logs": logs,
        "error_logs": errors,
        "node_timings": {"write_notes": timing},
    }
