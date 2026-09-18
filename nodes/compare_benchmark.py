from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from config import config
from prompts.benchmark_prompt import BENCHMARK_MATRIX_PROMPT
from state import ComparisonArtifact, ComparisonMetric, ComparisonPaper, ComparisonRow, PaperItem, ResearchState
from tools.llm_provider import get_llm, invoke_structured_output


def _dump_model(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        raw = value.model_dump()
    elif hasattr(value, "dict"):
        raw = value.dict()
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}
    return dict(raw) if isinstance(raw, dict) else {}


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9%]+", _normalise(value)) if len(token) > 1}


def _number_tokens(value: Any) -> set[str]:
    return {token.rstrip("%") for token in re.findall(r"[-+]?\d+(?:\.\d+)?%?", str(value or ""))}


def _source_evidence_matches(source: str, row: ComparisonRow, value: str, quote: str) -> Tuple[bool, str]:
    """Validate an LLM row against a retained PDF passage deterministically."""

    source_norm = _normalise(source)
    quote_norm = _normalise(quote)
    if not source_norm or not quote_norm:
        return False, "missing source evidence"
    quote_tokens = _tokens(quote_norm)
    source_tokens = _tokens(source_norm)
    if quote_norm not in source_norm:
        return False, "source quote is absent from the PDF passage"

    value_numbers = _number_tokens(value)
    if value_numbers and not value_numbers.issubset(_number_tokens(source)):
        return False, "reported value is absent from the PDF passage"

    dataset_tokens = _tokens(row.dataset)
    if not dataset_tokens or not dataset_tokens.issubset(source_tokens):
        return False, "dataset is absent from the PDF passage"
    if not dataset_tokens.issubset(quote_tokens):
        return False, "source quote does not identify the dataset"

    metric_tokens = _tokens(row.metric)
    if not metric_tokens or not (metric_tokens & source_tokens):
        return False, "metric is absent from the PDF passage"
    if not (metric_tokens & quote_tokens):
        return False, "source quote does not identify the metric"

    unit = _normalise(row.unit)
    unit_aliases = {
        "%": {"%", "percent", "percentage"},
        "percent": {"%", "percent", "percentage"},
        "percentage": {"%", "percent", "percentage"},
        "seconds": {"s", "sec", "secs", "second", "seconds"},
        "s": {"s", "sec", "secs", "second", "seconds"},
        "ms": {"ms", "millisecond", "milliseconds"},
    }
    unit_tokens = unit_aliases.get(unit, _tokens(unit))
    source_unit_tokens = _tokens(source_norm)
    source_has_percent = "%" in source_norm or "percent" in source_norm or "percentage" in source_norm
    source_has_ms = bool(re.search(r"\b(?:ms|msec|milliseconds?)\b", source_norm))
    source_has_seconds = bool(re.search(r"\b(?:s|sec|secs|second|seconds)\b", source_norm))
    unit_supported = (
        (unit in {"%", "percent", "percentage"} and source_has_percent)
        or (unit in {"ms", "millisecond", "milliseconds"} and source_has_ms)
        or (unit in {"s", "sec", "secs", "second", "seconds"} and source_has_seconds and not source_has_ms)
        or (unit not in {"%", "percent", "percentage", "ms", "millisecond", "milliseconds", "s", "sec", "secs", "second", "seconds"} and unit_tokens.issubset(source_unit_tokens))
    )
    if not unit_tokens or not unit_supported:
        return False, "unit is absent from the PDF passage"
    if unit in {"%", "percent", "percentage"} and not ("%" in quote_norm or "percent" in quote_norm or "percentage" in quote_norm):
        return False, "source quote does not identify the unit"
    if unit in {"ms", "millisecond", "milliseconds"} and not bool(re.search(r"\b(?:ms|msec|milliseconds?)\b", quote_norm)):
        return False, "source quote does not identify the unit"
    if unit in {"s", "sec", "secs", "second", "seconds"} and not bool(re.search(r"\b(?:s|sec|secs|second|seconds)\b", quote_norm)):
        return False, "source quote does not identify the unit"
    return True, ""


def _paper_aliases(paper: PaperItem) -> set[str]:
    return {
        _normalise(paper.paper_id),
        _normalise(paper.arxiv_id),
        _normalise(paper.title),
    } - {""}


def _resolve_paper_id(value: Any, papers: List[PaperItem]) -> str | None:
    candidate = _normalise(value)
    if not candidate:
        return None
    for paper in papers:
        if candidate in _paper_aliases(paper):
            return paper.paper_id
    # Accept a canonical ID with a harmless arxiv prefix.
    candidate = candidate.removeprefix("arxiv:")
    for paper in papers:
        if candidate in {_normalise(paper.paper_id).removeprefix("arxiv_"), _normalise(paper.arxiv_id)}:
            return paper.paper_id
    return None


def _compact_paper_context(paper: PaperItem) -> str:
    notes = paper.notes
    brief = notes.brief_summary if notes else ""
    result = notes.result if notes else paper.summary
    method = notes.method if notes else ""
    source = _source_evidence(paper)
    return (
        f"Paper ID: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Brief: {brief[:350]}\n"
        f"Method context: {method[:500]}\n"
        f"Result summary: {result[:700]}\n"
        f"PDF source excerpt (use only this for metric quotes): {source[:2400]}"
    )


def _source_evidence(paper: PaperItem) -> str:
    """Prefer parser output over LLM notes when validating source quotes."""

    if paper.sections.get("experiments"):
        return paper.sections["experiments"]
    if paper.extracted_text:
        return paper.extracted_text
    # A PMRL result is an LLM synthesis, not a source passage.  When the
    # parser did not retain source text, suppress numeric alignment rather
    # than treating that synthesis as independent evidence.
    return ""


def _fallback_artifact(papers: List[PaperItem]) -> ComparisonArtifact:
    return ComparisonArtifact(
        papers=[
            ComparisonPaper(
                paper_id=paper.paper_id,
                title=paper.title,
                findings=[],
                metrics=[],
            )
            for paper in papers
        ],
        rows=[],
        synthesis=["Not directly comparable: no validated shared metric evidence was returned."],
    )


def _validate_artifact(artifact: ComparisonArtifact, papers: List[PaperItem]) -> ComparisonArtifact:
    paper_by_id = {paper.paper_id: paper for paper in papers}
    resolved_papers: List[ComparisonPaper] = []
    for index, raw in enumerate(artifact.papers):
        paper_id = _resolve_paper_id(raw.paper_id, papers)
        if paper_id is None and index < len(papers):
            paper_id = papers[index].paper_id
        if paper_id is None:
            continue
        paper = paper_by_id[paper_id]
        metrics = []
        for metric in raw.metrics[:12]:
            row = ComparisonRow(
                metric=metric.metric,
                dataset=metric.dataset,
                unit=metric.unit,
                values={paper_id: metric.value},
                source_quotes={paper_id: metric.source_quote},
            )
            supported, _ = _source_evidence_matches(_source_evidence(paper), row, metric.value, metric.source_quote)
            if supported:
                metrics.append(metric)
        resolved_papers.append(
            ComparisonPaper(
                paper_id=paper_id,
                title=paper.title,
                findings=[str(item).strip()[:500] for item in raw.findings[:6] if str(item).strip()],
                metrics=metrics,
            )
        )

    validated_rows: List[ComparisonRow] = []
    for raw_row in artifact.rows[:32]:
        values: Dict[str, str] = {}
        quotes: Dict[str, str] = {}
        reasons: List[str] = []
        for raw_id, raw_value in raw_row.values.items():
            paper_id = _resolve_paper_id(raw_id, papers)
            if paper_id is not None:
                values[paper_id] = str(raw_value).strip()[:180]
        for raw_id, raw_quote in raw_row.source_quotes.items():
            paper_id = _resolve_paper_id(raw_id, papers)
            if paper_id is not None:
                quotes[paper_id] = str(raw_quote).strip()[:700]

        if not raw_row.metric.strip() or not raw_row.dataset.strip() or not raw_row.unit.strip():
            reasons.append("metric, dataset, and unit are required")
        if len(values) < 2:
            reasons.append("fewer than two paper values")
        for paper_id, value in values.items():
            paper = paper_by_id[paper_id]
            supported, reason = _source_evidence_matches(
                _source_evidence(paper),
                raw_row,
                value,
                quotes.get(paper_id, ""),
            )
            if not supported:
                reasons.append(f"{paper_id}: {reason}")
        if not raw_row.comparable:
            reasons.append("model did not mark this row comparable")
        comparable = bool(raw_row.comparable) and not reasons and len(values) >= 2
        reason = "" if comparable else "Not directly comparable: " + "; ".join(dict.fromkeys(reasons))
        validated_rows.append(
            ComparisonRow(
                metric=raw_row.metric.strip()[:160],
                dataset=raw_row.dataset.strip()[:160],
                unit=raw_row.unit.strip()[:80],
                values=values,
                source_quotes=quotes,
                comparable=comparable,
                reason=reason,
            )
        )

    synthesis = [str(item).strip()[:700] for item in artifact.synthesis[:6] if str(item).strip()]
    if not synthesis:
        synthesis = ["Not directly comparable: no validated shared metric evidence was returned."]
    return ComparisonArtifact(papers=resolved_papers, rows=validated_rows, synthesis=synthesis)


def _markdown_from_artifact(artifact: ComparisonArtifact, papers: List[PaperItem]) -> str:
    """Render the structured artifact without duplicating full PMRL prose."""

    if len(papers) < 2:
        return ""
    names = {paper.paper_id: paper.title for paper in papers}
    lines = ["| Metric | Dataset | Unit | " + " | ".join(names.values()) + " |", "|---|---|---|" + "---|" * len(papers)]
    if artifact.rows:
        for row in artifact.rows:
            cells = []
            for paper in papers:
                value = row.values.get(paper.paper_id, "—")
                if not row.comparable:
                    value = "Not directly comparable"
                cells.append(value.replace("|", "\\|"))
            lines.append(
                "| " + " | ".join(
                    [row.metric.replace("|", "\\|"), row.dataset.replace("|", "\\|"), row.unit.replace("|", "\\|")]
                    + cells
                ) + " |"
            )
            if not row.comparable and row.reason:
                lines.append("| Note |  |  | " + " | ".join([row.reason.replace("|", "\\|")] + ["" for _ in range(max(0, len(papers) - 1))]) + " |")
    else:
        lines.append("| Findings | — | — | " + " | ".join("See structured artifact" for _ in papers) + " |")
    lines.extend(["", "**Synthesis**", *[f"- {item}" for item in artifact.synthesis]])
    return "\n".join(lines)


def compare_benchmark_node(state: ResearchState) -> Dict[str, Any]:
    """Return one validated structured artifact and a Markdown projection."""

    papers = list(state.get("selected_papers", []) or [])
    started = time.perf_counter()
    logs = [f"[compare_benchmark] So sánh {len(papers)} bài báo"]
    if len(papers) < 2:
        logs.append("[compare_benchmark] Chỉ có 1 bài báo, bỏ qua benchmark.")
        return {
            "benchmark_matrix": None,
            "comparison_artifact": None,
            "trace_logs": logs,
            "node_timings": {"compare_benchmark": {"durationMs": max(0, int((time.perf_counter() - started) * 1000)), "llmMs": 0}},
        }

    prompt = BENCHMARK_MATRIX_PROMPT.format(
        papers_pmrl_summary="\n\n".join(_compact_paper_context(paper) for paper in papers),
    )
    llm_ms = 0
    try:
        llm = get_llm()
        llm_started = time.perf_counter()
        artifact = invoke_structured_output(
            prompt,
            ComparisonArtifact,
            llm=llm,
            provider=config.DEFAULT_PROVIDER,
        )
        llm_ms = max(0, int((time.perf_counter() - llm_started) * 1000))
        logs.append("[compare_benchmark] Đã có structured benchmark artifact.")
    except Exception:
        artifact = _fallback_artifact(papers)
        logs.append("[compare_benchmark] Benchmark artifact fallback; no shared metric claim.")

    artifact = _validate_artifact(artifact, papers)
    matrix = _markdown_from_artifact(artifact, papers)
    timing = {
        "durationMs": max(0, int((time.perf_counter() - started) * 1000)),
        "llmMs": llm_ms,
        "paperCount": len(papers),
        "rowCount": len(artifact.rows),
    }
    return {
        "benchmark_matrix": matrix,
        "comparison_artifact": _dump_model(artifact),
        "trace_logs": logs,
        "node_timings": {"compare_benchmark": timing},
    }
