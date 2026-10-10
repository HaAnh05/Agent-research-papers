"""Compare a candidate run against a frozen baseline on quality and latency.

Usage::

    .venv/bin/python bench/verify.py --baseline <dir> --candidate <dir>

Exit code 1 means a HARD failure (or a QUALITY/LATENCY failure under
``--strict`` / ``--strict-latency``).  The module has no dependency on the
graph code, so it works with any version of the nodes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PMRL_FIELDS = ("problem", "method", "result", "limitation")
QUALITY_RANK = {"complete": 2, "partial": 1}
SECTION_HEADING_RE = re.compile(r"^##\s+([1-5])\s*[.)]", re.MULTILINE)


# --- Data model ------------------------------------------------------------


@dataclass
class Check:
    name: str
    group: str  # HARD | QUALITY | LATENCY
    status: str  # PASS | FAIL | WARN | INFO
    detail: str = ""


@dataclass
class Thresholds:
    word_ratio_lo: float = 0.6
    word_ratio_hi: float = 1.6
    report_ratio_lo: float = 0.6
    report_ratio_hi: float = 1.6
    grounding_drop: float = 0.05
    max_equations_per_paper: int = 3
    target_seconds: float = 60.0


@dataclass
class Run:
    """Normalized view of a baseline or candidate artifact directory."""

    path: Path
    meta: dict[str, Any]
    paper_ids: list[str]
    notes: dict[str, dict[str, str]]
    summary_quality: dict[str, str]
    card_statuses: dict[str, dict[str, str]]
    report: str
    sources: dict[str, str]
    arxiv_ids: dict[str, str] = field(default_factory=dict)


# --- Loading ---------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_text(data: dict[str, Any]) -> str:
    parts = [str(data.get("extracted_text") or "")]
    sections = data.get("sections") or {}
    if isinstance(sections, dict):
        parts.extend(str(v) for v in sections.values() if isinstance(v, str))
    for table in data.get("tables") or []:  # structured HTML tables (caption + rows)
        if isinstance(table, dict):
            parts.append(str(table.get("caption") or ""))
            parts.extend(" | ".join(str(c) for c in row) for row in table.get("rows") or [] if isinstance(row, list))
    return "\n".join(parts)


def load_sources(directory: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    if directory.is_dir():
        for item in sorted(directory.glob("*.json")):
            data = _read_json(item)
            sources[str(data.get("paper_id") or item.stem)] = _source_text(data)
    return sources


def _arxiv_id(paper_id: str) -> str:
    return paper_id.removeprefix("arxiv_") if paper_id.startswith("arxiv_") else ""


def load_baseline(path: Path) -> Run:
    meta = _read_json(path / "run.json")
    artifacts = meta.get("artifacts", {})
    by_paper: dict[str, str] = artifacts.get("pmrl_by_paper", {})
    notes = {pid: _read_json(path / rel) for pid, rel in by_paper.items()}
    cards: dict[str, Any] = {}
    report_cache = artifacts.get("report_cache")
    if report_cache and (path / report_cache).is_file():
        cards = _read_json(path / report_cache).get("cards", {})
    quality = dict(meta.get("summary_quality") or {})
    for pid, entry in cards.items():
        quality.setdefault(pid, entry.get("quality", "unknown"))
    report_file = path / artifacts.get("report_markdown", "report.md")
    ids = list(by_paper) or list(quality)
    return Run(
        path=path, meta=meta, paper_ids=ids,
        notes={pid: {k: str(v or "") for k, v in n.items()} for pid, n in notes.items()},
        summary_quality=quality,
        card_statuses={pid: dict(e.get("card_statuses") or {}) for pid, e in cards.items()},
        report=report_file.read_text(encoding="utf-8") if report_file.is_file() else "",
        sources=load_sources(path / artifacts.get("source_text_dir", "source_text")),
        arxiv_ids={pid: _arxiv_id(pid) for pid in ids},
    )


def load_candidate(path: Path) -> Run:
    meta = _read_json(path / "run.json")
    state = _read_json(path / "state.json")
    papers = state.get("selected_papers") or []
    ids = [str(p["paper_id"]) for p in papers]
    notes: dict[str, dict[str, str]] = {}
    quality: dict[str, str] = {}
    statuses: dict[str, dict[str, str]] = {}
    for paper in papers:
        pid = str(paper["paper_id"])
        raw = paper.get("notes") or {}
        notes[pid] = {k: str(raw.get(k) or "") for k in (*PMRL_FIELDS, "brief_summary")}
        quality[pid] = str(raw.get("summary_quality") or "missing")
        statuses[pid] = dict(raw.get("summary_card_statuses") or {})
    report_file = path / "report.md"
    report = report_file.read_text(encoding="utf-8") if report_file.is_file() else str(state.get("final_report") or "")
    return Run(
        path=path, meta={**meta, "status": state.get("status", meta.get("status"))}, paper_ids=ids,
        notes=notes, summary_quality=quality, card_statuses=statuses, report=report,
        sources=load_sources(path / "source_text"),
        arxiv_ids={str(p["paper_id"]): str(p.get("arxiv_id") or _arxiv_id(str(p["paper_id"]))) for p in papers},
    )


def load_run(path: Path) -> Run:
    """Load a run_scenario output (has state.json) or a frozen baseline layout."""

    return load_candidate(path) if (path / "state.json").is_file() else load_baseline(path)


# --- Markdown helpers ------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks (BibTeX etc.) so headings/numbers inside are ignored."""

    return re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL)


def split_sections(report: str) -> dict[int, str]:
    """Map section number 1..5 to its body text (code fences removed)."""

    text = strip_code_fences(report)
    matches = list(re.finditer(r"^##\s+(\d+)\s*[.)].*$", text, flags=re.MULTILINE))
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.setdefault(int(match.group(1)), text[match.end():end])
    return sections


def paper_blocks(section2: str) -> list[str]:
    """Split section 2 into one block per ``###`` heading (heading line included)."""

    parts = re.split(r"(?m)^(?=###\s)", section2)
    return [part for part in parts if part.startswith("###")]


def count_display_equations(text: str) -> int:
    return len(re.findall(r"\$\$", text)) // 2


def has_table(text: str) -> bool:
    return bool(re.search(r"^\s*\|.*\|\s*\n\s*\|[\s:|-]*-[\s:|-]*\|", text, flags=re.MULTILINE))


# --- HARD checks -----------------------------------------------------------


def check_status(run: Run) -> Check:
    status = str(run.meta.get("status", "unknown"))
    return Check("status_not_error", "HARD", "FAIL" if status == "error" else "PASS", f"status={status}")


def check_paper_ids(baseline: Run, candidate: Run) -> Check:
    base, cand = set(baseline.paper_ids), set(candidate.paper_ids)
    if base == cand:
        return Check("same_paper_ids", "HARD", "PASS", f"{len(base)} papers")
    return Check("same_paper_ids", "HARD", "FAIL",
                 f"missing={sorted(base - cand)} extra={sorted(cand - base)}")


def check_notes_nonempty(run: Run) -> Check:
    bad = [
        f"{pid}.{name}"
        for pid in run.paper_ids for name in PMRL_FIELDS
        if not (run.notes.get(pid, {}).get(name) or "").strip()
    ]
    return Check("pmrl_notes_nonempty", "HARD", "FAIL" if bad else "PASS", f"empty={bad}" if bad else "all P/M/R/L set")


def check_report_structure(report: str, n_papers: int) -> Check:
    text = strip_code_fences(report)
    problems: list[str] = []
    if not re.search(r"^#\s+\S", text, flags=re.MULTILINE):
        problems.append("no H1")
    numbers = [int(m.group(1)) for m in SECTION_HEADING_RE.finditer(text)]
    if numbers != [1, 2, 3, 4]:
        problems.append(f"section headings {numbers} != [1,2,3,4]")
    section2 = split_sections(report).get(2, "")
    found = len(paper_blocks(section2))
    if found < n_papers:
        problems.append(f"### headings in section 2: {found} < {n_papers} papers")
    return Check("report_structure", "HARD", "FAIL" if problems else "PASS", "; ".join(problems) or f"H1, ## 1-4, {found} paper headings")


# "Key results" sub-bullets every paper block must carry (Vietnamese or English label).
RESULT_SUB_BULLETS = {
    "Insight": r"insights?",
    "Metrics": r"đại lượng đo|metrics?",
    "Setup": r"thiết lập|setup",
}


def check_results_format(report: str) -> Check:
    """Every paper block has the Insight, Metrics and Setup sub-bullets."""

    blocks = paper_blocks(split_sections(report).get(2, ""))
    if not blocks:
        return Check("results_format", "HARD", "FAIL", "no paper blocks in section 2")
    missing = [
        f"{block.splitlines()[0][:40]}: {name}"
        for block in blocks
        for name, pattern in RESULT_SUB_BULLETS.items()
        if not re.search(rf"(?mi)^[ \t]*[-*•][ \t]*\**[ \t]*(?:{pattern})[ \t]*\**[ \t]*:", block)
    ]
    return Check("results_format", "HARD", "FAIL" if missing else "PASS",
                 f"missing={missing[:4]}" if missing else f"{len(blocks)} blocks have Insight/Metrics/Setup")


def table_value_cells(block: str) -> list[str]:
    """Value cells (method and baseline columns) of the markdown tables in one paper block."""

    cells: list[str] = []
    in_table = False
    for line in block.splitlines():
        if not re.match(r"^\s*\|", line):
            in_table = False
            continue
        if re.match(r"^\s*\|[\s:|-]+\|\s*$", line):
            continue
        if not in_table:  # first row of a table is the header
            in_table = True
            continue
        row = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        for cell in row[2:]:
            # a baseline cell reads "value (baseline name)": the name is not a number claim
            cells.append(re.sub(r"(?<=\S)\s*\([^)]*\)\s*$", "", cell.strip()))
    return cells


def check_table_numbers(report: str, run: Run) -> Check:
    """Every number in a rendered results table occurs in that paper's source text."""

    blocks = paper_blocks(split_sections(report).get(2, ""))
    ids_by_paper = {pid: run.arxiv_ids.get(pid) or _arxiv_id(pid) for pid in run.paper_ids}
    bad: list[str] = []
    checked = 0
    for index, block in enumerate(blocks):
        cells = table_value_cells(block)
        if not cells:
            continue
        pid = next((p for p, a in ids_by_paper.items() if a and a in block), None) or (
            run.paper_ids[index] if index < len(run.paper_ids) else None)
        source = source_number_set({pid: run.sources[pid]}) if pid in run.sources else None
        if source is None:
            bad.append(f"{block.splitlines()[0][:40]}: no source text")
            continue
        for cell in cells:
            for match in _NUMBER_RE.finditer(cell):
                checked += 1
                if not number_variants(match.group(1), bool(match.group(2))) & source:
                    bad.append(f"{block.splitlines()[0][:30]}: {match.group(0).strip()}")
    return Check("results_table_numbers", "HARD", "FAIL" if bad else "PASS",
                 f"not in source: {bad[:5]}" if bad else f"{checked} table numbers found in source")


def check_arxiv_links(report: str, run: Run) -> Check:
    missing = [
        arxiv_id for arxiv_id in (run.arxiv_ids.get(pid) or _arxiv_id(pid) for pid in run.paper_ids)
        if arxiv_id and f"https://arxiv.org/abs/{arxiv_id}" not in report
    ]
    return Check("arxiv_links", "HARD", "FAIL" if missing else "PASS", f"missing={missing}" if missing else "all links present")


_INTERNAL_LINE_RE = re.compile(
    r"^\s*(?:synthesis requested|analysis scope:|language mode:|pmrl quality:|"
    r"delivered in .* per language mode|agent (?:status|self-review))\b"
    r"|\bPMRL\s+(?:quality|status)\b|\blanguage[- ]mode\b"
    r"|^\W*pipeline\s*[:=]|\bpipeline[_ ](?:version|status|error|stage)\b|\bSUMMARY_PIPELINE\w*"
    r"|\b(?:report|pmrl)_[0-9a-f]{8,}\b|\breport_[\w-]*\.md\b|\bsummary_quality\b"
    r"|\bnumber_not_in_source\b|\bcard_status(?:es)?\b",
    re.IGNORECASE,
)


def is_internal_line(line: str) -> bool:
    return bool(_INTERNAL_LINE_RE.search(line.lstrip(" -*•>_\t").strip()))


def check_no_internal_lines(report: str) -> Check:
    hits = [line.strip()[:80] for line in strip_code_fences(report).splitlines() if is_internal_line(line)]
    return Check("no_internal_lines", "HARD", "FAIL" if hits else "PASS", f"{len(hits)} hits: {hits[:3]}" if hits else "none")


# --- Number grounding ------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?))(\s*%)?")


def _canon(value: Decimal) -> str:
    return format(value.normalize(), "f")


def number_variants(token: str, percent: bool = False) -> set[str]:
    """Canonical readings of one number token (decimal comma/point, thousands)."""

    token = token.lstrip("+-")
    readings = {token.replace(",", ".")}
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        readings.add(token.replace(",", ""))
    if re.fullmatch(r"\d{1,3}\.\d{3}", token):
        readings.add(token.replace(".", ""))
    variants: set[str] = set()
    for reading in readings:
        try:
            value = Decimal(reading)
        except InvalidOperation:
            continue
        variants.add(_canon(value))
        if percent:
            variants.add(_canon(value / 100))
    return variants


def _scrub_for_numbers(text: str) -> str:
    text = strip_code_fences(text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]*\$", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", " ", text)  # arXiv ids
    text = re.sub(r"(?m)^\s*(?:#+\s*)?\d+[.)]\s+", "", text)  # list/heading numbering
    return text


def extract_report_numbers(report: str) -> list[tuple[str, set[str]]]:
    """Numbers worth grounding: skip bare integers below 10 (they match anything)."""

    result: list[tuple[str, set[str]]] = []
    for match in _NUMBER_RE.finditer(_scrub_for_numbers(report)):
        token, percent = match.group(1), bool(match.group(2))
        if re.fullmatch(r"\d", token) and not percent:
            continue
        result.append((token, number_variants(token, percent)))
    return result


def source_number_set(sources: dict[str, str]) -> set[str]:
    numbers: set[str] = set()
    for text in sources.values():
        for match in _NUMBER_RE.finditer(text):
            numbers |= number_variants(match.group(1))
    return numbers


def grounding_ratio(report: str, sources: dict[str, str]) -> tuple[float, int]:
    """Fraction of report numbers found in the union of source texts."""

    claims = extract_report_numbers(report)
    if not claims:
        return 1.0, 0
    source = source_number_set(sources)
    grounded = sum(1 for _, variants in claims if variants & source)
    return grounded / len(claims), len(claims)


# --- QUALITY checks --------------------------------------------------------


def _quality_status(ok: bool, strict: bool) -> str:
    return "PASS" if ok else ("FAIL" if strict else "WARN")


def _rank(value: str | None) -> int:
    return QUALITY_RANK.get(str(value or "").lower(), 0)


def check_summary_quality(baseline: Run, candidate: Run, strict: bool = False) -> Check:
    worse = [
        f"{pid}: {baseline.summary_quality.get(pid)}->{candidate.summary_quality.get(pid)}"
        for pid in baseline.paper_ids
        if _rank(candidate.summary_quality.get(pid)) < _rank(baseline.summary_quality.get(pid))
    ]
    return Check("summary_quality_rank", "QUALITY", _quality_status(not worse, strict), "; ".join(worse) or "not worse than baseline")


def count_complete_cards(run: Run) -> int:
    return sum(1 for statuses in run.card_statuses.values() for s in statuses.values() if s == "complete")


def check_card_statuses(baseline: Run, candidate: Run, strict: bool = False) -> Check:
    base, cand = count_complete_cards(baseline), count_complete_cards(candidate)
    return Check("complete_cards", "QUALITY", _quality_status(cand >= base, strict), f"candidate={cand} baseline={base}")


def word_count(text: str) -> int:
    return len(text.split())


def check_pmrl_word_counts(baseline: Run, candidate: Run, th: Thresholds, strict: bool = False) -> Check:
    bad: list[str] = []
    for pid in baseline.paper_ids:
        for name in PMRL_FIELDS:
            base = word_count(baseline.notes.get(pid, {}).get(name, ""))
            cand = word_count(candidate.notes.get(pid, {}).get(name, ""))
            if base and not th.word_ratio_lo <= cand / base <= th.word_ratio_hi:
                bad.append(f"{pid}.{name} {cand}/{base}={cand / base:.2f}x")
    return Check("pmrl_word_counts", "QUALITY", _quality_status(not bad, strict),
                 "; ".join(bad[:6]) or f"all within {th.word_ratio_lo}x-{th.word_ratio_hi}x")


def check_report_length(baseline: Run, candidate: Run, th: Thresholds, strict: bool = False) -> Check:
    base, cand = len(baseline.report), len(candidate.report)
    ratio = cand / base if base else 0.0
    ok = th.report_ratio_lo <= ratio <= th.report_ratio_hi
    return Check("report_length", "QUALITY", _quality_status(ok, strict), f"{cand}/{base} chars = {ratio:.2f}x")


def check_grounding(baseline: Run, candidate: Run, th: Thresholds, strict: bool = False) -> Check:
    base, base_n = grounding_ratio(baseline.report, baseline.sources)
    cand, cand_n = grounding_ratio(candidate.report, candidate.sources)
    ok = cand >= base - th.grounding_drop
    return Check("number_grounding", "QUALITY", _quality_status(ok, strict),
                 f"candidate={cand:.3f} ({cand_n} numbers) baseline={base:.3f} ({base_n}) allowed drop={th.grounding_drop}")


def check_equations(report: str, n_papers: int, th: Thresholds, strict: bool = False) -> Check:
    blocks = paper_blocks(split_sections(report).get(2, ""))
    counts = [count_display_equations(block) for block in blocks]
    if not counts:
        counts = [count_display_equations(report)]
        limit = th.max_equations_per_paper * max(n_papers, 1)
    else:
        limit = th.max_equations_per_paper
    ok = all(c <= limit for c in counts)
    return Check("display_equations", "QUALITY", _quality_status(ok, strict), f"per paper={counts} limit={limit}")


# --- LATENCY ---------------------------------------------------------------


def longest_llm_call(meta: dict[str, Any]) -> float:
    return max((float(c.get("seconds", 0)) for c in meta.get("llm_calls", [])), default=0.0)


def latency_checks(baseline: Run, candidate: Run, th: Thresholds) -> list[Check]:
    base_total = float(baseline.meta.get("total_seconds", 0) or 0)
    cand_total = float(candidate.meta.get("total_seconds", 0) or 0)
    speedup = base_total / cand_total if cand_total else 0.0
    checks = [
        Check("total_seconds", "LATENCY", "INFO", f"candidate={cand_total:.1f}s baseline={base_total:.1f}s speedup={speedup:.2f}x"),
        Check("longest_llm_call", "LATENCY", "INFO",
              f"candidate={longest_llm_call(candidate.meta):.1f}s baseline={longest_llm_call(baseline.meta):.1f}s"),
    ]
    base_nodes = baseline.meta.get("node_wall_seconds", {})
    cand_nodes = candidate.meta.get("node_wall_seconds", {})
    for node in dict.fromkeys([*base_nodes, *cand_nodes]):
        checks.append(Check(f"node:{node}", "LATENCY", "INFO",
                            f"candidate={float(cand_nodes.get(node, 0)):.1f}s baseline={float(base_nodes.get(node, 0)):.1f}s"))
    ok = 0 < cand_total <= th.target_seconds
    checks.append(Check("target_seconds", "LATENCY", "PASS" if ok else "FAIL",
                        f"{cand_total:.1f}s vs target {th.target_seconds:.0f}s"))
    return checks


# --- Orchestration ---------------------------------------------------------


def verify(baseline: Run, candidate: Run, th: Thresholds | None = None, *, strict: bool = False) -> list[Check]:
    th = th or Thresholds()
    n = len(baseline.paper_ids)
    checks = [
        check_status(candidate),
        check_paper_ids(baseline, candidate),
        check_notes_nonempty(candidate),
        check_report_structure(candidate.report, len(candidate.paper_ids)),
        check_results_format(candidate.report),
        check_table_numbers(candidate.report, candidate),
        check_arxiv_links(candidate.report, candidate),
        check_no_internal_lines(candidate.report),
        check_summary_quality(baseline, candidate, strict),
        check_card_statuses(baseline, candidate, strict),
        check_pmrl_word_counts(baseline, candidate, th, strict),
        check_report_length(baseline, candidate, th, strict),
        check_grounding(baseline, candidate, th, strict),
        check_equations(candidate.report, n, th, strict),
    ]
    return checks + latency_checks(baseline, candidate, th)


def exit_code(checks: list[Check], strict_latency: bool = False) -> int:
    """1 on any FAIL; LATENCY failures count only under ``strict_latency``."""

    return 1 if any(c.status == "FAIL" and (c.group != "LATENCY" or strict_latency) for c in checks) else 0


def format_table(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"{'GROUP':<8} {'STATUS':<6} {'CHECK':<{width}}  DETAIL"]
    lines += [f"{c.group:<8} {c.status:<6} {c.name:<{width}}  {c.detail}" for c in checks]
    return "\n".join(lines)


def default_output(candidate: Path) -> Path:
    if (candidate / "state.json").is_file():
        return candidate / "verify.json"
    return ROOT / "bench" / "runs" / f"{candidate.name}_verify.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="verify.json path (default: <candidate>/verify.json for runs)")
    parser.add_argument("--strict", action="store_true", help="QUALITY warnings become HARD failures")
    parser.add_argument("--strict-latency", action="store_true", help="a FAIL on --target-seconds makes the exit code 1")
    defaults = Thresholds()
    parser.add_argument("--target-seconds", type=float, default=defaults.target_seconds)
    parser.add_argument("--word-ratio", type=float, nargs=2, metavar=("LO", "HI"), default=[defaults.word_ratio_lo, defaults.word_ratio_hi])
    parser.add_argument("--report-ratio", type=float, nargs=2, metavar=("LO", "HI"), default=[defaults.report_ratio_lo, defaults.report_ratio_hi])
    parser.add_argument("--grounding-drop", type=float, default=defaults.grounding_drop)
    parser.add_argument("--max-equations", type=int, default=defaults.max_equations_per_paper)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    th = Thresholds(
        word_ratio_lo=args.word_ratio[0], word_ratio_hi=args.word_ratio[1],
        report_ratio_lo=args.report_ratio[0], report_ratio_hi=args.report_ratio[1],
        grounding_drop=args.grounding_drop, max_equations_per_paper=args.max_equations,
        target_seconds=args.target_seconds,
    )
    checks = verify(load_run(args.baseline), load_run(args.candidate), th, strict=args.strict)
    print(format_table(checks))
    code = exit_code(checks, args.strict_latency)
    out = args.out or default_output(args.candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exit_code": code, "thresholds": asdict(th), "checks": [asdict(c) for c in checks]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresult: {'FAIL' if code else 'PASS'} (verify.json: {out})")
    return code


if __name__ == "__main__":
    sys.exit(main())
