"""Per-stage LLM micro-benchmark: replay ONE pipeline stage with FIXED inputs.

Compares the current prompts ("v0": exact strings captured from today's code)
with a future prompt rewrite ("new": system+user builders).  Tokens are the
primary metric, because latency is driven by output + reasoning tokens and the
reasoning length varies several-fold between identical requests.

Stages: pmrl, section, overview.  Fixed inputs live in
``bench/prompt_variants/v0/inputs/`` (tracked), the captured v0 prompt strings
in ``bench/prompt_variants/v0/rendered/<stage>__<case>.json``.

Usage::

    # 1. (one time) copy inputs from two saved runs, then capture v0 prompts
    .venv/bin/python bench/prompt_bench.py --import-inputs --capture-v0

    # 2. benchmark (real provider calls)
    .venv/bin/python bench/prompt_bench.py --variants v0,new \
        --stages pmrl,section,overview --cases all --n 3 --concurrency 8 \
        --out bench/runs/prompt_bench_<ts>/

    # no provider calls: fake LLM with synthetic token counts
    .venv/bin/python bench/prompt_bench.py --variants v0 --dry-run --n 3

Variant "new" calls builders that must exist in the nodes (reported as
unavailable when missing; the v0 run is never affected):

* ``nodes.write_notes.build_pmrl_messages(paper) -> (system, user)``
* ``nodes.final_report.build_section_messages(paper, index, user_query, language_mode)``
* ``nodes.final_report.build_overview_messages(papers, user_query, title,
  language_mode)``

The structured stage (pmrl) goes through
``tools.llm_provider.invoke_structured_output(user, schema, system=system, ...)``;
text stages send ``[SystemMessage(system), HumanMessage(user)]``.  No retries:
a failed call is recorded as failed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from bench import verify  # noqa: E402
from bench.run_scenario import prompt_chars  # noqa: E402
from config import config, resolved_provider  # noqa: E402
from nodes.write_notes import PMRLCoreGeneration  # noqa: E402
from state import OverviewMulti, OverviewSingle, PaperItem  # noqa: E402
from text_utils import detect_response_language  # noqa: E402
from tools import llm_provider  # noqa: E402

VARIANTS_DIR = ROOT / "bench" / "prompt_variants"
STAGES = ("pmrl", "section", "overview")
STRUCTURED_SCHEMAS = {"pmrl": PMRLCoreGeneration}  # v0 (frozen) structured stages
# Variant "new": the overview call also returns JSON (OverviewSingle / OverviewMulti).
NEW_STRUCTURED_STAGES = ("pmrl", "overview")


def _new_schema(stage: str, run: "RunInput") -> Any:
    if stage == "overview":
        return OverviewSingle if len(run.papers) == 1 else OverviewMulti
    return STRUCTURED_SCHEMAS[stage]
KNOWN_VARIANTS = ("v0", "new")
# Saved runs the fixed inputs come from (name -> directory under bench/runs/).
DEFAULT_RUNS = {"single": "20261010_180106_single_v2", "multi": "20261010_180642_parallel_v2"}
# stage -> (module, builder name) for variant "new".
BUILDERS = {
    "pmrl": ("nodes.write_notes", "build_pmrl_messages"),
    "section": ("nodes.final_report", "build_section_messages"),
    "overview": ("nodes.final_report", "build_overview_messages"),
}
TOKEN_METRICS = ("output_tokens", "reasoning_tokens", "seconds", "input_tokens")
PMRL_FIELDS = verify.PMRL_FIELDS


# --- Fixed inputs ------------------------------------------------------------


@dataclass
class RunInput:
    """One saved run: the papers (with notes and source text) and report context."""

    name: str
    user_query: str
    language_mode: str
    title: str
    papers: list[PaperItem]
    sources: dict[str, str]  # paper_id -> source text (verify.load_sources layout)


@dataclass(frozen=True)
class Case:
    stage: str
    id: str  # unique per stage; overview cases use the run name
    run: str
    index: int = 0  # 1-based paper index inside the run (0 for overview)
    paper_id: str = ""


@dataclass(frozen=True)
class Task:
    variant: str
    case: Case
    rep: int

    @property
    def stage(self) -> str:
        return self.case.stage


def restore_paper(raw: dict[str, Any], source: dict[str, Any] | None) -> PaperItem:
    """Rebuild a PaperItem from state.json plus its source_text file."""

    data = {key: value for key, value in raw.items() if key not in ("extracted_text_chars", "sections_chars")}
    if source:
        data["extracted_text"] = source.get("extracted_text")
        data["sections"] = source.get("sections") or {}
        data["source_quality"] = source.get("source_quality") or data.get("source_quality") or {}
    return PaperItem(**data)


def load_run_input(directory: Path, name: str) -> RunInput:
    from nodes.final_report import report_title  # lazy: heavy import

    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    source_dir = directory / "source_text"
    papers: list[PaperItem] = []
    for raw in state["selected_papers"]:
        source_path = source_dir / f"{raw['paper_id']}.json"
        source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
        papers.append(restore_paper(raw, source))
    user_query = state.get("user_query") or "Research Analysis"
    # Mirrors final_report_node: language_mode and title.
    return RunInput(
        name=name,
        user_query=user_query,
        language_mode=detect_response_language(user_query),
        title=report_title(user_query, papers),
        papers=papers,
        sources=verify.load_sources(source_dir),
    )


def load_inputs(inputs_dir: Path) -> dict[str, RunInput]:
    runs: dict[str, RunInput] = {}
    for sub in sorted(p for p in inputs_dir.iterdir() if p.is_dir() and (p / "state.json").exists()):
        runs[sub.name] = load_run_input(sub, sub.name)
    return runs


def import_inputs(run_dirs: dict[str, Path], inputs_dir: Path) -> list[Path]:
    """Copy the files a case needs from saved runs into the tracked inputs dir.

    ``state.json`` is trimmed to query, intent and selected papers
    (the report text, timings and logs are not inputs).  Local file
    paths are blanked.
    """

    written: list[Path] = []
    for name, run_dir in run_dirs.items():
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        target = inputs_dir / name
        (target / "source_text").mkdir(parents=True, exist_ok=True)
        papers = []
        for paper in state["selected_papers"]:
            paper = dict(paper)
            paper["local_pdf_path"] = None
            papers.append(paper)
            source = json.loads((run_dir / "source_text" / f"{paper['paper_id']}.json").read_text(encoding="utf-8"))
            source.pop("local_pdf_path", None)
            path = target / "source_text" / f"{paper['paper_id']}.json"
            path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written.append(path)
        trimmed = {
            "source_run": run_dir.name,
            "user_query": state.get("user_query"),
            "intent": state.get("intent"),
            "selected_papers": papers,
        }
        path = target / "state.json"
        path.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def list_cases(runs: dict[str, RunInput], stages: tuple[str, ...] = STAGES) -> list[Case]:
    cases: list[Case] = []
    for stage in stages:
        for run in runs.values():
            if stage == "overview":
                cases.append(Case(stage, run.name, run.name))
                continue
            for index, paper in enumerate(run.papers, 1):
                case_id = f"{run.name}_{paper.paper_id.removeprefix('arxiv_')}"
                cases.append(Case(stage, case_id, run.name, index, paper.paper_id))
    return cases


def select_cases(cases: list[Case], selector: str) -> list[Case]:
    """``all`` or a comma list of case ids / run names."""

    tokens = {token.strip() for token in selector.split(",") if token.strip()}
    if not tokens or "all" in tokens:
        return cases
    return [case for case in cases if case.id in tokens or case.run in tokens]


# --- Step 1: capture the exact v0 prompts -----------------------------------


class CaptureLLM:
    """Fake chat model that records the strings sent to ``invoke``."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[Any] = []
        self.bound: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> "CaptureLLM":
        self.bound.append(kwargs)
        return self

    def with_structured_output(self, schema: Any) -> "CaptureLLM":
        return self

    def invoke(self, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(input)
        return SimpleNamespace(content=self.content)


_CAPTURE_RESPONSES = {
    "pmrl": json.dumps({
        "problem": "p ok", "method": "m ok", "result": "r ok", "limitation": "l ok",
    }),
    "section": "### Title\n- **Problem**: text\n- **Methodology**: text\n",
    "overview": json.dumps({"tldr": "t", "takeaway": "k"}),
}


@contextlib.contextmanager
def _capture_env() -> Iterator[None]:
    """Force the zai request shape and a throwaway PMRL cache directory."""

    old_provider, old_cache = config.DEFAULT_PROVIDER, config.SUMMARY_CACHE_DIR
    with tempfile.TemporaryDirectory(prefix="bench_capture_") as tmp:
        config.DEFAULT_PROVIDER = "zai"
        config.SUMMARY_CACHE_DIR = Path(tmp)
        try:
            yield
        finally:
            config.DEFAULT_PROVIDER, config.SUMMARY_CACHE_DIR = old_provider, old_cache


def _capture_one(case: Case, run: RunInput) -> str:
    """Call the production function for ``case`` with a capturing LLM; return its prompt."""

    from nodes import final_report as fr
    from nodes import write_notes as wn

    llm = CaptureLLM(_CAPTURE_RESPONSES[case.stage])
    paper = copy.deepcopy(run.papers[case.index - 1]) if case.index else None
    if case.stage == "pmrl":
        _, status, _ = wn.analyze_paper_notes(paper, llm=llm)
        if status != "ok":
            raise RuntimeError(f"capture failed for {case.stage}/{case.id}: {status}")
        if not llm.prompts[0].startswith(wn._notes_prompt(paper)):
            raise RuntimeError(f"captured pmrl prompt does not start with _notes_prompt: {case.id}")
    elif case.stage == "section":
        fr._write_paper_section(llm, paper, case.index, run.user_query, run.language_mode)
    else:
        fr._write_overview(
            llm, run.user_query, run.title, run.language_mode, copy.deepcopy(run.papers)
        )
    if len(llm.prompts) != 1 or not isinstance(llm.prompts[0], str) or not llm.prompts[0]:
        raise RuntimeError(f"expected exactly one non-empty string prompt for {case.stage}/{case.id}")
    return llm.prompts[0]


def render_v0(inputs_dir: Path, rendered_dir: Path) -> list[dict[str, Any]]:
    """Capture the exact strings today's code sends; write one JSON file per case."""

    runs = load_inputs(inputs_dir)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with _capture_env():
        for case in list_cases(runs):
            run = runs[case.run]
            structured = case.stage in STRUCTURED_SCHEMAS
            prompt = _capture_one(case, run)
            record = {
                "stage": case.stage,
                "case": case.id,
                "run": case.run,
                "paper_id": case.paper_id,
                "structured": structured,
                "schema": STRUCTURED_SCHEMAS[case.stage].__name__ if structured else None,
                "provider": "zai",
                "json_mode": structured,
                "prompt_chars": len(prompt),
                "prompt": prompt,
            }
            (rendered_dir / f"{case.stage}__{case.id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            results.append(record)
    return results


def load_rendered(rendered_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rendered: dict[tuple[str, str], dict[str, Any]] = {}
    if rendered_dir.is_dir():
        for path in sorted(rendered_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            rendered[(record["stage"], record["case"])] = record
    return rendered


# --- Instrumentation ---------------------------------------------------------

_TLS = threading.local()  # per worker thread: .calls (list of recorded LLM calls), .task


def _wrap_invoke(original: Callable[..., Any]) -> Callable[..., Any]:
    def timed(self: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response: Any = None
        error = ""
        try:
            response = original(self, input, *args, **kwargs)
            return response
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            usage = getattr(response, "usage_metadata", None) or {}
            details = usage.get("output_token_details") or {}
            calls = getattr(_TLS, "calls", None)
            if calls is not None:
                content = getattr(response, "content", "")
                calls.append({
                    "seconds": time.perf_counter() - start,
                    "prompt_chars": prompt_chars(input),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_tokens": details.get("reasoning"),
                    "text": llm_provider.extract_text_from_response(content) if content else "",
                    "error": error,
                })

    return timed


@contextlib.contextmanager
def instrument(*classes: type) -> Iterator[None]:
    """Patch ``invoke`` on each class to record timing, tokens and text per thread."""

    originals = [(cls, cls.invoke) for cls in classes]
    for cls, original in originals:
        cls.invoke = _wrap_invoke(original)  # type: ignore[method-assign]
    try:
        yield
    finally:
        for cls, original in originals:
            cls.invoke = original  # type: ignore[method-assign]


# --- Backends ----------------------------------------------------------------


def _stable(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _words(seed: int, count: int) -> str:
    return " ".join(f"w{(seed + i) % 97}" for i in range(count))


_FAKE_BASE_TOKENS = {"pmrl": 2400, "section": 1600, "overview": 1800}


class FakeLLM:
    """Offline stand-in: valid output per stage, synthetic usage metadata.

    Reasoning varies 1.7x-3.5x between reps (as seen with the real provider);
    variant "new" is made ~15% shorter so the delta columns are exercised.
    """

    def __init__(self, bound: dict[str, Any] | None = None) -> None:
        self.bound = bound or {}

    def bind(self, **kwargs: Any) -> "FakeLLM":
        return FakeLLM({**self.bound, **kwargs})

    def with_structured_output(self, schema: Any) -> "FakeLLM":
        return self

    def invoke(self, input: Any = None, *args: Any, **kwargs: Any) -> AIMessage:
        task: Task | None = getattr(_TLS, "task", None)
        stage = task.stage if task else "section"
        key = (task.variant, stage, task.case.id, task.rep) if task else ("x", stage, "", 0)
        seed = _stable(*key)
        text = self._text(stage, seed, task)
        out_tokens = int(_FAKE_BASE_TOKENS[stage] * (1.7 + 1.8 * ((_stable(*key, "r") % 1000) / 1000)))
        if task and task.variant == "new":
            out_tokens = int(out_tokens * 0.85)
        usage = {
            "input_tokens": max(1, prompt_chars(input) // 4),
            "output_tokens": out_tokens,
            "total_tokens": 0,
            "output_token_details": {"reasoning": int(out_tokens * 0.8)},
        }
        return AIMessage(content=text, usage_metadata=usage)

    def _text(self, stage: str, seed: int, task: Task | None) -> str:
        if stage == "pmrl":
            return json.dumps({
                "problem": _words(seed, 60), "method": _words(seed + 1, 90), "result": _words(seed + 2, 70),
                "limitation": _words(seed + 3, 40),
            })
        if stage == "section":
            return f"### Paper\n- **Problem**: {_words(seed, 80)}\n\n$$\nx = y\n$$\n\n- **Results**: {_words(seed + 1, 80)}\n"
        return json.dumps({
            "tldr": _words(seed, 40), "takeaway": _words(seed + 1, 30),
            "keyResults": [_words(seed + 2, 12)], "insights": [{"finding": _words(seed + 3, 10), "why": _words(seed + 4, 12)}],
        })


class Backend:
    """Chooses between the real provider and the offline fake."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def get_llm(self, json_mode: bool = False) -> Any:
        if self.dry_run:
            return FakeLLM({"response_format": {"type": "json_object"}} if json_mode else {})
        return llm_provider.get_llm(json_mode=json_mode)

    def instrumented_classes(self) -> tuple[type, ...]:
        if self.dry_run:
            return (FakeLLM,)
        from langchain_openai import ChatOpenAI

        return (ChatOpenAI,)


# --- Variants: running one call ---------------------------------------------


def resolve_builder(stage: str) -> tuple[Callable[..., Any] | None, str]:
    module_name, attr = BUILDERS[stage]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return None, f"cannot import {module_name}: {type(exc).__name__}"
    builder = getattr(module, attr, None)
    if not callable(builder):
        return None, f"{module_name}.{attr} is missing"
    return builder, ""


def new_unavailable_reason(stage: str) -> str:
    """Return why variant "new" cannot run ``stage`` (empty string when it can)."""

    builder, reason = resolve_builder(stage)
    if builder is None:
        return reason
    if stage in NEW_STRUCTURED_STAGES:
        params = inspect.signature(llm_provider.invoke_structured_output).parameters
        if "system" not in params and not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return "tools.llm_provider.invoke_structured_output has no `system` keyword"
    return ""


def _builder_args(case: Case, run: RunInput) -> tuple[Any, ...]:
    papers = copy.deepcopy(run.papers)
    if case.stage == "overview":
        return (papers, run.user_query, run.title, run.language_mode)
    paper = papers[case.index - 1]
    if case.stage == "section":
        return (paper, case.index, run.user_query, run.language_mode)
    return (paper,)


def _call_v0(task: Task, rendered: dict[str, Any], backend: Backend) -> None:
    prompt = rendered["prompt"]
    if rendered["structured"]:
        llm = backend.get_llm(json_mode=True)
        model: Any = llm
        # Same JSON-mode bind as tools.llm_provider.invoke_structured_output.
        if hasattr(llm, "bind"):
            try:
                model = llm.bind(response_format={"type": "json_object"})
            except Exception:
                model = llm
        model.invoke(prompt)
    else:
        backend.get_llm().invoke(prompt)


def _call_new(task: Task, run: RunInput, backend: Backend) -> None:
    builder, reason = resolve_builder(task.stage)
    if builder is None:
        raise RuntimeError(reason)
    system, user = builder(*_builder_args(task.case, run))
    if task.stage in NEW_STRUCTURED_STAGES:
        llm_provider.invoke_structured_output(
            user,
            _new_schema(task.stage, run),
            system=system,
            llm=backend.get_llm(json_mode=True),
            provider=config.DEFAULT_PROVIDER,
        )
    else:
        backend.get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])


# --- Quality metrics ---------------------------------------------------------


def _parse_model(text: str, schema: Any) -> Any:
    return llm_provider._validate_model(llm_provider._decode_json_object(text), schema)


def quality_pmrl(text: str) -> dict[str, Any]:
    try:
        generated = _parse_model(text, PMRLCoreGeneration)
    except Exception as exc:
        return {"parse_ok": False, "parse_error": type(exc).__name__}
    words = {name: verify.word_count(getattr(generated, name, "") or "") for name in PMRL_FIELDS}
    return {
        "parse_ok": True,
        "words": words,
        "all_nonempty": all(count > 0 for count in words.values()),
    }


def quality_text(text: str, sources: dict[str, str]) -> dict[str, Any]:
    ratio, claims = verify.grounding_ratio(text, sources)
    return {
        "parse_ok": bool(text.strip()),
        "words": verify.word_count(text),
        "equations": verify.count_display_equations(text),
        "grounding": round(ratio, 4),
        "numeric_claims": claims,
    }


def quality_overview(text: str, sources: dict[str, str], n_papers: int) -> dict[str, Any]:
    """Overview quality.  JSON (variant "new") needs tldr and takeaway; legacy Markdown (v0) needs ``## 1`` and ``## 3|4``."""

    result = quality_text(text, sources)
    try:
        data = llm_provider._decode_json_object(text)
    except Exception:
        data = None
    if isinstance(data, dict):
        missing = [key for key in ("tldr", "takeaway") if not str(data.get(key) or "").strip()]
    else:
        missing = [
            name for name, pattern in (("summary", r"(?m)^##\s*1\b"), ("conclusion", r"(?m)^##\s*[34]\b"))
            if not re.search(pattern, text)
        ]
    result.update({"headings_ok": not missing, "missing_headings": missing})
    return result


def compute_quality(task: Task, text: str, run: RunInput) -> dict[str, Any]:
    if not text.strip():
        return {"parse_ok": False, "parse_error": "empty_response"}
    stage = task.stage
    if stage == "pmrl":
        return quality_pmrl(text)
    if stage == "section":
        return quality_text(text, {task.case.paper_id: run.sources.get(task.case.paper_id, "")})
    return quality_overview(text, run.sources, len(run.papers))


def execute(task: Task, runs: dict[str, RunInput], rendered: dict[tuple[str, str], dict[str, Any]], backend: Backend) -> dict[str, Any]:
    """Run one call and return its full record (never raises)."""

    run = runs[task.case.run]
    _TLS.calls = []
    _TLS.task = task
    error = ""
    try:
        if task.variant == "v0":
            record = rendered.get((task.stage, task.case.id))
            if record is None:
                raise RuntimeError("v0 rendered prompt is missing")
            _call_v0(task, record, backend)
        else:
            _call_new(task, run, backend)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    calls = list(_TLS.calls)
    _TLS.calls, _TLS.task = None, None

    def total(key: str) -> int | None:
        values = [c[key] for c in calls if c[key] is not None]
        return sum(values) if values else None

    text = calls[-1]["text"] if calls else ""
    if not error and calls and calls[-1]["error"]:
        error = calls[-1]["error"]
    seconds = round(sum(c["seconds"] for c in calls), 3) if calls else None
    return {
        "variant": task.variant,
        "stage": task.stage,
        "case": task.case.id,
        "rep": task.rep,
        "ok": bool(text.strip()) and not error,
        "error": error,
        "seconds": seconds,
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "prompt_chars": total("prompt_chars"),
        "llm_calls": len(calls),
        "response_text": text,
        "quality": compute_quality(task, text, run) if not (error and not text) else {
            "parse_ok": False, "parse_error": error.split(":")[0] or "no_response",
        },
    }


# --- Statistics --------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy default); q in [0, 100]."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _dist(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": _round(statistics.median(values)) if values else None,
        "p90": _round(percentile(values, 90)),
    }


def _mean(values: list[float]) -> float | None:
    return _round(sum(values) / len(values), 4) if values else None


def _quality_aggregate(stage: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    qualities = [r["quality"] for r in records]
    parsed = [q for q in qualities if q.get("parse_ok")]
    out: dict[str, Any] = {"parse_rate": _mean([1.0 if q.get("parse_ok") else 0.0 for q in qualities])}
    if stage == "pmrl":
        out["all_nonempty_rate"] = _mean([1.0 if q.get("all_nonempty") else 0.0 for q in parsed])
        for name in PMRL_FIELDS:
            out[f"words_{name}_median"] = _round(statistics.median([q["words"][name] for q in parsed])) if parsed else None
    else:
        out["words_median"] = _round(statistics.median([q["words"] for q in parsed])) if parsed else None
        out["equations_mean"] = _mean([float(q["equations"]) for q in parsed])
        out["grounding_mean"] = _mean([q["grounding"] for q in parsed])
        if stage == "overview":
            out["headings_ok_rate"] = _mean([1.0 if q.get("headings_ok") else 0.0 for q in parsed])
    return out


def _rep_spread(records: list[dict[str, Any]]) -> float | None:
    """Median over cases of max/min output tokens across reps (reasoning variability)."""

    by_case: dict[str, list[int]] = {}
    for record in records:
        if record.get("output_tokens"):
            by_case.setdefault(record["case"], []).append(record["output_tokens"])
    ratios = [max(v) / min(v) for v in by_case.values() if len(v) >= 2 and min(v) > 0]
    return _round(statistics.median(ratios)) if ratios else None


def _delta_pct(new: float | None, base: float | None) -> float | None:
    if new is None or not base:
        return None
    return round((new - base) / base * 100, 1)


def summarize(records: list[dict[str, Any]], unavailable: dict[str, dict[str, str]], meta: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in STAGES:
        stage_records = [r for r in records if r["stage"] == stage]
        if not stage_records:
            continue
        per_variant: dict[str, Any] = {}
        for variant in sorted({r["variant"] for r in stage_records}):
            rows = [r for r in stage_records if r["variant"] == variant]
            metrics = {
                metric: _dist([float(r[metric]) for r in rows if r.get(metric) is not None])
                for metric in TOKEN_METRICS
            }
            per_variant[variant] = {
                "n_calls": len(rows),
                "n_failed": sum(1 for r in rows if not r["ok"]),
                "metrics": metrics,
                "output_tokens_rep_spread": _rep_spread(rows),
                "quality": _quality_aggregate(stage, rows),
            }
        entry: dict[str, Any] = {"variants": per_variant}
        base = per_variant.get("v0")
        if base:
            delta: dict[str, Any] = {}
            for variant, data in per_variant.items():
                if variant == "v0":
                    continue
                pct = {
                    metric: {
                        stat: _delta_pct(data["metrics"][metric][stat], base["metrics"][metric][stat])
                        for stat in ("median", "p90")
                    }
                    for metric in TOKEN_METRICS
                }
                quality_pct: dict[str, Any] = {}
                quality_abs: dict[str, Any] = {}
                for key, value in data["quality"].items():
                    base_value = base["quality"].get(key)
                    if not isinstance(value, (int, float)) or not isinstance(base_value, (int, float)):
                        continue
                    if key.endswith("_rate") or key == "grounding_mean":
                        quality_abs[key] = round(value - base_value, 4)
                    else:
                        quality_pct[key] = _delta_pct(value, base_value)
                delta[variant] = {"pct": pct, "quality_pct": quality_pct, "quality_abs": quality_abs}
            entry["delta_vs_v0"] = delta
        stages[stage] = entry
    return {"meta": meta, "unavailable": unavailable, "stages": stages}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _signed(value: float | None, suffix: str = "%") -> str:
    return "-" if value is None else f"{value:+g}{suffix}"


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "# Prompt bench summary",
        "",
        f"- variants: {', '.join(meta['variants'])} | stages: {', '.join(meta['stages'])} | n={meta['n']} | "
        f"concurrency={meta['concurrency']} | dry_run={meta['dry_run']}",
        f"- provider/model: {meta.get('provider')} / {meta.get('model')} | created: {meta['created']}",
        f"- calls: {meta['calls']} ({meta['failed']} failed). Percentiles use linear interpolation; "
        "failed calls without usage are excluded from token statistics.",
    ]
    if meta["dry_run"]:
        lines.append("- DRY RUN: fake LLM with synthetic token counts. Numbers below are not provider measurements.")
    if summary["unavailable"]:
        lines += ["", "## Unavailable", ""]
        for variant, reasons in summary["unavailable"].items():
            for stage, reason in reasons.items():
                lines.append(f"- variant `{variant}`, stage `{stage}`: {reason}")
    for stage, entry in summary["stages"].items():
        lines += ["", f"## {stage}", ""]
        lines += [
            "| variant | ok/calls | out tok med | out tok p90 | reasoning med | reasoning p90 | sec med | sec p90 | in tok med | out spread (max/min) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for variant, data in entry["variants"].items():
            m = data["metrics"]
            lines.append(
                f"| {variant} | {data['n_calls'] - data['n_failed']}/{data['n_calls']} "
                f"| {_fmt(m['output_tokens']['median'])} | {_fmt(m['output_tokens']['p90'])} "
                f"| {_fmt(m['reasoning_tokens']['median'])} | {_fmt(m['reasoning_tokens']['p90'])} "
                f"| {_fmt(m['seconds']['median'])} | {_fmt(m['seconds']['p90'])} "
                f"| {_fmt(m['input_tokens']['median'])} | {_fmt(data['output_tokens_rep_spread'])} |"
            )
        for variant, delta in entry.get("delta_vs_v0", {}).items():
            p = delta["pct"]
            lines.append(
                f"| **{variant} vs v0** | | {_signed(p['output_tokens']['median'])} | {_signed(p['output_tokens']['p90'])} "
                f"| {_signed(p['reasoning_tokens']['median'])} | {_signed(p['reasoning_tokens']['p90'])} "
                f"| {_signed(p['seconds']['median'])} | {_signed(p['seconds']['p90'])} "
                f"| {_signed(p['input_tokens']['median'])} | |"
            )
        lines += ["", "Quality:", ""]
        for variant, data in entry["variants"].items():
            parts = ", ".join(f"{key}={_fmt(value)}" for key, value in data["quality"].items())
            lines.append(f"- {variant}: {parts}")
        for variant, delta in entry.get("delta_vs_v0", {}).items():
            parts = [f"{key}={_signed(value, '')}" for key, value in delta["quality_abs"].items()]
            parts += [f"{key}={_signed(value)}" for key, value in delta["quality_pct"].items()]
            lines.append(f"- {variant} vs v0 (abs for rates/grounding, % otherwise): {', '.join(parts) or '-'}")
    return "\n".join(lines) + "\n"


# --- Orchestration -----------------------------------------------------------


def run_bench(
    *,
    variants: list[str],
    stages: list[str],
    case_selector: str,
    n: int,
    concurrency: int,
    out_dir: Path,
    variants_dir: Path = VARIANTS_DIR,
    dry_run: bool = False,
    backend: Backend | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    runs = load_inputs(variants_dir / "v0" / "inputs")
    rendered = load_rendered(variants_dir / "v0" / "rendered")
    cases = select_cases(list_cases(runs, tuple(stages)), case_selector)
    backend = backend or Backend(dry_run=dry_run)

    unavailable: dict[str, dict[str, str]] = {}
    tasks: list[Task] = []
    for case in cases:
        reason = ""
        for variant in variants:
            if variant == "v0" and (case.stage, case.id) not in rendered:
                unavailable.setdefault("v0", {})[case.stage] = (
                    f"no rendered prompt for case {case.id} (run --capture-v0)"
                )
                continue
            if variant == "new":
                reason = new_unavailable_reason(case.stage)
                if reason:
                    unavailable.setdefault("new", {})[case.stage] = reason
                    continue
            tasks.extend(Task(variant, case, rep) for rep in range(1, n + 1))
    # Adjacent in time per (case, rep) so variants see similar provider load.
    tasks.sort(key=lambda t: (t.stage, t.case.id, t.rep, t.variant))
    for variant, reasons in unavailable.items():
        for stage, reason in reasons.items():
            log(f"[unavailable] variant {variant}, stage {stage}: {reason}")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    with instrument(*backend.instrumented_classes()):
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(execute, task, runs, rendered, backend): task for task in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                log(f"[{done}/{len(tasks)}] {record['variant']} {record['stage']} {record['case']} "
                    f"r{record['rep']} {'ok' if record['ok'] else 'FAIL ' + record['error']}")
    records.sort(key=lambda r: (r["stage"], r["case"], r["rep"], r["variant"]))

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["response_text"]:
                suffix = "json" if record["stage"] in STRUCTURED_SCHEMAS else "md"
                name = f"{record['variant']}__{record['stage']}__{record['case']}__r{record['rep']}.{suffix}"
                (outputs_dir / name).write_text(record["response_text"], encoding="utf-8")

    meta = {
        "variants": variants,
        "stages": stages,
        "n": n,
        "concurrency": concurrency,
        "dry_run": dry_run,
        "provider": resolved_provider(),
        "model": config.DEFAULT_MODEL,
        "created": datetime.now().isoformat(timespec="seconds"),
        "calls": len(records),
        "failed": sum(1 for r in records if not r["ok"]),
    }
    summary = summarize(records, unavailable, meta)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


# --- CLI ---------------------------------------------------------------------


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="v0,new", help="comma list of: v0, new")
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma list of: {', '.join(STAGES)}")
    parser.add_argument("--cases", default="all", help="'all' or comma list of case ids / run names (single, multi)")
    parser.add_argument("--n", type=int, default=3, help="repetitions per variant and case")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", help="output dir (default bench/runs/prompt_bench_<ts>/)")
    parser.add_argument("--dry-run", action="store_true", help="fake LLM with synthetic tokens; no provider calls")
    parser.add_argument("--variants-dir", default=str(VARIANTS_DIR), help="dir holding v0/inputs and v0/rendered")
    parser.add_argument("--import-inputs", action="store_true",
                        help="copy fixed inputs from the saved runs into <variants-dir>/v0/inputs, then exit")
    parser.add_argument("--capture-v0", action="store_true",
                        help="render the exact v0 prompts from <variants-dir>/v0/inputs, then exit")
    parser.add_argument("--single-run", default=str(ROOT / "bench" / "runs" / DEFAULT_RUNS["single"]))
    parser.add_argument("--multi-run", default=str(ROOT / "bench" / "runs" / DEFAULT_RUNS["multi"]))
    args = parser.parse_args(argv)
    args.variants = _csv(args.variants)
    args.stages = _csv(args.stages)
    for variant in args.variants:
        if variant not in KNOWN_VARIANTS:
            parser.error(f"unknown variant {variant!r}; use {KNOWN_VARIANTS}")
    for stage in args.stages:
        if stage not in STAGES:
            parser.error(f"unknown stage {stage!r}; use {STAGES}")
    if args.n < 1:
        parser.error("--n must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants_dir = Path(args.variants_dir)
    if args.import_inputs or args.capture_v0:
        if args.import_inputs:
            written = import_inputs(
                {"single": Path(args.single_run), "multi": Path(args.multi_run)}, variants_dir / "v0" / "inputs"
            )
            print(f"imported {len(written)} input files into {variants_dir / 'v0' / 'inputs'}")
        if args.capture_v0:
            for record in render_v0(variants_dir / "v0" / "inputs", variants_dir / "v0" / "rendered"):
                print(f"{record['stage']}__{record['case']}: {record['prompt_chars']} chars")
        return 0
    out_dir = Path(args.out) if args.out else ROOT / "bench" / "runs" / f"prompt_bench_{datetime.now():%Y%m%d_%H%M%S}"
    summary = run_bench(
        variants=args.variants,
        stages=args.stages,
        case_selector=args.cases,
        n=args.n,
        concurrency=args.concurrency,
        out_dir=out_dir,
        variants_dir=variants_dir,
        dry_run=args.dry_run,
    )
    print(f"calls={summary['meta']['calls']} failed={summary['meta']['failed']} out={out_dir}")
    return 0 if summary["meta"]["calls"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
