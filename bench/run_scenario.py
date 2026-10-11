"""Run one full research-graph scenario with timing instrumentation.

The graph is treated as a black box: only ``graph.build_research_graph`` and
the public LangChain/requests entry points are touched.  Artifacts are written
in a layout that ``bench/verify.py`` can compare against a frozen baseline.

Usage::

    .venv/bin/python bench/run_scenario.py \
        --scenario bench/baselines/direct_compare_3papers_20261010/run.json \
        --label after_parallel
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DROPPED_PAPER_KEYS = ("extracted_text", "sections")


# --- Instrumentation -------------------------------------------------------


class CallRecorder:
    """Thread-safe list of timed calls, stamped relative to one start time."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.llm_calls: list[dict[str, Any]] = []
        self.http_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, kind: str, entry: dict[str, Any]) -> None:
        with self._lock:
            (self.llm_calls if kind == "llm" else self.http_calls).append(entry)


def prompt_chars(value: Any) -> int:
    """Best-effort character count of an LLM input (str, message list, PromptValue)."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(prompt_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(prompt_chars(item) for item in value.values())
    content = getattr(value, "content", None)
    if content is not None:
        return prompt_chars(content)
    to_messages = getattr(value, "to_messages", None)
    if callable(to_messages):
        try:
            return prompt_chars(to_messages())
        except Exception:
            return 0
    return len(str(value))


def _strip_query(url: str) -> str:
    return str(url).split("?", 1)[0][:200]


def _wrap_llm(original: Callable[..., Any], recorder: CallRecorder) -> Callable[..., Any]:
    def timed(self: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        start = recorder.now()
        response: Any = None
        try:
            response = original(self, input, *args, **kwargs)
            return response
        finally:
            usage = getattr(response, "usage_metadata", None) or {}
            output_details = usage.get("output_token_details") or {}
            recorder.add("llm", {
                "start_s": round(start, 2),
                "seconds": round(recorder.now() - start, 2),
                "prompt_chars": prompt_chars(input),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": output_details.get("reasoning"),
                "response_chars": prompt_chars(getattr(response, "content", "")),
                "thread": threading.current_thread().name,
            })

    return timed


def _wrap_http(original: Callable[..., Any], recorder: CallRecorder) -> Callable[..., Any]:
    def timed(self: Any, method: Any, url: Any = None, *args: Any, **kwargs: Any) -> Any:
        start = recorder.now()
        status: Any = "error"
        try:
            response = original(self, method, url, *args, **kwargs)
            status = getattr(response, "status_code", "ok")
            return response
        finally:
            recorder.add("http", {
                "method": str(method).upper(),
                "url": _strip_query(url if url is not None else kwargs.get("url", "")),
                "status": status,
                "start_s": round(start, 2),
                "seconds": round(recorder.now() - start, 2),
                "thread": threading.current_thread().name,
            })

    return timed


@contextmanager
def instrument(recorder: CallRecorder) -> Iterator[CallRecorder]:
    """Patch ``ChatOpenAI.invoke`` and ``requests.Session.request`` with timers."""

    import requests
    from langchain_openai import ChatOpenAI

    orig_llm = ChatOpenAI.invoke
    orig_http = requests.Session.request
    ChatOpenAI.invoke = _wrap_llm(orig_llm, recorder)  # type: ignore[method-assign]
    requests.Session.request = _wrap_http(orig_http, recorder)  # type: ignore[method-assign]
    recorder.t0 = time.perf_counter()  # exclude import time from the measured run
    try:
        yield recorder
    finally:
        ChatOpenAI.invoke = orig_llm  # type: ignore[method-assign]
        requests.Session.request = orig_http  # type: ignore[method-assign]


# --- Serialization ---------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Convert state values to JSON; drop paper bodies, keep their char lengths."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        result = {str(k): to_jsonable(v) for k, v in value.items()}
        if "paper_id" in result and any(key in result for key in DROPPED_PAPER_KEYS):
            text = result.pop("extracted_text", None)
            sections = result.pop("sections", None)
            result["extracted_text_chars"] = len(text or "")
            result["sections_chars"] = {
                str(k): len(v or "") if isinstance(v, str) else 0 for k, v in (sections or {}).items()
            }
        return result
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(dict(state))


def _paper_field(paper: Any, name: str, default: Any = None) -> Any:
    if isinstance(paper, dict):
        return paper.get(name, default)
    return getattr(paper, name, default)


def write_source_texts(papers: list[Any], text_cache_dir: Path, out_dir: Path) -> list[str]:
    """Copy each paper's TEXT cache json; synthesize it from state when absent."""

    target = out_dir / "source_text"
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for paper in papers:
        paper_id = str(_paper_field(paper, "paper_id", ""))
        if not paper_id:
            continue
        cached = text_cache_dir / f"{paper_id}.json"
        destination = target / f"{paper_id}.json"
        if cached.is_file():
            shutil.copyfile(cached, destination)
        else:
            payload = {
                "paper_id": paper_id,
                "title": _paper_field(paper, "title", ""),
                "sections": dict(_paper_field(paper, "sections", {}) or {}),
                "extracted_text": _paper_field(paper, "extracted_text", "") or "",
                "source_quality": dict(_paper_field(paper, "source_quality", {}) or {}),
            }
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(paper_id)
    return written


# --- Scenario execution ----------------------------------------------------


def load_scenario(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario:
        data = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
        return {
            "query": data.get("query", ""),
            "raw_inputs": list(data.get("raw_inputs", [])),
            "intent": data.get("intent"),
            "source": str(args.scenario),
        }
    return {"query": args.query or "", "raw_inputs": list(args.input or []), "intent": None, "source": "cli"}


def initial_state(query: str, raw_inputs: list[str]) -> dict[str, Any]:
    return {
        "user_query": query,
        "raw_inputs": raw_inputs,
        "retry_count": 0,
        "trace_logs": [],
        "error_logs": [],
        "node_timings": {},
        "conversation_context": "",
    }


def attribute_llm_nodes(llm_calls: list[dict[str, Any]], node_end_s: list[tuple[str, float]]) -> None:
    """Tag each LLM call with the first node whose completion follows the call."""

    for call in llm_calls:
        finished = call["start_s"] + call["seconds"]
        call["node"] = next((name for name, end in node_end_s if end + 0.05 >= finished), "unknown")


def run_graph(graph: Any, state: dict[str, Any], recorder: CallRecorder) -> tuple[dict[str, Any], dict[str, float], list[tuple[str, float]]]:
    """Stream the graph; return final state, per-node wall seconds and node end times."""

    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    node_wall: dict[str, float] = {}
    node_end_s: list[tuple[str, float]] = []
    previous = recorder.now()
    for update in graph.stream(state, cfg, stream_mode="updates"):
        now = recorder.now()
        for node in update:
            node_wall[node] = round(node_wall.get(node, 0.0) + (now - previous), 1)
            node_end_s.append((node, now))
            print(f"[{now:7.1f}s] {node}: {now - previous:6.1f}s", flush=True)
        previous = now
    return dict(graph.get_state(cfg).values), node_wall, node_end_s


def run_scenario(
    scenario: dict[str, Any],
    out_dir: Path,
    *,
    fresh_llm_cache: bool = True,
    label: str = "",
    graph: Any = None,
) -> dict[str, Any]:
    """Run the graph once and write ``run.json``, ``state.json``, ``report.md``, ``source_text/``."""

    from config import config

    out_dir.mkdir(parents=True, exist_ok=True)
    original_summary_dir = config.SUMMARY_CACHE_DIR
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if fresh_llm_cache:
        temp_dir = tempfile.TemporaryDirectory(prefix="bench_summaries_")
        config.SUMMARY_CACHE_DIR = Path(temp_dir.name)
    summary_dir = Path(config.SUMMARY_CACHE_DIR)
    try:
        if graph is None:
            from graph import build_research_graph

            graph = build_research_graph()
        recorder = CallRecorder()
        with instrument(recorder):
            state, node_wall, node_end_s = run_graph(
                graph, initial_state(scenario["query"], scenario["raw_inputs"]), recorder
            )
        total = round(recorder.now(), 1)
        attribute_llm_nodes(recorder.llm_calls, node_end_s)

        papers = list(state.get("selected_papers") or [])
        sources = write_source_texts(papers, Path(config.TEXT_CACHE_DIR), out_dir)
        cache_files: list[str] = []
        if summary_dir.is_dir():
            (out_dir / "cache").mkdir(exist_ok=True)
            for item in sorted(summary_dir.glob("*.json")):
                shutil.copyfile(item, out_dir / "cache" / item.name)
                cache_files.append(f"cache/{item.name}")

        report = str(state.get("final_report") or "")
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        (out_dir / "state.json").write_text(
            json.dumps(serialize_state(state), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run = {
            "description": label or "bench/run_scenario.py run",
            "intent": state.get("intent") or scenario.get("intent"),
            "query": scenario["query"],
            "raw_inputs": scenario["raw_inputs"],
            "scenario": scenario.get("source"),
            "fresh_llm_cache": fresh_llm_cache,
            "total_seconds": total,
            "status": state.get("status", "unknown"),
            "node_wall_seconds": node_wall,
            "node_timings": to_jsonable(state.get("node_timings") or {}),
            "llm_calls": recorder.llm_calls,
            "http_calls": recorder.http_calls,
            "artifacts": {
                "report_markdown": "report.md",
                "state": "state.json",
                "source_text_dir": "source_text",
                "source_text_papers": sources,
                "llm_cache_files": cache_files,
            },
        }
        (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return run
    finally:
        config.SUMMARY_CACHE_DIR = original_summary_dir
        if temp_dir is not None:
            temp_dir.cleanup()


# --- Dry run ---------------------------------------------------------------


class DryRunGraph:
    """Stub graph for ``--dry-run``: no network, no LLM; exercises the plumbing."""

    def __init__(self, raw_inputs: list[str]) -> None:
        self.raw_inputs = raw_inputs
        self._values: dict[str, Any] = {}

    def stream(self, state: dict[str, Any], cfg: dict[str, Any], stream_mode: str = "updates") -> Iterator[dict[str, Any]]:
        from state import PaperItem, PMRLNotes

        papers = []
        for index, raw in enumerate(self.raw_inputs or ["dry-run"]):
            arxiv_id = raw.rstrip("/").split("/")[-1] if "arxiv" in raw else f"0000.0000{index}"
            papers.append(PaperItem(
                paper_id=f"arxiv_{arxiv_id}", title=f"Dry run paper {index}", arxiv_id=arxiv_id,
                extracted_text="dry run text 1.5", sections={"abstract": "dry"},
                notes=PMRLNotes(problem="p", method="m", result="r", limitation="l"),
            ))
        yield {"router": {"intent": "direct_read"}}
        self._values = {**state, "intent": "direct_read", "status": "success", "selected_papers": papers,
                        "final_report": "# Dry run\n"}
        yield {"final_report": {"final_report": "# Dry run\n"}}

    def get_state(self, cfg: dict[str, Any]) -> Any:
        return type("Snapshot", (), {"values": self._values})()


# --- CLI -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="baseline run.json holding query and raw_inputs")
    parser.add_argument("--query", help="user query (when --scenario is absent)")
    parser.add_argument("--input", action="append", help="raw input (arXiv URL/id or PDF path); repeatable")
    parser.add_argument("--fresh-llm-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="use an empty temp SUMMARY_CACHE_DIR so PMRL/report caches miss (default on)")
    parser.add_argument("--label", default="run", help="label used in the default output directory name")
    parser.add_argument("--out", help="output dir (default bench/runs/<timestamp>_<label>/)")
    parser.add_argument("--dry-run", action="store_true", help="use a stub graph; no network, no LLM")
    args = parser.parse_args(argv)
    if not args.scenario and not args.query and not args.input:
        parser.error("give --scenario or --query/--input")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenario = load_scenario(args)
    out_dir = Path(args.out) if args.out else ROOT / "bench" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}"
    graph = DryRunGraph(scenario["raw_inputs"]) if args.dry_run else None
    run = run_scenario(scenario, out_dir, fresh_llm_cache=args.fresh_llm_cache, label=args.label, graph=graph)
    print(f"status={run['status']} total_seconds={run['total_seconds']} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
