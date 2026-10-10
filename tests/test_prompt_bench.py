"""Offline tests for bench/prompt_bench.py (fake LLM only; no provider calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from bench import prompt_bench as pb
from nodes import final_report as fr
from nodes import write_notes as wn
from state import PMRLGeneration
from tools import llm_provider

INPUTS_DIR = pb.VARIANTS_DIR / "v0" / "inputs"
RENDERED_DIR = pb.VARIANTS_DIR / "v0" / "rendered"


@pytest.fixture(scope="module")
def runs() -> dict[str, pb.RunInput]:
    return pb.load_inputs(INPUTS_DIR)


@pytest.fixture
def variants_dir(tmp_path: Path) -> Path:
    """A private copy of v0 inputs + freshly captured rendered prompts."""

    target = tmp_path / "variants"
    inputs = target / "v0" / "inputs"
    inputs.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(INPUTS_DIR, inputs)
    # v0 = the frozen one-string prompts committed under bench/.  Production now
    # sends [system, user] messages, so they can no longer be re-captured from
    # the live code (pb.render_v0 expects a single string).
    shutil.copytree(RENDERED_DIR, target / "v0" / "rendered")
    return target


def test_capture_renders_14_cases_with_non_empty_prompts(variants_dir: Path) -> None:
    files = sorted((variants_dir / "v0" / "rendered").glob("*.json"))
    assert len(files) == 14
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    counts: dict[str, int] = {}
    for record in records:
        assert record["prompt"].strip()
        assert record["prompt_chars"] == len(record["prompt"])
        assert record["structured"] == (record["stage"] in ("pmrl", "cards"))
        counts[record["stage"]] = counts.get(record["stage"], 0) + 1
    assert counts == {"pmrl": 4, "cards": 4, "section": 4, "overview": 2}
    assert {r["case"] for r in records if r["stage"] == "overview"} == {"single", "multi"}


def test_committed_rendered_dir_has_14_frozen_v0_prompts(runs: dict[str, pb.RunInput]) -> None:
    assert len(list(RENDERED_DIR.glob("*.json"))) == 14
    paper = runs["single"].papers[0]
    record = json.loads((RENDERED_DIR / f"pmrl__single_{paper.paper_id.removeprefix('arxiv_')}.json").read_text("utf-8"))
    # Frozen zai v0 request shape: one user string ending with the JSON Schema suffix.
    assert record["structured"] is True and record["prompt"].rstrip().endswith('"type":"object"}')
    assert "brief_summary" in record["prompt"]  # the v0 PMRL request still asked for it


def test_overview_cases_use_single_and_multi_prompts(runs: dict[str, pb.RunInput]) -> None:
    single = json.loads((RENDERED_DIR / "overview__single.json").read_text("utf-8"))["prompt"]
    multi = json.loads((RENDERED_DIR / "overview__multi.json").read_text("utf-8"))["prompt"]
    assert "BENCHMARK COMPARISON MATRIX" not in single
    assert "BENCHMARK COMPARISON MATRIX" in multi
    assert runs["multi"].benchmark_section.startswith(pb.BENCHMARK_HEADER)


class RecordingLLM:
    """Records every invoke input; replies with a valid stage-appropriate body."""

    log: list[tuple[Any, dict[str, Any]]] = []

    def __init__(self, bound: dict[str, Any] | None = None) -> None:
        self.bound = bound or {}

    def bind(self, **kwargs: Any) -> "RecordingLLM":
        return RecordingLLM({**self.bound, **kwargs})

    def invoke(self, input: Any = None, *args: Any, **kwargs: Any) -> AIMessage:
        RecordingLLM.log.append((input, dict(self.bound)))
        return pb.FakeLLM(self.bound).invoke(input)


class RecordingBackend(pb.Backend):
    def __init__(self) -> None:
        super().__init__(dry_run=True)

    def get_llm(self, json_mode: bool = False) -> Any:
        return RecordingLLM({"response_format": {"type": "json_object"}} if json_mode else {})

    def instrumented_classes(self) -> tuple[type, ...]:
        return (RecordingLLM,)


def test_v0_replay_sends_exact_captured_string(variants_dir: Path, tmp_path: Path) -> None:
    RecordingLLM.log = []
    summary = pb.run_bench(
        variants=["v0"], stages=list(pb.STAGES), case_selector="all", n=2, concurrency=4,
        out_dir=tmp_path / "out", variants_dir=variants_dir, backend=RecordingBackend(), log=lambda _: None,
    )
    rendered = pb.load_rendered(variants_dir / "v0" / "rendered")
    sent = [item[0] for item in RecordingLLM.log]
    assert len(sent) == 14 * 2
    assert all(isinstance(item, str) for item in sent)
    for record in rendered.values():
        assert sent.count(record["prompt"]) == 2
    for prompt, bound in RecordingLLM.log:
        structured = prompt.rstrip().endswith('"type":"object"}')
        assert ("response_format" in bound) == structured
    assert summary["meta"]["calls"] == 28 and summary["meta"]["failed"] == 0
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 28
    assert all(r["output_tokens"] and r["reasoning_tokens"] and r["response_text"] for r in rows)
    assert len(list((tmp_path / "out" / "outputs").iterdir())) == 28


def test_percentile_median_and_delta() -> None:
    assert pb.percentile([10, 20, 30], 50) == 20
    assert pb.percentile([10, 20, 30], 90) == 28
    assert pb.percentile([5], 90) == 5
    assert pb.percentile([], 90) is None
    assert pb._delta_pct(80, 100) == -20.0
    assert pb._delta_pct(None, 100) is None
    assert pb._delta_pct(5, 0) is None

    def rec(variant: str, rep: int, out: int, reason: int, secs: float) -> dict[str, Any]:
        return {
            "variant": variant, "stage": "section", "case": "c", "rep": rep, "ok": True, "error": "",
            "seconds": secs, "input_tokens": 100, "output_tokens": out, "reasoning_tokens": reason,
            "quality": {"parse_ok": True, "words": 100 if variant == "v0" else 50, "equations": 2,
                        "grounding": 0.9 if variant == "v0" else 0.8},
        }

    records = [rec("v0", 1, 100, 80, 2.0), rec("v0", 2, 200, 160, 4.0), rec("v0", 3, 300, 240, 6.0),
               rec("new", 1, 50, 40, 1.0), rec("new", 2, 100, 80, 2.0), rec("new", 3, 150, 120, 3.0)]
    summary = pb.summarize(records, {}, {"variants": ["v0", "new"], "stages": ["section"]})
    stage = summary["stages"]["section"]
    v0 = stage["variants"]["v0"]
    assert v0["metrics"]["output_tokens"] == {"n": 3, "median": 200.0, "p90": 280.0}
    assert v0["metrics"]["seconds"]["median"] == 4.0
    assert v0["output_tokens_rep_spread"] == 3.0
    delta = stage["delta_vs_v0"]["new"]
    assert delta["pct"]["output_tokens"] == {"median": -50.0, "p90": -50.0}
    assert delta["pct"]["reasoning_tokens"]["median"] == -50.0
    assert delta["quality_pct"]["words_median"] == -50.0
    assert delta["quality_abs"]["grounding_mean"] == pytest.approx(-0.1)


def test_missing_new_builders_reported_not_crashing(variants_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for module, name in pb.BUILDERS.values():
        monkeypatch.delattr(f"{module}.{name}", raising=False)
    summary = pb.run_bench(
        variants=["v0", "new"], stages=list(pb.STAGES), case_selector="all", n=1, concurrency=4,
        out_dir=tmp_path / "out", variants_dir=variants_dir, dry_run=True, log=lambda _: None,
    )
    assert set(summary["unavailable"]["new"]) == set(pb.STAGES)
    assert all("missing" in reason for reason in summary["unavailable"]["new"].values())
    assert summary["meta"]["calls"] == 14 and summary["meta"]["failed"] == 0
    assert all(set(entry["variants"]) == {"v0"} for entry in summary["stages"].values())
    markdown = (tmp_path / "out" / "summary.md").read_text("utf-8")
    assert "## Unavailable" in markdown and "build_pmrl_messages" in markdown


def test_new_without_system_kwarg_marks_structured_stages_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wn, "build_pmrl_messages", lambda paper: ("s", "u"), raising=False)
    monkeypatch.setattr(fr, "build_section_messages", lambda *a: ("s", "u"), raising=False)
    monkeypatch.setattr(
        llm_provider, "invoke_structured_output", lambda prompt, schema, *, llm=None, provider=None: None
    )
    assert "system" in pb.new_unavailable_reason("pmrl")
    assert pb.new_unavailable_reason("section") == ""


def test_new_variant_runs_through_builders(variants_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    def structured(user: str, schema: Any, *, system: str | None = None, llm: Any = None, provider: Any = None) -> Any:
        sent.append(("structured", system, user))
        return schema.model_validate(llm_provider._decode_json_object(llm.invoke([system, user])))

    monkeypatch.setattr(llm_provider, "invoke_structured_output", structured)
    monkeypatch.setattr(wn, "build_pmrl_messages", lambda paper: ("SYS-P", f"USER {paper.paper_id}"), raising=False)
    monkeypatch.setattr(fr, "build_cards_messages", lambda paper, lang: ("SYS-C", f"USER {lang}"), raising=False)
    monkeypatch.setattr(fr, "build_section_messages", lambda paper, i, q, lang: ("SYS-S", f"USER {i} {q}"), raising=False)
    monkeypatch.setattr(
        fr, "build_overview_messages", lambda papers, q, t, lang, b: ("SYS-O", f"USER {len(papers)} {t}"), raising=False
    )
    summary = pb.run_bench(
        variants=["v0", "new"], stages=list(pb.STAGES), case_selector="all", n=3, concurrency=4,
        out_dir=tmp_path / "out", variants_dir=variants_dir, dry_run=True, log=lambda _: None,
    )
    assert summary["unavailable"] == {}
    assert summary["meta"]["calls"] == 14 * 3 * 2 and summary["meta"]["failed"] == 0
    assert len(sent) == 8 * 3  # pmrl + cards, 4 cases each, 3 reps
    assert {s[1] for s in sent} == {"SYS-P", "SYS-C"}
    for entry in summary["stages"].values():
        assert entry["delta_vs_v0"]["new"]["pct"]["output_tokens"]["median"] is not None
    assert "new vs v0" in (tmp_path / "out" / "summary.md").read_text("utf-8")


def test_failed_call_is_recorded_not_retried(variants_dir: Path, tmp_path: Path) -> None:
    calls: list[Any] = []

    class Boom(RecordingLLM):
        def invoke(self, input: Any = None, *args: Any, **kwargs: Any) -> AIMessage:
            calls.append(input)
            raise TimeoutError("provider down")

    class BoomBackend(RecordingBackend):
        def get_llm(self, json_mode: bool = False) -> Any:
            return Boom()

        def instrumented_classes(self) -> tuple[type, ...]:
            return (Boom,)

    summary = pb.run_bench(
        variants=["v0"], stages=["section"], case_selector="single", n=1, concurrency=1,
        out_dir=tmp_path / "out", variants_dir=variants_dir, backend=BoomBackend(), log=lambda _: None,
    )
    assert len(calls) == 1  # no retry
    assert summary["meta"] == {**summary["meta"], "calls": 1, "failed": 1}
    row = json.loads((tmp_path / "out" / "results.jsonl").read_text("utf-8").splitlines()[0])
    assert row["ok"] is False and "TimeoutError" in row["error"]


def test_quality_uses_production_card_validation(runs: dict[str, pb.RunInput]) -> None:
    paper = runs["single"].papers[0]
    saved = paper.notes.summary_cards  # cards stored by the saved run
    quality = pb.quality_cards(json.dumps(saved.model_dump()), paper)
    assert quality["parse_ok"] is True
    assert set(quality["card_statuses"]) == set(wn.SUMMARY_CARD_FIELDS)
    assert quality["status"] in {"complete", "partial", "invalid", "missing"}
    assert pb.quality_cards("not json", paper) == {"parse_ok": False, "parse_error": "ValueError"}


def test_quality_text_and_overview_headings(runs: dict[str, pb.RunInput]) -> None:
    text = "## 1. S\nvalue 99999.5 here\n\n$$\nx\n$$\n\n## 4. K\nend"
    single = pb.quality_overview(text, runs["single"].sources, 1)
    assert single["headings_ok"] is True and single["equations"] == 1 and single["numeric_claims"] == 1
    assert single["grounding"] == 0.0
    multi = pb.quality_overview(text, runs["multi"].sources, 3)
    assert multi["headings_ok"] is False and multi["missing_headings"] == ["compare"]
    pmrl = pb.quality_pmrl(json.dumps({"problem": "a b", "method": "c", "result": "", "limitation": "d"}))
    assert pmrl["parse_ok"] and pmrl["words"]["problem"] == 2 and pmrl["all_nonempty"] is False


def test_cli_dry_run_writes_summary(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    assert pb.main(["--variants", "v0", "--dry-run", "--n", "3", "--out", str(out), "--stages", "pmrl,overview"]) == 0
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    assert summary["meta"]["calls"] == 6 * 3
    assert (out / "summary.md").read_text("utf-8").startswith("# Prompt bench summary")
    with pytest.raises(SystemExit):
        pb.parse_args(["--variants", "bogus"])
