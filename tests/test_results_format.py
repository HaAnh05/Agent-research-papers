"""PR3: grounded structured PMRL fields and the "Key results" block of a paper section."""

from __future__ import annotations

from typing import Any

from config import config
from nodes import final_report as fr
from nodes import write_notes as wn
from state import EvalSetting, Insight, PaperItem, PaperTable, PMRLNotes, Setup

TABLE = PaperTable(
    caption="Maze2D scores",
    label="Table 1",
    rows=[["Environment", "MPPI", "IQL", "Diffuser"], ["Maze2D Average", "16.2", "47.0", "119.5 ± 3.1"]],
)
SOURCE_SECTIONS = {
    "abstract": "We study planning with diffusion.",
    "experiments": "Diffuser scores 119.5 on Maze2D. We train on a single TPUv4 machine with 4 chips.",
}


def _paper(notes: PMRLNotes | None = None) -> PaperItem:
    return PaperItem(
        paper_id="arxiv_2205.09991",
        title="Diffuser: Planning with Diffusion",
        arxiv_id="2205.09991",
        sections=dict(SOURCE_SECTIONS),
        tables=[TABLE],
        source_quality={"coverageStatus": "complete"},
        notes=notes,
    )


def _row(**kw: Any) -> EvalSetting:
    base = dict(benchmark="Maze2D", metric="Normalized score", higher_is_better=True, method_value="119.5 ± 3.1",
                baseline_name="IQL", baseline_value="47.0", table_label="Table 1")
    return EvalSetting(**{**base, **kw})


class _FakeLLM:
    def __init__(self, generated: wn.PMRLCoreGeneration) -> None:
        self.generated = generated

    def with_structured_output(self, schema: Any) -> Any:
        generated = self.generated
        return type("S", (), {"invoke": lambda self, payload: generated})()


def test_grounding_drops_invented_numbers_and_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "gemini")
    monkeypatch.setattr(config, "SUMMARY_CACHE_DIR", tmp_path)
    generated = wn.PMRLCoreGeneration(
        problem="p", method="m", result="r", limitation="l",
        eval_settings=[
            _row(),
            _row(benchmark="Invented", method_value="91.37"),            # number absent from the source
            _row(benchmark="Wrong baseline", baseline_value="12.34"),     # baseline number absent
            _row(benchmark="Other table", table_label="Table 9", source_quote="made up quote"),
        ],
        setup=Setup(hardware="8x NVIDIA A100 80GB", compute="TPUv4 machine with 4 chips"),
        insights=[Insight(finding="f", why_it_matters="w")],
    )
    paper, status, _ = wn.analyze_paper_notes(_paper(), llm=_FakeLLM(generated))
    assert status == "ok"
    assert [r.benchmark for r in paper.notes.eval_settings] == ["Maze2D", "Other table"]
    assert paper.notes.eval_settings[1].table_label == "" and paper.notes.eval_settings[1].source_quote == ""
    assert paper.notes.eval_settings[0].table_label == "Table 1"
    assert paper.notes.setup.hardware == "" and paper.notes.setup.compute.startswith("TPUv4")
    assert len(paper.notes.insights) == 1
    assert paper.processing_metadata["evalSettings"] == {
        "proposed": 4, "kept": 2, "dropped": 2, "hardwareDropped": True, "computeDropped": False,
    }


def test_old_cached_pmrl_json_without_structured_fields_still_loads(monkeypatch):
    old = {"problem": "p", "method": "m", "result": "r", "limitation": "l", "brief_summary": ""}
    notes = wn._cached_pmrl_notes(old)
    assert notes is not None and notes.insights == [] and notes.eval_settings == [] and notes.setup == Setup()
    new = {**old, "insights": [{"finding": "f", "why_it_matters": None}], "eval_settings": "garbage"}
    loaded = wn._cached_pmrl_notes(new)
    assert loaded is not None and loaded.insights[0].why_it_matters == "" and loaded.eval_settings == []
    system, user = wn.build_pmrl_messages(_paper())
    assert "=== TABLES" in user and "[Table 1]" in user and "Maze2D Average | 16.2 | 47.0 | 119.5 ± 3.1" in user
    before = wn._pmrl_cache_entry("same prompt")[0]
    monkeypatch.setattr(wn, "PMRL_NOTES_VERSION", "next")
    assert wn._pmrl_cache_entry("same prompt")[0] != before                    # version is part of the cache key


LLM_VI = """### Diffuser (arXiv 2205.09991)
- **Vấn đề**: P.
- **Phương pháp**: M.
- **Kết quả chính**:
  - **Insight**:
    1. Một điều ta học được.
  - **Đại lượng đo**:
    - Normalized score: đo gì; cao hơn là tốt hơn.
  - **Thiết lập**: Datasets: Maze2D.
  - **Nhận xét**: Open-loop.
- **Hạn chế**: L.
"""


def test_results_table_inserted_before_remarks_and_localized():
    notes = PMRLNotes(
        problem="p", method="m", result="r", limitation="l", method_name="Diffuser",
        eval_settings=[_row(), _row(benchmark="Block", metric="Score", baseline_value="", baseline_name="", table_label="")],
    )
    block = fr._normalize_paper_section(LLM_VI, _paper(notes), "Vietnamese")
    lines = block.splitlines()
    header = next(i for i, line in enumerate(lines) if line.strip().startswith("| Benchmark"))
    remarks = next(i for i, line in enumerate(lines) if "**Nhận xét**" in line)
    assert header < remarks and lines[header].strip() == "| Benchmark | Metric | Diffuser | Baseline tốt nhất |"
    assert "| Maze2D | Normalized score ↑ | 119.5 ± 3.1 | 47.0 (IQL) |" in block
    assert "| Block | Score ↑ | 119.5 ± 3.1 | — |" in block
    assert "*Nguồn: Table 1 trong paper*" in block and block.index("*Nguồn") < block.index("**Nhận xét**")
    assert block.count("**Mã nguồn**") == 1                                   # deterministic tail kept once
    # no "Nhận xét" sub-bullet: the table goes at the end of the Key results block, before Hạn chế
    no_remarks = LLM_VI.replace("  - **Nhận xét**: Open-loop.\n", "")
    block = fr._normalize_paper_section(no_remarks, _paper(notes), "Vietnamese")
    assert block.index("| Benchmark") < block.index("**Hạn chế**")
    # no grounded rows: no table
    empty = _paper(PMRLNotes(problem="p", method="m", result="r", limitation="l", setup=Setup(hardware="TPUv4")))
    assert "| Benchmark" not in fr._normalize_paper_section(LLM_VI, empty, "Vietnamese")
    # fallback block has the new structure and the English labels/header
    fallback = fr._deterministic_paper_section(_paper(notes), "English")
    assert "**Insight**" in fallback and "**Metrics**" in fallback and "**Setup**" in fallback
    assert "| Benchmark | Metric | Diffuser | Best baseline |" in fallback
    assert "Insights:" in fr._paper_section_input(_paper(notes))


def test_hardware_line_enforced_only_when_paper_states_none():
    stated = PMRLNotes(problem="p", method="m", result="r", limitation="l", setup=Setup(hardware="TPUv4, 4 chips"))
    absent = PMRLNotes(problem="p", method="m", result="r", limitation="l")
    block = fr._normalize_paper_section(LLM_VI, _paper(absent), "Vietnamese")
    assert "Datasets: Maze2D; Hardware: paper không nêu" in block
    english = LLM_VI.replace("Thiết lập", "Setup").replace("Nhận xét", "Remarks")
    assert "Hardware: not stated in the paper" in fr._normalize_paper_section(english, _paper(absent), "English")
    assert block.count("Hardware") == 1
    already = LLM_VI.replace("Datasets: Maze2D.", "Datasets: Maze2D. Hardware: không rõ.")
    assert fr._normalize_paper_section(already, _paper(absent), "Vietnamese").count("Hardware") == 1
    assert "paper không nêu" not in fr._normalize_paper_section(LLM_VI, _paper(stated), "Vietnamese")
    # no Setup sub-bullet at all: the line is created
    missing = LLM_VI.replace("  - **Thiết lập**: Datasets: Maze2D.\n", "")
    assert "**Thiết lập**: Hardware: paper không nêu" in fr._normalize_paper_section(missing, _paper(absent), "Vietnamese")
