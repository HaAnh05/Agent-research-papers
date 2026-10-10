"""Offline tests for bench/verify.py and the bench/run_scenario.py plumbing."""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import pytest

from bench import run_scenario as rs
from bench import verify as v

BASELINE_DIR = Path(__file__).resolve().parent.parent / "bench" / "baselines" / "direct_compare_3papers_20261010"

GOOD_REPORT = """# Research Report: topic

## 1. Tom tat

Tong quan 46,9% cai thien.

## 2. Phan tich

### Paper A (arXiv 1111.11111)

- Ket qua 46,9%.
  - **Insight**: Dieu ta hoc duoc.
  - **Đại lượng đo**: Ti le thanh cong.
  - **Thiết lập**: Hardware: paper không nêu.

$$
x = y
$$

```bibtex
@article{a,
  title={## 9. fake},
}
```

### Paper B (arXiv 2222.22222)

- Ket qua.
  - **Insight**: Dieu ta hoc duoc.
  - **Metrics**: Success rate.
  - **Setup**: Hardware: not stated in the paper.

## 3. Boi canh nghien cuu

**Bai nen tang**

- Khong co.

## 4. Ket luan

Xong.

## 5. Tai lieu

[a](https://arxiv.org/abs/1111.11111) [b](https://arxiv.org/abs/2222.22222)
"""


def make_run(**overrides) -> v.Run:
    words = lambda n: " ".join(["w"] * n)  # noqa: E731
    base = dict(
        path=Path("."), meta={"status": "success", "total_seconds": 100.0, "llm_calls": [{"seconds": 30}],
                              "node_wall_seconds": {"read_paper": 10.0}},
        paper_ids=["arxiv_1111.11111", "arxiv_2222.22222"],
        notes={pid: {name: words(50) for name in v.PMRL_FIELDS} for pid in ("arxiv_1111.11111", "arxiv_2222.22222")},
        summary_quality={"arxiv_1111.11111": "complete", "arxiv_2222.22222": "partial"},
        card_statuses={"arxiv_1111.11111": {"tldr": "complete", "problem": "complete"},
                       "arxiv_2222.22222": {"tldr": "complete", "problem": "unsupported"}},
        report=GOOD_REPORT,
        sources={"arxiv_1111.11111": "The gain was 46.9 percent on 15 tasks.", "arxiv_2222.22222": "More 1,500 data."},
        arxiv_ids={"arxiv_1111.11111": "1111.11111", "arxiv_2222.22222": "2222.22222"},
    )
    base.update(overrides)
    return v.Run(**base)


def by_name(checks: list[v.Check]) -> dict[str, v.Check]:
    return {c.name: c for c in checks}


# --- HARD ---


def test_status_error_fails() -> None:
    assert v.check_status(make_run(meta={"status": "error"})).status == "FAIL"
    assert v.check_status(make_run(meta={"status": "degraded"})).status == "PASS"


def test_paper_id_mismatch_fails() -> None:
    other = make_run(paper_ids=["arxiv_1111.11111"])
    assert v.check_paper_ids(make_run(), other).status == "FAIL"
    assert v.check_paper_ids(make_run(), make_run()).status == "PASS"


def test_empty_note_field_fails() -> None:
    notes = make_run().notes
    notes["arxiv_2222.22222"]["limitation"] = "  "
    check = v.check_notes_nonempty(make_run(notes=notes))
    assert check.status == "FAIL" and "arxiv_2222.22222.limitation" in check.detail


def test_report_structure_passes_and_ignores_code_fences() -> None:
    assert v.check_report_structure(GOOD_REPORT, 2).status == "PASS"


@pytest.mark.parametrize("mutate", [
    lambda r: r.replace("# Research Report: topic\n", "", 1),
    lambda r: r.replace("## 4. Ket luan", "## 6. Ket luan"),
    lambda r: r.replace("### Paper B (arXiv 2222.22222)", "#### Paper B"),
])
def test_report_structure_failures(mutate) -> None:
    assert v.check_report_structure(mutate(GOOD_REPORT), 2).status == "FAIL"


def test_report_structure_requires_ordered_sections() -> None:
    swapped = GOOD_REPORT.replace("## 4. Ket luan", "## 5. X").replace("## 5. Tai lieu", "## 4. Y")
    assert v.check_report_structure(swapped, 2).status == "FAIL"


def test_context_section_check() -> None:
    assert v.check_context_section(GOOD_REPORT).status == "PASS"
    assert v.check_context_section(GOOD_REPORT.replace("**Bai nen tang**\n\n- Khong co.\n", "")).status == "FAIL"


def test_arxiv_link_missing_fails() -> None:
    run = make_run()
    assert v.check_arxiv_links(GOOD_REPORT, run).status == "PASS"
    assert v.check_arxiv_links(GOOD_REPORT.replace("https://arxiv.org/abs/2222.22222", ""), run).status == "FAIL"


@pytest.mark.parametrize("line", [
    "PMRL quality: complete",
    "- Language mode: vi",
    "Synthesis requested for 3 papers",
    "see report_3c814d1ff9bf7d50.md",
    "pipeline: v4",
    "summary_quality=partial",
])
def test_internal_lines_detected(line: str) -> None:
    assert v.check_no_internal_lines(GOOD_REPORT + "\n" + line + "\n").status == "FAIL"


def test_internal_lines_ignore_prose_and_code() -> None:
    prose = "Gom nhieu pipeline toi uu quy dao.\n```\nPMRL quality: x\n```\n"
    assert v.check_no_internal_lines(GOOD_REPORT + prose).status == "PASS"


def test_results_format_requires_insight_metrics_setup() -> None:
    assert v.check_results_format(GOOD_REPORT).status == "PASS"
    broken = GOOD_REPORT.replace("  - **Metrics**: Success rate.\n", "")
    check = v.check_results_format(broken)
    assert check.status == "FAIL" and "Paper B" in check.detail and "Metrics" in check.detail


def test_table_numbers_must_occur_in_the_paper_source() -> None:
    table = (
        "  | Benchmark | Metric | Diffuser | Best baseline |\n  |---|---|---|---|\n"
        "  | Maze2D U-Maze | Score ↑ | {value} | 33,2 (MPPI-2) |\n"
    )
    run = make_run(sources={"arxiv_1111.11111": "Score 46.9 +- 3.1 and 33.2.", "arxiv_2222.22222": "x"})
    good = GOOD_REPORT.replace("- Ket qua 46,9%.\n", "- Ket qua 46,9%.\n" + table.format(value="46,9 ± 3,1"), 1)
    assert v.check_table_numbers(good, run).status == "PASS"
    bad = GOOD_REPORT.replace("- Ket qua 46,9%.\n", "- Ket qua 46,9%.\n" + table.format(value="91,7"), 1)
    check = v.check_table_numbers(bad, run)
    assert check.status == "FAIL" and "91,7" in check.detail


# --- QUALITY ---


def test_summary_quality_rank() -> None:
    worse = make_run(summary_quality={"arxiv_1111.11111": "partial", "arxiv_2222.22222": "partial"})
    assert v.check_summary_quality(make_run(), worse).status == "WARN"
    assert v.check_summary_quality(make_run(), worse, strict=True).status == "FAIL"
    better = make_run(summary_quality={"arxiv_1111.11111": "complete", "arxiv_2222.22222": "complete"})
    assert v.check_summary_quality(make_run(), better).status == "PASS"


def test_complete_card_count() -> None:
    fewer = make_run(card_statuses={"arxiv_1111.11111": {"tldr": "complete"}})
    assert v.check_card_statuses(make_run(), fewer).status == "WARN"
    assert v.check_card_statuses(make_run(), make_run()).status == "PASS"


def test_pmrl_word_count_band() -> None:
    notes = make_run().notes
    notes["arxiv_1111.11111"]["method"] = " ".join(["w"] * 10)  # 0.2x
    th = v.Thresholds()
    assert v.check_pmrl_word_counts(make_run(), make_run(notes=notes), th).status == "WARN"
    notes["arxiv_1111.11111"]["method"] = " ".join(["w"] * 79)  # 1.58x
    assert v.check_pmrl_word_counts(make_run(), make_run(notes=notes), th).status == "PASS"
    notes["arxiv_1111.11111"]["method"] = " ".join(["w"] * 81)  # 1.62x
    assert v.check_pmrl_word_counts(make_run(), make_run(notes=notes), th).status == "WARN"


def test_report_length_band() -> None:
    th = v.Thresholds()
    assert v.check_report_length(make_run(), make_run(report=GOOD_REPORT[:100]), th).status == "WARN"
    assert v.check_report_length(make_run(), make_run(report=GOOD_REPORT * 2), th).status == "WARN"
    assert v.check_report_length(make_run(), make_run(), th).status == "PASS"


def test_number_normalization_tolerates_formats() -> None:
    assert v.number_variants("46,9") & v.number_variants("46.9")
    assert v.number_variants("1,500") & v.number_variants("1500")
    assert v.number_variants("1.500") & v.number_variants("1500")
    assert v.number_variants("0.469") & v.number_variants("46,9", percent=True)


def test_grounding_ignores_ids_urls_code_and_math() -> None:
    report = "## 1. X\nSee 2303.04137 and https://x.org/99999 and $$ 12345 $$ and $77777$.\n```\n88888\n```\nGain 46,9%.\n"
    claims = v.extract_report_numbers(report)
    assert [token for token, _ in claims] == ["46,9"]
    ratio, count = v.grounding_ratio(report, {"a": "Gain 46.9 percent"})
    assert (ratio, count) == (1.0, 1)
    assert v.grounding_ratio(report, {"a": "nothing"})[0] == 0.0


def test_grounding_drop_check() -> None:
    th = v.Thresholds()
    bad = make_run(report=GOOD_REPORT + "\nGain 91,2% and 777 items and 555 more.\n")
    assert v.check_grounding(make_run(), bad, th).status == "WARN"
    assert v.check_grounding(make_run(), bad, th, strict=True).status == "FAIL"
    assert v.check_grounding(make_run(), make_run(), th).status == "PASS"


def test_display_equation_limit() -> None:
    th = v.Thresholds()
    assert v.check_equations(GOOD_REPORT, 2, th).status == "PASS"
    many = GOOD_REPORT.replace("$$\nx = y\n$$", "\n".join(["$$\nx\n$$"] * 4))
    assert v.check_equations(many, 2, th).status == "WARN"


def test_comparison_table_check_is_removed() -> None:
    assert not hasattr(v, "check_comparison_table")


# --- LATENCY / orchestration ---


def test_latency_never_fails_exit_unless_strict() -> None:
    checks = v.verify(make_run(), make_run())
    assert by_name(checks)["target_seconds"].status == "FAIL"
    assert v.exit_code(checks) == 0
    assert v.exit_code(checks, strict_latency=True) == 1
    assert by_name(checks)["total_seconds"].detail.endswith("speedup=1.00x")


def test_hard_failure_sets_exit_code() -> None:
    checks = v.verify(make_run(), make_run(meta={"status": "error", "total_seconds": 1}))
    assert v.exit_code(checks) == 1


def test_quality_warning_does_not_fail_unless_strict() -> None:
    worse = make_run(summary_quality={"arxiv_1111.11111": "invalid", "arxiv_2222.22222": "partial"})
    assert v.exit_code(v.verify(make_run(), worse)) == 0
    assert v.exit_code(v.verify(make_run(), worse, strict=True)) == 1


def _five_section_copy(tmp_path: Path) -> Path:
    """Copy the frozen baseline, replacing its legacy comparison section (old ``## 3.``) with a context stub."""

    target = tmp_path / "baseline5"
    shutil.copytree(BASELINE_DIR, target)
    text = (target / "report.md").read_text(encoding="utf-8")
    text = re.sub(r"(?ms)^## 3\..*?(?=^## 4\.)", "## 3. Bối cảnh nghiên cứu\n\nChưa có dữ liệu trích dẫn.\n\n", text)
    (target / "report.md").write_text(text, encoding="utf-8")
    return target


def test_real_baseline_verifies_against_itself(tmp_path: Path) -> None:
    # The frozen baseline report has a legacy comparison as section 3; verify a
    # copy whose section 3 is replaced by a research-context stub.
    baseline = _five_section_copy(tmp_path)
    run = v.load_run(baseline)
    assert len(run.paper_ids) == 3 and run.sources and run.card_statuses
    checks = v.verify(run, run)
    # The frozen baseline predates the results format (Insight/Metrics/Setup), so skip those checks.
    failed = [c for c in checks if c.group in ("HARD", "QUALITY") and c.status != "PASS" and not c.name.startswith("results_")]
    assert not failed, failed
    out = tmp_path / "verify.json"
    code = v.main(["--baseline", str(baseline), "--candidate", str(baseline), "--out", str(out)])
    assert code == 1  # results_format FAILs on the legacy report; everything else passes
    assert [c["name"] for c in json.loads(out.read_text())["checks"] if c["status"] == "FAIL" and c["group"] != "LATENCY"] == ["results_format"]


# --- run_scenario plumbing (stub graph, no network, no LLM) ---


def test_dry_run_artifacts_load_as_candidate(tmp_path: Path) -> None:
    out = tmp_path / "run"
    scenario = {"query": "q", "raw_inputs": ["https://arxiv.org/abs/1111.11111"], "intent": None, "source": "test"}
    run = rs.run_scenario(scenario, out, graph=rs.DryRunGraph(scenario["raw_inputs"]))
    assert run["status"] == "success" and {"router", "final_report"} <= set(run["node_wall_seconds"])
    for name in ("run.json", "state.json", "report.md"):
        assert (out / name).is_file()
    state = json.loads((out / "state.json").read_text())
    paper = state["selected_papers"][0]
    assert "extracted_text" not in paper and paper["extracted_text_chars"] > 0 and paper["sections_chars"]
    candidate = v.load_run(out)
    assert candidate.paper_ids == ["arxiv_1111.11111"]
    assert (out / "source_text" / "arxiv_1111.11111.json").is_file()
    assert candidate.sources["arxiv_1111.11111"].startswith("dry run text")


def test_fresh_llm_cache_redirects_and_restores_summary_dir(tmp_path: Path) -> None:
    from config import config

    original = config.SUMMARY_CACHE_DIR
    seen: list[Path] = []

    class Spy(rs.DryRunGraph):
        def stream(self, *args, **kwargs):
            seen.append(Path(config.SUMMARY_CACHE_DIR))
            yield from super().stream(*args, **kwargs)

    scenario = {"query": "q", "raw_inputs": [], "intent": None, "source": "test"}
    rs.run_scenario(scenario, tmp_path / "a", graph=Spy([]))
    rs.run_scenario(scenario, tmp_path / "b", fresh_llm_cache=False, graph=Spy([]))
    assert seen[0] != original and seen[1] == original
    assert config.SUMMARY_CACHE_DIR == original


def test_instrumentation_records_llm_and_http_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from langchain_openai import ChatOpenAI

    monkeypatch.setattr(ChatOpenAI, "invoke", lambda self, input=None, *a, **k: "ok")
    monkeypatch.setattr(requests.Session, "request", lambda self, method, url=None, *a, **k: type("R", (), {"status_code": 200})())
    recorder = rs.CallRecorder()
    with rs.instrument(recorder):
        thread = threading.Thread(target=lambda: ChatOpenAI.invoke(object(), "x" * 25), name="worker-1")
        thread.start()
        thread.join()
        requests.get("https://example.org/a?secret=1", timeout=1)
    assert recorder.llm_calls[0]["prompt_chars"] == 25 and recorder.llm_calls[0]["thread"] == "worker-1"
    assert recorder.http_calls[0]["url"] == "https://example.org/a" and recorder.http_calls[0]["status"] == 200
    assert ChatOpenAI.invoke is not None and requests.Session.request.__name__ != "timed"


def test_llm_calls_attributed_to_nodes() -> None:
    calls = [{"start_s": 1.0, "seconds": 2.0}, {"start_s": 10.0, "seconds": 5.0}]
    rs.attribute_llm_nodes(calls, [("read_paper", 4.0), ("final_report", 16.0)])
    assert [c["node"] for c in calls] == ["read_paper", "final_report"]
