from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from httpx import ASGITransport, AsyncClient

import api
from config import config
from state import GitHubRepoInfo, PMRLNotes, PaperItem, SummaryCards


class FakeGraph:
    """Small deterministic graph double exercising the bridge contract."""

    def stream(self, state, graph_config, stream_mode=None):
        yield {"router": {"intent": "direct_read", "trace_logs": ["[router] routed"]}}
        yield {
            "final_report": "# Fake Research Report\n\nDone.",
            "status": "success",
            "trace_logs": ["[final_report] Báo cáo lưu: fake_report.md"],
        }

    def get_state(self, graph_config):
        return SimpleNamespace(values={
            "status": "success",
            "final_report": "# Fake Research Report\n\nDone.",
            "selected_papers": [],
            "trace_logs": ["[router] routed", "[final_report] Báo cáo lưu: fake_report.md"],
            "error_logs": [],
        })


@pytest.mark.asyncio
async def test_run_snapshot_preserves_long_report_while_masking_secrets():
    report = "# Long report\n\n" + ("research evidence " * 180) + "sk-report-secret-12345678\n\nEND-OF-REPORT"

    class LongReportGraph(FakeGraph):
        def stream(self, state, graph_config, stream_mode=None):
            yield {"final_report": report, "status": "success", "trace_logs": []}

        def get_state(self, graph_config):
            return SimpleNamespace(values={
                "status": "success",
                "final_report": report,
                "selected_papers": [],
                "trace_logs": [],
                "error_logs": [],
            })

    application = api.create_app(
        graph_factory=LongReportGraph,
        registry_instance=api.RunRegistry(),
    )
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post("/api/runs", data={"query": "long report"})
        assert response.status_code == 202
        run_id = response.json()["runId"]
        for _ in range(100):
            snapshot = (await client.get(f"/api/runs/{run_id}")).json()
            if snapshot["status"] != "running":
                break
            await asyncio.sleep(0.01)

    rendered_report = snapshot["result"]["report"]
    assert len(rendered_report) > 1200
    assert rendered_report.endswith("END-OF-REPORT")
    assert "sk-report-secret-12345678" not in rendered_report
    assert "[REDACTED]" in rendered_report


@pytest.mark.asyncio
async def test_comma_separated_paper_inputs_are_normalized_before_graph_execution():
    class CapturingGraph(FakeGraph):
        received_inputs: list[str] = []

        def stream(self, state, graph_config, stream_mode=None):
            type(self).received_inputs = list(state["raw_inputs"])
            yield from super().stream(state, graph_config, stream_mode)

    application = api.create_app(
        graph_factory=CapturingGraph,
        registry_instance=api.RunRegistry(),
    )
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post(
            "/api/runs",
            data={"paper_inputs": "1706.03762, 2005.14165"},
        )
        assert response.status_code == 202
        for _ in range(100):
            if CapturingGraph.received_inputs:
                break
            await asyncio.sleep(0.01)

    assert CapturingGraph.received_inputs == ["1706.03762", "2005.14165"]


@pytest.mark.asyncio
async def test_arxiv_id_entered_as_query_is_promoted_to_direct_paper_input():
    class CapturingGraph(FakeGraph):
        received_state: dict = {}

        def stream(self, state, graph_config, stream_mode=None):
            type(self).received_state = dict(state)
            yield from super().stream(state, graph_config, stream_mode)

    application = api.create_app(graph_factory=CapturingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post("/api/runs", data={"query": "1706.03762"})
        assert response.status_code == 202
        for _ in range(100):
            if CapturingGraph.received_state:
                break
            await asyncio.sleep(0.01)

    assert CapturingGraph.received_state["raw_inputs"] == ["1706.03762"]
    assert CapturingGraph.received_state["user_query"] == "Direct Paper Analysis"


@pytest.mark.asyncio
async def test_paper_urls_are_limited_to_arxiv_and_normalized_before_graph_execution():
    class CapturingGraph(FakeGraph):
        received_inputs: list[str] = []

        def stream(self, state, graph_config, stream_mode=None):
            type(self).received_inputs = list(state["raw_inputs"])
            yield from super().stream(state, graph_config, stream_mode)

    application = api.create_app(graph_factory=CapturingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        blocked = await client.post(
            "/api/runs",
            data={"paper_inputs": "http://169.254.169.254/latest/meta-data/"},
        )
        assert blocked.status_code == 422

        accepted = await client.post(
            "/api/runs",
            data={"paper_inputs": "https://arxiv.org/abs/1706.03762v7?download=1"},
        )
        assert accepted.status_code == 202
        for _ in range(100):
            if CapturingGraph.received_inputs:
                break
            await asyncio.sleep(0.01)

    assert CapturingGraph.received_inputs == ["1706.03762v7"]


@pytest.mark.asyncio
async def test_run_events_and_public_snapshot_exclude_internal_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.REPORTS_DIR.mkdir(parents=True)
    (config.REPORTS_DIR / "fake_report.md").write_text("# Fake Research Report\n", encoding="utf-8")
    registry = api.RunRegistry()
    application = api.create_app(graph_factory=FakeGraph, registry_instance=registry)

    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post(
            "/api/runs",
            data={"query": "transformers", "paper_inputs": "1706.03762"},
        )
        assert response.status_code == 202
        snapshot = response.json()
        run_id = snapshot["runId"]
        for _ in range(100):
            current = (await client.get(f"/api/runs/{run_id}")).json()
            if current["status"] != "running":
                break
            await asyncio.sleep(0.01)
        assert current["status"] == "success"
        assert "1706.03762" in current["input"]["paperInputs"]
        assert "extracted_text" not in json.dumps(current)
        events = await client.get(f"/api/runs/{run_id}/events", headers={"Accept": "text/event-stream"})
        assert events.status_code == 200
        assert "event: snapshot" in events.text
        assert events.text.count("event: ") == 1
        assert '"type":"step.started"' in events.text
        assert '"type":"step.completed"' in events.text
        assert '"type":"run.completed"' in events.text
        assert '"status":"completed"' in events.text


@pytest.mark.asyncio
async def test_pdf_upload_gets_uuid_name_and_non_pdf_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_CACHE_DIR", tmp_path / "cache")
    registry = api.RunRegistry()
    application = api.create_app(graph_factory=FakeGraph, registry_instance=registry)
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        bad = await client.post(
            "/api/runs",
            files={"pdfs": ("notes.txt", b"not a pdf", "text/plain")},
        )
        assert bad.status_code == 422
        good = await client.post(
            "/api/runs",
            files={"pdfs": ("paper.pdf", b"%PDF-1.7 fake", "application/pdf")},
        )
        assert good.status_code == 202
        run_input = good.json()["input"]
        assert run_input["pdfNames"] == ["paper.pdf"]
        assert run_input.get("paperInputs") == []
        uploaded = list((config.PDF_CACHE_DIR / "uploads").glob("*.pdf"))
        assert len(uploaded) == 1
        assert uploaded[0].name != "paper.pdf"


@pytest.mark.asyncio
async def test_single_active_run_returns_409(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class BlockingGraph(FakeGraph):
        def stream(self, state, graph_config, stream_mode=None):
            started.set()
            release.wait(timeout=2)
            yield from super().stream(state, graph_config, stream_mode)

    registry = api.RunRegistry()
    application = api.create_app(graph_factory=BlockingGraph, registry_instance=registry)
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        first = await client.post("/api/runs", data={"query": "first"})
        assert first.status_code == 202
        assert started.wait(timeout=1)
        second = await client.post("/api/runs", data={"query": "second"})
        assert second.status_code == 409
        assert second.json()["detail"]["activeRunId"] == first.json()["runId"]
        release.set()


@pytest.mark.asyncio
async def test_reports_reject_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.REPORTS_DIR.mkdir(parents=True)
    (config.REPORTS_DIR / "safe.md").write_text(
        "# Safe\n\napi_key=library-secret-value\n",
        encoding="utf-8",
    )
    (tmp_path / "secret.md").write_text("# Secret\n", encoding="utf-8")
    application = api.create_app(graph_factory=FakeGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        assert (await client.get("/api/reports")).status_code == 200
        detail = await client.get("/api/reports/safe.md")
        assert detail.status_code == 200
        assert "library-secret-value" not in detail.text
        assert "[REDACTED]" in detail.text
        download = await client.get("/api/reports/safe.md/download")
        assert download.status_code == 200
        assert "library-secret-value" not in download.text
        assert "[REDACTED]" in download.text
        assert (await client.get("/api/reports/%2e%2e%2fsecret.md")).status_code == 404
        assert (await client.get("/api/reports/%2e%2e%2fsecret.md/download")).status_code == 404


@pytest.mark.asyncio
async def test_reconnect_uses_after_seq_and_replays_only_new_frames():
    registry = api.RunRegistry()
    record = registry.create(
        api.RunInput(query="reconnect"),
        "zai",
        "glm-test",
        {"selected_papers": [], "trace_logs": [], "error_logs": []},
    )
    registry.append_event(record.run_id, "step.started", node="router", status="running", progress=0.1)
    registry.append_event(record.run_id, "step.completed", node="router", status="success", progress=0.1)
    registry.set_terminal(record.run_id, "success")
    registry.append_event(record.run_id, "run.completed", node="router", status="success", progress=1.0)
    application = api.create_app(graph_factory=FakeGraph, registry_instance=registry)
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.get(f"/api/runs/{record.run_id}/events?after_seq=2")
        assert response.status_code == 200
        event_lines = [line for line in response.text.splitlines() if line.startswith("event: ")]
        assert event_lines == ["event: snapshot"]


def test_public_dto_serialization_masks_internal_paper_fields():
    paper = PaperItem(
        paper_id="arxiv_1706.03762",
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        local_pdf_path="/private/upload.pdf",
        extracted_text="PRIVATE EXTRACTED PDF CONTENT",
        notes=PMRLNotes(problem="problem", method="method", result="result", limitation="limit"),
        github_repos=[GitHubRepoInfo(name="repo", url="https://github.com/example/repo")],
        bibtex="@article{attention}",
    )
    registry = api.RunRegistry()
    record = registry.create(
        api.RunInput(query="paper"),
        "zai",
        "glm-test",
        {"selected_papers": [paper], "trace_logs": [], "error_logs": []},
    )
    payload = registry.snapshot(record.run_id).model_dump(mode="json", by_alias=True)
    serialized = json.dumps(payload)
    assert "PRIVATE EXTRACTED PDF CONTENT" not in serialized
    assert "/private/upload.pdf" not in serialized
    assert payload["result"]["papers"][0]["githubRepos"][0]["url"] == "https://github.com/example/repo"


def test_public_dto_omits_non_arxiv_pdf_url():
    paper = PaperItem(
        paper_id="unsafe",
        title="Unsafe source",
        pdf_url="http://127.0.0.1:8000/internal",
    )
    registry = api.RunRegistry()
    record = registry.create(
        api.RunInput(query="paper"),
        "zai",
        "glm-test",
        {"selected_papers": [paper], "trace_logs": [], "error_logs": []},
    )
    payload = registry.snapshot(record.run_id).model_dump(mode="json", by_alias=True)
    assert payload["result"]["papers"][0]["pdfUrl"] is None


def test_sensitive_path_mask_does_not_corrupt_https_urls():
    value = "See https://arxiv.org/abs/1706.03762 and C:/private/cache/paper.pdf"
    masked = api._mask_sensitive(value)
    assert "https://arxiv.org/abs/1706.03762" in masked
    assert "C:/private" not in masked
    assert "[path omitted]" in masked
    assert api._mask_sensitive("https://example.com/?next=C:/tmp/file") == "https://example.com/?next=C:/tmp/file"


def test_paper_dto_validates_brief_and_marks_legacy_quality_unknown():
    valid = PaperItem(
        paper_id="arxiv_1706.03762",
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        notes=PMRLNotes(
            problem="problem",
            method="method",
            result="result",
            limitation="limit",
            brief_summary="Conclusion: the method is effective.\n- It improves the baseline.\n- It is evaluated on the reported data.\n- It remains sensitive to compute.",
        ),
        source_quality={"heading": "heading", "parser_status": "success"},
        notes_quality="complete",
    )
    valid_payload = api._paper_dto(valid).model_dump(mode="json", by_alias=True)
    assert valid_payload["notes"]["briefSummary"].startswith("Conclusion:")
    assert valid_payload["notes"]["status"] == "complete"
    assert valid_payload["pmrlStatus"] == "complete"
    assert valid_payload["notesQuality"] == "complete"
    assert valid_payload["sourceQuality"]["heading"] == "heading"
    assert valid_payload["sourceQuality"]["parserStatus"] == "success"

    invalid = PaperItem(
        paper_id="arxiv_2005.14165",
        title="Legacy",
        notes=PMRLNotes(problem="p", method="m", result="r", limitation="l", brief_summary="Conclusion only"),
    )
    invalid_payload = api._paper_dto(invalid).model_dump(mode="json", by_alias=True)
    assert invalid_payload["notes"]["briefSummary"] is None
    assert invalid_payload["notes"]["briefStatus"] == "invalid"
    assert invalid_payload["pmrlStatus"] == "unknown"

    legacy = PaperItem(
        paper_id="arxiv_2101.00001",
        title="Old cache",
        notes=PMRLNotes(problem="p", method="m", result="r", limitation="l"),
    )
    legacy_payload = api._paper_dto(legacy).model_dump(mode="json", by_alias=True)
    assert legacy_payload["notes"]["briefSummary"] is None
    assert legacy_payload["pmrlStatus"] == "unknown"
    assert legacy_payload["notesQuality"] == "unknown"
    assert legacy_payload["sourceQuality"]["heading"] == "unknown"
    assert legacy_payload["sourceQuality"]["parserStatus"] == "unknown"
    assert legacy_payload["sourceQuality"]["coverageStatus"] == "unknown"


def test_paper_dto_keeps_complete_summary_when_source_coverage_is_incomplete():
    cards = SummaryCards(
        tldr="Alpha contribution improves sequence learning.",
        problem="Beta problem limits parallel recurrent training.",
        method="Gamma method uses a self attention architecture.",
        key_results="Delta benchmark evidence improves the reported baseline.",
        why_it_matters="Epsilon impact enables faster practical modeling.",
    )
    paper = PaperItem(
        paper_id="incomplete-source",
        title="Incomplete source",
        source_quality={"coverageStatus": "incomplete", "missingSections": ["experiments"]},
        notes=PMRLNotes(
            problem="p", method="m", result="r", limitation="l",
            summary_cards=cards, summary_quality="complete",
        ),
    )

    payload = api._paper_dto(paper).model_dump(mode="json", by_alias=True)

    assert payload["summaryCards"]["status"] == "complete"
    assert payload["summaryCards"]["tldr"] == cards.tldr
    assert payload["sourceQuality"]["coverageStatus"] == "incomplete"


def test_paper_dto_preserves_valid_cards_from_partial_summary():
    paper = PaperItem(
        paper_id="partial-summary",
        title="Partial summary",
        source_quality={"coverageStatus": "incomplete"},
        notes=PMRLNotes(
            problem="p", method="m", result="r", limitation="l",
            summary_cards=SummaryCards(
                tldr="Alpha contribution improves sequence learning.",
                problem="Beta problem limits parallel recurrent training.",
                method="Gamma method uses a self attention architecture.",
                key_results="",
                why_it_matters="Epsilon impact enables faster practical modeling.",
            ),
            summary_quality="partial",
            summary_card_statuses={
                "tldr": "complete", "problem": "complete", "method": "complete",
                "key_results": "missing", "why_it_matters": "complete",
            },
            summary_card_reasons={"key_results": "number_not_in_source:62.5%"},
        ),
    )

    cards = api._paper_dto(paper).model_dump(mode="json", by_alias=True)["summaryCards"]

    assert cards["status"] == "partial"
    assert cards["tldr"]
    assert cards["keyResults"] == ""
    assert cards["cardStatuses"]["keyResults"] == "missing"
    assert cards["cardReasons"]["keyResults"] == "number_not_in_source:62.5%"


def test_structured_comparison_artifact_is_safe_and_gates_metric_comparability():
    result = api._result_from_state(
        {
            "status": "success",
            "selected_papers": [],
            "benchmark_matrix": "| Paper | Result |",
            "comparison_artifact": {
                "papers": [
                    {"paper_id": "p1", "title": "One", "findings": ["short finding"], "metrics": []},
                    {"paper_id": "p2", "title": "Two", "findings": ["another finding"], "metrics": []},
                ],
                "rows": [
                    {
                        "metric": "ATE",
                        "dataset": "Replica",
                        "unit": "cm",
                        "values": {"p1": "0.06", "p2": "0.12"},
                        "source_quotes": {"p1": "reported 0.06", "p2": "reported 0.12"},
                        "comparable": True,
                    },
                    {"metric": "PSNR", "values": {"p1": "42"}, "comparable": True},
                ],
                "synthesis": ["short synthesis"],
            },
        }
    )
    payload = result.model_dump(mode="json", by_alias=True)
    artifact = payload["comparisonArtifact"]
    assert artifact["available"] is True
    assert artifact["rows"][0]["comparable"] is True
    assert artifact["rows"][1]["comparable"] is False
    assert artifact["rows"][1]["reason"] == "Not directly comparable"
    assert payload["benchmark"] == "| Paper | Result |"


def test_trace_duration_and_snapshot_timings_are_numeric():
    paper = PaperItem(
        paper_id="arxiv_1706.03762",
        title="Timed paper",
        processing_metadata={"pdf_cache_ms": 17.5, "secret": "/tmp/private"},
    )
    registry = api.RunRegistry()
    record = registry.create(
        api.RunInput(query="timing"),
        "zai",
        "glm-test",
        {
            "status": "success",
            "selected_papers": [paper],
            "node_timings": {
                "write_notes": {"llm_ms": 123.4},
                "web_enrich": {"github.search_ms": 88},
            },
        },
    )
    event = registry.append_event(
        record.run_id,
        "step.updated",
        node="write_notes",
        status="running",
        facts={"timings": {"llm_ms": 123.4, "prompt": "private"}},
        duration_ms=42.8,
        timings={"llm_ms": 123.4},
    )
    snapshot = registry.snapshot(record.run_id).model_dump(mode="json", by_alias=True)
    event_payload = snapshot["traceEvents"][0]
    assert event_payload["durationMs"] == 42
    assert event_payload["timings"] == {"llm_ms": 123}
    assert event_payload["facts"]["timings"] == {"llm_ms": 123}
    assert snapshot["timings"]["write_notes.llm_ms"] == 123
    assert snapshot["timings"]["web_enrich.github.search_ms"] == 88
    assert snapshot["timings"]["paper.arxiv_1706.03762.pdf_cache_ms"] == 17
    assert "secret" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_report_links_preserve_valid_https_and_flatten_unknown_links(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.REPORTS_DIR.mkdir(parents=True)
    report = (
        "# Links\n\n"
        "[ArXiv](https://arxiv.org/abs/1706.03762)\n\n"
        "[GitHub](https://github.com/example/repo)\n\n"
        "[Local PDF](/home/user/private.pdf)\n\n"
        "[Unknown](https://example.invalid/paper)\n\n"
        "<https://example.invalid/autolink>\n\n"
        "https://example.invalid/bare\n"
    )
    (config.REPORTS_DIR / "links.md").write_text(report, encoding="utf-8")
    application = api.create_app(graph_factory=FakeGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        detail = await client.get("/api/reports/links.md")
        download = await client.get("/api/reports/links.md/download")
    assert detail.status_code == 200
    assert download.status_code == 200
    for body in (detail.json()["content"], download.text):
        assert "[ArXiv](https://arxiv.org/abs/1706.03762)" in body
        assert "[GitHub](https://github.com/example/repo)" in body
        assert "/home/user/private.pdf" not in body
        assert "Local PDF _(link unavailable: unverified source)_" in body
        assert "Unknown _(link unavailable: unverified source)_" in body
        assert "example.invalid" not in body


def test_public_link_normalizes_known_http_hosts_to_https():
    assert api._safe_link("http://arxiv.org/abs/1706.03762") == "https://arxiv.org/abs/1706.03762"
    assert api._safe_link("http://github.com/example/repo") == "https://github.com/example/repo"
    assert api._safe_link("https://evil.github.com/example/repo") is None
    assert api._safe_link("https://arxiv.org:8080/abs/1706.03762") is None


def test_snapshot_report_repairs_unambiguous_pdf_placeholder():
    paper = PaperItem(
        paper_id="arxiv_1706.03762",
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
    )
    registry = api.RunRegistry()
    record = registry.create(
        api.RunInput(query="paper"),
        "zai",
        "glm-test",
        {
            "status": "success",
            "selected_papers": [paper],
            "final_report": "# Report\n\n[Attention Is All You Need](paper.pdf)",
            "trace_logs": [],
        },
    )
    report = registry.snapshot(record.run_id).result.report
    assert report is not None
    assert "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)" in report
    assert "evil.invalid" not in api._sanitize_report_markdown("[Paper](https://evil.invalid/1706.03762)")


@pytest.mark.asyncio
async def test_config_exposes_provider_status_without_secrets(monkeypatch):
    secret = "zai-super-secret-value"
    monkeypatch.setattr(config, "DEFAULT_PROVIDER", "zai")
    monkeypatch.setattr(config, "ZAI_API_KEY", secret)
    application = api.create_app(graph_factory=FakeGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.get("/api/config")
        assert response.status_code == 200
        assert secret not in response.text
        body = response.json()
        assert isinstance(body["providers"], list)
        zai = next(item for item in body["providers"] if item["provider"] == "zai")
        assert zai["configured"] is True


def test_step_facts_are_sparse_allowlisted_and_task_events_stay_factless(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    config.REPORTS_DIR.mkdir(parents=True)
    (config.REPORTS_DIR / "run.md").write_text("# Run\n", encoding="utf-8")
    paper_one = PaperItem(
        paper_id="arxiv_1706.03762",
        title="One",
        github_repos=[GitHubRepoInfo(name="one", url="https://github.com/example/one")],
        bibtex="@article{one}",
    )
    paper_two = PaperItem(paper_id="arxiv_2005.14165", title="Two")

    assert api._step_facts("router", {}, {"intent": "direct_read", "secret": "/tmp/private"}) == {
        "intent": "direct_read"
    }
    assert api._step_facts("search_papers", {}, {"search_results": [paper_one, paper_two]}) == {"resultCount": 2}
    assert api._step_facts("eval_search", {}, {"selected_papers": [paper_one]}) == {"selectedCount": 1}
    assert api._step_facts("refine_query", {}, {"retry_count": 2}) == {"retryCount": 2}
    assert api._step_facts(
        "read_paper",
        {},
        {"selected_papers": [paper_one], "error_logs": ["Lỗi đọc bài báo 'Two': provider detail"]},
    ) == {"paperCount": 1, "failedPdfCount": 1}
    assert api._step_facts("web_enrich", {}, {"selected_papers": [paper_one, paper_two]}) == {
        "repositoryCount": 1,
        "bibtexCount": 1,
    }
    assert api._step_facts(
        "write_notes",
        {},
        {
            "trace_logs": [
                "[write_notes] arxiv_1706.03762: PMRL fallback (provider secret=/tmp/private)",
                "[write_notes] /tmp/private.pdf: PMRL fallback (must be ignored)",
            ]
        },
    ) == {"noteFallbackPaperIds": ["arxiv_1706.03762"]}
    assert api._step_facts(
        "compare_benchmark",
        {"selected_papers": [paper_one, paper_two]},
        {"benchmark_matrix": "| Paper | Result |"},
    ) == {
        "comparisonAvailable": True,
        "comparedPaperIds": ["arxiv_1706.03762", "arxiv_2005.14165"],
    }
    assert api._step_facts(
        "compare_benchmark",
        {"selected_papers": [paper_one, paper_two]},
        {"benchmark_matrix": "*(Không thể tạo ma trận so sánh tự động: api_key=secret)*"},
    ) == {"comparisonAvailable": False}
    assert api._valid_benchmark("| Paper | Result |\n| A | error unavailable metric is documented |") is True
    paper_with_reason = paper_one.model_copy(update={
        "notes": PMRLNotes(
            problem="p", method="m", result="r", limitation="l",
            summary_quality="partial",
            summary_card_statuses={"key_results": "unsupported"},
            summary_card_reasons={"key_results": "number_not_in_source:62.5%"},
        )
    })
    assert api._step_facts(
        "final_report",
        {"status": "success", "trace_logs": ["[final_report] saved: run.md"]},
        {"final_report": "# Run", "status": "success", "selected_papers": [paper_with_reason]},
    ) == {
        "reportAvailable": True,
        "reportId": "run.md",
        "summaryCardReasons": {
            "arxiv_1706.03762.keyResults": "number_not_in_source:62.5%",
        },
    }
    assert api._sanitize_facts({
        "summaryCardReasons": {
            "arxiv_1706.03762.keyResults": "number_not_in_source:62.5%",
            "arxiv_1706.03762.method": "/tmp/private source text",
        }
    }) == {
        "summaryCardReasons": {
            "arxiv_1706.03762.keyResults": "number_not_in_source:62.5%",
        }
    }
    assert api._valid_report("# Findings\n\nThe report explains an error in the final report.") is True
    assert api._valid_report("# Error Generating Final Report\n\nProvider unavailable.") is False

    registry = api.RunRegistry()
    record = registry.create(api.RunInput(query="facts"), "zai", "glm-test", {})
    lifecycle = registry.append_event(
        record.run_id,
        "step.completed",
        node="router",
        status="completed",
        facts={"intent": "search"},
    )
    assert lifecycle.facts is None
    api._append_node_update(
        record.run_id,
        "router",
        {},
        {},
        registry,
        details={"source": "task"},
        completed=True,
        task_id="task-1",
    )
    task_events = registry.snapshot(record.run_id).trace_events[-2:]
    assert [event.facts for event in task_events] == [None, None]


def test_result_dto_suppresses_failed_benchmark_and_report_placeholders():
    state = {
        "status": "error",
        "selected_papers": [],
        "benchmark_matrix": "*(Không thể tạo ma trận so sánh tự động: provider failed)*",
        "final_report": "# Error Generating Final Report\nprovider failed",
        "trace_logs": ["[final_report] saved: unsafe.md"],
        "error_logs": [],
    }
    result = api._result_from_state(state)
    assert result.benchmark is None
    assert result.report is None
    assert result.report_id is None
    assert result.answer is None


def test_public_failure_message_discards_exception_suffix_from_guard_prefix():
    raw_message = (
        "ArXiv không trả về bài báo nào sau 3 lượt tìm kiếm. "
        "Hãy thử 2-5 từ khóa kỹ thuật bằng tiếng Anh hoặc nhập ArXiv ID trực tiếp. "
        "provider prompt internals api_key=sk-abcdefghijklmnop at /tmp/private"
    )
    public_message = api._public_failure_message("eval_search", raw_message)
    assert public_message == (
        "ArXiv không trả về bài báo nào sau 3 lượt tìm kiếm. "
        "Hãy thử 2-5 từ khóa kỹ thuật bằng tiếng Anh hoặc nhập ArXiv ID trực tiếp."
    )
    assert "provider prompt" not in public_message
    assert "sk-abcdefghijklmnop" not in public_message
    assert "/tmp/private" not in public_message


@pytest.mark.asyncio
async def test_worker_exception_uses_safe_node_message_without_exception_details():
    raw_failure = "provider SYSTEM PROMPT internals api_key=sk-abcdefghijklmnop at /srv/private/paper.pdf"

    class ExplodingGraph:
        def stream(self, state, graph_config, stream_mode=None):
            yield {"router": {"intent": "search"}}
            raise RuntimeError(raw_failure)
            yield  # Keep this a streaming fake for type checkers.

    application = api.create_app(graph_factory=ExplodingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post("/api/runs", data={"query": "find papers"})
        assert response.status_code == 202
        run_id = response.json()["runId"]
        for _ in range(100):
            snapshot = (await client.get(f"/api/runs/{run_id}")).json()
            if snapshot["status"] != "running":
                break
            await asyncio.sleep(0.01)

    payload = json.dumps(snapshot)
    assert snapshot["status"] == "error"
    assert snapshot["error"] == "Could not understand the request. Check the request and start again."
    assert raw_failure not in payload
    assert "SYSTEM PROMPT" not in payload
    assert "sk-abcdefghijklmnop" not in payload
    assert "/srv/private" not in payload


@pytest.mark.asyncio
async def test_write_notes_exception_uses_provider_recovery_copy():
    class WriteNotesExplodingGraph:
        def stream(self, state, graph_config, stream_mode=None):
            yield {"router": {"intent": "search"}}
            yield {"write_notes": {"selected_papers": []}}
            raise RuntimeError("provider prompt=private details at /home/vha72/.cache/key")
            yield

    application = api.create_app(graph_factory=WriteNotesExplodingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post("/api/runs", data={"query": "find papers"})
        run_id = response.json()["runId"]
        for _ in range(100):
            snapshot = (await client.get(f"/api/runs/{run_id}")).json()
            if snapshot["status"] != "running":
                break
            await asyncio.sleep(0.01)

    assert snapshot["status"] == "error"
    assert snapshot["error"] == "Could not write structured notes. The language-model provider did not respond. Start again."
    assert "provider prompt" not in json.dumps(snapshot)
    assert "/home/vha72/.cache" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_multipart_parser_exception_returns_generic_public_detail(monkeypatch):
    raw_failure = "multipart parser leaked api_key=sk-abcdefghijklmnop at /tmp/private/form"

    async def broken_form(_request):
        raise RuntimeError(raw_failure)

    monkeypatch.setattr(api.Request, "form", broken_form)
    application = api.create_app(graph_factory=FakeGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://testserver") as client:
        response = await client.post("/api/runs", content=b"broken form", headers={"content-type": "multipart/form-data"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail == "Unable to parse the request form. Check the upload and try again."
    assert raw_failure not in response.text
    assert "sk-abcdefghijklmnop" not in response.text
    assert "/tmp/private" not in response.text
