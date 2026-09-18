from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
from httpx import ASGITransport, AsyncClient

import api
from state import PMRLNotes, PaperItem


async def wait_run(client: AsyncClient, run_id: str) -> dict:
    for _ in range(200):
        snapshot = (await client.get(f"/api/runs/{run_id}")).json()
        if snapshot["status"] != "running":
            return snapshot
        await asyncio.sleep(0.01)
    raise AssertionError("run did not settle")


class ContextGraph:
    states: list[dict] = []

    def stream(self, state, graph_config, stream_mode=None, version=None):
        type(self).states.append(dict(state))
        papers = [
            PaperItem(
                paper_id="arxiv_1111.11111",
                title="First paper",
                summary="First summary",
                arxiv_id="1111.11111",
                extracted_text="PRIVATE PDF TEXT",
                local_pdf_path="/tmp/private-first.pdf",
                notes=PMRLNotes(problem="P1 problem", method="P1 method", result="P1 result", limitation="P1 limit"),
                bibtex="@article{first, title={First}}",
            ),
            PaperItem(
                paper_id="arxiv_2222.22222",
                title="Second paper",
                summary="Second summary",
                arxiv_id="2222.22222",
                extracted_text="PRIVATE SECOND TEXT",
                local_pdf_path="/tmp/private-second.pdf",
                notes=PMRLNotes(problem="P2 problem", method="P2 method", result="P2 result", limitation="P2 limit"),
                bibtex="@article{second, title={Second}}",
            ),
        ]
        yield {"router": {"intent": "search"}}
        yield {"write_notes": {"selected_papers": papers, "benchmark_matrix": "Benchmark A vs B", "status": "success"}}
        yield {"final_report": {"selected_papers": papers, "final_report": "# Research answer\n\nDone.", "status": "success"}}


class V2Graph:
    calls: list[tuple[object, object]] = []

    def stream(self, state, graph_config, stream_mode=None, version=None):
        type(self).calls.append((stream_mode, version))
        yield {"type": "tasks", "data": {"id": "t1", "name": "final_report", "input": {"private": "RAW_TASK_INPUT"}}}
        yield {"type": "messages", "data": (SimpleNamespace(content="safe answer"), {"langgraph_node": "internal_node"})}
        yield {"type": "messages", "data": (SimpleNamespace(content="safe answer"), {"langgraph_node": "final_report"})}
        yield {"type": "tasks", "data": {"id": "t1", "name": "final_report", "result": {"private": "RAW_TASK_RESULT"}}}
        yield {"type": "updates", "data": {"final_report": {"final_report": "# Safe report", "status": "success"}}}


@pytest.mark.asyncio
async def test_threads_context_pmrl_and_run_isolation():
    ContextGraph.states = []
    app = api.create_app(graph_factory=ContextGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first_thread = (await client.post("/api/threads", json={"title": "Papers"})).json()["threadId"]
        first = await client.post(f"/api/threads/{first_thread}/messages", data={"message": "Compare papers"})
        assert first.status_code == 202
        run_one = await wait_run(client, first.json()["runId"])
        assert run_one["threadId"] == first_thread

        follow = await client.post(
            f"/api/threads/{first_thread}/messages",
            data={"message": "Explain Method of paper 2"},
        )
        assert follow.status_code == 202
        await wait_run(client, follow.json()["runId"])
        assert len(ContextGraph.states) >= 2
        context = ContextGraph.states[1]["conversation_context"]
        assert "2. Second paper" in context
        assert "Method: P2 method" in context
        assert "PRIVATE SECOND TEXT" not in context
        assert "/tmp/private-second.pdf" not in context

        second_thread = (await client.post("/api/threads", json={"title": "Isolated"})).json()["threadId"]
        isolated = await client.post(f"/api/threads/{second_thread}/messages", data={"message": "A separate run"})
        assert isolated.status_code == 202
        await wait_run(client, isolated.json()["runId"])
        detail = (await client.get(f"/api/threads/{first_thread}")).json()
        other = (await client.get(f"/api/threads/{second_thread}")).json()
        assert len(detail["runIds"]) == 2
        assert len(other["runIds"]) == 1
        assert len(other["messages"]) == 2


@pytest.mark.asyncio
async def test_v2_events_filter_raw_task_payload_and_stream_answer():
    V2Graph.calls = []
    app = api.create_app(graph_factory=V2Graph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/runs", data={"query": "answer"})
        assert response.status_code == 202
        run_id = response.json()["runId"]
        snapshot = await wait_run(client, run_id)
        payload = json.dumps(snapshot, ensure_ascii=False)
        assert snapshot["answer"] == "safe answer"
        assert "RAW_TASK_INPUT" not in payload
        assert "RAW_TASK_RESULT" not in payload
        event_types = [event["type"] for event in snapshot["traceEvents"]]
        assert "assistant.delta" in event_types
        assert "assistant.completed" in event_types
        assert all(event["progress"] == 0 for event in snapshot["traceEvents"] if event["type"].startswith("step."))
        assert V2Graph.calls[0] == (["tasks", "updates", "messages"], "v2")

        replay = await client.get(f"/api/runs/{run_id}/events?after_seq=1")
        assert replay.status_code == 200
        assert replay.text.count("event: snapshot") == 1
        assert "RAW_TASK_INPUT" not in replay.text


@pytest.mark.asyncio
async def test_provider_exception_masks_key_and_server_path():
    class FailingGraph:
        def stream(self, state, graph_config, stream_mode=None, version=None):
            raise RuntimeError("api_key=sk-abcdefghijklmnop at /tmp/private/uploads/paper.pdf")
            yield  # make this a streaming fake graph

    app = api.create_app(graph_factory=FailingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/runs", data={"query": "explain"})
        snapshot = await wait_run(client, response.json()["runId"])
        payload = json.dumps(snapshot)
        assert snapshot["status"] == "error"
        assert "sk-abcdefghijklmnop" not in payload
        assert "/tmp/private" not in payload


@pytest.mark.asyncio
async def test_thread_messages_share_global_active_run_gate():
    started = threading.Event()
    release = threading.Event()

    class BlockingGraph(ContextGraph):
        def stream(self, state, graph_config, stream_mode=None, version=None):
            started.set()
            release.wait(timeout=2)
            yield {"router": {"intent": "search"}}
            yield {"final_report": {"status": "success", "final_report": "# Done"}}

    app = api.create_app(graph_factory=BlockingGraph, registry_instance=api.RunRegistry())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first_thread = (await client.post("/api/threads")).json()["threadId"]
        second_thread = (await client.post("/api/threads")).json()["threadId"]
        first = await client.post(f"/api/threads/{first_thread}/messages", data={"message": "first"})
        assert first.status_code == 202
        assert started.wait(timeout=1)
        second = await client.post(f"/api/threads/{second_thread}/messages", data={"message": "second"})
        assert second.status_code == 409
        assert second.json()["detail"]["activeRunId"] == first.json()["runId"]
        release.set()
        await wait_run(client, first.json()["runId"])
