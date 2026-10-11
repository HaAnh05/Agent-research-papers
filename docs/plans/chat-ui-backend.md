# Chat UI — backend and frontend plan

Status: approved 2026-10-11. Target UI: `docs/ui-mockup/research-scout-v3.html` (desktop only).

## Goal

One chat screen: landing → chat. Each run shows an expandable progress line (5 stages with real facts). The assistant answer is a short summary of what a researcher must know. `report.md` (full report) and BibTeX are one click away. A right panel shows the related-papers graph. The chat bar stays for follow-ups: quick answers from the existing report, or a new analysis run in the same thread.

## Decisions

| # | Decision |
|---|---|
| D1 | `answerSummary` is structured JSON blocks, rendered by the frontend |
| D2 | Summary fields (one paper): `tldr`, `problem`, `intuition` (core idea and why it works), `keyResults` + code-built table, `metrics`, `setup` (hardware or "paper không nêu"), `caveats`, `takeaway`. Several papers: `tldr`, cross-paper insights, one card per paper (`intuition`, `keyResult`) |
| D3 | Summary text comes from the existing overview LLM call (now JSON output, report language). No new LLM call |
| D4 | Report prose follows ASD-STE100 adapted to Vietnamese, including its sentence-length limits; terms are fixed by a `<terms>` block. `bench/verify.py` measures sentence length, bullets and passive markers |
| D5 | Threads persist to disk as JSON |
| D6 | Keep `main.py`; remove Streamlit `app.py` |
| D7 | Rewrite the frontend in place in `frontend/`; delete unused components |
| D8 | No new unit tests; existing tests are only adjusted or deleted when they cover removed code. Validation is one e2e run on arXiv 2605.05737 (plus one topic query) after all PRs are coded, with user approval |

## PRs

| PR | Content |
|---|---|
| P1 | Cleanup: remove summary-cards LLM call and its schema/API/UI fields; remove legacy single-call report path (`FINAL_REPORT_PROMPT`, `REPORT_PARALLEL_SECTIONS`); remove deprecated compare fields and merge intent `direct_compare` into `direct_read`; remove `PMRLGeneration`, `brief_summary`, `RouterDecision.reasoning`; remove `app.py` |
| P2 | Graph: PMRL LLM call moves from `web_enrich` to `write_notes`; after `read_paper`, `write_notes` ∥ `citation_graph` ∥ `web_enrich` fan in to `final_report`; 5 UI stages with Vietnamese labels and facts per stage |
| P3 | Summary and report: PMRL adds `intuition`; overview returns JSON (D2, D3); section/overview prompts get STE rules and `<terms>`; `answerSummary` + structured fields in the API (camelCase graph); fix `_merge_answer`; store `reportId` in state; verify.py style checks |
| P4 | Follow-ups: router intent `followup` answers from the thread's report and notes (1 streamed LLM call); endpoint to analyse a paper from the graph; thread context includes report content; threads persist to disk |
| P5 | React frontend per mockup v3 |
| P6 | Early summary, live timers, per-call LLM metrics (see below) |
| — | One real-model e2e on arXiv 2605.05737 + one topic query; compare latency, reasoning tokens and style metrics with the 2026-10-10 runs |

## P6: early summary, live timers, LLM metrics

Measured on arXiv 2210.03629 (8:38 in total): read 0:01, then write_notes 1:36 ∥ citation_graph 1:31 ∥ web_enrich 0:01, then final_report about 7:00 (the user sees the answer only after it). Goal: the answer appears right after the notes, the UI shows live time, and every LLM call has measured numbers.

| # | Decision |
|---|---|
| E1 | New node `summarize` = overview LLM call + `build_answer_summary` (code). It reads the PMRL notes only. State keys: `overview` (JSON, `{}` when the call failed, `None` before) and `answer_summary` |
| E2 | `final_report` reuses `state["overview"]` (no second overview call), writes the paper sections and `report.md`, and patches `answer_summary` with `patch_answer_summary()`: `foundations`, `actions.graphNodes`, `actions.bibtexCount`, `actions.reportId`. A state without `overview` (direct call, bench) still makes the overview call itself. Call count per run is unchanged: PMRL (n), overview (1), section (n) |
| E3 | Topology: LangGraph runs in supersteps, so plain top-level `write_notes -> summarize` would make `summarize` wait for `citation_graph` (checked with a toy graph). The chain therefore runs in one top-level node `notes_branch` that invokes an inner graph. `final_report` joins `[notes_branch, citation_graph, web_enrich]`. The API streams with `subgraphs=True` so the inner nodes emit their own step events |
| E4 | API: new SSE event `answer.summary` (`answerSummary`, `reportReady`, `citationGraph`, `bibtex`), sent when `summarize` ends (with the lead as `assistant.delta`) and again when the graph or BibTeX arrives later. `run.completed` carries the final values. `GET /api/runs/{id}` returns the partial `answerSummary` while running and `result.reportReady` |
| E5 | Stages (6): read_paper, write_notes, summarize ("Tóm tắt nhanh"), citation_graph, web_enrich, final_report ("Viết báo cáo đầy đủ"). New facts: `keyResultCount`, `resultRowCount`, `llmCalls` |
| E6 | UI: ticking step and run timers (client time of `step.started`), server `durationMs` on completion; answer renders on `answer.summary`; "Tải report.md" grey until `reportReady`; follow-ups disabled with a hint until then; mock transport replays the new order |
| E7 | LLM metrics: `tools.llm_provider.track_llm_call` records `ms` and the token counts the provider returns (omitted when absent) for router, eval_search, refine_query, PMRL, overview, section and followup. Stored in `node_timings[node].llmCalls`, exposed in `facts.llmCalls` and in `data/threads/*.json` (`runs[].llmCalls`) |
| E8 | Versions: `REPORT_VERSION` p12 -> p13 (cache key changed: overview JSON in the report key, separate `overview_` key). Thread file format version stays 1 (new optional `llmCalls`) |

Checks without a real model (D8 still holds: no new tests): `.venv/bin/python -m pytest -q`; `cd frontend && npm run build`; an offline run of the real graph with a fake chat model (read/graph/web stubbed) shows `summarize` starting when `write_notes` ends, while `citation_graph` is still running; `frontend/scripts/screenshots_p6.mjs` takes mock-mode screenshots.
