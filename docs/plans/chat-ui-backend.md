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
| — | One real-model e2e on arXiv 2605.05737 + one topic query; compare latency, reasoning tokens and style metrics with the 2026-10-10 runs |
