# Research Agent v2 — backend plan (no UI)

Status: approved 2026-10-10. Frontend is out of scope (separate rework).

## 1. Product

Input: a topic, or a list of arXiv IDs.

Output (API + CLI):

| Artifact | Content |
|---|---|
| Markdown report | Summary of insights · per-paper analysis (insights, metric meaning, setup/hardware, results table from TeX) · research context from the citation graph · conclusion · references |
| Structured JSON | Per paper: `insights`, `eval_settings[]` (benchmark, metric, metric meaning, value, baseline, `source_quote`), `setup` (hardware, config), official repo link |
| Citation graph JSON | `nodes` (paper, year, citation count, selected flag), `edges` (cites / co-citation / coupling) for a future UI |

Side product: a search-recall bench (Recall@5, Recall@20, latency) run before and after each search change.

## 2. Pipeline

```
router
 └─ search ─► local BM25 on HF metadata (3.1M papers, to 2026-08-27)
             + arXiv API for newer papers (merged)
     └─ read_paper ─► ar5iv HTML ──fail──► arXiv HTML ──fail──► PDF (current path)
         └─ extract ─► sections · tabular tables · setup/hardware · repo link · bibliography
             ├─ write_notes ─► PMRL + insights + eval_settings + setup (code checks numbers against source)
             ├─ citation_graph ─► outgoing: HTML bibliography · incoming + co-citation: OpenAlex (cached)
             └─ web_enrich ─► only when TeX has no repo link
                 └─ final_report ─► Markdown + JSON
```

Removed: `compare_benchmark` node and the report "So sánh" section.

## 3. Report format

```
# Research Report: <topic>
## 1. Tóm tắt              cross-paper insights
## 2. Phân tích bài báo     per paper: Vấn đề · Phương pháp · Kết quả chính · Hạn chế · (code: repo, links, BibTeX)
     Kết quả chính = Insight (LLM) + Đại lượng đo + Thiết lập/hardware + table (code) + Nhận xét (LLM)
## 3. Bối cảnh nghiên cứu   foundational and follow-up papers (citation graph)
## 4. Kết luận
## 5. Tài liệu tham khảo
```

## 4. Data

| Item | Size | Note |
|---|---|---|
| HF `secemp9/arxiv-complete` `metadata` + BM25 index | 1.64 GB + index | `data/arxiv/`, git-ignored |
| HTML and OpenAlex cache | small | arXiv rate limit applies |
| Recall gold set | ~200 of 570 cs.AI surveys 2021–2026, bibliography via ar5iv HTML | fetched once |

Not downloaded: HF `paper_text`, `latex`, `source`, `pdf`.

**Source policy (checked 2026-10-10):** `arxiv.org/robots.txt` has `Disallow: /e-print`, `Disallow: /src`, `Crawl-delay: 15` and allows `/html`, `/pdf`, `/abs`. `export.arxiv.org/robots.txt` disallows everything. `ar5iv.labs.arxiv.org/robots.txt` only disallows `/log/`. The pipeline therefore never calls `e-print`; TeX-derived content comes from ar5iv / arXiv HTML (LaTeXML output: sections, `ltx_tabular` tables, `ltx_bibitem` references). Read order: ar5iv → arXiv HTML → PDF. OpenAlex needs a free API key.

## 5. Decisions

| # | Decision |
|---|---|
| Q1 | Data: ar5iv/arXiv HTML + OpenAlex + HF `metadata` (no e-print: robots.txt) |
| Q4 | Code builds the results table; LLM writes only Insight and Nhận xét |
| Q5 | Missing hardware → write "paper không nêu" |
| Q6 | "3. Bối cảnh nghiên cứu" is part of the Markdown report |
| Q7 | API keeps `benchmark`, `comparisonArtifact` (null) and facts `comparisonAvailable` (false), `comparedPaperIds` ([]) until the UI rework |
| Q8 | Keep intent `direct_compare` |
| Q9 | Local index and arXiv API are merged |
| Q10 | Gold set: cs.AI surveys 2021–2026, Recall@5/@20 |
| Test | Offline tests only (unit, fixtures, mocked LLM) until all PRs are coded; one real-model validation pass at the end, after user approval |

## 6. Roadmap (one PR each)

| # | PR | Done when |
|---|---|---|
| 1 | Remove `compare_benchmark` | Tests pass; report has 4 sections; API returns `benchmark: null` |
| 2 | TeX extraction | On the 3 bench papers (offline fixtures): tables, setup, full methodology |
| 3 | PMRL schema + "Kết quả chính" format | Every table number matches the source; `bench/verify.py` passes on fixtures |
| 4 | Recall bench | Baseline Recall table for the current arXiv search |
| 5 | Local BM25 index | Recall and latency re-measured against the baseline |
| 6 | Citation graph + section 3 | Graph JSON; OpenAlex vs HTML bibliography match rate on ~20 papers |

Order: 1 alone · 2 → 3 · 4 → 5 · 6 after 2.
